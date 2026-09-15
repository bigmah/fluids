//! Spray and the particle diagnostic, built on the GPU from the particles a
//! [`GpuFluid`] keeps there, straight into a mesh's vertex buffer. The CPU
//! solver draws the same thing as one entity per particle, which at
//! `slab.toml`'s 1.3 million particles is more than the ECS wants to carry.

use crate::gpu::{Binding, Kernel, cast};
use crate::sim_gpu::{GpuFluid, GpuScan, dispatch, staging_buffer, storage_buffer};
use bevy::render::render_resource::{
    BindingResource, Buffer, BufferDescriptor, BufferUsages, ComputePassDescriptor,
};
use bytemuck::{Pod, Zeroable};

/// Vertices per droplet: an octahedron, unindexed.
pub const DROPLET_VERTICES: u32 = 24;

/// Bytes per vertex in the spray mesh: position, normal.
pub const VERTEX_BYTES: u64 = 24;

/// `Params` in `spray.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SprayParams {
    n: u32,
    alpha: f32,
    radius: f32,
    scale: f32,
    all: u32,
    stride: u32,
    capacity: u32,
    vertex_start: u32,
}

const _: () = assert!(size_of::<SprayParams>() == 32);

const PARAMS: u32 = 0;
const PREVIOUS: u32 = 1;
const POSITIONS: u32 = 2;
const VELOCITIES: u32 = 3;
const SPRAY: u32 = 4;
const OFFSETS: u32 = 5;
const VERTICES: u32 = 6;
const DRAWN: u32 = 7;

/// How a rebuild draws the particles.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SprayStyle {
    /// Radius of a droplet at reference size.
    pub radius: f32,
    /// `render.particle_scale` over its reference, 2.4.
    pub scale: f32,
    /// Every `stride`th particle at rest size instead of the spray.
    pub diagnostic: Option<u32>,
}

pub struct GpuSpray {
    kernels: [Kernel; 3],
    params: Buffer,
    offsets: Buffer,
    scan: GpuScan,
    drawn: Buffer,
    total: Buffer,
    n: u32,
    revision: u64,
}

impl GpuSpray {
    pub fn new(fluid: &GpuFluid) -> Self {
        let gpu = fluid.gpu();
        let module = gpu.module("spray.wgsl", include_str!("spray.wgsl"));
        let params = (
            PARAMS,
            Binding::Uniform {
                size: size_of::<SprayParams>() as u64,
                dynamic: false,
            },
        );
        let storage = Binding::Storage;
        let kernels = [
            gpu.kernel(
                &module,
                "select_particles",
                &[params, (SPRAY, storage), (OFFSETS, storage)],
            ),
            gpu.kernel(
                &module,
                "place_droplets",
                &[
                    params,
                    (PREVIOUS, storage),
                    (POSITIONS, storage),
                    (VELOCITIES, storage),
                    (SPRAY, storage),
                    (OFFSETS, storage),
                    (VERTICES, storage),
                ],
            ),
            gpu.kernel(
                &module,
                "clear_droplets",
                &[
                    params,
                    (OFFSETS, storage),
                    (VERTICES, storage),
                    (DRAWN, storage),
                ],
            ),
        ];
        let n = fluid.len() as u32;
        let size = (n + 1).next_power_of_two();
        let offsets = storage_buffer(gpu, "droplet offsets", 4 * size as u64);
        Self {
            kernels,
            params: gpu.device.create_buffer(&BufferDescriptor {
                label: Some("spray params"),
                size: size_of::<SprayParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            scan: fluid.scan(&offsets, size),
            offsets,
            drawn: storage_buffer(gpu, "droplets drawn", 4),
            total: staging_buffer(gpu, "droplet count", 4),
            n,
            revision: fluid.revision(),
        }
    }

    /// Whether `fluid`'s buffers are still the ones this was built over.
    pub fn matches(&self, fluid: &GpuFluid) -> bool {
        self.revision == fluid.revision() && self.n == fluid.len() as u32
    }

    /// Writes a droplet per visible particle into `capacity` droplets' worth of
    /// `vertices` from vertex `start`, and waits for it. Returns how many
    /// droplets were wanted, which can be more than fit.
    pub fn rebuild(
        &self,
        fluid: &GpuFluid,
        alpha: f32,
        style: SprayStyle,
        vertices: &Buffer,
        start: u32,
        capacity: u32,
    ) -> u32 {
        let gpu = fluid.gpu();
        let params = SprayParams {
            n: self.n,
            alpha,
            radius: style.radius,
            scale: style.scale,
            all: style.diagnostic.is_some() as u32,
            stride: style.diagnostic.unwrap_or(1).max(1),
            capacity,
            vertex_start: start,
        };
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        fn whole(buffer: &Buffer) -> BindingResource<'_> {
            buffer.as_entire_binding()
        }
        let groups = [
            gpu.bind_group(
                &self.kernels[0],
                &[
                    (PARAMS, whole(&self.params)),
                    (SPRAY, whole(fluid.spray())),
                    (OFFSETS, whole(&self.offsets)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[1],
                &[
                    (PARAMS, whole(&self.params)),
                    (PREVIOUS, whole(fluid.previous())),
                    (POSITIONS, whole(fluid.positions())),
                    (VELOCITIES, whole(fluid.velocities())),
                    (SPRAY, whole(fluid.spray())),
                    (OFFSETS, whole(&self.offsets)),
                    (VERTICES, whole(vertices)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[2],
                &[
                    (PARAMS, whole(&self.params)),
                    (OFFSETS, whole(&self.offsets)),
                    (VERTICES, whole(vertices)),
                    (DRAWN, whole(&self.drawn)),
                ],
            ),
        ];
        let mut encoder = fluid.encoder();
        encoder.clear_buffer(&self.offsets, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            let run = |k: usize, pass: &mut bevy::render::render_resource::ComputePass, count| {
                pass.set_pipeline(&self.kernels[k].pipeline);
                pass.set_bind_group(0, &groups[k], &[]);
                dispatch(pass, count);
            };
            run(0, &mut pass, self.n);
            fluid.run_scan(&mut pass, &self.scan);
            run(1, &mut pass, self.n);
            run(2, &mut pass, capacity);
        }
        let total_at = 4 * self.n as u64;
        encoder.copy_buffer_to_buffer(&self.offsets, total_at, &self.drawn, 0, 4);
        encoder.copy_buffer_to_buffer(&self.offsets, total_at, &self.total, 0, 4);
        gpu.queue.submit([encoder.finish()]);
        gpu.read(&self.total, |bytes| cast::<u32>(bytes)[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::gpu::Gpu;
    use crate::sim::Fluid;
    use bevy::math::Vec3;

    fn droplets(gpu: &Gpu, solver: &GpuFluid, style: SprayStyle, capacity: u32) -> (u32, Vec<f32>) {
        let spray = GpuSpray::new(solver);
        let vertices = storage_buffer(
            gpu,
            "test droplets",
            VERTEX_BYTES * (DROPLET_VERTICES * capacity) as u64,
        );
        let wanted = spray.rebuild(solver, 1.0, style, &vertices, 0, capacity);
        let staging = staging_buffer(gpu, "test readback", vertices.size());
        let mut encoder = solver.encoder();
        encoder.copy_buffer_to_buffer(&vertices, 0, &staging, 0, None);
        gpu.queue.submit([encoder.finish()]);
        (
            wanted,
            gpu.read(&staging, |bytes| cast::<f32>(bytes).into_owned()),
        )
    }

    /// Droplets go where the spray is, closed and facing outwards, and the
    /// diagnostic places one at every sampled particle.
    #[test]
    fn droplets_mark_the_spray() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let config = Config::default();
        let mut fluid = Fluid::new(config.fluid_params());
        config.reset_fluid(&mut fluid);
        for (i, spray) in fluid.spray.iter_mut().enumerate() {
            *spray = if i % 5 == 0 { 0.8 } else { 0.05 };
        }
        for (i, v) in fluid.vel.iter_mut().enumerate() {
            *v = Vec3::new(0.0, 0.0, 600.0 * (i % 3) as f32);
        }
        let solver = GpuFluid::new(gpu.clone(), &fluid);
        let style = SprayStyle {
            radius: 3.0,
            scale: 1.0,
            diagnostic: None,
        };
        let expected: Vec<usize> = (0..fluid.len()).filter(|i| i % 5 == 0).collect();
        let (wanted, raw) = droplets(&gpu, &solver, style, expected.len() as u32);
        assert_eq!(wanted as usize, expected.len());
        for (k, &i) in expected.iter().enumerate() {
            let vertex = |m: usize| {
                let at = (k * DROPLET_VERTICES as usize + m) * 6;
                (
                    Vec3::from_slice(&raw[at..at + 3]),
                    Vec3::from_slice(&raw[at + 3..at + 6]),
                )
            };
            let centre = fluid.pos[i];
            let size = (0.3 + 0.8 * 0.45) * 3.0;
            let stretch = (fluid.vel[i].length() / 300.0).clamp(1.0, 2.8);
            let mut reach = 0.0f32;
            for face in 0..8 {
                let [(a, na), (b, _), (c, _)] = [0, 1, 2].map(|m| vertex(face * 3 + m));
                let outward = (b - a).cross(c - a).dot((a + b + c) / 3.0 - centre);
                assert!(outward > 0.0, "droplet {k} face {face} faces inwards");
                assert!((na.length() - 1.0).abs() < 1e-4);
                for corner in [a, b, c] {
                    reach = reach.max((corner - centre).length());
                }
            }
            assert!(
                (reach - size * stretch).abs() < 1e-3,
                "droplet {k} reaches {reach}, expected {}",
                size * stretch
            );
        }

        // The diagnostic ignores spray and samples by stride; capacity caps it.
        let diagnostic = SprayStyle {
            diagnostic: Some(4),
            ..style
        };
        let (wanted, raw) = droplets(&gpu, &solver, diagnostic, 16);
        assert_eq!(wanted as usize, fluid.len().div_ceil(4));
        let first = Vec3::from_slice(&raw[0..3]);
        assert!((first - fluid.pos[0]).length() <= 3.0 + 1e-3);
    }
}
