//! The surface in `surface.rs`, rebuilt on the GPU from the particles a
//! [`GpuFluid`] keeps there. Nothing comes back to the CPU but a triangle
//! count: the density volume and the triangles are written straight into
//! the texture and vertex buffer the water is drawn from.

use crate::gpu::{Binding, Gpu, Kernel, cast};
use crate::sim::Bounds;
use crate::sim_gpu::{GpuFluid, GpuGrid, GpuScan, dispatch, staging_buffer, storage_buffer};
use crate::surface::SurfaceLayout;
use bevy::render::render_resource::{
    BindGroup, BindingResource, Buffer, BufferDescriptor, BufferUsages, ComputePassDescriptor,
    TextureFormat, TextureView,
};
use bytemuck::{Pod, Zeroable};

/// The format of the density volume the water shader samples.
pub const DENSITY_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// Bytes per vertex in the water mesh: position, normal, colour.
pub const VERTEX_BYTES: u64 = 40;

/// `Params` in `surface.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    origin: [f32; 4],
    grid_origin: [f32; 4],
    n: u32,
    alpha: f32,
    radius2: f32,
    reference: f32,
    voxels_x: u32,
    voxels_y: u32,
    voxels_z: u32,
    voxel_count: u32,
    dims_x: i32,
    dims_y: i32,
    dims_z: i32,
    cube_count: u32,
    vertex_start: u32,
    vertex_capacity: u32,
    _pad: [u32; 2],
}

const _: () = assert!(size_of::<SurfaceParams>() == 96);

const PARAMS: u32 = 0;
const PREVIOUS: u32 = 1;
const POSITIONS: u32 = 2;
const FOAM: u32 = 3;
const INTERP: u32 = 4;
const GRID_START: u32 = 5;
const SORTED: u32 = 6;
const SAMPLES: u32 = 7;
const SAMPLE_FOAM: u32 = 8;
const DENSITY: u32 = 9;
const OFFSETS: u32 = 10;
const VERTICES: u32 = 11;
const DRAWN: u32 = 12;

/// Where a rebuild writes: a range of a vertex buffer laid out as the water
/// mesh, and a volume texture as large as the voxel grid.
pub struct SurfaceTarget<'a> {
    pub vertices: &'a Buffer,
    /// First vertex of the range, in vertices.
    pub start: u32,
    /// Vertices in the range.
    pub capacity: u32,
    pub density: &'a TextureView,
}

pub struct GpuSurface {
    kernels: [Kernel; 5],
    params: Buffer,
    interp: Buffer,
    grid: GpuGrid,
    samples: Buffer,
    sample_foam: Buffer,
    offsets: Buffer,
    scan: GpuScan,
    drawn: Buffer,
    total: Buffer,
    base: SurfaceParams,
    /// The solver buffers this was built against; see [`GpuFluid::revision`].
    revision: u64,
}

impl GpuSurface {
    /// The reconstruction `surface` lays out, over the particles in `fluid`.
    pub fn new(fluid: &GpuFluid, surface: &SurfaceLayout, bounds: Bounds) -> Self {
        let gpu = fluid.gpu();
        let module = gpu.module("surface.wgsl", include_str!("surface.wgsl"));
        let params = Binding::Uniform {
            size: size_of::<SurfaceParams>() as u64,
            dynamic: false,
        };
        let storage = Binding::Storage;
        let kernels = [
            gpu.kernel(
                &module,
                "interpolate",
                &[
                    (PARAMS, params),
                    (PREVIOUS, storage),
                    (POSITIONS, storage),
                    (FOAM, storage),
                    (INTERP, storage),
                ],
            ),
            gpu.kernel(
                &module,
                "sample_density",
                &[
                    (PARAMS, params),
                    (INTERP, storage),
                    (GRID_START, storage),
                    (SORTED, storage),
                    (SAMPLES, storage),
                    (SAMPLE_FOAM, storage),
                    (DENSITY, Binding::VolumeOut(DENSITY_FORMAT)),
                ],
            ),
            gpu.kernel(
                &module,
                "classify",
                &[(PARAMS, params), (SAMPLES, storage), (OFFSETS, storage)],
            ),
            gpu.kernel(
                &module,
                "emit",
                &[
                    (PARAMS, params),
                    (SAMPLES, storage),
                    (SAMPLE_FOAM, storage),
                    (OFFSETS, storage),
                    (VERTICES, storage),
                ],
            ),
            gpu.kernel(
                &module,
                "clear_tail",
                &[
                    (PARAMS, params),
                    (OFFSETS, storage),
                    (VERTICES, storage),
                    (DRAWN, storage),
                ],
            ),
        ];

        let n = fluid.len() as u32;
        let [vx, vy, vz] = surface.dims.map(|d| d as u32);
        let voxel_count = vx * vy * vz;
        let cube_count = (vx - 1) * (vy - 1) * (vz - 1);
        let offsets_size = (cube_count + 1).next_power_of_two();
        let interp = storage_buffer(gpu, "surface particles", 16 * n as u64);
        let grid = fluid.grid(&interp, n, bounds, surface.radius);
        let offsets = storage_buffer(gpu, "triangle offsets", 4 * offsets_size as u64);
        let scan = fluid.scan(&offsets, offsets_size);
        let dims = {
            let size = bounds.size();
            let cell = surface.radius;
            [
                (size.x / cell).ceil() as i32 + 1,
                (size.y / cell).ceil() as i32 + 1,
                (size.z / cell).ceil() as i32 + 1,
            ]
        };
        Self {
            kernels,
            params: gpu.device.create_buffer(&BufferDescriptor {
                label: Some("surface params"),
                size: size_of::<SurfaceParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            interp,
            grid,
            samples: storage_buffer(gpu, "surface samples", 16 * voxel_count as u64),
            sample_foam: storage_buffer(gpu, "surface foam", 4 * voxel_count as u64),
            offsets,
            scan,
            drawn: storage_buffer(gpu, "triangles drawn", 4),
            total: staging_buffer(gpu, "triangle count", 4),
            base: SurfaceParams {
                origin: surface.origin.extend(surface.cell).to_array(),
                grid_origin: bounds.min.extend(surface.radius).to_array(),
                n,
                alpha: 1.0,
                radius2: surface.radius * surface.radius,
                reference: surface.reference,
                voxels_x: vx,
                voxels_y: vy,
                voxels_z: vz,
                voxel_count,
                dims_x: dims[0],
                dims_y: dims[1],
                dims_z: dims[2],
                cube_count,
                vertex_start: 0,
                vertex_capacity: 0,
                _pad: [0; 2],
            },
            revision: fluid.revision(),
        }
    }

    /// Whether `fluid`'s buffers are still the ones this was built over.
    pub fn matches(&self, fluid: &GpuFluid) -> bool {
        self.revision == fluid.revision() && self.base.n == fluid.len() as u32
    }

    /// Rebuilds the surface from the particles `alpha` of the way from where
    /// the last step started to where it ended, and waits for it. Returns the
    /// triangles the surface needed, which can be more than `target` holds.
    pub fn rebuild(&self, fluid: &GpuFluid, alpha: f32, target: &SurfaceTarget) -> u32 {
        let gpu = fluid.gpu();
        let params = SurfaceParams {
            alpha,
            vertex_start: target.start,
            vertex_capacity: target.capacity,
            ..self.base
        };
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let groups = self.bind_groups(gpu, fluid, target);
        let b = &self.base;
        let mut encoder = fluid.encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, &groups, 0, b.n);
        }
        fluid.rebuild_grid(&mut encoder, &self.grid);
        encoder.clear_buffer(&self.offsets, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, &groups, 1, b.voxel_count);
            self.run(&mut pass, &groups, 2, b.cube_count);
            fluid.run_scan(&mut pass, &self.scan);
            self.run(&mut pass, &groups, 3, b.cube_count);
            self.run(&mut pass, &groups, 4, target.capacity);
        }
        let total_at = 4 * b.cube_count as u64;
        encoder.copy_buffer_to_buffer(&self.offsets, total_at, &self.drawn, 0, 4);
        encoder.copy_buffer_to_buffer(&self.offsets, total_at, &self.total, 0, 4);
        gpu.queue.submit([encoder.finish()]);
        gpu.read(&self.total, |bytes| cast::<u32>(bytes)[0])
    }

    fn run(
        &self,
        pass: &mut bevy::render::render_resource::ComputePass,
        groups: &[BindGroup; 5],
        k: usize,
        count: u32,
    ) {
        pass.set_pipeline(&self.kernels[k].pipeline);
        pass.set_bind_group(0, &groups[k], &[]);
        dispatch(pass, count);
    }

    fn bind_groups(&self, gpu: &Gpu, fluid: &GpuFluid, target: &SurfaceTarget) -> [BindGroup; 5] {
        fn whole(buffer: &Buffer) -> BindingResource<'_> {
            buffer.as_entire_binding()
        }
        let params = || (PARAMS, whole(&self.params));
        [
            gpu.bind_group(
                &self.kernels[0],
                &[
                    params(),
                    (PREVIOUS, whole(fluid.previous())),
                    (POSITIONS, whole(fluid.positions())),
                    (FOAM, whole(fluid.foam())),
                    (INTERP, whole(&self.interp)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[1],
                &[
                    params(),
                    (INTERP, whole(&self.interp)),
                    (GRID_START, whole(self.grid.starts())),
                    (SORTED, whole(self.grid.sorted())),
                    (SAMPLES, whole(&self.samples)),
                    (SAMPLE_FOAM, whole(&self.sample_foam)),
                    (DENSITY, BindingResource::TextureView(target.density)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[2],
                &[
                    params(),
                    (SAMPLES, whole(&self.samples)),
                    (OFFSETS, whole(&self.offsets)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[3],
                &[
                    params(),
                    (SAMPLES, whole(&self.samples)),
                    (SAMPLE_FOAM, whole(&self.sample_foam)),
                    (OFFSETS, whole(&self.offsets)),
                    (VERTICES, whole(target.vertices)),
                ],
            ),
            gpu.bind_group(
                &self.kernels[4],
                &[
                    params(),
                    (OFFSETS, whole(&self.offsets)),
                    (VERTICES, whole(target.vertices)),
                    (DRAWN, whole(&self.drawn)),
                ],
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::sim::Fluid;
    use crate::surface::Surface;
    use bevy::math::Vec3;
    use bevy::mesh::Mesh;
    use bevy::render::render_resource::{
        Extent3d, TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };
    use std::collections::HashMap;

    struct Rebuilt {
        triangles: u32,
        /// Non-degenerate triangles: three positions, then three normals.
        faces: Vec<[Vec3; 6]>,
        samples: Vec<[f32; 4]>,
        foam: Vec<f32>,
    }

    fn rebuild_on_gpu(gpu: &Gpu, fluid: &Fluid, surface: &Surface, capacity: u32) -> Rebuilt {
        let solver = GpuFluid::new(gpu.clone(), fluid);
        let on_gpu = GpuSurface::new(&solver, &surface.layout(), fluid.params.bounds);
        let vertices = storage_buffer(gpu, "test vertices", VERTEX_BYTES * capacity as u64);
        let [x, y, z] = surface.dims.map(|d| d as u32);
        let texture = gpu.device.create_texture(&TextureDescriptor {
            label: Some("test density"),
            size: Extent3d {
                width: x,
                height: y,
                depth_or_array_layers: z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: DENSITY_FORMAT,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let target = SurfaceTarget {
            vertices: &vertices,
            start: 0,
            capacity,
            density: &view,
        };
        let triangles = on_gpu.rebuild(&solver, 1.0, &target);
        let fetch = |buffer: &Buffer| {
            let staging = staging_buffer(gpu, "test readback", buffer.size());
            let mut encoder = solver.encoder();
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, None);
            gpu.queue.submit([encoder.finish()]);
            gpu.read(&staging, |bytes| bytes.to_vec())
        };
        let raw: Vec<f32> = cast(&fetch(&vertices)).into_owned();
        let faces = raw
            .chunks_exact(30)
            .take(triangles.min(capacity / 3) as usize)
            .filter_map(|t| {
                let v = |k: usize, o: usize| {
                    Vec3::new(t[k * 10 + o], t[k * 10 + o + 1], t[k * 10 + o + 2])
                };
                let face = [v(0, 0), v(1, 0), v(2, 0), v(0, 3), v(1, 3), v(2, 3)];
                (face[0] != face[1] || face[0] != face[2]).then_some(face)
            })
            .collect();
        Rebuilt {
            triangles,
            faces,
            samples: cast(&fetch(&on_gpu.samples)).into_owned(),
            foam: cast(&fetch(&on_gpu.sample_foam)).into_owned(),
        }
    }

    fn cpu_faces(mesh: &Mesh) -> Vec<[Vec3; 6]> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .unwrap()
            .as_float3()
            .unwrap();
        positions
            .chunks_exact(3)
            .zip(normals.chunks_exact(3))
            .map(|(p, n)| [p[0], p[1], p[2], n[0], n[1], n[2]].map(Vec3::from))
            .collect()
    }

    /// The CPU reconstruction and the GPU one agree voxel for voxel and
    /// triangle for triangle, on a disordered dam break and a flume mid-wave.
    #[test]
    fn gpu_surface_matches_the_cpu() {
        let Some(gpu) = Gpu::headless() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let mut slab = Config::load("slab.toml").unwrap();
        slab.world.depth = 8.0 * slab.fluid.spacing;
        for (config, steps) in [(Config::default(), 45), (slab, 120)] {
            let mut fluid = Fluid::new(config.fluid_params());
            config.reset_fluid(&mut fluid);
            for _ in 0..steps {
                fluid.step(1.0 / 60.0);
            }
            // Foam gives the colour channel something to carry.
            for (i, foam) in fluid.foam.iter_mut().enumerate() {
                *foam = (i % 7) as f32 / 7.0;
            }
            let mut surface = Surface::new(
                config.bounds(),
                config.fluid.spacing,
                config.render.surface_resolution,
            );
            let mesh = surface.rebuild(&fluid);
            let cpu = cpu_faces(&mesh);
            let gpu_result =
                rebuild_on_gpu(&gpu, &fluid, &surface, 3 * surface.triangles as u32 * 2);

            for ((density, normal, foam), (s, f)) in surface
                .samples()
                .zip(gpu_result.samples.iter().zip(&gpu_result.foam))
            {
                assert!(
                    (density - s[3]).abs() <= 1e-4 * density.max(1.0),
                    "density {density} vs {}",
                    s[3]
                );
                assert!(
                    (normal - Vec3::new(s[0], s[1], s[2])).length()
                        <= 1e-3 * normal.length().max(1.0)
                );
                assert!((foam - f).abs() <= 1e-3, "foam {foam} vs {f}");
            }

            // Degenerate triangles are dropped on the CPU and collapsed on the
            // GPU, so the counts differ by those alone.
            assert!(gpu_result.triangles as usize >= cpu.len());
            let key = |face: &[Vec3; 6]| {
                let c = (face[0] + face[1] + face[2]) / 3.0;
                (c * 10.0).round().as_ivec3().to_array()
            };
            let mut by_centre: HashMap<[i32; 3], Vec<&[Vec3; 6]>> = HashMap::new();
            for face in &gpu_result.faces {
                by_centre.entry(key(face)).or_default().push(face);
            }
            let near = |a: &[Vec3; 6], b: &[Vec3; 6]| {
                (0..3).all(|k| (a[k] - b[k]).length() < 1e-2)
                    && (3..6).all(|k| (a[k] - b[k]).length() < 1e-3)
            };
            let mut matched = 0;
            for face in &cpu {
                let c = key(face);
                let found = (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| {
                        (-1..=1).any(|dz| {
                            by_centre
                                .get(&[c[0] + dx, c[1] + dy, c[2] + dz])
                                .is_some_and(|list| list.iter().any(|g| near(face, g)))
                        })
                    })
                });
                matched += found as usize;
            }
            assert!(
                matched == cpu.len() && gpu_result.faces.len() <= cpu.len() + cpu.len() / 1000,
                "{} of {} CPU triangles found among {} on the GPU",
                matched,
                cpu.len(),
                gpu_result.faces.len()
            );
        }
    }

    /// What a frame's surface costs at full scale. Ignored; run with
    /// `cargo test --release surface_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn surface_cost() {
        let Some(gpu) = Gpu::headless() else {
            return;
        };
        for name in ["config.toml", "wide.toml", "slab.toml"] {
            let config = Config::load(name).unwrap();
            let mut fluid = Fluid::new(config.fluid_params());
            config.reset_fluid(&mut fluid);
            let mut solver = GpuFluid::new(gpu.clone(), &fluid);
            for _ in 0..3 {
                solver.step(&mut fluid, 1.0 / 60.0);
            }
            let layout = SurfaceLayout::new(
                config.bounds(),
                config.fluid.spacing,
                config.render.surface_resolution,
            );
            let surface = GpuSurface::new(&solver, &layout, config.bounds());
            let [x, y, z] = layout.dims.map(|d| d as u32);
            let texture = gpu.device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: x,
                    height: y,
                    depth_or_array_layers: z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: DENSITY_FORMAT,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            let sizing = storage_buffer(&gpu, "sizing", VERTEX_BYTES);
            let needed = surface.rebuild(
                &solver,
                1.0,
                &SurfaceTarget {
                    vertices: &sizing,
                    start: 0,
                    capacity: 0,
                    density: &view,
                },
            );
            let vertices = storage_buffer(&gpu, "vertices", VERTEX_BYTES * 3 * needed as u64);
            let target = SurfaceTarget {
                vertices: &vertices,
                start: 0,
                capacity: 3 * needed,
                density: &view,
            };
            surface.rebuild(&solver, 0.5, &target);
            let start = std::time::Instant::now();
            for _ in 0..10 {
                surface.rebuild(&solver, 0.5, &target);
            }
            let gpu_ms = start.elapsed().as_secs_f32() * 100.0;
            solver.download(&mut fluid);
            let mut cpu = Surface::new(
                config.bounds(),
                config.fluid.spacing,
                config.render.surface_resolution,
            );
            let start = std::time::Instant::now();
            for _ in 0..3 {
                cpu.rebuild_interpolated(&fluid, 0.5);
            }
            let cpu_ms = start.elapsed().as_secs_f32() * 1000.0 / 3.0;
            println!(
                "{name:>12}: {:>8} particles  {:>9} voxels  {:>8} triangles  cpu {cpu_ms:>6.1} ms  gpu {gpu_ms:>5.1} ms",
                fluid.len(),
                x * y * z,
                needed,
            );
        }
    }
}
