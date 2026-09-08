//! Reconstruct a 3D isosurface from the particle density. Unlike a height field,
//! this preserves overhangs, disconnected splashes, and the air inside a barrel.

use crate::sim::{Bounds, Fluid, Grid};
use bevy::{asset::RenderAssetUsages, mesh::PrimitiveTopology, prelude::*};
use rayon::prelude::*;

pub const ISO: f32 = 0.45;

#[derive(Clone, Copy, Default)]
struct Sample {
    density: f32,
    normal: Vec3,
    foam: f32,
}

#[derive(Clone, Copy)]
struct Vertex {
    p: Vec3,
    n: Vec3,
    foam: f32,
}

#[derive(Default)]
struct Triangles {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}

impl Triangles {
    fn push(&mut self, a: Vertex, mut b: Vertex, mut c: Vertex) {
        let face = (b.p - a.p).cross(c.p - a.p);
        if face.length_squared() < 1e-10 {
            return;
        }
        if face.dot(a.n + b.n + c.n) < 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        for v in [a, b, c] {
            self.positions.push(v.p.to_array());
            self.normals.push(v.n.normalize_or_zero().to_array());
            self.colors.push([v.foam, 0.0, 0.0, 1.0]);
        }
    }
}

#[derive(Resource)]
pub struct Surface {
    pub origin: Vec3,
    pub cell: f32,
    pub dims: [usize; 3],
    radius: f32,
    reference: f32,
    grid: Grid,
    samples: Vec<Sample>,
    pub triangles: usize,
}

impl Surface {
    pub fn new(bounds: Bounds, spacing: f32, resolution: f32) -> Self {
        let radius = spacing * 2.4;
        let cell = spacing * resolution;
        let origin = bounds.min - Vec3::splat(radius);
        let extent = bounds.size() + Vec3::splat(2.0 * radius);
        let dims = (extent / cell).ceil().as_uvec3() + UVec3::ONE;
        let dims = [dims.x as usize, dims.y as usize, dims.z as usize];
        let mut reference = 0.0;
        for z in -3..=3 {
            for y in -3..=3 {
                for x in -3..=3 {
                    let r2 = Vec3::new(x as f32, y as f32, z as f32).length_squared()
                        * spacing
                        * spacing;
                    reference += (1.0 - r2 / (radius * radius)).max(0.0).powi(3);
                }
            }
        }
        Self {
            origin,
            cell,
            dims,
            radius,
            reference,
            grid: Grid::new(bounds, radius),
            samples: vec![Sample::default(); dims.iter().product()],
            triangles: 0,
        }
    }

    pub fn texture_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .map(|s| (s.density.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }

    #[cfg(test)]
    pub fn density_at(&self, p: Vec3) -> f32 {
        let v = ((p - self.origin) / self.cell).round().as_uvec3();
        self.samples[v.x as usize + self.dims[0] * (v.y as usize + self.dims[1] * v.z as usize)]
            .density
    }

    pub fn rebuild(&mut self, fluid: &Fluid) -> Mesh {
        self.rebuild_interpolated(fluid, 1.0)
    }

    pub fn rebuild_interpolated(&mut self, fluid: &Fluid, alpha: f32) -> Mesh {
        let positions: Vec<Vec3> = fluid
            .previous_pos
            .iter()
            .zip(&fluid.pos)
            .map(|(a, b)| a.lerp(*b, alpha))
            .collect();
        self.grid.rebuild(&positions);
        let [nx, ny, nz] = self.dims;
        let plane = nx * ny;
        let (origin, cell, radius2, reference) =
            (self.origin, self.cell, self.radius.powi(2), self.reference);
        let grid = &self.grid;
        self.samples
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let p = origin
                    + Vec3::new((i % nx) as f32, ((i / nx) % ny) as f32, (i / plane) as f32) * cell;
                let mut sample = Sample::default();
                grid.for_each_candidate(p, |j| {
                    let r = p - positions[j as usize];
                    let q = 1.0 - r.length_squared() / radius2;
                    if q > 0.0 {
                        let w = q * q * q;
                        sample.density += w;
                        sample.normal += r * (q * q);
                        sample.foam += fluid.foam[j as usize] * w;
                    }
                });
                sample.foam /= sample.density.max(1e-6);
                sample.density /= reference;
                *out = sample;
            });

        let samples = &self.samples;
        let slices: Vec<_> = (0..nz - 1)
            .into_par_iter()
            .map(|z| {
                let mut out = Triangles::default();
                for y in 0..ny - 1 {
                    for x in 0..nx - 1 {
                        let base = x + y * nx + z * plane;
                        let indices = [
                            base,
                            base + 1,
                            base + nx,
                            base + nx + 1,
                            base + plane,
                            base + plane + 1,
                            base + plane + nx,
                            base + plane + nx + 1,
                        ];
                        let s = indices.map(|i| samples[i]);
                        if s.iter().all(|s| s.density < ISO) || s.iter().all(|s| s.density >= ISO) {
                            continue;
                        }
                        let p: [Vec3; 8] = std::array::from_fn(|i| {
                            origin
                                + Vec3::new(
                                    (x + (i & 1)) as f32,
                                    (y + ((i >> 1) & 1)) as f32,
                                    (z + ((i >> 2) & 1)) as f32,
                                ) * cell
                        });
                        // A consistent body diagonal makes neighboring cube faces agree.
                        for tet in [
                            [0, 1, 3, 7],
                            [0, 3, 2, 7],
                            [0, 2, 6, 7],
                            [0, 6, 4, 7],
                            [0, 4, 5, 7],
                            [0, 5, 1, 7],
                        ] {
                            polygonize(tet, &p, &s, &mut out);
                        }
                    }
                }
                out
            })
            .collect();
        let total = slices.iter().map(|s| s.positions.len()).sum();
        let mut all = Triangles {
            positions: Vec::with_capacity(total),
            normals: Vec::with_capacity(total),
            colors: Vec::with_capacity(total),
        };
        for mut slice in slices {
            all.positions.append(&mut slice.positions);
            all.normals.append(&mut slice.normals);
            all.colors.append(&mut slice.colors);
        }
        self.triangles = total / 3;
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, all.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, all.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, all.colors)
    }
}

fn polygonize(tet: [usize; 4], p: &[Vec3; 8], s: &[Sample; 8], out: &mut Triangles) {
    let mut inside = [0; 4];
    let mut outside = [0; 4];
    let (mut ni, mut no) = (0, 0);
    for i in tet {
        if s[i].density >= ISO {
            inside[ni] = i;
            ni += 1;
        } else {
            outside[no] = i;
            no += 1;
        }
    }
    let edge = |a: usize, b: usize| {
        let t = ((ISO - s[a].density) / (s[b].density - s[a].density)).clamp(0.0, 1.0);
        Vertex {
            p: p[a].lerp(p[b], t),
            n: s[a].normal.lerp(s[b].normal, t),
            foam: s[a].foam + (s[b].foam - s[a].foam) * t,
        }
    };
    match ni {
        1 => out.push(
            edge(inside[0], outside[0]),
            edge(inside[0], outside[1]),
            edge(inside[0], outside[2]),
        ),
        3 => out.push(
            edge(outside[0], inside[0]),
            edge(outside[0], inside[1]),
            edge(outside[0], inside[2]),
        ),
        2 => {
            let a = edge(inside[0], outside[0]);
            let b = edge(inside[0], outside[1]);
            let c = edge(inside[1], outside[0]);
            let d = edge(inside[1], outside[1]);
            out.push(a, b, c);
            out.push(b, d, c);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn reconstructed_water_is_closed_with_outward_normals() {
        let mut f = Fluid::new(Config::default().fluid_params());
        f.fill_block([6, 6, 6]);
        let mut surface = Surface::new(f.params.bounds, f.params.spacing, 1.0);
        let mesh = surface.rebuild(&f);
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
        assert!(positions.len() > 100);
        let mut volume = 0.0;
        let mut edges = std::collections::HashMap::new();
        let key = |p: [f32; 3]| p.map(|v| (v * 100.0).round() as i32);
        for (p, n) in positions.chunks_exact(3).zip(normals.chunks_exact(3)) {
            let [a, b, c] = [Vec3::from(p[0]), Vec3::from(p[1]), Vec3::from(p[2])];
            assert!(
                (b - a)
                    .cross(c - a)
                    .dot(Vec3::from(n[0]) + Vec3::from(n[1]) + Vec3::from(n[2]))
                    > 0.0
            );
            volume += a.dot(b.cross(c)) / 6.0;
            for (a, b) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                let (a, b) = (key(a), key(b));
                if a != b {
                    *edges
                        .entry(if a < b { (a, b) } else { (b, a) })
                        .or_insert(0) += 1;
                }
            }
        }
        assert!(
            edges.values().all(|&count| count == 2),
            "surface has cracks or overlapping faces"
        );
        let expected = f.len() as f32 * f.params.spacing.powi(3);
        assert!(
            (0.65..1.15).contains(&(volume / expected)),
            "reconstruction lost volume: {}",
            volume / expected
        );
        assert!(
            normals
                .iter()
                .all(|n| (Vec3::from(*n).length() - 1.0).abs() < 1e-4)
        );
    }
}
