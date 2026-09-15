//! The solver in `sim.rs`, run on the GPU.
//!
//! [`Fluid`] stays the reference, and the owner of the scene: it fills the tank,
//! holds the parameters, and still steps on the CPU when asked to. This runs
//! the same substep as compute passes in `sim.wgsl`. The particles stay on the
//! GPU, where the surface and spray are built from them (`surface_gpu.rs`,
//! `spray_gpu.rs`), and where the mouse and the window title query them. Only
//! the tests copy them back.
//!
//! The one structural difference is the neighbour lists. The CPU counts first
//! and then fills lists of exactly the right length; here every particle gets a
//! fixed number of slots, which avoids a readback per substep to size the
//! buffer. The capacity is generous, grows if it is ever exceeded, and the
//! largest neighbourhood seen is tracked so tests can prove it never was.

use crate::gpu::{Binding, Gpu, Kernel, cast};
use crate::sim::{Bounds, Fluid};
use bevy::log::warn;
use bevy::math::Vec3;
use bevy::prelude::Resource;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, Buffer, BufferBinding,
    BufferDescriptor, BufferSize, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePass, ComputePassDescriptor,
};
use bytemuck::{Pod, Zeroable};
use core::f32::consts::PI;

pub(crate) const WORKGROUP: u32 = 256;

/// Steps between renumbering the particles into grid order; see
/// [`GpuFluid::encode_renumber`].
const RENUMBER_EVERY: u32 = 1;
const MAX_GROUPS: u32 = 65535;

const WAVE: u32 = 1;
const SINGLE: u32 = 2;
const CLAMP: u32 = 4;
const MAKER: u32 = 8;

/// `Params` in `sim.wgsl`: vec4s first, then scalars, so nothing is padded
/// implicitly on either side.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    bounds_size: [f32; 4],
    wall_min: [f32; 4],
    wall_max: [f32; 4],
    gravity: [f32; 4],
    grid_origin: [f32; 4],
    direction: [f32; 4],
    impulse: [f32; 4],
    ray: [f32; 4],
    n: u32,
    capacity: u32,
    dims_x: i32,
    dims_y: i32,
    dims_z: i32,
    cells: u32,
    flags: u32,
    tensile_n: i32,
    reduce_size: u32,
    dt: f32,
    time: f32,
    h: f32,
    h2: f32,
    poly6: f32,
    spiky: f32,
    inv_rho0: f32,
    epsilon: f32,
    tensile_scale: f32,
    tensile_w: f32,
    relax: f32,
    margin: f32,
    touch: f32,
    friction: f32,
    viscosity: f32,
    speed_scale: f32,
    interior_scale: f32,
    strength: f32,
    reef_height: f32,
    reef_start: f32,
    reef_width: f32,
    reef_skew: f32,
    origin_x: f32,
    amplitude: f32,
    omega: f32,
    wavenumber: f32,
    depth: f32,
    level: f32,
    generation_width: f32,
    period: f32,
    beach_start: f32,
    beach_width: f32,
    _pad: [f32; 3],
}

const _: () = assert!(size_of::<GpuParams>() == 336);

/// `Level` in `sim.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Level {
    stride: u32,
    count: u32,
    top: u32,
    down: u32,
}

/// A binding in `sim.wgsl`. The discriminant is the binding number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Params = 0,
    Level = 1,
    Positions = 2,
    Velocities = 4,
    Predicted = 5,
    Lambdas = 6,
    Deltas = 7,
    Boundary = 8,
    Scratch = 9,
    Foam = 10,
    Spray = 11,
    Cells = 12,
    GridCount = 13,
    GridStart = 14,
    GridCursor = 15,
    Sorted = 16,
    Neighbors = 17,
    NeighborCount = 18,
    MaxNeighbors = 19,
    Stats = 20,
    Top = 21,
    Vectors = 23,
    Scalars = 24,
    Gathered = 25,
}

/// A compute pass: one entry point in `sim.wgsl`, and the bindings it touches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pass {
    Impulse,
    Predict,
    Bin,
    Count,
    Prefix,
    Scatter,
    SortCells,
    Neighbors,
    Lambda,
    Delta,
    ApplyDelta,
    Velocity,
    Viscosity,
    Foam,
    MeasureTop,
    Measure,
    Reduce,
    RayHits,
    ReduceNearest,
    GatherVectors,
    GatherScalars,
}

impl Pass {
    const ALL: [Pass; 21] = [
        Pass::Impulse,
        Pass::Predict,
        Pass::Bin,
        Pass::Count,
        Pass::Prefix,
        Pass::Scatter,
        Pass::SortCells,
        Pass::Neighbors,
        Pass::Lambda,
        Pass::Delta,
        Pass::ApplyDelta,
        Pass::Velocity,
        Pass::Viscosity,
        Pass::Foam,
        Pass::MeasureTop,
        Pass::Measure,
        Pass::Reduce,
        Pass::RayHits,
        Pass::ReduceNearest,
        Pass::GatherVectors,
        Pass::GatherScalars,
    ];

    /// The passes that sort positions into a grid, bound per [`GpuGrid`].
    const GRID: [Pass; 5] = [
        Pass::Bin,
        Pass::Count,
        Pass::Prefix,
        Pass::Scatter,
        Pass::SortCells,
    ];

    fn entry(self) -> &'static str {
        match self {
            Pass::Impulse => "apply_impulse",
            Pass::Predict => "predict",
            Pass::Bin => "bin_particles",
            Pass::Count => "count_cells",
            Pass::Prefix => "prefix_sum",
            Pass::Scatter => "scatter_particles",
            Pass::SortCells => "sort_cells",
            Pass::Neighbors => "find_neighbors",
            Pass::Lambda => "solve_lambda",
            Pass::Delta => "solve_delta",
            Pass::ApplyDelta => "apply_delta",
            Pass::Velocity => "update_velocity",
            Pass::Viscosity => "apply_viscosity",
            Pass::Foam => "update_foam",
            Pass::MeasureTop => "measure_top",
            Pass::Measure => "measure",
            Pass::Reduce => "reduce",
            Pass::RayHits => "ray_hits",
            Pass::ReduceNearest => "reduce_nearest",
            Pass::GatherVectors => "gather_vectors",
            Pass::GatherScalars => "gather_scalars",
        }
    }

    fn slots(self) -> &'static [Slot] {
        use Slot::*;
        match self {
            Pass::Impulse => &[Params, Positions, Velocities],
            Pass::Predict => &[Params, Positions, Velocities, Predicted],
            Pass::Bin => &[Params, Predicted, Cells],
            Pass::Count => &[Params, Cells, GridCount],
            Pass::Prefix => &[Level, GridStart],
            Pass::Scatter => &[Params, Cells, GridCursor, Sorted],
            Pass::SortCells => &[Params, GridStart, Sorted],
            Pass::Neighbors => &[
                Params,
                Predicted,
                GridStart,
                Sorted,
                Neighbors,
                NeighborCount,
                MaxNeighbors,
            ],
            Pass::Lambda => &[
                Params,
                Predicted,
                Neighbors,
                NeighborCount,
                Lambdas,
                Boundary,
            ],
            Pass::Delta => &[
                Params,
                Predicted,
                Neighbors,
                NeighborCount,
                Lambdas,
                Boundary,
                Deltas,
            ],
            Pass::ApplyDelta => &[Params, Predicted, Deltas],
            Pass::Velocity => &[Params, Predicted, Positions, Velocities],
            Pass::Viscosity => &[
                Params,
                Positions,
                Velocities,
                Neighbors,
                NeighborCount,
                Scratch,
            ],
            Pass::Foam => &[
                Params,
                Positions,
                Velocities,
                Neighbors,
                NeighborCount,
                Foam,
                Spray,
            ],
            Pass::MeasureTop => &[Params, Positions, Stats],
            Pass::Measure => &[
                Params,
                Positions,
                Velocities,
                Neighbors,
                NeighborCount,
                Boundary,
                Stats,
                Top,
            ],
            Pass::Reduce | Pass::ReduceNearest => &[Level, Stats],
            Pass::RayHits => &[Params, Positions, Stats],
            Pass::GatherVectors => &[Params, Sorted, Vectors, Scratch],
            Pass::GatherScalars => &[Params, Sorted, Scalars, Gathered],
        }
    }
}

/// A table of scan levels in one uniform buffer, addressed by dynamic offset.
struct Levels {
    buffer: Buffer,
    /// Offset and pair count of each level, in the order they run.
    runs: Vec<(u32, u32)>,
}

impl Levels {
    /// Up-sweep levels over a power-of-two `size`, then the down-sweep if asked.
    fn new(gpu: &Gpu, size: u32, down_sweep: bool) -> Self {
        let align = gpu.device.limits().min_uniform_buffer_offset_alignment;
        let top = size / 2;
        let mut levels = Vec::new();
        let mut stride = 1;
        while stride < size {
            levels.push(Level {
                stride,
                count: size / (2 * stride),
                top,
                down: 0,
            });
            stride *= 2;
        }
        if down_sweep {
            let mut stride = top;
            while stride >= 1 {
                levels.push(Level {
                    stride,
                    count: size / (2 * stride),
                    top,
                    down: 1,
                });
                stride /= 2;
            }
        }
        let mut bytes = vec![0u8; levels.len() * align as usize];
        let mut runs = Vec::with_capacity(levels.len());
        for (k, level) in levels.iter().enumerate() {
            let offset = k * align as usize;
            bytes[offset..offset + size_of::<Level>()].copy_from_slice(bytemuck::bytes_of(level));
            runs.push((offset as u32, level.count));
        }
        let buffer = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("scan levels"),
            size: bytes.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, &bytes);
        Self { buffer, runs }
    }

    fn binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: BufferSize::new(size_of::<Level>() as u64),
        })
    }
}

pub(crate) fn storage_buffer(gpu: &Gpu, label: &str, bytes: u64) -> Buffer {
    gpu.device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4).next_multiple_of(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub(crate) fn staging_buffer(gpu: &Gpu, label: &str, bytes: u64) -> Buffer {
    gpu.device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4).next_multiple_of(4),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The dimensions of the CPU's [`crate::sim::Grid`] over `bounds`, so cell
/// indices agree.
fn grid_dims(bounds: Bounds, cell: f32) -> [i32; 3] {
    let size = bounds.size();
    [
        (size.x / cell).ceil() as i32 + 1,
        (size.y / cell).ceil() as i32 + 1,
        (size.z / cell).ceil() as i32 + 1,
    ]
}

/// A dense grid over a buffer of positions, rebuilt on the GPU by counting
/// sort, then a sort within each cell so cells list particles in index order as
/// the CPU grid does. The solver bins its predictions into one; the surface
/// bins interpolated particles into another, with wider cells.
pub(crate) struct GpuGrid {
    n: u32,
    cells: u32,
    params: Buffer,
    cells_of: Buffer,
    starts: Buffer,
    cursor: Buffer,
    sorted: Buffer,
    levels: Levels,
    bind_groups: Vec<BindGroup>,
}

impl GpuGrid {
    /// Offset of each cell's first particle in [`Self::sorted`], plus a
    /// trailing total.
    pub(crate) fn starts(&self) -> &Buffer {
        &self.starts
    }

    /// Particle indices ordered by cell.
    pub(crate) fn sorted(&self) -> &Buffer {
        &self.sorted
    }
}

/// An in-place exclusive prefix sum over a power-of-two buffer of `u32`s.
pub(crate) struct GpuScan {
    levels: Levels,
    bind_group: BindGroup,
}

/// Everything sized by the particle count.
struct Buffers {
    n: usize,
    capacity: u32,
    dims: [i32; 3],
    cells: u32,
    /// Power-of-two length of the particle reductions.
    reduce_size: u32,
    params: Buffer,
    positions: Buffer,
    previous: Buffer,
    velocities: Buffer,
    predicted: Buffer,
    lambdas: Buffer,
    deltas: Buffer,
    boundary: Buffer,
    scratch: Buffer,
    foam: Buffer,
    spray: Buffer,
    neighbors: Buffer,
    neighbor_count: Buffer,
    max_neighbors: Buffer,
    stats: Buffer,
    top: Buffer,
    /// The particle, in `Fluid`'s numbering, each slot holds; see
    /// [`GpuFluid::encode_renumber`].
    ids: Buffer,
    gathered: Buffer,
    reduce_levels: Levels,
    grid: GpuGrid,
    /// One reduction result, or the largest neighbourhood.
    probe: Buffer,
}

impl Buffers {
    fn slot(&self, slot: Slot) -> BindingResource<'_> {
        match slot {
            Slot::Params => self.params.as_entire_binding(),
            Slot::Level => self.reduce_levels.binding(),
            Slot::Positions => self.positions.as_entire_binding(),
            Slot::Velocities => self.velocities.as_entire_binding(),
            Slot::Predicted => self.predicted.as_entire_binding(),
            Slot::Lambdas => self.lambdas.as_entire_binding(),
            Slot::Deltas => self.deltas.as_entire_binding(),
            Slot::Boundary => self.boundary.as_entire_binding(),
            Slot::Scratch => self.scratch.as_entire_binding(),
            Slot::Foam => self.foam.as_entire_binding(),
            Slot::Spray => self.spray.as_entire_binding(),
            Slot::GridStart => self.grid.starts.as_entire_binding(),
            Slot::Sorted => self.grid.sorted.as_entire_binding(),
            Slot::Neighbors => self.neighbors.as_entire_binding(),
            Slot::NeighborCount => self.neighbor_count.as_entire_binding(),
            Slot::MaxNeighbors => self.max_neighbors.as_entire_binding(),
            Slot::Stats => self.stats.as_entire_binding(),
            Slot::Top => self.top.as_entire_binding(),
            Slot::Gathered => self.gathered.as_entire_binding(),
            Slot::Cells | Slot::GridCount | Slot::GridCursor => {
                unreachable!("{slot:?} is bound by the grid")
            }
            Slot::Vectors | Slot::Scalars => unreachable!("{slot:?} is bound per gathered buffer"),
        }
    }
}

/// A push or pull from the mouse, waiting for the next step.
#[derive(Clone, Copy)]
struct Impulse {
    center: Vec3,
    radius: f32,
    strength: f32,
}

/// The title readout, measured on the GPU from the last step's state.
#[derive(Clone, Copy, Debug)]
pub struct Readout {
    /// As [`Fluid::compression_error`].
    pub compression: f32,
    /// As [`Fluid::interior_density_ratio`]; NaN when the water has no bulk.
    pub bulk: f32,
    /// As [`Fluid::max_speed`].
    pub peak_speed: f32,
}

#[derive(Resource)]
pub struct GpuFluid {
    gpu: Gpu,
    kernels: Vec<Kernel>,
    buffers: Buffers,
    /// One per pass, except the grid passes, which each grid binds itself, and
    /// the gathers, which bind once per buffer they renumber.
    bind_groups: Vec<Option<BindGroup>>,
    /// Positions, previous positions, velocities, foam, spray, ids.
    gather_groups: Vec<BindGroup>,
    /// Steps since the particles were last renumbered into grid order.
    since_renumber: u32,
    /// The [`Fluid::generation`] last uploaded.
    generation: u64,
    /// Neighbour lists exist once a step has run since the last upload.
    stepped: bool,
    /// Largest neighbourhood seen since the last upload, capped or not.
    max_neighbors: u32,
    impulses: Vec<Impulse>,
    /// Bumped whenever the particle buffers are replaced, so bind groups made
    /// elsewhere know to follow.
    revision: u64,
}

impl GpuFluid {
    pub fn new(gpu: Gpu, fluid: &Fluid) -> Self {
        let module = gpu.module("sim.wgsl", include_str!("sim.wgsl"));
        let kernels: Vec<Kernel> = Pass::ALL
            .iter()
            .map(|&pass| {
                let bindings: Vec<_> = pass
                    .slots()
                    .iter()
                    .map(|&slot| {
                        let kind = match slot {
                            Slot::Params => Binding::Uniform {
                                size: size_of::<GpuParams>() as u64,
                                dynamic: false,
                            },
                            Slot::Level => Binding::Uniform {
                                size: size_of::<Level>() as u64,
                                dynamic: true,
                            },
                            _ => Binding::Storage,
                        };
                        (slot as u32, kind)
                    })
                    .collect();
                gpu.kernel(&module, pass.entry(), &bindings)
            })
            .collect();
        let mut solver = Self {
            buffers: Self::buffers(&gpu, &kernels, fluid, neighbor_capacity(fluid)),
            gpu,
            kernels,
            bind_groups: Vec::new(),
            gather_groups: Vec::new(),
            since_renumber: 0,
            generation: 0,
            stepped: false,
            max_neighbors: 0,
            impulses: Vec::new(),
            revision: 0,
        };
        solver.bind();
        solver.upload(fluid);
        solver
    }

    fn buffers(gpu: &Gpu, kernels: &[Kernel], fluid: &Fluid, capacity: u32) -> Buffers {
        let n = fluid.len().max(1);
        let bounds = fluid.params.bounds;
        let h = fluid.params.smoothing_radius;
        let dims = grid_dims(bounds, h);
        let cells = (dims[0] * dims[1] * dims[2]) as u32;
        let reduce_size = (n as u32).next_power_of_two().max(2);
        let n64 = n as u64;
        let predicted = storage_buffer(gpu, "predicted positions", 16 * n64);
        let grid = Self::make_grid(gpu, kernels, &predicted, n as u32, bounds, h);
        Buffers {
            n,
            capacity,
            dims,
            cells,
            reduce_size,
            params: gpu.device.create_buffer(&BufferDescriptor {
                label: Some("solver params"),
                size: size_of::<GpuParams>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            positions: storage_buffer(gpu, "positions", 16 * n64),
            previous: storage_buffer(gpu, "previous positions", 16 * n64),
            velocities: storage_buffer(gpu, "velocities", 16 * n64),
            predicted,
            lambdas: storage_buffer(gpu, "lambdas", 4 * n64),
            deltas: storage_buffer(gpu, "deltas", 16 * n64),
            boundary: storage_buffer(gpu, "boundary support", 16 * n64),
            scratch: storage_buffer(gpu, "velocity scratch", 16 * n64),
            foam: storage_buffer(gpu, "foam", 4 * n64),
            spray: storage_buffer(gpu, "spray", 4 * n64),
            neighbors: storage_buffer(gpu, "neighbours", 4 * n64 * capacity as u64),
            neighbor_count: storage_buffer(gpu, "neighbour counts", 4 * n64),
            max_neighbors: storage_buffer(gpu, "max neighbours", 4),
            stats: storage_buffer(gpu, "stats", 16 * reduce_size as u64),
            top: storage_buffer(gpu, "top", 16),
            ids: storage_buffer(gpu, "particle ids", 4 * n64),
            gathered: storage_buffer(gpu, "gathered scalars", 4 * n64),
            reduce_levels: Levels::new(gpu, reduce_size, false),
            grid,
            probe: staging_buffer(gpu, "probe", 16),
        }
    }

    fn make_grid(
        gpu: &Gpu,
        kernels: &[Kernel],
        positions: &Buffer,
        n: u32,
        bounds: Bounds,
        cell: f32,
    ) -> GpuGrid {
        let dims = grid_dims(bounds, cell);
        let cells = (dims[0] * dims[1] * dims[2]) as u32;
        let size = (cells + 1).next_power_of_two();
        let params = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("grid params"),
            size: size_of::<GpuParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(
            &params,
            0,
            bytemuck::bytes_of(&GpuParams {
                grid_origin: bounds.min.extend(cell).to_array(),
                n,
                dims_x: dims[0],
                dims_y: dims[1],
                dims_z: dims[2],
                cells,
                ..Zeroable::zeroed()
            }),
        );
        let mut grid = GpuGrid {
            n,
            cells,
            params,
            cells_of: storage_buffer(gpu, "particle cells", 4 * n as u64),
            starts: storage_buffer(gpu, "grid starts", 4 * size as u64),
            cursor: storage_buffer(gpu, "grid cursor", 4 * size as u64),
            sorted: storage_buffer(gpu, "sorted particles", 4 * n as u64),
            levels: Levels::new(gpu, size, true),
            bind_groups: Vec::new(),
        };
        grid.bind_groups = Pass::GRID
            .iter()
            .map(|&pass| {
                bind(
                    gpu,
                    &kernels[pass as usize].layout,
                    pass,
                    |slot| match slot {
                        Slot::Params => grid.params.as_entire_binding(),
                        Slot::Level => grid.levels.binding(),
                        Slot::Predicted => positions.as_entire_binding(),
                        Slot::Cells => grid.cells_of.as_entire_binding(),
                        Slot::GridCount | Slot::GridStart => grid.starts.as_entire_binding(),
                        Slot::GridCursor => grid.cursor.as_entire_binding(),
                        Slot::Sorted => grid.sorted.as_entire_binding(),
                        _ => unreachable!("{slot:?} is not a grid binding"),
                    },
                )
            })
            .collect();
        grid
    }

    fn bind(&mut self) {
        let b = &self.buffers;
        self.gather_groups = [
            (Pass::GatherVectors, &b.positions),
            (Pass::GatherVectors, &b.previous),
            (Pass::GatherVectors, &b.velocities),
            (Pass::GatherScalars, &b.foam),
            (Pass::GatherScalars, &b.spray),
            (Pass::GatherScalars, &b.ids),
        ]
        .into_iter()
        .map(|(pass, source)| {
            bind(
                &self.gpu,
                &self.kernels[pass as usize].layout,
                pass,
                |slot| match slot {
                    Slot::Vectors | Slot::Scalars => source.as_entire_binding(),
                    _ => b.slot(slot),
                },
            )
        })
        .collect();
        self.bind_groups = Pass::ALL
            .iter()
            .map(|&pass| {
                let gather = matches!(pass, Pass::GatherVectors | Pass::GatherScalars);
                (!Pass::GRID.contains(&pass) && !gather).then(|| {
                    bind(
                        &self.gpu,
                        &self.kernels[pass as usize].layout,
                        pass,
                        |slot| self.buffers.slot(slot),
                    )
                })
            })
            .collect();
    }

    /// A grid over `n` positions in `positions`, with cells `cell` wide.
    pub(crate) fn grid(&self, positions: &Buffer, n: u32, bounds: Bounds, cell: f32) -> GpuGrid {
        Self::make_grid(&self.gpu, &self.kernels, positions, n, bounds, cell)
    }

    /// Encodes a rebuild of `grid` from its positions as they are when it runs.
    pub(crate) fn rebuild_grid(&self, encoder: &mut CommandEncoder, grid: &GpuGrid) {
        let group = |pass: Pass| {
            let k = Pass::GRID.iter().position(|p| *p == pass).unwrap();
            &grid.bind_groups[k]
        };
        encoder.clear_buffer(&grid.starts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            for which in [Pass::Bin, Pass::Count] {
                pass.set_pipeline(&self.kernels[which as usize].pipeline);
                pass.set_bind_group(0, group(which), &[]);
                dispatch(&mut pass, grid.n);
            }
            self.run_scan_levels(&mut pass, group(Pass::Prefix), &grid.levels);
        }
        encoder.copy_buffer_to_buffer(&grid.starts, 0, &grid.cursor, 0, None);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        for (which, count) in [(Pass::Scatter, grid.n), (Pass::SortCells, grid.cells)] {
            pass.set_pipeline(&self.kernels[which as usize].pipeline);
            pass.set_bind_group(0, group(which), &[]);
            dispatch(&mut pass, count);
        }
    }

    /// An exclusive prefix sum over `buffer`, which holds a power-of-two `size`
    /// of `u32`s: afterwards each entry is the sum of the ones before it.
    pub(crate) fn scan(&self, buffer: &Buffer, size: u32) -> GpuScan {
        let levels = Levels::new(&self.gpu, size, true);
        let bind_group = bind(
            &self.gpu,
            &self.kernels[Pass::Prefix as usize].layout,
            Pass::Prefix,
            |slot| match slot {
                Slot::Level => levels.binding(),
                _ => buffer.as_entire_binding(),
            },
        );
        GpuScan { levels, bind_group }
    }

    pub(crate) fn run_scan(&self, pass: &mut ComputePass, scan: &GpuScan) {
        self.run_scan_levels(pass, &scan.bind_group, &scan.levels);
    }

    fn run_scan_levels(&self, pass: &mut ComputePass, group: &BindGroup, levels: &Levels) {
        pass.set_pipeline(&self.kernels[Pass::Prefix as usize].pipeline);
        for &(offset, count) in &levels.runs {
            pass.set_bind_group(0, group, &[offset]);
            dispatch(pass, count);
        }
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn len(&self) -> usize {
        self.buffers.n
    }

    pub(crate) fn positions(&self) -> &Buffer {
        &self.buffers.positions
    }

    pub(crate) fn previous(&self) -> &Buffer {
        &self.buffers.previous
    }

    pub(crate) fn velocities(&self) -> &Buffer {
        &self.buffers.velocities
    }

    pub(crate) fn foam(&self) -> &Buffer {
        &self.buffers.foam
    }

    pub(crate) fn spray(&self) -> &Buffer {
        &self.buffers.spray
    }

    /// Copies the particles in `fluid` to the GPU, replacing whatever was there.
    pub fn upload(&mut self, fluid: &Fluid) {
        if fluid.len().max(1) != self.buffers.n {
            self.buffers = Self::buffers(&self.gpu, &self.kernels, fluid, self.buffers.capacity);
            self.bind();
            self.revision += 1;
        }
        let b = &self.buffers;
        let pack =
            |v: &[Vec3]| -> Vec<[f32; 4]> { v.iter().map(|p| p.extend(0.0).to_array()).collect() };
        let queue = &self.gpu.queue;
        queue.write_buffer(&b.positions, 0, bytemuck::cast_slice(&pack(&fluid.pos)));
        queue.write_buffer(
            &b.previous,
            0,
            bytemuck::cast_slice(&pack(&fluid.previous_pos)),
        );
        queue.write_buffer(&b.velocities, 0, bytemuck::cast_slice(&pack(&fluid.vel)));
        queue.write_buffer(&b.foam, 0, bytemuck::cast_slice(&fluid.foam));
        queue.write_buffer(&b.spray, 0, bytemuck::cast_slice(&fluid.spray));
        let ids: Vec<u32> = (0..fluid.len() as u32).collect();
        queue.write_buffer(&b.ids, 0, bytemuck::cast_slice(&ids));
        let mut encoder = self.encoder();
        // The dam break never writes boundary support, and relies on it being zero.
        for buffer in [&b.boundary, &b.neighbor_count, &b.max_neighbors] {
            encoder.clear_buffer(buffer, 0, None);
        }
        queue.submit([encoder.finish()]);
        if fluid.generation != self.generation {
            self.max_neighbors = 0;
        }
        self.generation = fluid.generation;
        self.stepped = false;
        self.impulses.clear();
    }

    /// Uploads `fluid` if it has been reset since the GPU last saw it. Stepping
    /// does this itself; anything that reads the particles while paused should
    /// call it first.
    pub fn sync(&mut self, fluid: &Fluid) {
        if fluid.generation != self.generation || fluid.len().max(1) != self.buffers.n {
            self.upload(fluid);
        }
    }

    /// Queues the mouse push of [`Fluid::apply_radial_impulse`] for the next step.
    pub fn apply_radial_impulse(&mut self, center: Vec3, radius: f32, strength: f32) {
        self.impulses.push(Impulse {
            center,
            radius,
            strength,
        });
    }

    /// Advances the fluid by `dt` seconds, exactly as [`Fluid::step`] does, and
    /// waits for the GPU to finish. The particles stay on the GPU: `fluid` gets
    /// only its clock advanced.
    pub fn step(&mut self, fluid: &mut Fluid, dt: f32) {
        if fluid.pos.is_empty() || dt <= 0.0 {
            return;
        }
        self.sync(fluid);
        let n = self.buffers.n as u32;
        let substeps = fluid.params.substeps.max(1);
        let sub_dt = dt / substeps as f32;

        // The CPU applies these straight to the velocities between steps; the
        // positions they read are the same ones this step starts from.
        for impulse in std::mem::take(&mut self.impulses) {
            self.write_params(fluid, sub_dt, fluid.elapsed, Some(impulse), Vec3::ZERO);
            let mut encoder = self.encoder();
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                self.run(&mut pass, Pass::Impulse, n);
            }
            self.gpu.queue.submit([encoder.finish()]);
        }

        let mut encoder = self.encoder();
        self.since_renumber += 1;
        if self.stepped && self.since_renumber >= RENUMBER_EVERY {
            self.write_params(fluid, sub_dt, fluid.elapsed, None, Vec3::ZERO);
            self.encode_renumber(&mut encoder);
            self.since_renumber = 0;
        }
        encoder.copy_buffer_to_buffer(&self.buffers.positions, 0, &self.buffers.previous, 0, None);
        for _ in 0..substeps {
            fluid.elapsed += sub_dt;
            // Written per submission: a queued write lands before the commands
            // submitted with it, so each substep needs its own.
            self.write_params(fluid, sub_dt, fluid.elapsed, None, Vec3::ZERO);
            self.encode_substep(&mut encoder, fluid.params.iterations);
            self.gpu.queue.submit([encoder.finish()]);
            encoder = self.encoder();
        }
        self.write_params(fluid, dt, fluid.elapsed, None, Vec3::ZERO);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::Foam, n);
        }
        encoder.copy_buffer_to_buffer(&self.buffers.max_neighbors, 0, &self.buffers.probe, 0, 4);
        self.gpu.queue.submit([encoder.finish()]);
        self.stepped = true;
        // Waiting here is what keeps the solver from queueing steps faster than
        // the GPU runs them, and it makes the step's cost measurable.
        let seen = self
            .gpu
            .read(&self.buffers.probe, |bytes| cast::<u32>(bytes)[0]);
        self.note_neighbors(seen);
    }

    /// Renumbers the particles into the cell order of the last substep's grid.
    /// As the flow mixes, fill order stops putting neighbours near each other
    /// in memory, and a late `slab.toml` step costs twice an early one; in grid
    /// order it costs barely more. Neighbour sums then run in a different order
    /// from the CPU's, which changes nothing but the last bit of rounding.
    fn encode_renumber(&self, encoder: &mut CommandEncoder) {
        let b = &self.buffers;
        let targets = [
            (Pass::GatherVectors, &b.positions, &b.scratch),
            (Pass::GatherVectors, &b.previous, &b.scratch),
            (Pass::GatherVectors, &b.velocities, &b.scratch),
            (Pass::GatherScalars, &b.foam, &b.gathered),
            (Pass::GatherScalars, &b.spray, &b.gathered),
            (Pass::GatherScalars, &b.ids, &b.gathered),
        ];
        for ((pass_kind, target, out), group) in targets.into_iter().zip(&self.gather_groups) {
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                pass.set_pipeline(&self.kernels[pass_kind as usize].pipeline);
                pass.set_bind_group(0, group, &[]);
                dispatch(&mut pass, b.n as u32);
            }
            encoder.copy_buffer_to_buffer(out, 0, target, 0, target.size());
        }
    }

    fn encode_substep(&self, encoder: &mut CommandEncoder, iterations: u32) {
        let b = &self.buffers;
        let n = b.n as u32;
        {
            // 1. Forces and prediction.
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::Predict, n);
        }
        // 2. Neighbourhoods.
        self.rebuild_grid(encoder, &b.grid);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::Neighbors, n);
            // 3. Jacobi iterations.
            for _ in 0..iterations {
                self.run(&mut pass, Pass::Lambda, n);
                self.run(&mut pass, Pass::Delta, n);
                self.run(&mut pass, Pass::ApplyDelta, n);
            }
            // 4. Velocity, 5. viscosity.
            self.run(&mut pass, Pass::Velocity, n);
            self.run(&mut pass, Pass::Viscosity, n);
        }
        encoder.copy_buffer_to_buffer(&b.scratch, 0, &b.velocities, 0, None);
    }

    /// Grows the neighbour lists if a step outran them. Only the lists move;
    /// the particle buffers others have bound stay where they are.
    fn note_neighbors(&mut self, seen: u32) {
        self.max_neighbors = self.max_neighbors.max(seen);
        let capacity = self.buffers.capacity;
        if seen > capacity {
            let grown = (seen + seen / 4).next_multiple_of(8);
            warn!(
                "a particle had {seen} neighbours, past the GPU solver's {capacity} slots; \
                 growing to {grown}. That step ran on truncated neighbourhoods."
            );
            let n = self.buffers.n as u64;
            self.buffers.neighbors = storage_buffer(&self.gpu, "neighbours", 4 * n * grown as u64);
            self.buffers.capacity = grown;
            self.bind();
        }
    }

    /// Copies the particles back into `fluid`: positions, the positions the
    /// last step started from, velocities, foam and spray.
    #[cfg(test)]
    pub fn download(&self, fluid: &mut Fluid) {
        let b = &self.buffers;
        let n = b.n as u64;
        // Made per call: nothing in the app downloads, so it should not carry
        // a staging buffer the size of the particles.
        let readback = staging_buffer(&self.gpu, "particle readback", 60 * n);
        let mut encoder = self.encoder();
        encoder.copy_buffer_to_buffer(&b.positions, 0, &readback, 0, 16 * n);
        encoder.copy_buffer_to_buffer(&b.previous, 0, &readback, 16 * n, 16 * n);
        encoder.copy_buffer_to_buffer(&b.velocities, 0, &readback, 32 * n, 16 * n);
        encoder.copy_buffer_to_buffer(&b.foam, 0, &readback, 48 * n, 4 * n);
        encoder.copy_buffer_to_buffer(&b.spray, 0, &readback, 52 * n, 4 * n);
        encoder.copy_buffer_to_buffer(&b.ids, 0, &readback, 56 * n, 4 * n);
        self.gpu.queue.submit([encoder.finish()]);
        let n = n as usize;
        self.gpu.read(&readback, |bytes| {
            // Slots back into `Fluid`'s numbering.
            let ids = cast::<u32>(&bytes[56 * n..60 * n]);
            let vectors = |src: &[u8]| {
                let mut out = vec![Vec3::ZERO; n];
                for (s, &id) in cast::<[f32; 4]>(src).iter().zip(ids.iter()) {
                    out[id as usize] = Vec3::new(s[0], s[1], s[2]);
                }
                out
            };
            let scalars = |src: &[u8]| {
                let mut out = vec![0.0; n];
                for (s, &id) in cast::<f32>(src).iter().zip(ids.iter()) {
                    out[id as usize] = *s;
                }
                out
            };
            fluid.pos = vectors(&bytes[..16 * n]);
            fluid.previous_pos = vectors(&bytes[16 * n..32 * n]);
            fluid.vel = vectors(&bytes[32 * n..48 * n]);
            fluid.foam = scalars(&bytes[48 * n..52 * n]);
            fluid.spray = scalars(&bytes[52 * n..56 * n]);
        });
    }

    /// Compression, bulk density and peak speed, computed on the GPU. `None`
    /// until a step has built neighbour lists for the current particles.
    pub fn readout(&self, fluid: &Fluid) -> Option<Readout> {
        if !self.stepped || fluid.generation != self.generation {
            return None;
        }
        let b = &self.buffers;
        self.write_params(fluid, 0.0, fluid.elapsed, None, Vec3::ZERO);
        let last = 16 * (b.reduce_size as u64 - 1);
        let mut encoder = self.encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::MeasureTop, b.reduce_size);
            self.run_levels(&mut pass, Pass::Reduce);
        }
        encoder.copy_buffer_to_buffer(&b.stats, last, &b.top, 0, 16);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::Measure, b.reduce_size);
            self.run_levels(&mut pass, Pass::Reduce);
        }
        encoder.copy_buffer_to_buffer(&b.stats, last, &b.probe, 0, 16);
        self.gpu.queue.submit([encoder.finish()]);
        let [compression, bulk, samples, peak_speed] =
            self.gpu.read(&b.probe, |bytes| cast::<[f32; 4]>(bytes)[0]);
        Some(Readout {
            compression: compression / b.n as f32,
            bulk: if samples > 0.0 {
                bulk / samples
            } else {
                f32::NAN
            },
            peak_speed,
        })
    }

    /// [`Fluid::nearest_along_ray`], over the particles on the GPU.
    pub fn nearest_along_ray(
        &self,
        fluid: &Fluid,
        origin: Vec3,
        direction: Vec3,
        radius: f32,
    ) -> Option<Vec3> {
        let direction = direction.normalize_or_zero();
        if direction == Vec3::ZERO {
            return None;
        }
        let b = &self.buffers;
        let ray = Impulse {
            center: origin,
            radius,
            strength: 0.0,
        };
        self.write_params(fluid, 0.0, fluid.elapsed, Some(ray), direction);
        let mut encoder = self.encoder();
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            self.run(&mut pass, Pass::RayHits, b.reduce_size);
            self.run_levels(&mut pass, Pass::ReduceNearest);
        }
        encoder.copy_buffer_to_buffer(&b.stats, 16 * (b.reduce_size as u64 - 1), &b.probe, 0, 16);
        self.gpu.queue.submit([encoder.finish()]);
        let [x, y, z, along] = self.gpu.read(&b.probe, |bytes| cast::<[f32; 4]>(bytes)[0]);
        (along > -3.0e38).then_some(Vec3::new(x, y, z))
    }

    /// The largest neighbourhood seen since the last upload. The GPU solver
    /// matches the CPU only while this stays within [`Self::capacity`].
    #[cfg(test)]
    pub fn max_neighbors(&self) -> u32 {
        self.max_neighbors
    }

    #[cfg(test)]
    pub fn capacity(&self) -> u32 {
        self.buffers.capacity
    }

    pub(crate) fn encoder(&self) -> CommandEncoder {
        self.gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default())
    }

    fn run(&self, pass: &mut ComputePass, which: Pass, count: u32) {
        let group = self.bind_groups[which as usize]
            .as_ref()
            .expect("grid passes run through rebuild_grid");
        pass.set_pipeline(&self.kernels[which as usize].pipeline);
        pass.set_bind_group(0, group, &[]);
        dispatch(pass, count);
    }

    fn run_levels(&self, pass: &mut ComputePass, which: Pass) {
        let group = self.bind_groups[which as usize].as_ref().unwrap();
        pass.set_pipeline(&self.kernels[which as usize].pipeline);
        for &(offset, count) in &self.buffers.reduce_levels.runs {
            pass.set_bind_group(0, group, &[offset]);
            dispatch(pass, count);
        }
    }

    fn write_params(&self, fluid: &Fluid, dt: f32, time: f32, impulse: Option<Impulse>, ray: Vec3) {
        let b = &self.buffers;
        let p = &fluid.params;
        let bounds = p.bounds;
        let margin = p.spacing * 0.5;
        let k = &fluid.kernels;
        let mut flags = 0;
        if fluid.wave.is_some() {
            flags |= WAVE;
        }
        if let Some(maker) = fluid.wavemaker {
            flags |= MAKER;
            if maker.single.is_some() {
                flags |= SINGLE;
            }
        }
        if p.clamp_constraint {
            flags |= CLAMP;
        }
        let wave = fluid.wave.unwrap_or_default();
        let maker = fluid.wavemaker;
        let from = |v: Vec3| v.extend(0.0).to_array();
        let impulse = impulse.unwrap_or(Impulse {
            center: Vec3::ZERO,
            radius: 0.0,
            strength: 0.0,
        });
        let params = GpuParams {
            bounds_min: from(bounds.min),
            bounds_max: from(bounds.max),
            bounds_size: from(bounds.size()),
            wall_min: from(bounds.min + Vec3::splat(margin)),
            wall_max: from(bounds.max - Vec3::splat(margin)),
            gravity: from(p.gravity),
            grid_origin: bounds.min.extend(p.smoothing_radius).to_array(),
            direction: from(maker.map_or(Vec3::X, |m| m.direction)),
            impulse: impulse.center.extend(impulse.radius).to_array(),
            ray: from(ray),
            n: b.n as u32,
            capacity: b.capacity,
            dims_x: b.dims[0],
            dims_y: b.dims[1],
            dims_z: b.dims[2],
            cells: b.cells,
            flags,
            tensile_n: p.tensile_n,
            reduce_size: b.reduce_size,
            dt,
            time,
            h: k.h,
            h2: k.h2,
            poly6: k.poly6,
            spiky: k.spiky,
            inv_rho0: 1.0 / fluid.rest_density,
            epsilon: fluid.epsilon,
            tensile_scale: fluid.tensile_scale,
            tensile_w: fluid.tensile_w,
            relax: p.jacobi_relax,
            margin,
            touch: p.spacing * 0.25,
            friction: p.wall_friction,
            viscosity: p.viscosity,
            speed_scale: (p.gravity.length() * p.spacing).sqrt().max(1.0),
            interior_scale: 1.0 / (4.0 / 3.0 * PI * k.h.powi(3)) / (1.0 / p.spacing.powi(3)),
            strength: impulse.strength,
            reef_height: wave.reef_height,
            reef_start: wave.reef_start,
            reef_width: wave.reef_width,
            reef_skew: wave.reef_skew,
            origin_x: maker.map_or(0.0, |m| m.origin_x),
            amplitude: maker.map_or(0.0, |m| m.amplitude),
            omega: maker.map_or(0.0, |m| m.omega),
            wavenumber: maker.map_or(0.0, |m| m.k),
            depth: maker.map_or(0.0, |m| m.depth),
            level: maker.map_or(0.0, |m| m.level),
            generation_width: maker.map_or(1.0, |m| m.generation_width),
            period: maker.map_or(1.0, |m| m.period),
            beach_start: maker.map_or(0.0, |m| m.beach_start),
            beach_width: maker.map_or(1.0, |m| m.beach_width),
            _pad: [0.0; 3],
        };
        self.gpu
            .queue
            .write_buffer(&b.params, 0, bytemuck::bytes_of(&params));
    }
}

/// A bind group for `pass`, taking each of its slots from `resource`.
fn bind<'a>(
    gpu: &Gpu,
    layout: &BindGroupLayout,
    pass: Pass,
    resource: impl Fn(Slot) -> BindingResource<'a>,
) -> BindGroup {
    let entries: Vec<_> = pass
        .slots()
        .iter()
        .map(|&slot| BindGroupEntry {
            binding: slot as u32,
            resource: resource(slot),
        })
        .collect();
    gpu.device.create_bind_group(pass.entry(), layout, &entries)
}

/// Room for this many neighbours per particle. A settled fluid averages the
/// kernel ball's volume in rest spacings, 33 at the shipped 2:1 ratio; the
/// layer packed against a wall runs about half as dense again.
fn neighbor_capacity(fluid: &Fluid) -> u32 {
    let ratio = fluid.params.smoothing_radius / fluid.params.spacing;
    let ball = 4.0 / 3.0 * PI * ratio.powi(3);
    ((ball * 2.0).ceil() as u32).next_multiple_of(8).max(32)
}

/// Dispatches enough workgroups for `count` invocations, spilling into a second
/// dimension past the per-dimension limit. Shaders fold it back with `ROW`.
pub(crate) fn dispatch(pass: &mut ComputePass, count: u32) {
    let groups = count.div_ceil(WORKGROUP);
    if groups > 0 {
        pass.dispatch_workgroups(groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS), 1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::wave::tests::overturned_columns;

    /// A headless device, or `None` with a note, so machines without a GPU
    /// skip these instead of failing.
    fn device() -> Option<Gpu> {
        let gpu = Gpu::headless();
        if gpu.is_none() {
            eprintln!("no GPU adapter available; skipping");
        }
        gpu
    }

    impl GpuFluid {
        /// Runs only the neighbour search, on `positions` as the predictions, and
        /// returns the lists in the CPU's flattened layout.
        fn neighbor_lists(&mut self, fluid: &Fluid, positions: &[Vec3]) -> (Vec<u32>, Vec<u32>) {
            self.upload(fluid);
            let b = &self.buffers;
            let n = b.n as u32;
            let packed: Vec<[f32; 4]> =
                positions.iter().map(|p| p.extend(0.0).to_array()).collect();
            self.gpu
                .queue
                .write_buffer(&b.predicted, 0, bytemuck::cast_slice(&packed));
            self.write_params(fluid, 0.0, 0.0, None, Vec3::ZERO);
            let mut encoder = self.encoder();
            self.rebuild_grid(&mut encoder, &b.grid);
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                self.run(&mut pass, Pass::Neighbors, n);
            }
            self.gpu.queue.submit([encoder.finish()]);
            let seen: u32 = cast(&self.fetch(&b.max_neighbors))[0];
            self.max_neighbors = self.max_neighbors.max(seen);
            let counts: Vec<u32> = cast(&self.fetch(&b.neighbor_count)).into_owned();
            let slots: Vec<u32> = cast(&self.fetch(&b.neighbors)).into_owned();
            let mut starts = vec![0u32];
            let mut lists = Vec::new();
            for (i, &count) in counts.iter().enumerate() {
                let base = i * b.capacity as usize;
                lists.extend_from_slice(&slots[base..base + count as usize]);
                starts.push(lists.len() as u32);
            }
            (starts, lists)
        }

        /// A copy of a storage buffer's contents.
        fn fetch(&self, buffer: &Buffer) -> Vec<u8> {
            let staging = self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some("test readback"),
                size: buffer.size(),
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self.encoder();
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, None);
            self.gpu.queue.submit([encoder.finish()]);
            self.gpu.read(&staging, |bytes| bytes.to_vec())
        }

        /// Replaces the neighbour lists with narrower ones, keeping the particles.
        fn shrink_lists(&mut self, fluid: &Fluid, capacity: u32) {
            self.buffers = Self::buffers(&self.gpu, &self.kernels, fluid, capacity);
            self.bind();
            self.upload(fluid);
        }

        /// A step, with the particles copied back as the CPU solver leaves them.
        fn step_back(&mut self, fluid: &mut Fluid, dt: f32) {
            self.step(fluid, dt);
            self.download(fluid);
        }
    }

    /// The narrow slab flume of `wave::tests::a_single_wave_plunges_over_the_reef`.
    fn slab_flume() -> Config {
        let mut c = Config::load("slab.toml").unwrap();
        c.world.depth = 8.0 * c.fluid.spacing;
        c.wave.reef_skew = 0.0;
        c.validate().unwrap();
        c
    }

    /// The narrow swell flume of `wave::tests::incoming_swell_reaches_the_reef_and_resets_cleanly`.
    fn swell_flume() -> Config {
        let mut c = Config::load("config.toml").unwrap();
        c.world.depth = 8.0 * c.fluid.spacing;
        c.wave.reef_skew = 0.0;
        c.validate().unwrap();
        c
    }

    fn fresh(config: &Config) -> Fluid {
        let mut fluid = Fluid::new(config.fluid_params());
        config.reset_fluid(&mut fluid);
        fluid
    }

    fn copy_state(from: &Fluid, to: &mut Fluid) {
        to.pos.clone_from(&from.pos);
        to.previous_pos.clone_from(&from.previous_pos);
        to.vel.clone_from(&from.vel);
        to.foam.clone_from(&from.foam);
        to.spray.clone_from(&from.spray);
        to.elapsed = from.elapsed;
        to.params.gravity = from.params.gravity;
    }

    fn largest_gap(a: &[Vec3], b: &[Vec3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0, f32::max)
    }

    fn largest_difference(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn neighbour_search_matches_the_cpu_exactly() {
        let Some(gpu) = device() else { return };
        for config in [Config::default(), slab_flume()] {
            // Disordered positions rather than the fill lattice: a second of motion.
            let mut cpu = fresh(&config);
            for _ in 0..60 {
                cpu.step(1.0 / 60.0);
            }
            let positions = cpu.pos.clone();
            cpu.build_neighbors_from(&positions);
            let mut solver = GpuFluid::new(gpu.clone(), &cpu);
            let (starts, lists) = solver.neighbor_lists(&cpu, &positions);
            assert!(solver.max_neighbors() <= solver.capacity());
            // Same neighbours in the same order: sums over them round the same way.
            assert_eq!(starts, cpu.neighbor_start, "neighbour counts differ");
            assert_eq!(lists, cpu.neighbors, "neighbour lists differ");
        }
    }

    /// Every term, from the same state: the dam break's artificial pressure and
    /// walls, plus a mouse push; the swell's wavemaker, ramped up; the slab's
    /// reef, solid support and beach.
    #[test]
    fn one_step_matches_the_cpu() {
        let Some(gpu) = device() else { return };
        let mut swell = swell_flume();
        let generating = 1.5 * swell.swell.period;
        swell.scene.replay_after = 0.0;
        for (name, config, clock) in [
            ("dam break", Config::default(), None),
            ("swell", swell, Some(generating)),
            ("slab", slab_flume(), None),
        ] {
            let mut cpu = fresh(&config);
            for _ in 0..30 {
                cpu.step(1.0 / 60.0);
            }
            if let Some(time) = clock {
                cpu.elapsed = time;
            }
            let mut twin = fresh(&config);
            copy_state(&cpu, &mut twin);
            let mut solver = GpuFluid::new(gpu.clone(), &twin);
            if name == "dam break" {
                let center = cpu.pos[cpu.len() / 2];
                cpu.apply_radial_impulse(center, 110.0, 400.0);
                solver.apply_radial_impulse(center, 110.0, 400.0);
            }
            cpu.step(1.0 / 60.0);
            solver.step_back(&mut twin, 1.0 / 60.0);

            let d = config.fluid.spacing;
            let pos = largest_gap(&cpu.pos, &twin.pos);
            let vel = largest_gap(&cpu.vel, &twin.vel);
            let foam = largest_difference(&cpu.foam, &twin.foam);
            let spray = largest_difference(&cpu.spray, &twin.spray);
            assert!(pos < 1e-4 * d, "{name}: positions differ by {pos}");
            assert!(vel < 1e-2 * d, "{name}: velocities differ by {vel}");
            assert!(
                foam < 1e-4 && spray < 1e-4,
                "{name}: foam {foam}, spray {spray}"
            );
            assert_eq!(twin.elapsed, cpu.elapsed);
            assert_eq!(twin.previous_pos, cpu.previous_pos);
            assert!(solver.max_neighbors() <= solver.capacity());

            let readout = solver.readout(&twin).unwrap();
            let close = |a: f32, b: f32| (a - b).abs() <= 1e-3 * a.abs().max(b.abs()).max(1e-3);
            assert!(
                close(readout.compression, cpu.compression_error()),
                "{name}: {readout:?}"
            );
            assert!(
                close(readout.peak_speed, cpu.max_speed()),
                "{name}: {readout:?}"
            );
            let bulk = cpu.interior_density_ratio();
            assert!(
                close(readout.bulk, bulk) || (readout.bulk.is_nan() && bulk.is_nan()),
                "{name}: bulk {} against {bulk}",
                readout.bulk
            );
        }
    }

    /// The comparison that matters over time. Rounding alone makes the two runs
    /// part company within a second -- the `sensitivity` diagnostic shows the CPU
    /// drifting from itself just as fast after a nudge a millionth of a spacing
    /// wide -- so positions cannot be compared far apart. Instead, a GPU step from
    /// the CPU's exact state has to agree all the way through the break, and a
    /// GPU run left to itself has to throw its lip over the reef when the CPU's does.
    #[test]
    fn the_slab_flume_matches_the_cpu_through_the_break() {
        let Some(gpu) = device() else { return };
        let config = slab_flume();
        let (b, d) = (config.bounds(), config.fluid.spacing);
        let reef = b.min.x + config.world.width * config.wave.reef_start;
        let beach = b.max.x - config.world.width * config.wave.beach_width;
        let mut cpu = fresh(&config);
        let mut free = fresh(&config);
        let mut free_solver = GpuFluid::new(gpu.clone(), &free);
        let mut shadow = fresh(&config);
        let mut shadow_solver = GpuFluid::new(gpu, &shadow);

        let overturn = |fluid: &Fluid| {
            let lip: Vec<_> = overturned_columns(fluid, config.wave)
                .into_iter()
                .filter(|x| *x > reef)
                .collect();
            assert!(
                lip.iter().all(|x| *x < beach),
                "overturned on the beach: {lip:?}"
            );
            lip.len() >= 2
        };
        let (mut cpu_break, mut gpu_break) = (None, None);
        let mut worst = 0.0f32;
        for step in 1..=20 * 60 {
            let checkpoint = step % 30 == 0;
            if checkpoint {
                copy_state(&cpu, &mut shadow);
                shadow_solver.upload(&shadow);
            }
            cpu.step(1.0 / 60.0);
            free_solver.step_back(&mut free, 1.0 / 60.0);
            if checkpoint {
                shadow_solver.step_back(&mut shadow, 1.0 / 60.0);
                let gap = largest_gap(&cpu.pos, &shadow.pos);
                worst = worst.max(gap);
                assert!(
                    gap < 3e-4 * d,
                    "at {:.2}s a GPU step from the CPU's state is {gap} away",
                    cpu.elapsed
                );
            }
            if step % 6 == 0 {
                if cpu_break.is_none() && overturn(&cpu) {
                    cpu_break = Some(cpu.elapsed);
                }
                if gpu_break.is_none() && overturn(&free) {
                    gpu_break = Some(free.elapsed);
                }
                if cpu_break.is_some() && gpu_break.is_some() {
                    break;
                }
            }
        }
        let (Some(cpu_break), Some(gpu_break)) = (cpu_break, gpu_break) else {
            panic!("did not overturn over the reef: cpu {cpu_break:?}, gpu {gpu_break:?}");
        };
        println!(
            "worst one-step gap {:.1e} spacings; lip thrown at {cpu_break:.2}s on the CPU, {gpu_break:.2}s on the GPU",
            worst / d
        );
        assert!(
            (cpu_break - gpu_break).abs() <= 0.5,
            "the GPU threw its lip at {gpu_break:.2}s, the CPU at {cpu_break:.2}s"
        );
        assert!(free.pos.iter().all(|p| p.is_finite()));
        assert!(free_solver.max_neighbors() <= free_solver.capacity());
        assert!(shadow_solver.max_neighbors() <= shadow_solver.capacity());
    }

    /// Replays and captures depend on this: the threads scatter particles into
    /// cells in any order, and sorting each cell is what makes the result repeat.
    #[test]
    fn gpu_runs_repeat_exactly() {
        let Some(gpu) = device() else { return };
        let config = slab_flume();
        let run = |fluid: &mut Fluid, solver: &mut GpuFluid| {
            for _ in 0..90 {
                solver.step(fluid, 1.0 / 60.0);
            }
            solver.download(fluid);
            fluid.pos.clone()
        };
        let mut a = fresh(&config);
        let mut solver_a = GpuFluid::new(gpu.clone(), &a);
        let mut b = fresh(&config);
        let mut solver_b = GpuFluid::new(gpu, &b);
        let first = run(&mut a, &mut solver_a);
        assert_eq!(first, run(&mut b, &mut solver_b));
        // A reset is picked up without an explicit upload, and replays the same.
        config.reset_fluid(&mut a);
        assert_eq!(first, run(&mut a, &mut solver_a));
    }

    /// Renumbering moves particles between slots and back: a step taken after
    /// the particles were shuffled into grid order agrees, particle for
    /// particle, with the same step taken in the original numbering.
    #[test]
    fn renumbering_keeps_every_particle_its_own() {
        let Some(gpu) = device() else { return };
        let config = slab_flume();
        let mut cpu = fresh(&config);
        for _ in 0..60 {
            cpu.step(1.0 / 60.0);
        }
        let run = |renumber: bool| {
            let mut fluid = fresh(&config);
            copy_state(&cpu, &mut fluid);
            let mut solver = GpuFluid::new(gpu.clone(), &fluid);
            solver.step_back(&mut fluid, 1.0 / 60.0);
            if !renumber {
                // A fresh upload has no grid to renumber from.
                solver.upload(&fluid);
            }
            solver.step_back(&mut fluid, 1.0 / 60.0);
            fluid
        };
        let (plain, shuffled) = (run(false), run(true));
        let d = config.fluid.spacing;
        assert!(largest_gap(&plain.pos, &shuffled.pos) < 1e-4 * d);
        assert!(largest_gap(&plain.previous_pos, &shuffled.previous_pos) < 1e-4 * d);
        assert!(largest_gap(&plain.vel, &shuffled.vel) < 1e-2 * d);
        assert!(largest_difference(&plain.foam, &shuffled.foam) < 1e-4);
        assert!(largest_difference(&plain.spray, &shuffled.spray) < 1e-4);
    }

    #[test]
    fn crowded_neighbourhoods_widen_the_lists() {
        let Some(gpu) = device() else { return };
        let config = Config::default();
        let mut cpu = fresh(&config);
        for _ in 0..30 {
            cpu.step(1.0 / 60.0);
        }
        let mut twin = fresh(&config);
        copy_state(&cpu, &mut twin);
        let mut solver = GpuFluid::new(gpu, &twin);
        solver.shrink_lists(&twin, 8);
        solver.step_back(&mut twin, 1.0 / 60.0);
        assert!(solver.max_neighbors() > 8);
        assert!(solver.capacity() >= solver.max_neighbors());

        // Wide enough again, it goes back to matching.
        cpu.step(1.0 / 60.0);
        copy_state(&cpu, &mut twin);
        solver.upload(&twin);
        cpu.step(1.0 / 60.0);
        solver.step_back(&mut twin, 1.0 / 60.0);
        assert!(largest_gap(&cpu.pos, &twin.pos) < 1e-4 * config.fluid.spacing);
    }

    /// How far a GPU run drifts from the CPU over the slab flume, against how far
    /// the CPU drifts from itself after a rounding-sized nudge, plus a GPU step
    /// from the CPU's exact state along the way. Ignored; run with
    /// `cargo test --release sensitivity -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sensitivity() {
        let Some(gpu) = device() else { return };
        let config = slab_flume();
        let d = config.fluid.spacing;
        let mut cpu = fresh(&config);
        let mut nudged = fresh(&config);
        for (i, p) in nudged.pos.iter_mut().enumerate() {
            let s = ((i as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
            *p += Vec3::splat(s * 2e-5 * d);
        }
        let mut free = fresh(&config);
        let mut free_solver = GpuFluid::new(gpu.clone(), &free);
        let mut shadow = fresh(&config);
        let mut shadow_solver = GpuFluid::new(gpu, &shadow);
        let mean = |a: &[Vec3], b: &[Vec3]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| (*a - *b).length())
                .sum::<f32>()
                / a.len() as f32
                / d
        };
        println!(
            "{:>6} {:>11} {:>11} {:>11} {:>11} {:>11}",
            "t", "step max", "gpu max", "gpu mean", "nudge max", "nudge mean"
        );
        for step in 1..=20 * 60 {
            let checkpoint = step % 30 == 0;
            if checkpoint {
                copy_state(&cpu, &mut shadow);
                shadow_solver.upload(&shadow);
            }
            cpu.step(1.0 / 60.0);
            nudged.step(1.0 / 60.0);
            free_solver.step_back(&mut free, 1.0 / 60.0);
            if checkpoint {
                shadow_solver.step_back(&mut shadow, 1.0 / 60.0);
                println!(
                    "{:>6.2} {:>11.2e} {:>11.2e} {:>11.2e} {:>11.2e} {:>11.2e}",
                    cpu.elapsed,
                    largest_gap(&cpu.pos, &shadow.pos) / d,
                    largest_gap(&cpu.pos, &free.pos) / d,
                    mean(&cpu.pos, &free.pos),
                    largest_gap(&cpu.pos, &nudged.pos) / d,
                    mean(&cpu.pos, &nudged.pos),
                );
            }
        }
    }

    /// CPU against GPU per step at full `slab.toml` scale, and what the copy
    /// back costs. Ignored; run with
    /// `cargo test --release speedup -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn speedup() {
        let Some(gpu) = device() else { return };
        for name in ["config.toml", "wide.toml", "slab.toml"] {
            let config = Config::load(name).unwrap();
            let mut fluid = fresh(&config);
            let mut solver = GpuFluid::new(gpu.clone(), &fluid);
            let dt = 1.0 / 60.0;
            // Warm up for a second of wall-clock time: an idle GPU takes a
            // moment to reach full speed, which otherwise lands on the first
            // preset measured.
            let warm = std::time::Instant::now();
            while warm.elapsed().as_secs_f32() < 1.0 {
                solver.step(&mut fluid, dt);
            }
            let time = |steps: u32, mut f: Box<dyn FnMut() + '_>| {
                let start = std::time::Instant::now();
                for _ in 0..steps {
                    f();
                }
                start.elapsed().as_secs_f32() * 1000.0 / steps as f32
            };
            let gpu_only = time(20, Box::new(|| solver.step(&mut fluid, dt)));
            let with_copy = time(20, Box::new(|| solver.step_back(&mut fluid, dt)));
            let mut cpu = fresh(&config);
            let cpu_steps = if cpu.len() > 500_000 { 3 } else { 20 };
            let cpu_ms = time(cpu_steps, Box::new(|| cpu.step(dt)));
            println!(
                "{name:>12}: {:>8} particles  cpu {cpu_ms:>7.1} ms  gpu {gpu_only:>6.1} ms  gpu+copy {with_copy:>6.1} ms  ({:.1}x)",
                fluid.len(),
                cpu_ms / gpu_only,
            );
        }
    }

    /// Where a full-scale `slab.toml` substep spends its GPU time, from timestamp
    /// queries. Ignored; `cargo test --release profile_phases -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_phases() {
        let Some(gpu) = Gpu::headless_with(wgpu::Features::TIMESTAMP_QUERY) else {
            return;
        };
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            eprintln!("no timestamp queries on this adapter; skipping");
            return;
        }
        let config = Config::load("slab.toml").unwrap();
        let mut fluid = fresh(&config);
        let mut solver = GpuFluid::new(gpu.clone(), &fluid);
        for _ in 0..3 {
            solver.step(&mut fluid, 1.0 / 60.0);
        }
        let b = &solver.buffers;
        let n = b.n as u32;
        let iterations = fluid.params.iterations;
        let mut phases: Vec<(&str, Option<Pass>)> = vec![
            ("predict", Some(Pass::Predict)),
            ("grid", None),
            ("neighbours", Some(Pass::Neighbors)),
        ];
        for _ in 0..iterations {
            phases.push(("lambda", Some(Pass::Lambda)));
            phases.push(("delta", Some(Pass::Delta)));
            phases.push(("apply delta", Some(Pass::ApplyDelta)));
        }
        phases.push(("velocity", Some(Pass::Velocity)));
        phases.push(("viscosity", Some(Pass::Viscosity)));
        phases.push(("foam", Some(Pass::Foam)));
        let queries = 2 * phases.len() as u32;
        let set = gpu
            .device
            .wgpu_device()
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("phases"),
                ty: wgpu::QueryType::Timestamp,
                count: queries,
            });
        let resolve = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("timestamps"),
            size: 8 * queries as u64,
            usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("timestamps readback"),
            size: 8 * queries as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut totals: Vec<(&str, f64)> = Vec::new();
        let rounds = 5;
        for _ in 0..rounds {
            fluid.elapsed += 1.0 / 60.0;
            solver.write_params(&fluid, 1.0 / 60.0, fluid.elapsed, None, Vec3::ZERO);
            let mut encoder = solver.encoder();
            for (k, (_, which)) in phases.iter().enumerate() {
                let timestamps = |k: usize| wgpu::ComputePassTimestampWrites {
                    query_set: &set,
                    beginning_of_pass_write_index: Some(2 * k as u32),
                    end_of_pass_write_index: Some(2 * k as u32 + 1),
                };
                match which {
                    Some(which) => {
                        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(timestamps(k)),
                        });
                        solver.run(&mut pass, *which, n);
                    }
                    None => {
                        // The grid rebuild spans several passes; time it with an
                        // empty pass either side.
                        encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: &set,
                                beginning_of_pass_write_index: Some(2 * k as u32),
                                end_of_pass_write_index: None,
                            }),
                        });
                        solver.rebuild_grid(&mut encoder, &b.grid);
                        encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: &set,
                                beginning_of_pass_write_index: None,
                                end_of_pass_write_index: Some(2 * k as u32 + 1),
                            }),
                        });
                    }
                }
                if phases[k].0 == "viscosity" {
                    encoder.copy_buffer_to_buffer(&b.scratch, 0, &b.velocities, 0, None);
                }
            }
            encoder.resolve_query_set(&set, 0..queries, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, None);
            gpu.queue.submit([encoder.finish()]);
            let stamps: Vec<u64> = gpu.read(&staging, |bytes| cast::<u64>(bytes).into_owned());
            let period = gpu.queue.get_timestamp_period() as f64;
            for (k, (name, _)) in phases.iter().enumerate() {
                let ms = (stamps[2 * k + 1] - stamps[2 * k]) as f64 * period / 1e6;
                match totals.iter_mut().find(|(n, _)| n == name) {
                    Some(entry) => entry.1 += ms,
                    None => totals.push((name, ms)),
                }
            }
        }
        let sum: f64 = totals.iter().map(|(_, ms)| ms).sum::<f64>() / rounds as f64;
        for (name, ms) in &totals {
            let ms = ms / rounds as f64;
            println!("{name:>12}: {ms:>7.2} ms  {:>4.1}%", 100.0 * ms / sum);
        }
        println!(
            "{:>12}: {sum:>7.2} ms per substep, {} particles",
            "total", n
        );
    }

    /// Cost per step as the `slab.toml` flow develops, which is what the
    /// renumbering is for. Ignored; run with
    /// `cargo test --release sustained -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sustained() {
        let Some(gpu) = device() else { return };
        let config = Config::load("slab.toml").unwrap();
        let mut fluid = fresh(&config);
        let mut solver = GpuFluid::new(gpu, &fluid);
        let mut start = std::time::Instant::now();
        for step in 1..=720 {
            solver.step(&mut fluid, 1.0 / 60.0);
            if step % 60 == 0 {
                println!(
                    "t={:5.2}s  {:6.1} ms/step",
                    fluid.elapsed,
                    start.elapsed().as_secs_f32() * 1000.0 / 60.0
                );
                start = std::time::Instant::now();
            }
        }
    }
}
