//! A 3D Position Based Fluids solver (Macklin & Müller, SIGGRAPH 2013).
//!
//! PBF resolves incompressibility as a positional constraint solved by Jacobi
//! iteration rather than as a pressure force, which is what lets it stay stable
//! at a 1/60 s step. A force-based SPH solver at this particle count would need
//! a step in the 1e-4 range to avoid blowing up.
//!
//! Particle state lives in flat arrays instead of in the ECS: every solver
//! iteration touches each particle's whole neighbourhood at random, which is far
//! cheaper over `Vec`s than over archetype storage -- and it lets the three hot
//! loops run under rayon, which 3D needs. A neighbourhood here holds roughly
//! twice what the 2D version did, because a ball has more room in it than a
//! disc, so the same particle count costs about twice as much per step.

use bevy::math::Vec3;
use bevy::prelude::Resource;
use core::f32::consts::PI;
use rayon::prelude::*;

/// An axis-aligned box. The fluid lives inside one of these.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}

impl Bounds {
    /// A box of the given size, centred on the origin.
    pub fn from_size(size: Vec3) -> Self {
        let half = size * 0.5;
        Self {
            min: -half,
            max: half,
        }
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

/// Smoothing kernels, with their normalisation constants folded in at build
/// time so the hot loops stay free of `powi`.
///
/// These constants are 3D. The 2D poly6 normalises over a disc and the 3D one
/// over a ball, so they are genuinely different numbers -- carrying the 2D ones
/// into 3D would leave the rest density silently wrong.
#[derive(Clone, Copy)]
struct Kernels {
    h: f32,
    h2: f32,
    poly6: f32,
    spiky: f32,
}

impl Kernels {
    fn new(h: f32) -> Self {
        Self {
            h,
            h2: h * h,
            // 3D normalisations: poly6 integrates to 1 over the ball of radius
            // h, spiky is the derivative constant -45/(pi h^6).
            poly6: 315.0 / (64.0 * PI * h.powi(9)),
            spiky: -45.0 / (PI * h.powi(6)),
        }
    }

    /// Poly6, taking r^2 to keep the square root out of the density loop.
    #[inline]
    fn poly6(&self, r2: f32) -> f32 {
        if r2 >= self.h2 {
            return 0.0;
        }
        let d = self.h2 - r2;
        self.poly6 * d * d * d
    }

    /// Gradient of the spiky kernel. Points from `i` towards `j` for an offset
    /// of `r = p_i - p_j`, because the constant is negative.
    #[inline]
    fn spiky_grad(&self, r: Vec3) -> Vec3 {
        let len = r.length();
        if len <= 1e-6 || len >= self.h {
            return Vec3::ZERO;
        }
        let d = self.h - len;
        r * (self.spiky * d * d / len)
    }
}

/// Solver tunables. Lengths are world units.
#[derive(Resource, Clone, Copy)]
pub struct FluidParams {
    /// Kernel radius. Everything else is expressed relative to this.
    pub smoothing_radius: f32,
    /// Rest spacing of the initial lattice. Sets how many neighbours a particle
    /// sees, and calibrates the rest density.
    pub spacing: f32,
    /// Jacobi iterations per substep. More iterations drive the compression
    /// down roughly linearly, at a matching cost: see the `sweep` test.
    pub iterations: u32,
    /// Solver steps per frame. Each one re-predicts positions and rebuilds
    /// neighbourhoods, so this is the knob that keeps the fastest particle from
    /// crossing a kernel radius in a single step.
    pub substeps: u32,
    /// Constraint force mixing, as a fraction of the lattice gradient reference.
    /// Softens the constraint so surface particles don't get huge multipliers.
    pub relaxation: f32,
    /// XSPH viscosity. 0 is inviscid and jittery, ~0.1 reads as water.
    pub viscosity: f32,
    /// Artificial pressure strength, as a fraction of the reference multiplier.
    /// Counteracts the clumping that the density constraint causes at the free
    /// surface (the "tensile instability").
    pub tensile_k: f32,
    /// Distance the artificial pressure is referenced against, in units of h.
    pub tensile_q: f32,
    /// Exponent on the artificial pressure term.
    pub tensile_n: i32,
    pub gravity: Vec3,
    /// Under-relaxation on the positional correction. Jacobi updates every
    /// particle against stale neighbours, so applying the full correction
    /// overshoots and rings; Gauss-Seidel would not need this.
    pub jacobi_relax: f32,
    /// Resolve compression only, leaving under-dense particles alone.
    ///
    /// A particle near the free surface has part of its kernel sticking out
    /// into nothing, so its density reads low even when the fluid is at rest
    /// density. Acting on that deficit pulls the surface inwards and squeezes
    /// the whole body -- which a deep 2D pool mostly hides, and a shallow 3D
    /// one does not, since almost every particle is within a kernel radius of
    /// a surface.
    pub clamp_constraint: bool,
    /// Fraction of tangential velocity kept when a particle touches a wall.
    pub wall_friction: f32,
    pub bounds: Bounds,
}

/// Uniform grid over the simulation bounds, rebuilt each step by counting sort.
/// The bounds are fixed and the cell size is the kernel radius, so a dense grid
/// beats a hash here: no modulo, no collisions, and neighbours land contiguously.
struct Grid {
    cell_size: f32,
    origin: Vec3,
    dims: [i32; 3],
    /// Start offset of each cell into `sorted`, plus a trailing total.
    cell_start: Vec<u32>,
    /// Scratch copy of `cell_start` used as a write cursor while scattering.
    cursor: Vec<u32>,
    /// Particle indices ordered by cell.
    sorted: Vec<u32>,
}

impl Grid {
    fn new(bounds: Bounds, cell_size: f32) -> Self {
        let size = bounds.size();
        let dims = [
            (size.x / cell_size).ceil() as i32 + 1,
            (size.y / cell_size).ceil() as i32 + 1,
            (size.z / cell_size).ceil() as i32 + 1,
        ];
        let cells = (dims[0] * dims[1] * dims[2]) as usize;
        Self {
            cell_size,
            origin: bounds.min,
            dims,
            cell_start: vec![0; cells + 1],
            cursor: vec![0; cells + 1],
            sorted: Vec::new(),
        }
    }

    #[inline]
    fn cell_of(&self, p: Vec3) -> [i32; 3] {
        let local = (p - self.origin) / self.cell_size;
        [
            (local.x as i32).clamp(0, self.dims[0] - 1),
            (local.y as i32).clamp(0, self.dims[1] - 1),
            (local.z as i32).clamp(0, self.dims[2] - 1),
        ]
    }

    #[inline]
    fn index(&self, c: [i32; 3]) -> usize {
        ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize
    }

    fn rebuild(&mut self, positions: &[Vec3]) {
        self.cell_start.fill(0);
        for &p in positions {
            let cell = self.index(self.cell_of(p));
            // Counts are written one slot high so the prefix sum below turns
            // them directly into start offsets.
            self.cell_start[cell + 1] += 1;
        }
        for i in 1..self.cell_start.len() {
            self.cell_start[i] += self.cell_start[i - 1];
        }
        self.cursor.copy_from_slice(&self.cell_start);
        self.sorted.resize(positions.len(), 0);
        for (i, &p) in positions.iter().enumerate() {
            let cell = self.index(self.cell_of(p));
            let slot = &mut self.cursor[cell];
            self.sorted[*slot as usize] = i as u32;
            *slot += 1;
        }
    }

    /// Runs `f` over every particle in the 27 cells around `p`.
    #[inline]
    fn for_each_candidate(&self, p: Vec3, mut f: impl FnMut(u32)) {
        let c = self.cell_of(p);
        let lo = [(c[0] - 1).max(0), (c[1] - 1).max(0), (c[2] - 1).max(0)];
        let hi = [
            (c[0] + 1).min(self.dims[0] - 1),
            (c[1] + 1).min(self.dims[1] - 1),
            (c[2] + 1).min(self.dims[2] - 1),
        ];
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                // The x run is contiguous in memory, so take it as one slice
                // rather than a cell at a time.
                let from = self.cell_start[self.index([lo[0], y, z])] as usize;
                let to = self.cell_start[self.index([hi[0], y, z]) + 1] as usize;
                for &j in &self.sorted[from..to] {
                    f(j);
                }
            }
        }
    }
}

/// The fluid. `pos` is the state you render; everything else is solver scratch.
#[derive(Resource)]
pub struct Fluid {
    pub params: FluidParams,
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// Predicted positions, which the constraint solver actually moves.
    pred: Vec<Vec3>,
    lambda: Vec<f32>,
    delta: Vec<Vec3>,
    vel_scratch: Vec<Vec3>,
    kernels: Kernels,
    grid: Grid,
    /// Neighbour lists, flattened. Built once per substep and reused across all
    /// solver iterations, which is where most of the speed comes from.
    neighbors: Vec<u32>,
    neighbor_start: Vec<u32>,
    /// Calibrated from the rest lattice, see [`lattice_reference`].
    rest_density: f32,
    epsilon: f32,
    tensile_scale: f32,
    /// Poly6 evaluated at `tensile_q * h`, the artificial pressure reference.
    tensile_w: f32,
}

impl Fluid {
    pub fn new(params: FluidParams) -> Self {
        let kernels = Kernels::new(params.smoothing_radius);
        let (rest_density, grad_ref) = lattice_reference(&kernels, params.spacing);
        let dq = params.tensile_q * params.smoothing_radius;
        Self {
            pos: Vec::new(),
            vel: Vec::new(),
            pred: Vec::new(),
            lambda: Vec::new(),
            delta: Vec::new(),
            vel_scratch: Vec::new(),
            grid: Grid::new(params.bounds, params.smoothing_radius),
            neighbors: Vec::new(),
            neighbor_start: Vec::new(),
            rest_density,
            epsilon: params.relaxation * grad_ref,
            // The artificial pressure has to share units with lambda, and
            // lambda scales like 1 / sum|grad C|^2. Deriving it from the same
            // reference keeps `tensile_k` meaningful if h or spacing change.
            tensile_scale: params.tensile_k / grad_ref,
            tensile_w: kernels.poly6(dq * dq),
            kernels,
            params,
        }
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Fills a corner of the bounds with a lattice of particles, which then
    /// collapses into a dam break.
    pub fn fill_block(&mut self, counts: [usize; 3]) {
        self.pos.clear();
        self.vel.clear();
        let d = self.params.spacing;
        let corner = self.params.bounds.min + Vec3::splat(d);
        for y in 0..counts[1] {
            for z in 0..counts[2] {
                for x in 0..counts[0] {
                    // Nudge alternating layers so the lattice isn't perfectly
                    // axis-aligned; a perfect grid takes a while to break
                    // symmetry, and in 3D it can sit there noticeably long.
                    let stagger = if y % 2 == 0 { 0.0 } else { d * 0.5 };
                    self.pos.push(
                        corner
                            + Vec3::new(
                                x as f32 * d + stagger,
                                y as f32 * d,
                                z as f32 * d + stagger,
                            ),
                    );
                    self.vel.push(Vec3::ZERO);
                }
            }
        }
        let n = self.pos.len();
        self.pred = vec![Vec3::ZERO; n];
        self.lambda = vec![0.0; n];
        self.delta = vec![Vec3::ZERO; n];
        self.vel_scratch = vec![Vec3::ZERO; n];
        // These describe the configuration we just threw away. Leaving them
        // would let `compression_error` report densities for the old state,
        // and would index out of bounds if the particle count changed.
        self.neighbors.clear();
        self.neighbor_start.clear();
    }

    /// The frontmost particle within `radius` of the ray, if any.
    ///
    /// A screen position names a ray, not a point, so an interaction in 3D has
    /// to get its depth from somewhere. Taking it from the fluid itself is what
    /// makes the mouse feel like the 2D version did: you push the water you are
    /// pointing at, rather than whatever happens to lie on some reference
    /// plane. Frontmost rather than nearest-to-the-ray, so pointing at a deep
    /// pool pushes its surface instead of reaching through to the far side.
    pub fn nearest_along_ray(&self, origin: Vec3, direction: Vec3, radius: f32) -> Option<Vec3> {
        let dir = direction.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let r2 = radius * radius;
        self.pos
            .par_iter()
            .filter_map(|p| {
                let rel = *p - origin;
                let along = rel.dot(dir);
                if along <= 0.0 {
                    return None;
                }
                // Perpendicular distance to the ray, via Pythagoras on the
                // component along it -- no square roots in the filter.
                let perp2 = rel.length_squared() - along * along;
                (perp2 < r2).then_some((along, *p))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, p)| p)
    }

    /// Pushes particles within `radius` of `center` radially outwards, with a
    /// smooth falloff to zero at the rim. A negative `strength` pulls inwards.
    /// This is the mouse.
    pub fn apply_radial_impulse(&mut self, center: Vec3, radius: f32, strength: f32) {
        let r2 = radius * radius;
        for (p, v) in self.pos.iter().zip(self.vel.iter_mut()) {
            let offset = *p - center;
            let d2 = offset.length_squared();
            if d2 < r2 && d2 > 1e-6 {
                let falloff = 1.0 - d2 / r2;
                *v += offset.normalize() * (strength * falloff);
            }
        }
    }

    /// Advances the fluid by `dt` seconds, in `params.substeps` equal steps.
    pub fn step(&mut self, dt: f32) {
        if self.pos.is_empty() || dt <= 0.0 {
            return;
        }
        let substeps = self.params.substeps.max(1);
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            self.substep(sub_dt);
        }
    }

    fn substep(&mut self, dt: f32) {
        // 1. Apply external forces and predict where particles want to be.
        let (min, max) = self.wall_limits();
        let gravity = self.params.gravity;
        self.pred
            .par_iter_mut()
            .zip(self.vel.par_iter_mut())
            .zip(self.pos.par_iter())
            .for_each(|((pred, vel), pos)| {
                *vel += gravity * dt;
                *pred = (*pos + *vel * dt).clamp(min, max);
            });

        // 2. Neighbourhoods, from the predicted positions.
        self.grid.rebuild(&self.pred);
        self.build_neighbors();

        // 3. Jacobi-solve the density constraint.
        for _ in 0..self.params.iterations {
            self.solve_density();
        }

        // 4. Derive velocity from the positions the solver settled on, which is
        //    what makes wall collisions and constraint corrections energy-safe:
        //    a particle pushed out of a wall simply *has* less velocity
        //    afterwards, with no restitution coefficient to tune.
        let inv_dt = 1.0 / dt;
        let friction = self.params.wall_friction;
        let touch = self.params.spacing * 0.25;
        self.vel
            .par_iter_mut()
            .zip(self.pos.par_iter_mut())
            .zip(self.pred.par_iter())
            .for_each(|((vel, pos), pred)| {
                *vel = (*pred - *pos) * inv_dt;
                *pos = *pred;
                // Tangential drag, applied here rather than during prediction:
                // step 4 overwrites the whole velocity, so anything damped
                // earlier in the substep is simply discarded.
                if pos.x <= min.x + touch || pos.x >= max.x - touch {
                    vel.y *= friction;
                    vel.z *= friction;
                }
                if pos.y <= min.y + touch || pos.y >= max.y - touch {
                    vel.x *= friction;
                    vel.z *= friction;
                }
                if pos.z <= min.z + touch || pos.z >= max.z - touch {
                    vel.x *= friction;
                    vel.y *= friction;
                }
            });

        self.apply_viscosity();
    }

    /// The box the particles are actually held inside: the bounds, inset by
    /// half a spacing so a particle's own volume stays in the room.
    #[inline]
    fn wall_limits(&self) -> (Vec3, Vec3) {
        let margin = Vec3::splat(self.params.spacing * 0.5);
        (
            self.params.bounds.min + margin,
            self.params.bounds.max - margin,
        )
    }

    /// Builds every particle's neighbour list, in parallel.
    ///
    /// Counted first, then filled, because the lists are variable length and a
    /// single shared output vector cannot be appended to from several threads.
    /// Counting costs a second pass over the same candidates, which is still
    /// far cheaper than doing the whole thing on one core: in 3D each particle
    /// screens around a hundred candidates across 27 cells, and left serial
    /// this was the majority of the frame.
    fn build_neighbors(&mut self) {
        let (grid, pred, h2) = (&self.grid, &self.pred, self.kernels.h2);
        let n = pred.len();

        // Pass 1: how many neighbours each particle has. Written into the
        // starts array one slot high, so the scan below turns the counts
        // directly into offsets.
        self.neighbor_start.clear();
        self.neighbor_start.resize(n + 1, 0);
        self.neighbor_start[1..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, count)| {
                let pi = pred[i];
                let mut c = 0;
                grid.for_each_candidate(pi, |j| {
                    if j as usize != i && (pi - pred[j as usize]).length_squared() < h2 {
                        c += 1;
                    }
                });
                *count = c;
            });
        for i in 1..=n {
            self.neighbor_start[i] += self.neighbor_start[i - 1];
        }

        // Pass 2: fill. Each task owns a contiguous run of particles and the
        // matching contiguous run of the output, carved out up front by
        // `split_at_mut`, so the writes cannot overlap.
        let starts = &self.neighbor_start;
        self.neighbors.clear();
        self.neighbors.resize(starts[n] as usize, 0);

        const CHUNK: usize = 256;
        let mut rest = &mut self.neighbors[..];
        let mut tasks = Vec::with_capacity(n / CHUNK + 1);
        for from in (0..n).step_by(CHUNK) {
            let to = (from + CHUNK).min(n);
            let len = (starts[to] - starts[from]) as usize;
            let (head, tail) = rest.split_at_mut(len);
            tasks.push((from, to, head));
            rest = tail;
        }

        tasks.into_par_iter().for_each(|(from, to, out)| {
            let base = starts[from];
            for i in from..to {
                let pi = pred[i];
                let mut w = (starts[i] - base) as usize;
                grid.for_each_candidate(pi, |j| {
                    if j as usize != i && (pi - pred[j as usize]).length_squared() < h2 {
                        out[w] = j;
                        w += 1;
                    }
                });
            }
        });
    }

    fn solve_density(&mut self) {
        let inv_rho0 = 1.0 / self.rest_density;
        let kernels = self.kernels;
        let epsilon = self.epsilon;
        let tensile_scale = self.tensile_scale;
        let tensile_w = self.tensile_w;
        let tensile_n = self.params.tensile_n;
        let relax = self.params.jacobi_relax;
        let clamp = self.params.clamp_constraint;
        let (min, max) = self.wall_limits();
        let pred = &self.pred;
        let neighbors = &self.neighbors;
        let starts = &self.neighbor_start;

        // Density, constraint value, and the multiplier that will correct it.
        self.lambda.par_iter_mut().enumerate().for_each(|(i, out)| {
            let pi = pred[i];
            let mut rho = kernels.poly6(0.0);
            // Gradient of C_i with respect to p_i, and the sum of squared
            // gradients with respect to every particle in the neighbourhood.
            let mut grad_self = Vec3::ZERO;
            let mut sum_sq = 0.0;
            for &j in &neighbors[starts[i] as usize..starts[i + 1] as usize] {
                let r = pi - pred[j as usize];
                rho += kernels.poly6(r.length_squared());
                let g = kernels.spiky_grad(r) * inv_rho0;
                grad_self += g;
                sum_sq += g.length_squared();
            }
            sum_sq += grad_self.length_squared();
            let c = rho * inv_rho0 - 1.0;
            if clamp && c <= 0.0 {
                *out = 0.0;
            } else {
                *out = -c / (sum_sq + epsilon);
            }
        });

        // Positional correction.
        let lambda = &self.lambda;
        self.delta.par_iter_mut().enumerate().for_each(|(i, out)| {
            let pi = pred[i];
            let li = lambda[i];
            let mut d = Vec3::ZERO;
            for &j in &neighbors[starts[i] as usize..starts[i + 1] as usize] {
                let r = pi - pred[j as usize];
                let ratio = kernels.poly6(r.length_squared()) / tensile_w;
                let s_corr = -tensile_scale * ratio.powi(tensile_n);
                d += kernels.spiky_grad(r) * (li + lambda[j as usize] + s_corr);
            }
            *out = d * (inv_rho0 * relax);
        });

        let delta = &self.delta;
        self.pred
            .par_iter_mut()
            .zip(delta.par_iter())
            .for_each(|(p, d)| *p = (*p + *d).clamp(min, max));
    }

    /// XSPH: nudge each particle towards its neighbourhood's mean velocity.
    /// Cheap, and it's what stops the surface looking like sand.
    fn apply_viscosity(&mut self) {
        let inv_rho0 = 1.0 / self.rest_density;
        let kernels = self.kernels;
        let viscosity = self.params.viscosity;
        let pos = &self.pos;
        let vel = &self.vel;
        let neighbors = &self.neighbors;
        let starts = &self.neighbor_start;

        self.vel_scratch
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out)| {
                let (pi, vi) = (pos[i], vel[i]);
                let mut dv = Vec3::ZERO;
                for &j in &neighbors[starts[i] as usize..starts[i + 1] as usize] {
                    let w = kernels.poly6((pi - pos[j as usize]).length_squared());
                    dv += (vel[j as usize] - vi) * w;
                }
                *out = vi + dv * (viscosity * inv_rho0);
            });
        self.vel.copy_from_slice(&self.vel_scratch);
    }

    /// Mean density error as a fraction of rest density, over compressed
    /// particles only. Surface particles are legitimately under-dense, so
    /// including them would report a large error for a perfectly good fluid.
    pub fn compression_error(&self) -> f32 {
        // Neighbour lists are only valid between a `step` and the next
        // `fill_block`; without them there is nothing to measure.
        if self.neighbor_start.len() != self.pos.len() + 1 {
            return 0.0;
        }
        let total: f32 = (0..self.pos.len())
            .into_par_iter()
            .map(|i| {
                let pi = self.pos[i];
                let mut rho = self.kernels.poly6(0.0);
                for &j in &self.neighbors
                    [self.neighbor_start[i] as usize..self.neighbor_start[i + 1] as usize]
                {
                    rho += self
                        .kernels
                        .poly6((pi - self.pos[j as usize]).length_squared());
                }
                (rho / self.rest_density - 1.0).max(0.0)
            })
            .sum();
        total / self.pos.len() as f32
    }

    /// Local number density, as a multiple of the rest lattice's, averaged over
    /// particles at least one kernel radius from any surface.
    ///
    /// This is the honest test of whether the fluid holds its volume. Unlike
    /// `compression_error`, which reads the SPH density estimate, this counts
    /// particles in a ball and divides by its volume, so it cannot be fooled by
    /// the kernel truncation that makes every particle near a free surface look
    /// under-dense. 1.0 is a fluid at rest density.
    pub fn interior_density_ratio(&self) -> f32 {
        if self.neighbor_start.len() != self.pos.len() + 1 {
            return 1.0;
        }
        let h = self.kernels.h;
        let ball = 4.0 / 3.0 * PI * h.powi(3);
        let rest_n = 1.0 / self.params.spacing.powi(3);
        let (min, max) = (self.params.bounds.min, self.params.bounds.max);

        // A particle is interior if it is a kernel radius from every wall and
        // has fluid a kernel radius above it.
        let top = self
            .pos
            .par_iter()
            .map(|p| p.y)
            .reduce(|| f32::MIN, f32::max);
        let samples: Vec<f32> = self
            .pos
            .par_iter()
            .enumerate()
            .filter(|(_, p)| {
                p.x > min.x + h
                    && p.x < max.x - h
                    && p.z > min.z + h
                    && p.z < max.z - h
                    && p.y > min.y + h
                    && p.y < top - h
            })
            .map(|(i, _)| {
                let count = (self.neighbor_start[i + 1] - self.neighbor_start[i]) as f32;
                (count + 1.0) / ball / rest_n
            })
            .collect();
        if samples.is_empty() {
            // Too little water to have a bulk: every particle is within a
            // kernel radius of a surface. Nothing to report rather than NaN.
            return f32::NAN;
        }
        samples.iter().sum::<f32>() / samples.len() as f32
    }

    pub fn max_speed(&self) -> f32 {
        self.vel
            .par_iter()
            .map(|v| v.length())
            .reduce(|| 0.0f32, f32::max)
    }
}

/// Density and constraint-gradient magnitude a particle would see at the centre
/// of an unbounded cubic lattice of the given spacing, with unit mass.
///
/// Calibrating against this instead of hard-coding a rest density means the
/// tunables keep their meaning when `smoothing_radius` or `spacing` change --
/// otherwise every one of them has to be retuned by hand. It is also what makes
/// the move from 2D to 3D survivable: the kernel constants and the neighbour
/// count both changed, and this recomputes the target density from them.
fn lattice_reference(kernels: &Kernels, spacing: f32) -> (f32, f32) {
    let reach = (kernels.h / spacing).ceil() as i32;
    let mut rho = 0.0;
    let mut sum_sq = 0.0;
    let mut grad_self = Vec3::ZERO;
    for gz in -reach..=reach {
        for gy in -reach..=reach {
            for gx in -reach..=reach {
                let d = Vec3::new(gx as f32, gy as f32, gz as f32) * spacing;
                rho += kernels.poly6(d.length_squared());
                if gx != 0 || gy != 0 || gz != 0 {
                    let g = kernels.spiky_grad(d);
                    grad_self += g;
                    sum_sq += g.length_squared();
                }
            }
        }
    }
    // The 1/rho0 factor in grad C is applied here, once rho0 is known.
    let grad_ref = (sum_sq + grad_self.length_squared()) / (rho * rho);
    (rho, grad_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    /// The tuned defaults, so the tests exercise the configuration that ships.
    fn params() -> FluidParams {
        Config::default().fluid_params()
    }

    fn block() -> [usize; 3] {
        Config::default().fluid.block
    }

    fn settled(steps: u32) -> Fluid {
        let mut fluid = Fluid::new(params());
        fluid.fill_block(block());
        for _ in 0..steps {
            fluid.step(1.0 / 60.0);
        }
        fluid
    }

    #[test]
    fn rest_density_matches_the_seed_lattice() {
        // The 3D kernels have different normalisation constants from the 2D
        // ones, and the neighbourhood holds roughly twice as many particles.
        // This is the check that the calibration followed the move across.
        let mut fluid = Fluid::new(params());
        fluid.fill_block(block());
        // One tiny step just to populate neighbour lists; the lattice has not
        // had time to move, so the interior should sit at rest density.
        fluid.step(1.0 / 600.0);
        assert!(
            fluid.compression_error() < 0.05,
            "seed lattice is not near rest density: {}",
            fluid.compression_error()
        );
    }

    #[test]
    fn stays_finite_and_bounded() {
        let fluid = settled(400);
        let b = fluid.params.bounds;
        for (i, p) in fluid.pos.iter().enumerate() {
            assert!(p.is_finite(), "particle {i} left the reals: {p:?}");
            assert!(
                p.cmpge(b.min - 1.0).all() && p.cmple(b.max + 1.0).all(),
                "particle {i} escaped the box: {p:?}"
            );
        }
    }

    #[test]
    fn settles_without_gaining_energy() {
        let fluid = settled(600);
        // Ten seconds in, a dam break should have sloshed out nearly all of its
        // kinetic energy. A solver that is pumping energy shows up here first.
        assert!(
            fluid.max_speed() < 500.0,
            "fluid is gaining energy, max speed {}",
            fluid.max_speed()
        );
        assert!(
            fluid.compression_error() < 0.06,
            "fluid is compressing, error {}",
            fluid.compression_error()
        );
    }

    #[test]
    fn settles_into_a_flat_pool() {
        // A dam break that has come to rest should have run out to every wall
        // and levelled off. This is the check that says "this behaves like
        // water" rather than merely "this is stable".
        let fluid = settled(600);
        let b = fluid.params.bounds;
        let lo = fluid.pos.iter().copied().reduce(Vec3::min).unwrap();
        let hi = fluid.pos.iter().copied().reduce(Vec3::max).unwrap();
        let spread = (hi - lo) / b.size();
        assert!(
            spread.x > 0.9 && spread.z > 0.9,
            "the fluid did not spread across the floor: {spread:?}"
        );

        // Flat, not heaped: the top of the pool should be within a couple of
        // particle layers everywhere it is sampled.
        let surface_of = |keep: fn(&Vec3) -> bool| {
            fluid
                .pos
                .iter()
                .filter(|p| keep(p))
                .map(|p| p.y)
                .fold(f32::MIN, f32::max)
        };
        let left = surface_of(|p| p.x < 0.0);
        let right = surface_of(|p| p.x >= 0.0);
        assert!(
            (left - right).abs() < 3.0 * fluid.params.spacing,
            "the pool is not level: {left:.1} on one side, {right:.1} on the other"
        );
    }

    #[test]
    fn the_bulk_holds_its_rest_density() {
        // Measured by counting particles in a ball, not by the SPH density
        // estimate: near a free surface the kernel is truncated and reads low
        // whatever the fluid is really doing, so `compression_error` cannot
        // settle this on its own.
        //
        // Only the bulk is asserted. The layer against the floor packs about
        // 50% too densely, because a hard wall truncates the kernel the same
        // way a free surface does and there are no boundary particles to
        // complete it -- so the settled pool sits around 10% shallower than its
        // rest volume implies. Fixing that is a boundary-handling job, not a
        // solver one; see the README.
        let fluid = settled(600);
        let ratio = fluid.interior_density_ratio();
        assert!(
            (0.95..1.08).contains(&ratio),
            "bulk density is {ratio:.3} of rest, expected about 1.0"
        );
    }

    #[test]
    fn a_ray_finds_the_near_face_of_the_fluid() {
        let mut fluid = Fluid::new(params());
        fluid.fill_block(block());
        let lo = fluid.pos.iter().copied().reduce(Vec3::min).unwrap();
        let hi = fluid.pos.iter().copied().reduce(Vec3::max).unwrap();
        let centre = (lo + hi) * 0.5;

        // Fire at the block from well outside it, down the +x axis.
        let origin = Vec3::new(lo.x - 500.0, centre.y, centre.z);
        let hit = fluid
            .nearest_along_ray(origin, Vec3::X, 30.0)
            .expect("a ray through the middle of the block should hit it");
        assert!(
            (hit.x - lo.x).abs() < 3.0 * fluid.params.spacing,
            "hit the far side at x={:.1}, near face is at {:.1}",
            hit.x,
            lo.x
        );

        // Aimed away from the fluid it should find nothing, rather than
        // reporting the closest particle behind the camera.
        assert!(fluid.nearest_along_ray(origin, -Vec3::X, 30.0).is_none());
        // And a ray that passes wide misses.
        let wide = Vec3::new(lo.x - 500.0, hi.y + 300.0, centre.z);
        assert!(fluid.nearest_along_ray(wide, Vec3::X, 30.0).is_none());
    }

    #[test]
    fn radial_impulse_pushes_out_and_pulls_in() {
        let mut fluid = Fluid::new(params());
        fluid.fill_block(block());
        let center = fluid.pos[fluid.len() / 2];

        let probe = fluid
            .pos
            .iter()
            .position(|p| {
                let d = (*p - center).length();
                d > 20.0 && d < 60.0
            })
            .expect("no particle in the probe annulus");
        let offset = fluid.pos[probe] - center;

        fluid.apply_radial_impulse(center, 130.0, 100.0);
        let pushed = fluid.vel[probe];
        assert!(
            pushed.dot(offset) > 0.0,
            "push sent the particle inwards: {pushed:?} against offset {offset:?}"
        );

        // The same impulse negated should cancel it exactly.
        fluid.apply_radial_impulse(center, 130.0, -100.0);
        assert!(
            fluid.vel[probe].length() < 1e-3,
            "pull did not cancel the push: {:?}",
            fluid.vel[probe]
        );

        let far = fluid
            .pos
            .iter()
            .position(|p| (*p - center).length() > 200.0)
            .expect("no particle outside the radius");
        assert_eq!(fluid.vel[far], Vec3::ZERO);
    }

    /// Parameter sweep. Ignored by default; run with
    /// `cargo test --release sweep -- --ignored --nocapture`.
    ///
    /// 3D is far more forgiving on compression than 2D was -- a neighbourhood
    /// holds about twice as many particles, so each Jacobi pass carries more
    /// information -- and correspondingly far tighter on time. This is the
    /// sweep that spends the surplus accuracy on frame budget.
    #[test]
    #[ignore]
    fn sweep() {
        println!(
            "{:>6} {:>5}  {:>9} {:>9} {:>10} {:>9}",
            "relax", "iter", "ms/step", "peak_spd", "compress", "interior"
        );
        for &relax in &[0.4f32, 0.5] {
            for &iters in &[4u32, 6, 8, 10, 12] {
                let sweep_params = FluidParams {
                    jacobi_relax: relax,
                    iterations: iters,
                    ..params()
                };
                let mut fluid = Fluid::new(sweep_params);
                fluid.fill_block(block());

                let mut peak = 0.0f32;
                let start = std::time::Instant::now();
                for step in 0..500 {
                    fluid.step(1.0 / 60.0);
                    if step >= 60 {
                        peak = peak.max(fluid.max_speed());
                    }
                    if !fluid.max_speed().is_finite() {
                        break;
                    }
                }
                let ms = start.elapsed().as_secs_f32() * 1000.0 / 500.0;

                // Where the pool sits against where its own volume says it
                // should. Compared on MEAN height rather than on the surface:
                // a pool whose fluid volume is `depth` deep has its particle
                // centres spread from half a spacing off the floor to half a
                // spacing below the surface, so they average `depth / 2`. That
                // half-spacing offset at each end is real, and comparing a
                // surface reading against the raw volume depth quietly reports
                // a correct fluid as 19% short.
                let interior = fluid.interior_density_ratio();

                println!(
                    "{:>6.2} {:>5}  {:>9.2} {:>9.1} {:>10.4} {:>9.3}",
                    relax,
                    iters,
                    ms,
                    peak,
                    fluid.compression_error(),
                    interior
                );
            }
        }
    }

    /// How the per-step cost scales with particle count, which is what decides
    /// how far the config can be pushed. Ignored by default; run with
    /// `cargo test --release scaling -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn scaling() {
        println!(
            "{:>14} {:>10} {:>10} {:>5} {:>10} {:>8}",
            "block", "particles", "ms/step", "sub", "compress", "60Hz?"
        );
        let max = Config::default().max_block();
        for block in [
            [18usize, 20, 18],
            [24, 26, 24],
            [30, 29, 30],
            [max[0], max[1], max[2]],
        ] {
            let mut config = Config::default();
            config.fluid.block = block;
            // A taller block falls further and so lands faster, which is its
            // own reason to substep -- nothing to do with resolution.
            config.solver.substeps = config.required_substeps();
            config.validate().expect("scaling config should be valid");

            let mut fluid = Fluid::new(config.fluid_params());
            fluid.fill_block(block);
            // Warm up past the initial collapse, where the fluid is at its
            // densest and the neighbour lists are longest.
            for _ in 0..120 {
                fluid.step(1.0 / 60.0);
            }
            let start = std::time::Instant::now();
            for _ in 0..150 {
                fluid.step(1.0 / 60.0);
            }
            let ms = start.elapsed().as_secs_f32() * 1000.0 / 150.0;
            println!(
                "{:>14} {:>10} {:>10.2} {:>5} {:>10.4} {:>8}",
                format!("[{}, {}, {}]", block[0], block[1], block[2]),
                fluid.len(),
                ms,
                config.solver.substeps,
                fluid.compression_error(),
                if ms < 16.7 { "yes" } else { "no" }
            );
        }
    }

    /// How hard the mouse has to push to visibly move the water. Ignored by
    /// default; run with
    /// `cargo test --release mouse_response -- --ignored --nocapture`.
    ///
    /// Measured headless because the windowed version cannot see the signal:
    /// the pool is still sloshing from its own dam break at the point a hand
    /// would reach for the mouse, and that swamps the push.
    #[test]
    #[ignore]
    fn mouse_response() {
        let mut settled = Fluid::new(params());
        settled.fill_block(block());
        for _ in 0..1200 {
            settled.step(1.0 / 60.0);
        }
        let rest = settled.pos.clone();
        println!("settled at peak {:.1} u/s", settled.max_speed());
        println!("{:>10} {:>8}  {:>10} {:>10}", "strength", "radius", "peak", "surface_up");

        for &radius in &[85.0f32, 110.0] {
            for &strength in &[4_000.0f32, 12_000.0, 30_000.0, 60_000.0] {
                let mut fluid = Fluid::new(params());
                fluid.fill_block(block());
                fluid.pos.copy_from_slice(&rest);
                fluid.vel.iter_mut().for_each(|v| *v = Vec3::ZERO);

                // Push just under the surface, in the middle of the pool,
                // holding the button for a third of a second.
                let b = fluid.params.bounds;
                let top = fluid.pos.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                let at = Vec3::new(0.0, top - fluid.params.spacing, 0.0);
                let before = top;
                let mut peak = 0.0f32;
                for _ in 0..20 {
                    fluid.apply_radial_impulse(at, radius, strength / 60.0);
                    fluid.step(1.0 / 60.0);
                    peak = peak.max(fluid.max_speed());
                }
                for _ in 0..20 {
                    fluid.step(1.0 / 60.0);
                    peak = peak.max(fluid.max_speed());
                }
                let after = fluid.pos.iter().map(|p| p.y).fold(f32::MIN, f32::max);
                println!(
                    "{:>10.0} {:>8.0}  {:>10.1} {:>10.1}",
                    strength,
                    radius,
                    peak,
                    after - before
                );
                let _ = b;
            }
        }
    }

    /// Diagnostic trace, plus the per-step cost at the shipped particle count.
    /// Ignored by default; run with
    /// `cargo test --release report -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn report() {
        let mut fluid = Fluid::new(params());
        fluid.fill_block(block());
        println!(
            "particles={} rest_density={:.6}",
            fluid.len(),
            fluid.rest_density
        );
        let start = std::time::Instant::now();
        for step in 1..=600 {
            fluid.step(1.0 / 60.0);
            if step % 100 == 0 {
                println!(
                    "t={:5.2}s  compression={:.4}  interior_n={:.3}  max_speed={:7.1}",
                    step as f32 / 60.0,
                    fluid.compression_error(),
                    fluid.interior_density_ratio(),
                    fluid.max_speed()
                );
            }
        }
        // The solver has a 16.7 ms budget per frame at 60 Hz.
        println!(
            "mean step cost {:.2} ms",
            start.elapsed().as_secs_f32() * 1000.0 / 600.0
        );
    }
}
