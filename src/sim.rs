//! A 2D Position Based Fluids solver (Macklin & Müller, SIGGRAPH 2013).
//!
//! PBF resolves incompressibility as a positional constraint solved by Jacobi
//! iteration rather than as a pressure force, which is what lets it stay stable
//! at a 1/60 s step. A force-based SPH solver at this particle count would need
//! a step in the 1e-4 range to avoid blowing up.
//!
//! Particle state lives in flat arrays instead of in the ECS: every solver
//! iteration touches each particle's whole neighbourhood at random, which is far
//! cheaper over `Vec`s than over archetype storage.

use bevy::math::{Rect, Vec2};
use bevy::prelude::Resource;
use core::f32::consts::PI;

/// Smoothing kernels, with their normalisation constants folded in at build
/// time so the hot loops stay free of `powi`.
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
            // 2D normalisations: poly6 integrates to 1 over the disc of radius
            // h, spiky is the derivative constant -30/(pi h^5).
            poly6: 4.0 / (PI * h.powi(8)),
            spiky: -30.0 / (PI * h.powi(5)),
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
    fn spiky_grad(&self, r: Vec2) -> Vec2 {
        let len = r.length();
        if len <= 1e-6 || len >= self.h {
            return Vec2::ZERO;
        }
        let d = self.h - len;
        r * (self.spiky * d * d / len)
    }
}

/// Solver tunables. Lengths are world units; the window is sized to match.
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
    /// crossing a kernel radius in a single step. It is what a finer `spacing`
    /// needs: peak speed is set by gravity and box height, not by resolution,
    /// so halving the kernel radius doubles the distance travelled per step
    /// measured in kernel radii. Above 1 the neighbour lists stop describing
    /// reality and the fluid explodes.
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
    pub gravity: Vec2,
    /// Under-relaxation on the positional correction. Jacobi updates every
    /// particle against stale neighbours, so applying the full correction
    /// overshoots and rings; Gauss-Seidel would not need this. Values at or
    /// above 0.6 diverge for some iteration counts -- 0.5 is stable across the
    /// whole sweep, so treat it as the ceiling rather than a starting point.
    pub jacobi_relax: f32,
    /// Fraction of tangential velocity kept when a particle hits a wall.
    pub wall_friction: f32,
    pub bounds: Rect,
}

/// Uniform grid over the simulation bounds, rebuilt each step by counting sort.
/// The bounds are fixed and the cell size is the kernel radius, so a dense grid
/// beats a hash here: no modulo, no collisions, and neighbours land contiguously.
struct Grid {
    cell_size: f32,
    origin: Vec2,
    cols: i32,
    rows: i32,
    /// Start offset of each cell into `sorted`, plus a trailing total.
    cell_start: Vec<u32>,
    /// Scratch copy of `cell_start` used as a write cursor while scattering.
    cursor: Vec<u32>,
    /// Particle indices ordered by cell.
    sorted: Vec<u32>,
}

impl Grid {
    fn new(bounds: Rect, cell_size: f32) -> Self {
        let size = bounds.size();
        let cols = (size.x / cell_size).ceil() as i32 + 1;
        let rows = (size.y / cell_size).ceil() as i32 + 1;
        let cells = (cols * rows) as usize;
        Self {
            cell_size,
            origin: bounds.min,
            cols,
            rows,
            cell_start: vec![0; cells + 1],
            cursor: vec![0; cells + 1],
            sorted: Vec::new(),
        }
    }

    #[inline]
    fn cell_of(&self, p: Vec2) -> (i32, i32) {
        let local = (p - self.origin) / self.cell_size;
        (
            (local.x as i32).clamp(0, self.cols - 1),
            (local.y as i32).clamp(0, self.rows - 1),
        )
    }

    fn rebuild(&mut self, positions: &[Vec2]) {
        self.cell_start.fill(0);
        for &p in positions {
            let (cx, cy) = self.cell_of(p);
            // Counts are written one slot high so the prefix sum below turns
            // them directly into start offsets.
            self.cell_start[(cy * self.cols + cx) as usize + 1] += 1;
        }
        for i in 1..self.cell_start.len() {
            self.cell_start[i] += self.cell_start[i - 1];
        }
        self.cursor.copy_from_slice(&self.cell_start);
        self.sorted.resize(positions.len(), 0);
        for (i, &p) in positions.iter().enumerate() {
            let (cx, cy) = self.cell_of(p);
            let slot = &mut self.cursor[(cy * self.cols + cx) as usize];
            self.sorted[*slot as usize] = i as u32;
            *slot += 1;
        }
    }
}

/// The fluid. `pos` is the state you render; everything else is solver scratch.
#[derive(Resource)]
pub struct Fluid {
    pub params: FluidParams,
    pub pos: Vec<Vec2>,
    pub vel: Vec<Vec2>,
    /// Predicted positions, which the constraint solver actually moves.
    pred: Vec<Vec2>,
    lambda: Vec<f32>,
    delta: Vec<Vec2>,
    vel_scratch: Vec<Vec2>,
    kernels: Kernels,
    grid: Grid,
    /// Neighbour lists, flattened. Built once per step and reused across all
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

    /// Fills the lower-left region of the bounds with a lattice of particles,
    /// which then collapses into a dam break.
    pub fn fill_block(&mut self, cols: usize, rows: usize) {
        self.pos.clear();
        self.vel.clear();
        let d = self.params.spacing;
        let corner = self.params.bounds.min + Vec2::splat(d);
        for row in 0..rows {
            for col in 0..cols {
                // Nudge alternating rows so the lattice isn't perfectly
                // axis-aligned; a perfect grid takes a while to break symmetry.
                let stagger = if row % 2 == 0 { 0.0 } else { d * 0.5 };
                self.pos
                    .push(corner + Vec2::new(col as f32 * d + stagger, row as f32 * d));
                self.vel.push(Vec2::ZERO);
            }
        }
        let n = self.pos.len();
        self.pred = vec![Vec2::ZERO; n];
        self.lambda = vec![0.0; n];
        self.delta = vec![Vec2::ZERO; n];
        self.vel_scratch = vec![Vec2::ZERO; n];
        // These describe the configuration we just threw away. Leaving them
        // would let `compression_error` report densities for the old state,
        // and would index out of bounds if the particle count changed.
        self.neighbors.clear();
        self.neighbor_start.clear();
    }

    /// Pushes particles within `radius` of `center` radially outwards, with a
    /// smooth falloff to zero at the rim. A negative `strength` pulls inwards.
    /// This is the mouse.
    pub fn apply_radial_impulse(&mut self, center: Vec2, radius: f32, strength: f32) {
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
        for i in 0..self.pos.len() {
            self.vel[i] += self.params.gravity * dt;
            self.pred[i] = self.pos[i] + self.vel[i] * dt;
            self.confine(i);
        }

        // 2. Neighbourhoods, from the predicted positions.
        self.grid.rebuild(&self.pred);
        self.build_neighbors();

        // 3. Jacobi-solve the density constraint.
        for _ in 0..self.params.iterations {
            self.solve_density();
        }

        // 4. Derive velocity from the positions the solver settled on, which is
        //    what makes wall collisions and constraint corrections energy-safe.
        let inv_dt = 1.0 / dt;
        for i in 0..self.pos.len() {
            self.vel[i] = (self.pred[i] - self.pos[i]) * inv_dt;
            self.pos[i] = self.pred[i];
        }

        self.apply_viscosity();
    }

    /// Clamps a predicted position back inside the bounds and bleeds off
    /// tangential speed, so walls feel like walls rather than mirrors.
    #[inline]
    fn confine(&mut self, i: usize) {
        let margin = self.params.spacing * 0.5;
        let min = self.params.bounds.min + margin;
        let max = self.params.bounds.max - margin;
        let p = &mut self.pred[i];
        if p.x < min.x {
            p.x = min.x;
            self.vel[i].y *= self.params.wall_friction;
        } else if p.x > max.x {
            p.x = max.x;
            self.vel[i].y *= self.params.wall_friction;
        }
        if p.y < min.y {
            p.y = min.y;
            self.vel[i].x *= self.params.wall_friction;
        } else if p.y > max.y {
            p.y = max.y;
            self.vel[i].x *= self.params.wall_friction;
        }
    }

    fn build_neighbors(&mut self) {
        let (grid, pred, kernels) = (&self.grid, &self.pred, &self.kernels);
        let neighbors = &mut self.neighbors;
        let starts = &mut self.neighbor_start;
        neighbors.clear();
        starts.clear();
        starts.reserve(pred.len() + 1);

        for (i, &pi) in pred.iter().enumerate() {
            starts.push(neighbors.len() as u32);
            let (cx, cy) = grid.cell_of(pi);
            for gy in (cy - 1).max(0)..=(cy + 1).min(grid.rows - 1) {
                for gx in (cx - 1).max(0)..=(cx + 1).min(grid.cols - 1) {
                    let cell = (gy * grid.cols + gx) as usize;
                    let (from, to) = (grid.cell_start[cell], grid.cell_start[cell + 1]);
                    for &j in &grid.sorted[from as usize..to as usize] {
                        if j as usize != i && (pi - pred[j as usize]).length_squared() < kernels.h2
                        {
                            neighbors.push(j);
                        }
                    }
                }
            }
        }
        starts.push(neighbors.len() as u32);
    }

    #[inline]
    fn neighbors_of(&self, i: usize) -> &[u32] {
        let from = self.neighbor_start[i] as usize;
        let to = self.neighbor_start[i + 1] as usize;
        &self.neighbors[from..to]
    }

    fn solve_density(&mut self) {
        let inv_rho0 = 1.0 / self.rest_density;

        // Density, constraint value, and the multiplier that will correct it.
        for i in 0..self.pred.len() {
            let pi = self.pred[i];
            let mut rho = self.kernels.poly6(0.0);
            // Gradient of C_i with respect to p_i, and the sum of squared
            // gradients with respect to every particle in the neighbourhood.
            let mut grad_self = Vec2::ZERO;
            let mut sum_sq = 0.0;
            for &j in self.neighbors_of(i) {
                let r = pi - self.pred[j as usize];
                rho += self.kernels.poly6(r.length_squared());
                let g = self.kernels.spiky_grad(r) * inv_rho0;
                grad_self += g;
                sum_sq += g.length_squared();
            }
            sum_sq += grad_self.length_squared();
            let c = rho * inv_rho0 - 1.0;
            self.lambda[i] = -c / (sum_sq + self.epsilon);
        }

        // Positional correction.
        for i in 0..self.pred.len() {
            let pi = self.pred[i];
            let li = self.lambda[i];
            let mut d = Vec2::ZERO;
            for &j in self.neighbors_of(i) {
                let r = pi - self.pred[j as usize];
                let ratio = self.kernels.poly6(r.length_squared()) / self.tensile_w;
                let s_corr = -self.tensile_scale * ratio.powi(self.params.tensile_n);
                d += self.kernels.spiky_grad(r) * (li + self.lambda[j as usize] + s_corr);
            }
            self.delta[i] = d * (inv_rho0 * self.params.jacobi_relax);
        }

        for i in 0..self.pred.len() {
            self.pred[i] += self.delta[i];
            self.confine(i);
        }
    }

    /// XSPH: nudge each particle towards its neighbourhood's mean velocity.
    /// Cheap, and it's what stops the surface looking like sand.
    fn apply_viscosity(&mut self) {
        let inv_rho0 = 1.0 / self.rest_density;
        for i in 0..self.pos.len() {
            let (pi, vi) = (self.pos[i], self.vel[i]);
            let mut dv = Vec2::ZERO;
            for &j in self.neighbors_of(i) {
                let w = self
                    .kernels
                    .poly6((pi - self.pos[j as usize]).length_squared());
                dv += (self.vel[j as usize] - vi) * w;
            }
            self.vel_scratch[i] = vi + dv * (self.params.viscosity * inv_rho0);
        }
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
        let mut total = 0.0;
        for i in 0..self.pos.len() {
            let pi = self.pos[i];
            let mut rho = self.kernels.poly6(0.0);
            for &j in self.neighbors_of(i) {
                rho += self
                    .kernels
                    .poly6((pi - self.pos[j as usize]).length_squared());
            }
            total += (rho / self.rest_density - 1.0).max(0.0);
        }
        total / self.pos.len() as f32
    }

    pub fn max_speed(&self) -> f32 {
        self.vel.iter().fold(0.0f32, |m, v| m.max(v.length()))
    }
}

/// Density and constraint-gradient magnitude a particle would see at the centre
/// of an unbounded square lattice of the given spacing, with unit mass.
///
/// Calibrating against this instead of hard-coding a rest density means the
/// tunables keep their meaning when `smoothing_radius` or `spacing` change --
/// otherwise every one of them has to be retuned by hand.
fn lattice_reference(kernels: &Kernels, spacing: f32) -> (f32, f32) {
    let reach = (kernels.h / spacing).ceil() as i32;
    let mut rho = 0.0;
    let mut sum_sq = 0.0;
    let mut grad_self = Vec2::ZERO;
    for gy in -reach..=reach {
        for gx in -reach..=reach {
            let d = Vec2::new(gx as f32, gy as f32) * spacing;
            rho += kernels.poly6(d.length_squared());
            if gx != 0 || gy != 0 {
                let g = kernels.spiky_grad(d);
                grad_self += g;
                sum_sq += g.length_squared();
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

    fn block() -> (usize, usize) {
        let c = Config::default();
        (c.fluid.columns, c.fluid.rows)
    }

    fn settled(steps: u32) -> Fluid {
        let mut fluid = Fluid::new(params());
        let (cols, rows) = block();
        fluid.fill_block(cols, rows);
        for _ in 0..steps {
            fluid.step(1.0 / 60.0);
        }
        fluid
    }

    #[test]
    fn rest_density_matches_the_seed_lattice() {
        let mut fluid = Fluid::new(params());
        let (cols, rows) = block();
        fluid.fill_block(cols, rows);
        // One step just to populate neighbour lists; the lattice has not had
        // time to move, so the interior should already sit at rest density.
        fluid.step(1.0 / 600.0);
        assert!(
            fluid.compression_error() < 0.05,
            "seed lattice is not near rest density: {}",
            fluid.compression_error()
        );
    }

    #[test]
    fn stays_finite_and_bounded() {
        let fluid = settled(600);
        let b = fluid.params.bounds;
        for (i, p) in fluid.pos.iter().enumerate() {
            assert!(p.is_finite(), "particle {i} left the reals: {p:?}");
            assert!(
                p.x >= b.min.x - 1.0
                    && p.x <= b.max.x + 1.0
                    && p.y >= b.min.y - 1.0
                    && p.y <= b.max.y + 1.0,
                "particle {i} escaped the box: {p:?}"
            );
        }
    }

    #[test]
    fn settles_without_gaining_energy() {
        let fluid = settled(900);
        // Fifteen seconds in, a dam break should have sloshed out nearly all of
        // its kinetic energy. A solver that is pumping energy shows up here
        // first: the tuned parameters settle around 60 units/s.
        assert!(
            fluid.max_speed() < 500.0,
            "fluid is gaining energy, max speed {}",
            fluid.max_speed()
        );
        assert!(
            fluid.compression_error() < 0.05,
            "fluid is compressing, error {}",
            fluid.compression_error()
        );
    }

    #[test]
    fn radial_impulse_pushes_out_and_pulls_in() {
        let mut fluid = Fluid::new(params());
        let (cols, rows) = block();
        fluid.fill_block(cols, rows);
        let center = fluid.pos[fluid.len() / 2];

        // A particle offset from the centre should be pushed directly away.
        let probe = fluid
            .pos
            .iter()
            .position(|p| {
                let d = *p - center;
                d.length() > 20.0 && d.length() < 60.0
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

        // Nothing beyond the radius should have been touched.
        let far = fluid
            .pos
            .iter()
            .position(|p| (*p - center).length() > 200.0)
            .expect("no particle outside the radius");
        assert_eq!(fluid.vel[far], Vec2::ZERO);
    }

    #[test]
    fn settles_into_a_flat_pool() {
        let fluid = settled(900);
        // An incompressible fluid that has come to rest should spread across
        // the floor to the depth its own volume implies. This is the check that
        // actually says "this looks like water" rather than "this is stable":
        // a gassy solver settles too high, a collapsing one too low.
        let b = fluid.params.bounds;
        let area = fluid.len() as f32 * fluid.params.spacing.powi(2);
        let expected = b.min.y + area / b.size().x;
        let stragglers = fluid
            .pos
            .iter()
            .filter(|p| p.y > expected + fluid.params.spacing)
            .count();
        assert!(
            stragglers * 100 < fluid.len(),
            "{stragglers} of {} particles are above the expected surface at y={expected:.1}",
            fluid.len()
        );
    }

    /// Parameter sweep. Ignored by default; run with
    /// `cargo test --release sweep -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn sweep() {
        println!(
            "{:>6} {:>6} {:>5}  {:>9} {:>9} {:>9} {:>8} {:>9}",
            "relax", "tens", "iter", "peak_spd", "mean_spd", "compress", "surface", "ms/step"
        );
        for &relax in &[0.4f32, 0.5, 0.6] {
            for &tens in &[0.04f32] {
                for &iters in &[8u32, 10, 12, 16] {
                    let sweep_params = FluidParams {
                        jacobi_relax: relax,
                        tensile_k: tens,
                        iterations: iters,
                        ..params()
                    };
                    let mut fluid = Fluid::new(sweep_params);
                    let (cols, rows) = block();
                    fluid.fill_block(cols, rows);

                    // Where the pool surface should end up if the fluid keeps
                    // its rest volume and spreads across the full floor.
                    let b = fluid.params.bounds;
                    let area = fluid.len() as f32 * fluid.params.spacing.powi(2);
                    let expected = b.min.y + area / b.size().x;

                    let start = std::time::Instant::now();
                    let mut peak = 0.0f32;
                    let mut speed_sum = 0.0f32;
                    let mut comp_sum = 0.0f32;
                    let mut samples = 0.0f32;
                    for step in 0..900 {
                        fluid.step(1.0 / 60.0);
                        if step >= 60 {
                            peak = peak.max(fluid.max_speed());
                        }
                        if step >= 840 {
                            speed_sum += fluid.max_speed();
                            comp_sum += fluid.compression_error();
                            samples += 1.0;
                        }
                        if !fluid.max_speed().is_finite() {
                            break;
                        }
                    }
                    let ms = start.elapsed().as_secs_f32() * 1000.0 / 900.0;

                    // Fraction of particles more than one spacing above the
                    // expected surface: high means the pool never settled.
                    let above = fluid
                        .pos
                        .iter()
                        .filter(|p| p.y > expected + fluid.params.spacing)
                        .count() as f32
                        / fluid.len() as f32;

                    println!(
                        "{:>6.2} {:>6.3} {:>5}  {:>9.1} {:>9.1} {:>9.4} {:>8.3} {:>9.2}",
                        relax,
                        tens,
                        iters,
                        peak,
                        speed_sum / samples,
                        comp_sum / samples,
                        above,
                        ms
                    );
                }
            }
        }
    }

    /// How the per-step cost scales, along the two different axes that the
    /// particle count can be raised on. Ignored by default; run with
    /// `cargo test --release scaling -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn scaling() {
        fn measure(config: &Config) -> (f32, f32, f32) {
            let mut fluid = Fluid::new(config.fluid_params());
            fluid.fill_block(config.fluid.columns, config.fluid.rows);
            // Warm up past the initial collapse, where the fluid is at its
            // densest and the neighbour lists are longest.
            for _ in 0..120 {
                fluid.step(1.0 / 60.0);
            }
            let start = std::time::Instant::now();
            for _ in 0..200 {
                fluid.step(1.0 / 60.0);
            }
            let ms = start.elapsed().as_secs_f32() * 1000.0 / 200.0;
            (ms, fluid.compression_error(), fluid.max_speed())
        }

        // Axis 1: more water at the shipped resolution. Nothing else has to
        // change, and the ceiling is simply the box being full.
        println!("more water, spacing {}:", Config::default().fluid.spacing);
        println!(
            "{:>10} {:>10} {:>10} {:>5} {:>10} {:>8}",
            "block", "particles", "ms/step", "sub", "compress", "60Hz?"
        );
        let (max_cols, max_rows) = Config::default().max_block();
        for fraction in [0.5f32, 0.7, 0.85, 1.0] {
            let mut config = Config::default();
            config.fluid.columns = (max_cols as f32 * fraction) as usize;
            config.fluid.rows = (max_rows as f32 * fraction) as usize;
            // A taller starting block falls further and so lands faster, which
            // is its own reason to substep -- nothing to do with resolution.
            config.solver.substeps = config.required_substeps();
            config.validate().expect("config should be valid");
            let (ms, compress, _) = measure(&config);
            println!(
                "{:>10} {:>10} {:>10.2} {:>5} {:>10.4} {:>8}",
                format!("{}x{}", config.fluid.columns, config.fluid.rows),
                config.fluid.columns * config.fluid.rows,
                ms,
                config.solver.substeps,
                compress,
                if ms < 16.7 { "yes" } else { "no" }
            );
        }

        // Axis 2: the same water, resolved more finely. Costs substeps as well
        // as particles, because peak speed does not fall with the kernel.
        println!("\nfiner resolution, same volume of water:");
        println!(
            "{:>8} {:>10} {:>10} {:>9} {:>10} {:>10} {:>8}",
            "spacing", "particles", "ms/step", "sub x it", "compress", "max_speed", "60Hz?"
        );
        for spacing in [14.0f32, 12.0, 10.0, 8.0, 6.5, 5.0, 4.0] {
            let mut config = Config::default();
            config.fluid.spacing = spacing;
            config.fluid.smoothing_radius = spacing * 2.4;
            let (max_cols, max_rows) = config.max_block();
            config.fluid.columns = (max_cols as f32 * 0.63) as usize;
            config.fluid.rows = (max_rows as f32 * 0.8) as usize;
            // Take the solver's own advice, which is what the config error
            // tells a user to do. Iterations stay put: they cannot be traded
            // away for substeps, because a finer grid makes the pool deeper in
            // particles and Jacobi needs the passes to carry pressure up it.
            config.solver.substeps = config.required_substeps();
            config.validate().expect("config should be valid");
            let (ms, compress, peak) = measure(&config);
            println!(
                "{:>8.1} {:>10} {:>10.2} {:>9} {:>10.4} {:>10.1} {:>8}",
                spacing,
                config.fluid.columns * config.fluid.rows,
                ms,
                format!("{}x{}", config.solver.substeps, config.solver.iterations),
                compress,
                peak,
                if ms < 16.7 { "yes" } else { "no" }
            );
        }
    }

    /// Diagnostic trace, plus the per-step cost at the shipped particle count.
    /// Ignored by default; run with
    /// `cargo test --release report -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn report() {
        let mut fluid = Fluid::new(params());
        let (cols, rows) = block();
        fluid.fill_block(cols, rows);
        println!(
            "particles={} rest_density={:.6}",
            fluid.len(),
            fluid.rest_density
        );
        let start = std::time::Instant::now();
        for step in 1..=900 {
            fluid.step(1.0 / 60.0);
            if step % 150 == 0 {
                println!(
                    "t={:5.2}s  compression={:.4}  max_speed={:7.1}",
                    step as f32 / 60.0,
                    fluid.compression_error(),
                    fluid.max_speed()
                );
            }
        }
        // The solver has a 16.7 ms budget per frame at 60 Hz.
        println!(
            "mean step cost {:.2} ms",
            start.elapsed().as_secs_f32() * 1000.0 / 900.0
        );
    }
}
