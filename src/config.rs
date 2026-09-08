//! Everything tunable, loaded from `config.toml`.
//!
//! The solver deliberately knows nothing about this module: [`Config`] is the
//! file format, and [`Config::fluid_params`] converts it into the plain struct
//! the solver actually takes. Keeping the two apart means the file can grow
//! sections like `[render]` that the physics has no business seeing.
//!
//! Every section and every field is optional. A missing `config.toml` is not an
//! error -- the defaults here are the tuned ones -- but an unknown or malformed
//! field is, because silently ignoring a typo in a tuning file is how you spend
//! an afternoon wondering why a parameter does nothing.

use bevy::math::{Rect, Vec2};
use bevy::prelude::Resource;
use serde::Deserialize;
use std::path::Path;

use crate::sim::FluidParams;

pub const DEFAULT_PATH: &str = "config.toml";

/// Solver rate, in Hz. Mirrors `SIM_HZ` in `main`; here so the stability check
/// can reason about the real substep length.
const SIM_HZ: f32 = 60.0;

/// Largest Courant number the solver is known to survive. The tuned config
/// measures 0.90 and is stable; 1.13 explodes. Held slightly under 1.0 so a
/// config that merely *approaches* the cliff is refused rather than shipped.
const COURANT_LIMIT: f32 = 0.95;

#[derive(Debug, Clone, Default, Deserialize, Resource)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub world: World,
    pub fluid: FluidBlock,
    pub solver: Solver,
    pub render: Render,
    pub input: Input,
}

/// The box the fluid lives in. The window is sized to match.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct World {
    pub width: f32,
    pub height: f32,
    /// Acceleration in world units per second squared. The box is `height`
    /// units tall, so scale this with the box if you change it.
    pub gravity: [f32; 2],
}

/// How much water there is, and how finely it is resolved.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FluidBlock {
    /// Particles across and up in the starting block. This is the knob for
    /// *how much water*: the block collapses and spreads over the floor, so
    /// more of it means a deeper pool.
    pub columns: usize,
    pub rows: usize,
    /// Rest distance between neighbouring particles. This is the knob for
    /// *resolution*: halving it packs four times as many particles into the
    /// same volume of water. Change `smoothing_radius` with it.
    pub spacing: f32,
    /// Kernel radius: how far a particle looks for neighbours. What matters is
    /// its ratio to `spacing`, which sets the neighbour count and with it the
    /// solver's whole calibration. Keep it at roughly 2.4x `spacing`.
    pub smoothing_radius: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Solver {
    /// Jacobi iterations per substep. Drives compression down roughly
    /// linearly, at a matching cost.
    pub iterations: u32,
    /// Solver steps per frame. Raise this when you lower `fluid.spacing`: peak
    /// speed is set by gravity and box height rather than by resolution, so a
    /// smaller kernel means a particle covers more of it per step. Let it get
    /// too low for the resolution and the fluid explodes -- `validate` refuses
    /// the combination rather than letting you find out at runtime.
    pub substeps: u32,
    /// Under-relaxation on the positional correction. See [`FluidParams`];
    /// 0.5 is a ceiling, not a starting point.
    pub jacobi_relax: f32,
    pub relaxation: f32,
    pub viscosity: f32,
    pub tensile_k: f32,
    pub tensile_q: f32,
    pub tensile_n: i32,
    pub wall_friction: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Render {
    /// Particle sprite diameter, as a multiple of `spacing`. Below about 2.0
    /// neighbouring particles stop overlapping and the fluid reads as a dot
    /// screen rather than a body of water.
    pub particle_scale: f32,
    /// Speed at which a particle is drawn as full white foam.
    pub foam_speed: f32,
    /// Linear RGB, 0..1.
    pub background: [f32; 3],
    pub deep_color: [f32; 3],
    pub mid_color: [f32; 3],
    pub foam_color: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Input {
    pub mouse_radius: f32,
    /// Velocity change per second at the centre of the mouse's influence.
    pub mouse_strength: f32,
}

impl Default for World {
    fn default() -> Self {
        Self {
            width: 1240.0,
            height: 760.0,
            gravity: [0.0, -1400.0],
        }
    }
}

impl Default for FluidBlock {
    fn default() -> Self {
        Self {
            columns: 78,
            rows: 60,
            spacing: 10.0,
            smoothing_radius: 24.0,
        }
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            iterations: 12,
            substeps: 1,
            jacobi_relax: 0.5,
            relaxation: 0.05,
            viscosity: 0.08,
            tensile_k: 0.04,
            tensile_q: 0.2,
            tensile_n: 4,
            wall_friction: 0.98,
        }
    }
}

impl Default for Render {
    fn default() -> Self {
        Self {
            particle_scale: 2.4,
            foam_speed: 900.0,
            background: [0.05, 0.06, 0.10],
            deep_color: [0.06, 0.25, 0.75],
            mid_color: [0.25, 0.72, 0.98],
            foam_color: [0.92, 0.98, 1.00],
        }
    }
}

impl Default for Input {
    fn default() -> Self {
        Self {
            mouse_radius: 130.0,
            mouse_strength: 5200.0,
        }
    }
}

impl Config {
    /// Reads and validates a config file. A missing file yields the defaults;
    /// a malformed or invalid one is an error, so a typo is loud rather than
    /// silently ignored.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let config = Self::default();
                config.validate()?;
                Ok(config)
            }
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }

    /// Parses and validates config text. Fields left out fall back to the
    /// defaults above, so a file naming one value is a legal config.
    pub fn parse(text: &str) -> Result<Self, String> {
        let config: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        config.validate()?;
        Ok(config)
    }

    /// The simulation box, centred on the origin.
    pub fn bounds(&self) -> Rect {
        let half = Vec2::new(self.world.width, self.world.height) * 0.5;
        Rect {
            min: -half,
            max: half,
        }
    }

    pub fn fluid_params(&self) -> FluidParams {
        FluidParams {
            smoothing_radius: self.fluid.smoothing_radius,
            spacing: self.fluid.spacing,
            iterations: self.solver.iterations,
            substeps: self.solver.substeps,
            relaxation: self.solver.relaxation,
            viscosity: self.solver.viscosity,
            tensile_k: self.solver.tensile_k,
            tensile_q: self.solver.tensile_q,
            tensile_n: self.solver.tensile_n,
            gravity: Vec2::from(self.world.gravity),
            jacobi_relax: self.solver.jacobi_relax,
            wall_friction: self.solver.wall_friction,
            bounds: self.bounds(),
        }
    }

    /// Rough peak speed the fluid will reach, from the free-fall of the
    /// starting block collapsing to the floor: v = sqrt(2 g h). Measured peaks
    /// land within about 20% of this across the sweep.
    pub fn expected_peak_speed(&self) -> f32 {
        let g = Vec2::from(self.world.gravity).length();
        let fall = (self.fluid.rows as f32 * self.fluid.spacing).min(self.world.height);
        (2.0 * g * fall).sqrt()
    }

    /// How far the fastest particle travels in one solver substep, in units of
    /// the kernel radius. Neighbour lists are rebuilt once per substep, so once
    /// this approaches 1 a particle can cross a neighbour before anyone looks
    /// again, and the density constraint starts acting on stale information.
    pub fn courant_number(&self) -> f32 {
        let sub_dt = 1.0 / (SIM_HZ * self.solver.substeps as f32);
        self.expected_peak_speed() * sub_dt / self.fluid.smoothing_radius
    }

    /// Substeps needed to keep [`Self::courant_number`] under the safe limit.
    pub fn required_substeps(&self) -> u32 {
        let one_step = {
            let mut probe = self.clone();
            probe.solver.substeps = 1;
            probe.courant_number()
        };
        (one_step / COURANT_LIMIT).ceil().max(1.0) as u32
    }

    /// Largest starting block that fits inside the bounds at the current
    /// spacing. Mirrors the placement in `Fluid::fill_block`, including the
    /// half-spacing row stagger and the collision margin.
    pub fn max_block(&self) -> (usize, usize) {
        let d = self.fluid.spacing;
        let cols = (self.world.width / d - 1.0).floor().max(0.0) as usize;
        let rows = (self.world.height / d - 0.5).floor().max(0.0) as usize;
        (cols, rows)
    }

    /// Checks the invariants the solver relies on. Called by [`Self::load`] and
    /// [`Self::parse`]; public so a config assembled in code can be checked too.
    pub fn validate(&self) -> Result<(), String> {
        let f = &self.fluid;
        if f.columns == 0 || f.rows == 0 {
            return Err("fluid.columns and fluid.rows must both be at least 1".into());
        }
        if !f.spacing.is_finite() || f.spacing <= 0.0 {
            return Err(format!("fluid.spacing must be positive, got {}", f.spacing));
        }
        if !self.world.width.is_finite()
            || !self.world.height.is_finite()
            || self.world.width <= 0.0
            || self.world.height <= 0.0
        {
            return Err("world.width and world.height must be positive".into());
        }

        // The neighbourhood has to be big enough for a density estimate to mean
        // anything. Below about 1.5 spacings a particle sees only its four
        // nearest neighbours and the fluid behaves like loose sand.
        let ratio = f.smoothing_radius / f.spacing;
        if !ratio.is_finite() || ratio < 1.5 {
            return Err(format!(
                "fluid.smoothing_radius must be at least 1.5x fluid.spacing \
                 (got {:.2}x); the tuned ratio is 2.4x",
                ratio
            ));
        }

        // A block that does not fit would be clamped into the walls at spawn,
        // starting the sim from a badly compressed state.
        let (max_cols, max_rows) = self.max_block();
        if f.columns > max_cols || f.rows > max_rows {
            return Err(format!(
                "the starting block does not fit in the world: {}x{} particles at \
                 spacing {} needs a {:.0}x{:.0} box, but world is {:.0}x{:.0}. \
                 At this spacing the box holds at most {}x{} ({} particles); \
                 either lower fluid.columns/fluid.rows, lower fluid.spacing, or \
                 raise world.width/world.height",
                f.columns,
                f.rows,
                f.spacing,
                (f.columns as f32 + 1.0) * f.spacing,
                (f.rows as f32 + 0.5) * f.spacing,
                self.world.width,
                self.world.height,
                max_cols,
                max_rows,
                max_cols * max_rows,
            ));
        }

        let s = &self.solver;
        if s.iterations == 0 {
            return Err("solver.iterations must be at least 1".into());
        }
        if s.substeps == 0 {
            return Err("solver.substeps must be at least 1".into());
        }

        // The trap this catches: raising the particle count by lowering
        // `spacing` shrinks the kernel, but the peak speed comes from gravity
        // and the drop height, so the distance covered per step grows in kernel
        // radii until neighbour lists no longer describe reality. Measured: the
        // default config sits at 0.90, and 1.13 explodes to 10^6 units/s.
        let courant = self.courant_number();
        if courant > COURANT_LIMIT {
            return Err(format!(
                "the fluid would move too far per solver step to stay stable \
                 (Courant number {courant:.2}, limit {COURANT_LIMIT}): at spacing {} the \
                 kernel radius is {:.1} units, but the fluid is expected to peak \
                 near {:.0} units/s. Set solver.substeps = {} (currently {}), or \
                 raise fluid.spacing, or lower world.gravity",
                self.fluid.spacing,
                self.fluid.smoothing_radius,
                self.expected_peak_speed(),
                self.required_substeps(),
                s.substeps,
            ));
        }
        // Measured, not guessed: the `sweep` test diverges at 0.6 for some
        // iteration counts. Refusing the value beats shipping an explosion.
        if !s.jacobi_relax.is_finite() || s.jacobi_relax <= 0.0 || s.jacobi_relax > 0.55 {
            return Err(format!(
                "solver.jacobi_relax must be in (0, 0.55]; got {}. Values at or \
                 above 0.6 diverge for some iteration counts",
                s.jacobi_relax
            ));
        }
        if !s.relaxation.is_finite() || s.relaxation <= 0.0 {
            return Err("solver.relaxation must be positive".into());
        }
        if !s.wall_friction.is_finite() || !(0.0..=1.0).contains(&s.wall_friction) {
            return Err(format!(
                "solver.wall_friction must be in [0, 1]; got {}",
                s.wall_friction
            ));
        }
        if self.render.particle_scale <= 0.0 {
            return Err("render.particle_scale must be positive".into());
        }
        if self.render.foam_speed <= 0.0 {
            return Err("render.foam_speed must be positive".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default()
            .validate()
            .expect("shipped defaults are invalid");
    }

    #[test]
    fn a_missing_file_yields_the_defaults() {
        let config = Config::load("does-not-exist.toml").expect("missing file should be fine");
        assert_eq!(config.fluid.columns, Config::default().fluid.columns);
    }

    #[test]
    fn a_partial_config_keeps_the_other_defaults() {
        let config = Config::parse("[solver]\niterations = 4\n").unwrap();
        assert_eq!(config.solver.iterations, 4);
        // Untouched, both in the section that was named and in one that wasn't.
        assert_eq!(
            config.solver.jacobi_relax,
            Config::default().solver.jacobi_relax
        );
        assert_eq!(config.fluid.spacing, Config::default().fluid.spacing);
    }

    #[test]
    fn an_unknown_field_is_rejected() {
        // The whole point of `deny_unknown_fields`: a typo in a tuning file
        // must not silently do nothing.
        let err = Config::parse("[solver]\niteratons = 4\n").unwrap_err();
        assert!(err.contains("iteratons"), "unhelpful error: {err}");
    }

    #[test]
    fn a_block_too_big_for_the_world_is_rejected() {
        let err = Config::parse("[fluid]\ncolumns = 500\n").unwrap_err();
        assert!(
            err.contains("does not fit") && err.contains("at most"),
            "error should say what fits: {err}"
        );
    }

    #[test]
    fn the_largest_reported_block_actually_fits() {
        // `max_block` is quoted to the user in the error above, so it had
        // better be accurate at a range of spacings.
        for spacing in [4.0f32, 7.5, 10.0, 20.0] {
            let mut config = Config::default();
            config.fluid.spacing = spacing;
            config.fluid.smoothing_radius = spacing * 2.4;
            let (cols, rows) = config.max_block();
            config.fluid.columns = cols;
            config.fluid.rows = rows;
            // Isolate the geometry question from the stability one: a fine
            // spacing needs substeps, and that is a different test.
            config.solver.substeps = config.required_substeps();
            config
                .validate()
                .unwrap_or_else(|e| panic!("max_block does not fit at spacing {spacing}: {e}"));

            config.fluid.columns = cols + 1;
            assert!(
                config.validate().is_err(),
                "one more column than max_block reports should not fit at spacing {spacing}"
            );
        }
    }

    #[test]
    fn non_finite_numbers_are_rejected() {
        // TOML can express these, and a NaN that reaches the solver poisons
        // every particle it touches within a step or two.
        for text in [
            "[fluid]\nspacing = nan\n",
            "[fluid]\nspacing = inf\n",
            "[solver]\njacobi_relax = nan\n",
            "[world]\nwidth = nan\n",
        ] {
            assert!(
                Config::parse(text).is_err(),
                "accepted a non-finite value: {text:?}"
            );
        }
    }

    #[test]
    fn a_diverging_relaxation_is_rejected() {
        let err = Config::parse("[solver]\njacobi_relax = 0.9\n").unwrap_err();
        assert!(err.contains("diverge"), "unhelpful error: {err}");
    }

    #[test]
    fn too_few_neighbours_is_rejected() {
        let err = Config::parse("[fluid]\nsmoothing_radius = 11.0\n").unwrap_err();
        assert!(err.contains("smoothing_radius"), "unhelpful error: {err}");
    }
}

#[cfg(test)]
mod stability_tests {
    use super::*;

    #[test]
    fn the_shipped_config_is_inside_the_stability_limit() {
        let config = Config::default();
        assert!(
            config.courant_number() <= COURANT_LIMIT,
            "shipped config sits at Courant {:.2}",
            config.courant_number()
        );
    }

    #[test]
    fn a_finer_spacing_without_substeps_is_rejected() {
        // The exact trap someone falls into when asking for more particles:
        // halve the spacing, keep everything else, watch it explode. Measured
        // at spacing 8 with one substep: 10^6 units/s.
        let err = Config::parse("[fluid]\nspacing = 5.0\nsmoothing_radius = 12.0\n").unwrap_err();
        assert!(
            err.contains("solver.substeps"),
            "the error should name the fix: {err}"
        );
    }

    #[test]
    fn required_substeps_is_enough_to_pass() {
        // Whatever the error message tells the user to set must actually work,
        // at every spacing, or the advice is worse than useless.
        for spacing in [12.0f32, 10.0, 8.0, 6.5, 5.0, 4.0, 3.0] {
            let mut config = Config::default();
            config.fluid.spacing = spacing;
            config.fluid.smoothing_radius = spacing * 2.4;
            let (max_cols, max_rows) = config.max_block();
            config.fluid.columns = config.fluid.columns.min(max_cols);
            config.fluid.rows = config.fluid.rows.min(max_rows);
            config.solver.substeps = config.required_substeps();
            config.validate().unwrap_or_else(|e| {
                panic!("required_substeps is not sufficient at spacing {spacing}: {e}")
            });
        }
    }
}
