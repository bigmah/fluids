//! A directional swell generation zone and a steep reef in a particle wave tank.
//! Only the offshore zone is driven; shoaling and breaking are solved by PBF.

use crate::sim::Bounds;
use bevy::math::Vec3;
use serde::Deserialize;
use std::f32::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Swell {
    /// Direction of travel in degrees: 0 = +X, positive turns toward +Z.
    pub direction: f32,
    /// Target offshore crest-to-trough height, in world units.
    pub height: f32,
    /// Period in simulation seconds, independent of playback speed.
    pub period: f32,
}

impl Default for Swell {
    fn default() -> Self {
        Self {
            direction: 0.0,
            height: 40.0,
            period: 1.8,
        }
    }
}

impl Swell {
    /// Solve omega^2 = g k tanh(k h). Bisection avoids Newton overshoot.
    pub fn wavenumber(&self, gravity: f32, depth: f32) -> f32 {
        let omega2 = (TAU / self.period).powi(2);
        let mut lo = 0.0;
        let mut hi = (omega2 / gravity).max((omega2 / (gravity * depth)).sqrt()) * 2.0;
        for _ in 0..40 {
            let k = (lo + hi) * 0.5;
            if gravity * k * (k * depth).tanh() > omega2 {
                hi = k;
            } else {
                lo = k;
            }
        }
        (lo + hi) * 0.5
    }

    pub fn wavelength(&self, gravity: f32, depth: f32) -> f32 {
        TAU / self.wavenumber(gravity, depth)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Wave {
    pub water_depth: f32,
    pub reef_height: f32,
    /// Start and width of the steep rise, as fractions of world.width.
    pub reef_start: f32,
    pub reef_width: f32,
    /// Change in reef X per unit Z. The reef stays fixed as swell direction changes.
    pub reef_skew: f32,
    /// Fraction of world.width reserved for the shoreward damping beach.
    pub beach_width: f32,
}

impl Default for Wave {
    fn default() -> Self {
        Self {
            water_depth: 190.0,
            reef_height: 174.0,
            reef_start: 0.30,
            reef_width: 0.24,
            reef_skew: 0.12,
            beach_width: 0.22,
        }
    }
}

impl Wave {
    pub fn validate(
        &self,
        swell: Swell,
        bounds: Bounds,
        spacing: f32,
        gravity: Vec3,
    ) -> Result<(), String> {
        for (name, value) in [
            ("water_depth", self.water_depth),
            ("reef_height", self.reef_height),
            ("reef_start", self.reef_start),
            ("reef_width", self.reef_width),
            ("beach_width", self.beach_width),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("wave.{name} must be finite and nonnegative"));
            }
        }
        if !self.reef_skew.is_finite() || self.reef_skew.abs() > 1.0 {
            return Err("wave.reef_skew must be finite and in [-1, 1]".into());
        }
        if gravity.x != 0.0 || gravity.z != 0.0 || !gravity.y.is_finite() || gravity.y >= 0.0 {
            return Err("the wave tank requires downward world.gravity = [0, -g, 0]".into());
        }
        if !swell.direction.is_finite() || !(-60.0..=60.0).contains(&swell.direction) {
            return Err("swell.direction must be in [-60, 60] degrees: 0 travels along +X toward the reef, positive toward +Z".into());
        }
        if !swell.period.is_finite() || swell.period <= 0.0 {
            return Err("swell.period must be finite and positive (simulation seconds)".into());
        }
        if !swell.height.is_finite() || swell.height < 0.0 {
            return Err(
                "swell.height must be finite and nonnegative (crest-to-trough world units)".into(),
            );
        }
        if swell.height > 0.0 && swell.height < 3.0 * spacing {
            return Err("swell.height must resolve at least three particle spacings; lower fluid.spacing and smoothing_radius together (or set height = 0 for still water)".into());
        }
        if self.water_depth < self.reef_height + 2.0 * spacing {
            return Err(
                "wave.water_depth must leave at least two particle layers above the reef".into(),
            );
        }
        if self.water_depth + 2.0 * swell.height + spacing > bounds.size().y {
            return Err("world.height must fit wave.water_depth + 2 * swell.height + fluid.spacing to leave room for breaking crests".into());
        }
        let wavelength = swell.wavelength(-gravity.y, self.water_depth);
        if !wavelength.is_finite() || wavelength < 12.0 * spacing {
            return Err("swell.period is too short to resolve its wavelength; increase it or lower fluid.spacing".into());
        }
        if self.water_depth < wavelength * 0.5 {
            return Err(format!(
                "deep-water swell needs wave.water_depth >= half its wavelength ({:.1}); increase depth or shorten swell.period",
                wavelength * 0.5
            ));
        }
        if swell.height / wavelength > 0.12 {
            return Err("swell.height / wavelength must be <= 0.12 so the incoming swell does not already break offshore; lower height or increase period and depth".into());
        }
        let skew = self.reef_skew.abs() * bounds.size().z * 0.5;
        let offshore = self.reef_start * bounds.size().x - skew;
        let maker = Wavemaker::new(*self, swell, bounds, -gravity.y);
        if offshore < maker.generation_width + 4.0 * spacing {
            return Err("wave.reef_start must leave at least four particle spacings of deep water beyond the generation zone; increase world.width or reef_start, or shorten swell.period".into());
        }
        if self.reef_width * bounds.size().x < 2.0 * spacing {
            return Err("wave.reef_width must resolve at least two particle spacings".into());
        }
        if !(0.1..=0.4).contains(&self.beach_width)
            || (self.reef_start + self.reef_width + self.beach_width) * bounds.size().x
                + skew
                + 4.0 * spacing
                >= bounds.size().x
        {
            return Err("wave.beach_width must be in [0.1, 0.4] and start at least four spacings beyond the whole reef rise".into());
        }
        if bounds.size().z < 8.0 * spacing {
            return Err("world.depth must resolve at least eight particle spacings".into());
        }
        Ok(())
    }

    pub fn water_level(&self, bounds: Bounds) -> f32 {
        bounds.min.y + self.water_depth
    }

    /// The collision surface and visible mesh use exactly the same bathymetry.
    pub fn floor(&self, p: Vec3, bounds: Bounds) -> f32 {
        let start = bounds.min.x + bounds.size().x * self.reef_start + self.reef_skew * p.z;
        let t = ((p.x - start) / (bounds.size().x * self.reef_width)).clamp(0.0, 1.0);
        bounds.min.y + self.reef_height * t * t * (3.0 - 2.0 * t)
    }

    pub fn normal(&self, p: Vec3, bounds: Bounds) -> Vec3 {
        let width = bounds.size().x * self.reef_width;
        let start = bounds.min.x + bounds.size().x * self.reef_start + self.reef_skew * p.z;
        let t = ((p.x - start) / width).clamp(0.0, 1.0);
        let slope = self.reef_height * 6.0 * t * (1.0 - t) / width;
        Vec3::new(-slope, 1.0, slope * self.reef_skew).normalize()
    }

    /// Project along the bed normal. A vertical-only clamp on a steep reef
    /// turns horizontal pressure corrections into an artificial upward jet.
    pub fn project(&self, p: &mut Vec3, bounds: Bounds, margin: f32) {
        for _ in 0..8 {
            let penetration = self.floor(*p, bounds) + margin - p.y;
            if penetration <= 1e-4 {
                break;
            }
            let width = bounds.size().x * self.reef_width;
            let start = bounds.min.x + bounds.size().x * self.reef_start + self.reef_skew * p.z;
            let t = ((p.x - start) / width).clamp(0.0, 1.0);
            let slope = self.reef_height * 6.0 * t * (1.0 - t) / width;
            let normal = Vec3::new(-slope, 1.0, slope * self.reef_skew);
            *p += normal * (penetration / normal.length_squared());
            *p = p.clamp(
                bounds.min + Vec3::splat(margin),
                bounds.max - Vec3::splat(margin),
            );
        }
        p.y = p.y.max(self.floor(*p, bounds) + margin);
    }
}

/// A first-order wave relaxation zone, derived once at reset. Its target is
/// the finite-depth Airy orbital velocity field. Forcing fades out in deep
/// water, leaving the approach, reef and shelf entirely to the particle solver.
/// https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#3132-relaxation-zone-rz
#[derive(Debug, Clone, Copy)]
pub struct Wavemaker {
    origin_x: f32,
    amplitude: f32,
    omega: f32,
    k: f32,
    direction: Vec3,
    depth: f32,
    level: f32,
    generation_width: f32,
    pub period: f32,
    beach_start: f32,
    beach_width: f32,
}

impl Wavemaker {
    pub fn new(wave: Wave, swell: Swell, bounds: Bounds, gravity: f32) -> Self {
        let k = swell.wavenumber(gravity, wave.water_depth);
        let (sin, cos) = swell.direction.to_radians().sin_cos();
        Self {
            origin_x: bounds.min.x,
            amplitude: swell.height * 0.5,
            omega: TAU / swell.period,
            k,
            direction: Vec3::new(cos, 0.0, sin),
            depth: wave.water_depth,
            level: wave.water_level(bounds),
            generation_width: 0.7 * TAU / k,
            period: swell.period,
            beach_start: bounds.max.x - wave.beach_width * bounds.size().x,
            beach_width: wave.beach_width * bounds.size().x,
        }
    }

    pub fn ramp(&self, time: f32) -> f32 {
        let t = ((time - self.period) / (2.0 * self.period)).clamp(0.0, 1.0);
        0.5 - 0.5 * (PI * t).cos()
    }

    pub fn orbital_velocity(&self, p: Vec3, time: f32) -> Vec3 {
        let z = (p.y - self.level).clamp(-self.depth, 0.0);
        // Ratios cosh(k(z+h))/sinh(kh), sinh(k(z+h))/sinh(kh)
        // written as exponentials so large kh cannot overflow.
        let a = (self.k * z).exp();
        let b = (-self.k * (z + 2.0 * self.depth)).exp();
        let denominator = 1.0 - (-2.0 * self.k * self.depth).exp();
        let phase = self.k * (p - Vec3::X * self.origin_x).dot(self.direction)
            - self.omega * (time - self.period);
        let scale = self.amplitude * self.omega * self.ramp(time) / denominator;
        self.direction * (scale * (a + b) * phase.cos()) + Vec3::Y * (scale * (a - b) * phase.sin())
    }

    pub fn generation_weight(&self, p: Vec3) -> f32 {
        let q = ((p.x - self.origin_x) / self.generation_width).clamp(0.0, 1.0);
        // Smooth on both ends, with a long taper into unforced deep water.
        (PI * q).sin().powi(2)
    }

    pub fn drive(&self, p: Vec3, velocity: Vec3, time: f32, dt: f32) -> Vec3 {
        let blend = 1.0 - (-12.0 * self.generation_weight(p) * dt / self.period).exp();
        velocity.lerp(self.orbital_velocity(p, time), blend)
    }

    pub fn peak_speed(&self) -> f32 {
        self.amplitude * self.omega / (self.k * self.depth).tanh()
    }

    pub fn damping(&self, p: Vec3, dt: f32) -> f32 {
        let q = ((p.x - self.beach_start) / self.beach_width).clamp(0.0, 1.0);
        (-6.0 * q * q * dt / self.period).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, sim::Fluid};

    fn config() -> Config {
        Config::parse(include_str!("../config.toml")).unwrap()
    }

    #[test]
    fn dispersion_sets_wavelength_from_period_and_depth() {
        for period in [0.7, 1.8, 5.0] {
            for depth in [10.0, 190.0, 1000.0] {
                let swell = Swell {
                    period,
                    ..Swell::default()
                };
                let k = swell.wavenumber(700.0, depth);
                let omega2 = (TAU / period).powi(2);
                assert!((700.0 * k * (k * depth).tanh() / omega2 - 1.0).abs() < 1e-5);
            }
        }
        let s = Swell::default();
        assert!(
            (s.wavelength(700.0, 1000.0) / (700.0 * s.period.powi(2) / TAU) - 1.0).abs() < 1e-5
        );
    }

    #[test]
    fn wavemaker_obeys_height_period_and_direction() {
        let c = config();
        let make = |s| Wavemaker::new(c.wave, s, c.bounds(), 700.0);
        let a = make(c.swell);
        let b = make(Swell {
            height: c.swell.height * 2.0,
            ..c.swell
        });
        let p = Vec3::new(-400.0, c.wave.water_level(c.bounds()), 40.0);
        let t = 4.25 * c.swell.period;
        assert_eq!(a.orbital_velocity(p, 0.0), Vec3::ZERO);
        assert!((b.orbital_velocity(p, t) - 2.0 * a.orbital_velocity(p, t)).length() < 1e-5);
        assert!(
            (a.orbital_velocity(p, t) - a.orbital_velocity(p, t + c.swell.period)).length() < 1e-3
        );
        assert_eq!(a.orbital_velocity(p, t).z, 0.0);
        let left = make(Swell {
            direction: 30.0,
            ..c.swell
        });
        let right = make(Swell {
            direction: -30.0,
            ..c.swell
        });
        let v = left.orbital_velocity(p, t);
        let mirrored = right.orbital_velocity(p * Vec3::new(1.0, 1.0, -1.0), t);
        assert!((v - mirrored * Vec3::new(1.0, 1.0, -1.0)).length() < 1e-4);
        assert!((v.z / v.x - 30.0f32.to_radians().tan()).abs() < 1e-5);
        assert!(
            a.orbital_velocity(Vec3::new(p.x, c.bounds().min.y, p.z), t)
                .y
                .abs()
                < 1e-5
        );
        let still = make(Swell {
            height: 0.0,
            ..c.swell
        });
        assert_eq!(still.orbital_velocity(p, t), Vec3::ZERO);
        assert!(
            a.generation_weight(Vec3::ZERO) < 1e-10,
            "reef must be unforced"
        );
        assert_eq!(
            a.damping(Vec3::ZERO, 0.1),
            1.0,
            "beach must not damp the reef"
        );
        assert!(a.damping(c.bounds().max, 0.1) < 1.0);
        assert!(
            (a.damping(c.bounds().max, 0.1).powi(2) - a.damping(c.bounds().max, 0.2)).abs() < 1e-6
        );
    }

    #[test]
    fn invalid_swell_and_tank_settings_are_rejected() {
        for edit in [
            "direction = nan",
            "direction = 90.0",
            "direction = -61.0",
            "height = -1.0",
            "height = inf",
            "height = 1.0",
            "height = 500.0",
            "period = 0.0",
            "period = -1.0",
            "period = nan",
            "period = 0.1",
            "period = 10.0",
        ] {
            let field = edit.split(' ').next().unwrap();
            let mut c = config();
            let text = format!("[swell]\n{edit}\n");
            c.swell = toml::from_str::<Config>(&text).unwrap().swell;
            assert!(c.validate().is_err(), "accepted swell.{field}: {edit}");
        }
        let mut c = config();
        c.swell.height = 0.0;
        c.validate().unwrap();
        c.world.gravity = [0.0, 0.0, 0.0];
        assert!(c.validate().unwrap_err().contains("gravity"));
        let mut c = config();
        c.wave.reef_height = c.wave.water_depth;
        assert!(c.validate().is_err());
        let mut c = config();
        c.wave.reef_start = 0.1;
        assert!(c.validate().is_err());
        let mut c = config();
        c.wave.beach_width = 0.5;
        assert!(c.validate().is_err());
        assert!(
            Config::parse("[wave]\ncurl = 2.5").is_err(),
            "obsolete crest controls must not silently do nothing"
        );
    }

    fn gauge(fluid: &Fluid, x: f32) -> f32 {
        let d = fluid.params.spacing;
        let mut heights: Vec<_> = fluid
            .pos
            .iter()
            .filter(|p| (p.x - x).abs() < d && p.z.abs() < 3.0 * d)
            .map(|p| p.y)
            .collect();
        heights.sort_by(|a, b| b.total_cmp(a));
        assert!(!heights.is_empty(), "water is missing at the gauge");
        heights.iter().take(12).sum::<f32>() / heights.len().clamp(1, 12) as f32
    }

    #[test]
    fn incoming_swell_reaches_the_reef_and_resets_cleanly() {
        let mut c = config();
        // A narrow flume retains the shipped cross-section and particle resolution.
        c.world.depth = 80.0;
        c.wave.reef_skew = 0.0;
        c.validate().unwrap();
        let mut f = Fluid::new(c.fluid_params());
        c.reset_fluid(&mut f);
        let initial = f.pos.clone();
        assert!(f.vel.iter().all(|v| *v == Vec3::ZERO));
        assert!(f.pos.iter().all(|p| p.y <= c.wave.water_level(c.bounds())));
        let mut surface =
            crate::surface::Surface::new(c.bounds(), c.fluid.spacing, c.render.surface_resolution);
        surface.rebuild(&f);
        let probe = Vec3::new(
            50.0,
            c.wave.water_level(c.bounds()) + c.fluid.spacing * 2.0,
            0.0,
        );
        assert!(
            surface.density_at(probe) < crate::surface::ISO,
            "a crest must not exist at reset"
        );
        let mut near_min = f32::INFINITY;
        let mut near_max = f32::NEG_INFINITY;
        let mut peak = 0.0f32;
        for step in 1..=900 {
            f.step(1.0 / 60.0);
            peak = peak.max(f.max_speed());
            assert!(f.pos.iter().all(|p| p.is_finite()
                && p.cmpge(c.bounds().min).all()
                && p.cmple(c.bounds().max).all()
                && p.y + 1e-3 >= c.wave.floor(*p, c.bounds()) + c.fluid.spacing * 0.5
                && p.x + 1e-3 >= c.bounds().min.x + c.fluid.spacing * 0.5));
            if step > 360 {
                let near = gauge(&f, 50.0);
                near_min = near_min.min(near);
                near_max = near_max.max(near);
            }
        }
        assert!(
            near_max - near_min > c.fluid.spacing,
            "swell did not reach the shelf"
        );
        assert!(
            peak < c.expected_peak_speed(),
            "wavemaker injected excessive energy: {peak}"
        );
        assert!(f.compression_error() < 0.1);
        assert_eq!(f.len(), initial.len(), "wave tank must conserve particles");
        let wave_range = near_max - near_min;
        c.reset_fluid(&mut f);
        assert_eq!(f.pos, initial);
        assert_eq!(f.elapsed, 0.0);
        assert!(f.foam.iter().all(|&v| v == 0.0));
        assert!(f.vel.iter().all(|&v| v == Vec3::ZERO));

        // A control run distinguishes arriving swell from startup settling or
        // a reef collision that generates its own waves without any input.
        c.swell.height = 0.0;
        c.reset_fluid(&mut f);
        let mut still_min = f32::INFINITY;
        let mut still_max = f32::NEG_INFINITY;
        for step in 1..=900 {
            f.step(1.0 / 60.0);
            if step > 360 {
                let near = gauge(&f, 50.0);
                still_min = still_min.min(near);
                still_max = still_max.max(near);
            }
        }
        assert!(
            wave_range > 2.0 * (still_max - still_min),
            "reef motion must come from swell: wave range {wave_range}, still-water range {}",
            still_max - still_min
        );
        assert!(
            (gauge(&f, -250.0) - c.wave.water_level(c.bounds())).abs() < 1.5 * c.fluid.spacing,
            "solid boundary support must preserve offshore water depth"
        );
    }
}
