//! A deliberately initialized plunging crest, released into the PBF solver.
//! This is a barrel study, not a prediction of offshore swell shoaling at Pipeline.
//! The hollow is air: no particles or prescribed animation inside it.

use crate::sim::Bounds;
use bevy::math::Vec3;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Wave {
    pub water_depth: f32,
    pub height: f32,
    /// Lip thickness as a fraction of crest height.
    pub lip_thickness: f32,
    /// Initial forward speed at the crest, in world units / second.
    pub speed: f32,
    /// Angle of the pitching lip in radians; pi/2 is directly overhead.
    pub curl: f32,
    /// Variation of curl along the crest, giving a progressively breaking lip.
    pub peel: f32,
    pub reef_height: f32,
}

impl Default for Wave {
    fn default() -> Self {
        Self {
            water_depth: 66.0,
            height: 155.0,
            lip_thickness: 0.23,
            speed: 280.0,
            curl: 2.25,
            peel: 0.6,
            reef_height: 28.0,
        }
    }
}

impl Wave {
    pub fn validate(&self, bounds: Bounds, spacing: f32) -> Result<(), String> {
        for (name, value) in [
            ("water_depth", self.water_depth),
            ("height", self.height),
            ("lip_thickness", self.lip_thickness),
            ("speed", self.speed),
            ("curl", self.curl),
            ("peel", self.peel),
            ("reef_height", self.reef_height),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("wave.{name} must be finite and nonnegative"));
            }
        }
        if self.water_depth < self.reef_height + 2.0 * spacing {
            return Err(
                "wave.water_depth must leave at least two particle layers above the reef".into(),
            );
        }
        if self.height < 8.0 * spacing
            || self.height * (0.8 + self.lip_thickness).max(1.0) + self.water_depth + spacing
                > bounds.size().y
        {
            return Err("wave.height must be at least eight particle spacings and fit above wave.water_depth".into());
        }
        if !(0.15..=0.4).contains(&self.lip_thickness)
            || self.height * self.lip_thickness < 2.0 * spacing
        {
            return Err("wave.lip_thickness must be in [0.15, 0.4] and resolve at least two particle layers".into());
        }
        if self.height > bounds.size().x * 0.4 || bounds.size().z < 8.0 * spacing {
            return Err("wave needs world.width >= 2.5 * wave.height and at least eight layers across world.depth".into());
        }
        if self.curl - self.peel * 0.5 < 1.5 || self.curl + self.peel * 0.5 > 2.95 {
            return Err("wave.curl +/- half wave.peel must stay in [1.5, 2.95]".into());
        }
        Ok(())
    }

    pub fn water_level(&self, bounds: Bounds) -> f32 {
        bounds.min.y + self.water_depth
    }

    /// A smooth reef shelf. Shared by collision geometry and its visible mesh.
    pub fn floor(&self, p: Vec3, bounds: Bounds) -> f32 {
        let t = ((p.x - 0.12 * p.z + bounds.size().x * 0.10) / (bounds.size().x * 0.35))
            .clamp(0.0, 1.0);
        bounds.min.y + self.reef_height * t * t * (3.0 - 2.0 * t)
    }

    pub fn sample(&self, p: Vec3, bounds: Bounds) -> Option<Vec3> {
        let level = self.water_level(bounds);
        let center_x = -bounds.size().x * 0.04 + 0.12 * p.z;
        let rx = self.height * 0.48;
        let ry = self.height * 0.40;
        let thickness = self.height * self.lip_thickness;
        let x = p.x - center_x;
        let y = p.y - (level + ry);
        let inner = (x / rx).powi(2) + (y / ry).powi(2);
        let outer = (x / (rx + thickness)).powi(2) + (y / (ry + thickness)).powi(2);
        let angle = (y / ry).atan2(-x / rx);
        let tip = self.curl - self.peel * p.z / bounds.size().z;
        let back_height = self.height * (-(x / (self.height * 0.9)).powi(2)).exp();
        let pool = p.y <= level;
        let back = x <= 0.0 && p.y <= level + back_height && inner >= 1.0;
        let lip =
            outer <= 1.0 && inner >= 1.0 && angle <= tip && angle >= -std::f32::consts::FRAC_PI_2;
        if !(pool || back || lip) {
            return None;
        }

        let rise = ((p.y - level) / (self.height * 0.65)).clamp(0.0, 1.0);
        let envelope = (-(x / self.height).powi(2)).exp();
        let speed = self.speed * rise * envelope;
        Some(Vec3::new(
            speed * angle.sin().max(0.15),
            speed * 0.60 * angle.cos(),
            0.0,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, sim::Fluid, surface::Surface};

    fn config() -> Config {
        Config::parse(include_str!("../config.toml")).unwrap()
    }

    #[test]
    fn the_barrel_contains_air_below_a_resolved_lip() {
        let c = config();
        let w = c.wave;
        let b = c.bounds();
        let center = Vec3::new(-b.size().x * 0.04, w.water_level(b) + w.height * 0.40, 0.0);
        assert!(
            w.sample(center, b).is_none(),
            "the barrel cavity must be air"
        );
        assert!(
            w.sample(center + Vec3::Y * w.height * 0.48, b).is_some(),
            "missing overhead lip"
        );
        let mut fluid = Fluid::new(c.fluid_params());
        c.reset_fluid(&mut fluid);
        let radius = w.height * 0.30;
        assert!(fluid.pos.iter().all(|p| {
            p.z.abs() > c.fluid.spacing
                || Vec3::new(p.x - center.x, p.y - center.y, 0.0).length() > radius
        }));
        // Reconstruction must not bridge the air cavity or lose the lip.
        let mut surface = Surface::new(b, c.fluid.spacing, c.render.surface_resolution);
        surface.rebuild(&fluid);
        assert!(surface.triangles > 1000);
        assert!(
            surface.density_at(center) < crate::surface::ISO,
            "reconstruction filled the air cavity"
        );
        assert!(
            surface.density_at(center + Vec3::Y * w.height * 0.48) > crate::surface::ISO,
            "reconstruction lost the overhead lip"
        );
    }

    #[test]
    fn a_plunging_crest_stays_finite_and_above_the_reef() {
        let c = config();
        let mut fluid = Fluid::new(c.fluid_params());
        c.reset_fluid(&mut fluid);
        let n = fluid.len();
        let initial = fluid.pos.clone();
        for _ in 0..150 {
            fluid.step(1.0 / 60.0);
            assert!(
                fluid.max_speed() < 2000.0,
                "wave solver gained excessive energy: {}",
                fluid.max_speed()
            );
            for p in &fluid.pos {
                assert!(p.is_finite());
                assert!(p.cmpge(c.bounds().min).all() && p.cmple(c.bounds().max).all());
                assert!(p.y + 1e-3 >= c.wave.floor(*p, c.bounds()) + c.fluid.spacing * 0.5);
            }
        }
        assert_eq!(fluid.len(), n, "wave loses particles");
        assert!(
            fluid
                .pos
                .iter()
                .zip(&initial)
                .any(|(a, b)| a.distance(*b) > c.wave.height * 0.5)
        );
        assert!(
            fluid.compression_error() < 0.10,
            "wave compression {}",
            fluid.compression_error()
        );
        c.reset_fluid(&mut fluid);
        assert_eq!(fluid.pos, initial);
        assert_eq!(fluid.elapsed, 0.0);
        assert!(fluid.foam.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn invalid_wave_geometry_is_rejected() {
        let c = config();
        for w in [
            Wave {
                height: f32::NAN,
                ..c.wave
            },
            Wave {
                water_depth: 5.0,
                ..c.wave
            },
            Wave {
                lip_thickness: 0.01,
                ..c.wave
            },
            Wave {
                curl: 3.5,
                ..c.wave
            },
            Wave {
                height: 1000.0,
                ..c.wave
            },
        ] {
            assert!(w.validate(c.bounds(), c.fluid.spacing).is_err());
        }
    }
}
