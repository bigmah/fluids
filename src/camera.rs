//! A minimal orbit camera: drag to turn, scroll to zoom.
//!
//! Orbiting is on the right button, leaving the left one for the fluid so the
//! mouse works the way it did in 2D.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use bevy::{camera::Hdr, core_pipeline::tonemapping::Tonemapping};
use core::f32::consts::FRAC_PI_2;

use crate::config::Config;

/// Spherical coordinates around the origin.
#[derive(Component)]
pub struct OrbitCamera {
    /// The point the camera looks at and swings around.
    pub target: Vec3,
    pub radius: f32,
    /// Angle around the vertical axis.
    pub yaw: f32,
    /// Angle above the horizontal. Clamped short of the poles, where the
    /// up-vector becomes ambiguous and the view snaps.
    pub pitch: f32,
}

const MIN_PITCH: f32 = -0.2;
const MAX_PITCH: f32 = FRAC_PI_2 - 0.05;
const ORBIT_SENSITIVITY: f32 = 0.006;
const ZOOM_SENSITIVITY: f32 = 0.12;

pub struct OrbitCameraPlugin;

impl Plugin for OrbitCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, orbit);
    }
}

fn spawn_camera(mut commands: Commands, config: Res<Config>) {
    let bounds = config.bounds();
    let size = bounds.size();
    // Aimed a little below the middle of the tank rather than at its centre:
    // the water spends most of its time in the bottom half, and pointing at
    // the geometric centre wastes the upper third of the frame on empty space.
    let wave = config.scene.scenario == crate::config::Scenario::Wave;
    let orbit = OrbitCamera {
        target: Vec3::new(
            if wave {
                bounds.min.x + size.x * (config.wave.reef_start + config.wave.reef_width)
            } else {
                0.0
            },
            if wave {
                bounds.min.y + config.wave.water_depth + config.swell.height * 0.32
            } else {
                bounds.min.y + size.y * 0.30
            },
            0.0,
        ),
        // Focus on the reef break; the dam break keeps the whole tank in view.
        radius: size.length() * if wave { 0.55 } else { 1.05 },
        yaw: if wave { 0.58 } else { 0.7 },
        pitch: 0.30,
    };
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: (size.length() * 12.0).max(10000.0),
            ..default()
        }),
        Hdr,
        Tonemapping::AcesFitted,
        place(&orbit),
        orbit,
    ));
}

fn place(orbit: &OrbitCamera) -> Transform {
    let (sy, cy) = orbit.yaw.sin_cos();
    let (sp, cp) = orbit.pitch.sin_cos();
    let eye = orbit.target + Vec3::new(cp * sy, sp, cp * cy) * orbit.radius;
    Transform::from_translation(eye).looking_at(orbit.target, Vec3::Y)
}

fn orbit(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut camera: Single<(&mut OrbitCamera, &mut Transform)>,
) {
    let (orbit, transform) = &mut *camera;
    let mut moved = false;

    if buttons.pressed(MouseButton::Right) && motion.delta != Vec2::ZERO {
        orbit.yaw -= motion.delta.x * ORBIT_SENSITIVITY;
        orbit.pitch =
            (orbit.pitch + motion.delta.y * ORBIT_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
        moved = true;
    }
    if scroll.delta.y != 0.0 {
        // Scale the step by the current distance, so zooming feels the same
        // whether you are close in or far out.
        orbit.radius =
            (orbit.radius * (1.0 - scroll.delta.y * ZOOM_SENSITIVITY)).clamp(50.0, 8000.0);
        moved = true;
    }
    if moved {
        **transform = place(orbit);
    }
}
