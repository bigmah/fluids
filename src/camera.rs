//! A free-flying camera: drag to look around, WASD to move.
//!
//! Nothing in the scene answers to the mouse, so either button turns the view.

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::{camera::Hdr, core_pipeline::tonemapping::Tonemapping};
use core::f32::consts::FRAC_PI_2;

use crate::config::Config;

/// Where the camera faces, kept as angles rather than read back off the
/// transform, so looking up and down can stop short of vertical, where turning
/// left and right would start to roll the view.
#[derive(Component)]
pub struct FlyCamera {
    /// Angle around the vertical axis.
    yaw: f32,
    /// Angle above the horizontal.
    pitch: f32,
    /// Cruising speed in units per second, sized to the opening shot.
    speed: f32,
    velocity: Vec3,
}

const MAX_PITCH: f32 = FRAC_PI_2 - 0.02;
const LOOK_SENSITIVITY: f32 = 0.004;
/// Shift multiplies the cruising speed by this.
const BOOST: f32 = 4.0;
/// How quickly the camera takes up the speed the keys ask for, per second.
/// Easing in and out keeps starts and stops from jolting the view.
const RESPONSE: f32 = 10.0;

pub struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_camera)
            .add_systems(Update, fly);
    }
}

fn spawn_camera(mut commands: Commands, config: Res<Config>) {
    let bounds = config.bounds();
    let size = bounds.size();
    // Aimed a little below the middle of the tank rather than at its centre:
    // the water spends most of its time in the bottom half, and pointing at
    // the geometric centre wastes the upper third of the frame on empty space.
    let wave = config.scene.scenario == crate::config::Scenario::Wave;
    let target = Vec3::new(
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
    );
    // Focus on the reef break; the dam break keeps the whole tank in view.
    // A wave is only ever about a tenth of its own wavelength tall, so a shot
    // framing the whole tank makes any swell look small no matter how big it
    // is. This one sits close enough to the reef that the offshore half runs
    // off the sides, and frames at most four tank-heights of a long tank, so a
    // long approach does not push the break into the distance. It stands back
    // a little further than a wide tank is deep, so a long crest runs away from
    // the camera instead of starting right under it.
    let distance = if wave {
        (Vec3::new(size.x.min(4.0 * size.y), size.y, size.z).length() * 0.38).max(size.z * 1.15)
    } else {
        size.length() * 1.05
    };
    let (yaw, pitch) = if wave { (0.58, 0.30) } else { (0.7, 0.30) };
    let (sy, cy) = f32::sin_cos(yaw);
    let (sp, cp) = f32::sin_cos(pitch);
    let eye = target + Vec3::new(cp * sy, sp, cp * cy) * distance;
    let transform = Transform::from_translation(eye).looking_at(target, Vec3::Y);
    let (yaw, pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            far: (size.length() * 12.0).max(10000.0),
            ..default()
        }),
        Hdr,
        Tonemapping::AcesFitted,
        transform,
        FlyCamera {
            yaw,
            pitch,
            // Covers the opening shot's distance in under two seconds, so the
            // pace suits the dam break's small box and the slab's long one alike.
            speed: distance * 0.6,
            velocity: Vec3::ZERO,
        },
    ));
}

fn fly(
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    // Real time, so slow motion slows the water and not the camera.
    time: Res<Time<Real>>,
    mut camera: Single<(&mut FlyCamera, &mut Transform)>,
) {
    let (fly, transform) = &mut *camera;
    let mut turned = false;
    if buttons.any_pressed([MouseButton::Left, MouseButton::Right]) && motion.delta != Vec2::ZERO {
        fly.yaw -= motion.delta.x * LOOK_SENSITIVITY;
        fly.pitch = (fly.pitch - motion.delta.y * LOOK_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
        turned = true;
    }
    let rotation = Quat::from_euler(EulerRot::YXZ, fly.yaw, fly.pitch, 0.0);

    // Forward follows the view, so W flies where you are looking; up and down
    // stay vertical whichever way that is.
    let held = |a, b| keys.any_pressed([a, b]);
    let mut heading = Vec3::ZERO;
    if held(KeyCode::KeyW, KeyCode::ArrowUp) {
        heading += rotation * Vec3::NEG_Z;
    }
    if held(KeyCode::KeyS, KeyCode::ArrowDown) {
        heading -= rotation * Vec3::NEG_Z;
    }
    if held(KeyCode::KeyD, KeyCode::ArrowRight) {
        heading += rotation * Vec3::X;
    }
    if held(KeyCode::KeyA, KeyCode::ArrowLeft) {
        heading -= rotation * Vec3::X;
    }
    if keys.pressed(KeyCode::KeyE) {
        heading += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyQ) {
        heading -= Vec3::Y;
    }
    let boost = if held(KeyCode::ShiftLeft, KeyCode::ShiftRight) {
        BOOST
    } else {
        1.0
    };
    let wanted = heading.normalize_or_zero() * fly.speed * boost;

    let dt = time.delta_secs();
    fly.velocity = fly.velocity.lerp(wanted, 1.0 - (-RESPONSE * dt).exp());
    // Settle to a stop rather than creeping forever towards one.
    if wanted == Vec3::ZERO && fly.velocity.length() < fly.speed * 1e-3 {
        fly.velocity = Vec3::ZERO;
    }
    if turned || fly.velocity != Vec3::ZERO {
        transform.rotation = rotation;
        transform.translation += fly.velocity * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / 60.0,
            )))
            .insert_resource(Config::load("config.toml").unwrap())
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<AccumulatedMouseMotion>()
            .add_plugins(FlyCameraPlugin);
        app.update();
        app
    }

    fn pose(app: &mut App) -> Transform {
        *app.world_mut()
            .query_filtered::<&Transform, With<FlyCamera>>()
            .single(app.world())
            .unwrap()
    }

    fn frames(app: &mut App, count: usize) {
        for _ in 0..count {
            app.update();
        }
    }

    #[test]
    fn holds_still_until_asked_then_flies_where_it_looks() {
        let mut app = app();
        let start = pose(&mut app);
        frames(&mut app, 30);
        assert_eq!(pose(&mut app), start, "the camera drifted with no input");

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyW);
        frames(&mut app, 60);
        let flown = pose(&mut app);
        let travel = flown.translation - start.translation;
        assert!(
            travel.normalize().dot(*start.forward()) > 0.999,
            "W flew {travel:?}, looking along {:?}",
            start.forward()
        );
        // Rebuilt from its angles, so equal to rounding rather than exactly.
        assert!(flown.rotation.abs_diff_eq(start.rotation, 1e-5));

        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(KeyCode::KeyW);
        frames(&mut app, 120);
        let stopped = pose(&mut app);
        frames(&mut app, 10);
        assert_eq!(pose(&mut app), stopped, "the camera never came to rest");
    }

    #[test]
    fn dragging_right_turns_right() {
        let mut app = app();
        let start = pose(&mut app);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(40.0, 0.0);
        frames(&mut app, 1);
        let turned = pose(&mut app);
        assert!(turned.forward().dot(*start.right()) > 0.0);
        assert_eq!(turned.translation, start.translation);
    }
}
