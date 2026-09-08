//! A 3D water simulation: Position Based Fluids, drawn with Bevy.
//!
//! Everything tunable lives in `config.toml`; see `config.rs` for the fields
//! and their defaults. Pass a different path as the first argument.
//!
//! Controls:
//!   left drag     orbit the camera
//!   scroll        zoom
//!   right mouse   push the water away from the cursor
//!   shift + right pull the water towards the cursor
//!   space         pause / resume
//!   R             reset to the starting dam break
//!   G             flip gravity

mod camera;
mod config;
mod render;
mod sim;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use camera::{OrbitCamera, OrbitCameraPlugin};
use config::Config;
use render::FluidRenderPlugin;
use sim::Fluid;

/// Solver rate. The solver is stable here; see the tests in `sim`.
const SIM_HZ: f64 = 60.0;

#[derive(Resource, Default)]
struct Paused(bool);

/// Throttles the window-title readout. Recomputing the compression error costs
/// a full neighbourhood pass, so it is not something to do every frame.
#[derive(Resource)]
struct HudTimer(Timer);

impl Default for HudTimer {
    fn default() -> Self {
        Self(Timer::from_seconds(0.25, TimerMode::Repeating))
    }
}

fn main() {
    // A bad config is worth a clean message on stderr rather than a panic
    // backtrace: this file is meant to be edited by hand.
    let loaded = match std::env::args().nth(1) {
        Some(path) => Config::load(path),
        None => Config::load_default(),
    };
    let config = match loaded {
        Ok(config) => config,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let mut fluid = Fluid::new(config.fluid_params());
    fluid.fill_block(config.fluid.block);
    let size = config.bounds().size();
    println!(
        "{} particles, {}x{}x{} world, {} substeps x {} iterations",
        config.particle_count(),
        size.x,
        size.y,
        size.z,
        config.solver.substeps,
        config.solver.iterations
    );

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Fluids".into(),
                resolution: (1280, 800).into(),
                // Without this the window can open on whichever monitor the WM
                // feels like, including off-screen.
                position: WindowPosition::Centered(MonitorSelection::Primary),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(SIM_HZ))
        .insert_resource(config)
        .insert_resource(fluid)
        .init_resource::<Paused>()
        .init_resource::<HudTimer>()
        .add_plugins((OrbitCameraPlugin, FluidRenderPlugin))
        .add_systems(FixedUpdate, step_fluid.run_if(running))
        .add_systems(
            Update,
            (handle_keys, handle_mouse.run_if(running), update_title),
        )
        .run();
}

fn running(paused: Res<Paused>) -> bool {
    !paused.0
}

fn step_fluid(mut fluid: ResMut<Fluid>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs();
    fluid.step(dt);
}

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<Config>,
    mut fluid: ResMut<Fluid>,
    mut paused: ResMut<Paused>,
) {
    if keys.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        fluid.fill_block(config.fluid.block);
    }
    if keys.just_pressed(KeyCode::KeyG) {
        fluid.params.gravity = -fluid.params.gravity;
    }
}

/// Pushes or pulls the water at the cursor.
///
/// A screen position names a ray, not a point, so the depth has to come from
/// somewhere. It is taken from the plane through the camera's focus point
/// facing the camera, which is the reading that matches what the cursor looks
/// like it is over.
fn handle_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform, &OrbitCamera)>,
    config: Res<Config>,
    time: Res<Time>,
    mut fluid: ResMut<Fluid>,
) {
    if !buttons.pressed(MouseButton::Right) {
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let (camera, camera_transform, orbit) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    // The plane the camera is already focused on, so the push lands where the
    // cursor looks like it is pointing.
    let plane = InfinitePlane3d::new(camera_transform.forward());
    let Some(distance) = ray.intersect_plane(orbit.target, plane) else {
        return;
    };
    let point = ray.get_point(distance);

    let pull = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    // Scaled by frame time so the push feels the same regardless of frame rate.
    let strength = config.input.mouse_strength * time.delta_secs() * if pull { -1.0 } else { 1.0 };
    fluid.apply_radial_impulse(point, config.input.mouse_radius, strength);
}

/// Reports the live state of the solver in the window title: how far the fluid
/// is from incompressible, and how fast the quickest particle is moving.
fn update_title(
    fluid: Res<Fluid>,
    paused: Res<Paused>,
    time: Res<Time>,
    mut timer: ResMut<HudTimer>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    window.title = format!(
        "Fluids - {} particles - compression {:.1}% - bulk {:.0}% - peak {:.0} u/s{}",
        fluid.len(),
        fluid.compression_error() * 100.0,
        // Counted geometrically rather than read off the SPH estimate, which
        // is truncated near every surface. 100% is a fluid at rest density.
        fluid.interior_density_ratio() * 100.0,
        fluid.max_speed(),
        if paused.0 { " - PAUSED" } else { "" },
    );
}
