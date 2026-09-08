//! A 3D water simulation: Position Based Fluids, drawn with Bevy.
//!
//! Everything tunable lives in `config.toml`; see `config.rs` for the fields
//! and their defaults. Pass a different path as the first argument.
//!
//! Controls, with the fluid on the left button as it was in 2D:
//!   left mouse    push the water away from the cursor
//!   shift + left  pull the water towards the cursor
//!   right drag    orbit the camera
//!   scroll        zoom
//!   space         pause / resume
//!   R             reset to the starting dam break
//!   G             flip gravity

mod camera;
mod config;
mod render;
mod sim;

use bevy::prelude::*;
use bevy::window::{PresentMode, PrimaryWindow};

use camera::{OrbitCamera, OrbitCameraPlugin};
use config::Config;
use render::FluidRenderPlugin;
use sim::Fluid;

/// Solver rate. The solver is stable here; see the tests in `sim`.
const SIM_HZ: f64 = 60.0;

#[derive(Resource, Default)]
struct Paused(bool);

/// Exponentially smoothed frame and solver timings, for the title readout.
///
/// Worth carrying because the two answer different questions: the solver time
/// is what a GPU port would attack, and the gap between it and the frame time
/// is what a GPU port would leave untouched.
#[derive(Resource, Default)]
struct Timings {
    frame_ms: f32,
    solver_ms: f32,
}

impl Timings {
    fn feed(slot: &mut f32, sample: f32) {
        *slot = if *slot == 0.0 {
            sample
        } else {
            *slot * 0.9 + sample * 0.1
        };
    }
}

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
                present_mode: if config.render.vsync {
                    PresentMode::AutoVsync
                } else {
                    PresentMode::AutoNoVsync
                },
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
        .init_resource::<Timings>()
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

fn step_fluid(mut fluid: ResMut<Fluid>, time: Res<Time<Fixed>>, mut timings: ResMut<Timings>) {
    let dt = time.delta_secs();
    let start = std::time::Instant::now();
    fluid.step(dt);
    Timings::feed(&mut timings.solver_ms, start.elapsed().as_secs_f32() * 1000.0);
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

/// Pushes or pulls the water at the cursor, on the left button as in 2D.
///
/// The depth comes from the fluid: the impulse lands on the frontmost water
/// under the cursor. Falling back to a plane through the camera's focus point
/// keeps a drag going when the cursor slides off the water mid-stroke, instead
/// of the push cutting out.
fn handle_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform, &OrbitCamera)>,
    config: Res<Config>,
    time: Res<Time>,
    mut fluid: ResMut<Fluid>,
) {
    if !buttons.pressed(MouseButton::Left) {
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let (camera, camera_transform, orbit) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    let reach = config.input.mouse_radius;
    let point = match fluid.nearest_along_ray(ray.origin, *ray.direction, reach) {
        Some(hit) => hit,
        None => {
            let plane = InfinitePlane3d::new(camera_transform.forward());
            let Some(distance) = ray.intersect_plane(orbit.target, plane) else {
                return;
            };
            ray.get_point(distance)
        }
    };

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
    real: Res<Time<Real>>,
    mut timings: ResMut<Timings>,
    mut timer: ResMut<HudTimer>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    Timings::feed(&mut timings.frame_ms, real.delta_secs() * 1000.0);
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    // Counted geometrically rather than read off the SPH estimate, which is
    // truncated near every surface. 100% is a fluid at rest density; a pool too
    // thin to have a bulk has nothing to report.
    let bulk = fluid.interior_density_ratio();
    let bulk = if bulk.is_finite() {
        format!("{:.0}%", bulk * 100.0)
    } else {
        "n/a".to_string()
    };
    window.title = format!(
        "Fluids - {} particles - {:.1} ms/frame ({:.1} solver) - compression {:.1}% \
         - bulk {bulk} - peak {:.0} u/s{}",
        fluid.len(),
        timings.frame_ms,
        timings.solver_ms,
        fluid.compression_error() * 100.0,
        fluid.max_speed(),
        if paused.0 { " - PAUSED" } else { "" },
    );
}
