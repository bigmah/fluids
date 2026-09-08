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
//!   R             replay the current scene
//!   G             flip gravity
//!   S             normal / slow playback
//!   P / B         particle view / tank bounds
//!   F12           screenshot

mod camera;
mod config;
mod render;
mod sim;
mod surface;
mod wave;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, PrimaryWindow};

use camera::{OrbitCamera, OrbitCameraPlugin};
use config::Config;
use render::FluidRenderPlugin;
use sim::Fluid;

/// Solver rate. The solver is stable here; see the tests in `sim`.
const SIM_HZ: f64 = 60.0;

#[derive(Resource, Default)]
struct Paused(bool);

#[derive(Resource)]
struct Playback {
    scale: f32,
}

/// Optional deterministic capture for visual regression checks. The window
/// still renders normally; capture freezes at a requested simulation time.
#[derive(Resource)]
struct Capture {
    at: f32,
    path: String,
    warmup: u32,
    requested: bool,
}

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
    config.reset_fluid(&mut fluid);
    let size = config.bounds().size();
    println!(
        "{} particles, {}x{}x{} world, {} substeps x {} iterations",
        fluid.len(),
        size.x,
        size.y,
        size.z,
        config.solver.substeps,
        config.solver.iterations
    );

    let capture = std::env::var("FLUIDS_CAPTURE_PATH")
        .ok()
        .map(|path| Capture {
            at: std::env::var("FLUIDS_CAPTURE_AT")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|t| t.is_finite() && *t >= 0.0)
                .unwrap_or(0.0),
            path,
            warmup: 0,
            requested: false,
        });
    let mut app = App::new();
    if let Some(capture) = capture {
        app.insert_resource(capture);
    }
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
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
    .insert_resource(Playback {
        scale: config.scene.time_scale,
    })
    .insert_resource(config)
    .insert_resource(fluid)
    .init_resource::<Paused>()
    .init_resource::<HudTimer>()
    .init_resource::<Timings>()
    .add_plugins((OrbitCameraPlugin, FluidRenderPlugin))
    .add_systems(FixedUpdate, step_fluid.run_if(running))
    .add_systems(
        Update,
        (
            handle_keys,
            handle_mouse.run_if(running),
            update_title,
            capture_frame,
            playback_clock,
        ),
    )
    .run();
}

fn running(paused: Res<Paused>) -> bool {
    !paused.0
}

fn step_fluid(
    mut fluid: ResMut<Fluid>,
    time: Res<Time<Fixed>>,
    mut timings: ResMut<Timings>,
    config: Res<Config>,
    capture: Option<Res<Capture>>,
) {
    let mut dt = time.delta_secs();
    if let Some(ref capture) = capture {
        dt = dt.min((capture.at - fluid.elapsed).max(0.0));
    }
    if dt <= 0.0 {
        return;
    }
    let start = std::time::Instant::now();
    fluid.step(dt);
    if capture.is_none()
        && config.scene.replay_after > 0.0
        && fluid.elapsed >= config.scene.replay_after
    {
        config.reset_fluid(&mut fluid);
    }
    Timings::feed(
        &mut timings.solver_ms,
        start.elapsed().as_secs_f32() * 1000.0,
    );
}

fn playback_clock(playback: Res<Playback>, mut time: ResMut<Time<Virtual>>) {
    if playback.is_changed() {
        time.set_relative_speed(playback.scale);
    }
}

fn handle_keys(
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<Config>,
    mut fluid: ResMut<Fluid>,
    mut paused: ResMut<Paused>,
    mut playback: ResMut<Playback>,
) {
    if keys.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        config.reset_fluid(&mut fluid);
    }
    if keys.just_pressed(KeyCode::KeyG) {
        fluid.params.gravity = -fluid.params.gravity;
    }
    if keys.just_pressed(KeyCode::KeyS) {
        playback.scale = if playback.scale < 0.99 {
            1.0
        } else {
            config.scene.time_scale.min(0.25)
        };
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
#[allow(
    clippy::too_many_arguments,
    reason = "Bevy injects resources as independent system parameters"
)]
fn update_title(
    fluid: Res<Fluid>,
    paused: Res<Paused>,
    time: Res<Time>,
    real: Res<Time<Real>>,
    mut timings: ResMut<Timings>,
    mut timer: ResMut<HudTimer>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    surface: Res<render::SurfaceTiming>,
    playback: Res<Playback>,
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
        "{} | {:.2}s / {:.2}x | {} particles | {:.1} ms ({:.1} solve + {:.1} surface) | compression {:.1}% | bulk {bulk} | peak {:.0}{} | SPACE pause · R replay · S speed · P particles · B bounds",
        if fluid.wave.is_some() {
            "Slab / breaking wave"
        } else {
            "Fluids / dam break"
        },
        fluid.elapsed,
        playback.scale,
        fluid.len(),
        timings.frame_ms,
        timings.solver_ms,
        surface.0,
        fluid.compression_error() * 100.0,
        fluid.max_speed(),
        if paused.0 { " - PAUSED" } else { "" },
    );
}

fn capture_frame(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    fluid: Res<Fluid>,
    capture: Option<ResMut<Capture>>,
    timings: Res<Timings>,
    surface: Res<render::SurfaceTiming>,
) {
    if keys.just_pressed(KeyCode::F12) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(format!("fluids-{stamp}.png")));
    }
    let Some(mut capture) = capture else {
        return;
    };
    if capture.requested || fluid.elapsed + 1e-5 < capture.at {
        return;
    }
    capture.warmup += 1;
    if capture.warmup < 30 {
        return;
    }
    capture.requested = true;
    println!(
        "Capture at {:.3}s: {:.2} ms solver, {:.2} ms surface",
        fluid.elapsed, timings.solver_ms, surface.0
    );
    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(capture.path.clone()))
        .observe(
            |_: On<bevy::render::view::screenshot::ScreenshotCaptured>,
             mut exit: MessageWriter<AppExit>| {
                exit.write(AppExit::Success);
            },
        );
}

#[cfg(test)]
mod playback_tests {
    use super::*;
    use bevy::time::TimeUpdateStrategy;
    use std::time::Duration;

    fn after_thirty_steps(scale: f32) -> Vec<Vec3> {
        let mut config = Config::default();
        config.fluid.block = [6, 8, 6];
        let mut fluid = Fluid::new(config.fluid_params());
        config.reset_fluid(&mut fluid);
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1.0 / SIM_HZ,
            )))
            .insert_resource(Time::<Fixed>::from_hz(SIM_HZ))
            .insert_resource(Playback { scale })
            .insert_resource(config)
            .insert_resource(fluid)
            .init_resource::<Timings>()
            .add_systems(FixedUpdate, step_fluid)
            .add_systems(Update, playback_clock);
        for _ in 0..200 {
            app.update();
            let fluid = app.world().resource::<Fluid>();
            if fluid.elapsed >= 0.5 - 1e-5 {
                assert!((fluid.elapsed - 0.5).abs() < 1e-5);
                return fluid.pos.clone();
            }
        }
        panic!("playback never reached the target simulation time");
    }

    #[test]
    fn slow_motion_preserves_the_physics() {
        assert_eq!(after_thirty_steps(1.0), after_thirty_steps(0.25));
    }
}
