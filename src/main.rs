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
mod gpu;
mod render;
mod sim;
mod sim_gpu;
mod spray_gpu;
mod surface;
mod surface_gpu;
mod wave;

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::{PresentMode, PrimaryWindow};

use camera::{OrbitCamera, OrbitCameraPlugin};
use config::Config;
use gpu::Gpu;
use render::FluidRenderPlugin;
use sim::Fluid;
use sim_gpu::GpuFluid;

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
/// With `until` set, it saves a numbered frame every `every` simulated seconds
/// from `at` to `until`, holding the simulation at each one, so the sequence
/// plays back at the simulation's own speed whatever the machine's frame rate.
#[derive(Resource)]
struct Capture {
    at: f32,
    start: f32,
    until: Option<f32>,
    every: f32,
    frame: u32,
    path: String,
    warmup: u32,
    requested: bool,
}

impl Capture {
    /// Within a quarter step counts as arrived. Landing exactly would take a
    /// sliver of a step, and PBF reads velocity back out of the position change,
    /// so a tiny `dt` turns ordinary density corrections into a velocity spike.
    const SLACK: f32 = 0.25 / SIM_HZ as f32;

    pub(crate) fn reached(&self, elapsed: f32) -> bool {
        elapsed + Self::SLACK >= self.at
    }

    /// Sequence frames land on whole solver steps, so the recorded run takes
    /// the same steps as an uncaptured one.
    fn frame_time(&self) -> f32 {
        let t = self.start + self.frame as f32 * self.every;
        (t * SIM_HZ as f32).round() / SIM_HZ as f32
    }
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
        "{} particles, {}x{}x{} world, {} substeps x {} iterations on the {}",
        fluid.len(),
        size.x,
        size.y,
        size.z,
        config.solver.substeps,
        config.solver.iterations,
        match config.solver.backend {
            config::Backend::Gpu => "GPU",
            config::Backend::Cpu => "CPU",
        }
    );
    if config.scene.scenario == config::Scenario::Wave && config.swell.waves == 1 {
        println!(
            "Single wave: height {:.1} units in {:.1} of water; shelf depth {:.1} units",
            config.swell.height,
            config.wave.water_depth,
            config.wave.water_depth - config.wave.reef_height,
        );
    } else if config.scene.scenario == config::Scenario::Wave {
        println!(
            "Swell: travel {:.1} degrees toward +X/+Z, height {:.1} units, period {:.2}s, wavelength {:.1} units; shelf depth {:.1} units",
            config.swell.direction,
            config.swell.height,
            config.swell.period,
            config
                .swell
                .wavelength(-config.world.gravity[1], config.wave.water_depth),
            config.wave.water_depth - config.wave.reef_height,
        );
    }

    let env_time = |name| {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|t| t.is_finite() && *t >= 0.0)
    };
    let capture = std::env::var("FLUIDS_CAPTURE_PATH").ok().map(|path| {
        let start = env_time("FLUIDS_CAPTURE_AT").unwrap_or(0.0);
        let mut capture = Capture {
            at: start,
            start,
            until: env_time("FLUIDS_CAPTURE_UNTIL"),
            every: 1.0
                / env_time("FLUIDS_CAPTURE_FPS")
                    .filter(|f| *f > 0.0)
                    .unwrap_or(30.0),
            frame: 0,
            path,
            warmup: 0,
            requested: false,
        };
        if capture.until.is_some() {
            capture.at = capture.frame_time();
            if let Err(e) = std::fs::create_dir_all(&capture.path) {
                eprintln!("error: cannot create {}: {e}", capture.path);
                std::process::exit(1);
            }
        }
        capture
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
    .add_systems(Startup, start_gpu_solver)
    .add_systems(FixedUpdate, step_fluid.run_if(running))
    .add_systems(
        Update,
        (
            handle_keys,
            sync_gpu_fluid.after(handle_keys),
            handle_mouse.run_if(running),
            update_title,
            playback_clock,
        ),
    )
    // After the surface has been rebuilt for this frame, so a screenshot never
    // sees an interpolated mesh left over from before the capture time advanced.
    .add_systems(PostUpdate, capture_frame)
    .run();
}

fn running(paused: Res<Paused>) -> bool {
    !paused.0
}

/// Moves the solver onto the renderer's own GPU when the config asks for it.
/// The render device exists by the time `Startup` runs; without one, the CPU
/// solver carries on.
fn start_gpu_solver(
    mut commands: Commands,
    config: Res<Config>,
    fluid: Res<Fluid>,
    device: Option<Res<RenderDevice>>,
    queue: Option<Res<RenderQueue>>,
) {
    if config.solver.backend != config::Backend::Gpu {
        return;
    }
    let (Some(device), Some(queue)) = (device, queue) else {
        warn!("no render device; the solver stays on the CPU");
        return;
    };
    let gpu = Gpu::new(device.clone(), queue.clone());
    commands.insert_resource(GpuFluid::new(gpu, &fluid));
}

/// A reset while paused has no step to carry it to the GPU; this does.
fn sync_gpu_fluid(fluid: Res<Fluid>, gpu: Option<ResMut<GpuFluid>>) {
    if let Some(mut gpu) = gpu {
        gpu.sync(&fluid);
    }
}

fn step_fluid(
    mut fluid: ResMut<Fluid>,
    gpu: Option<ResMut<GpuFluid>>,
    time: Res<Time<Fixed>>,
    mut timings: ResMut<Timings>,
    config: Res<Config>,
    capture: Option<Res<Capture>>,
) {
    let mut dt = time.delta_secs();
    if let Some(ref capture) = capture {
        if capture.reached(fluid.elapsed) {
            return;
        }
        dt = dt.min(capture.at - fluid.elapsed);
    }
    if dt <= 0.0 {
        return;
    }
    let start = std::time::Instant::now();
    // A reset below is picked up by the GPU solver at its next step.
    match gpu {
        Some(mut gpu) => gpu.step(&mut fluid, dt),
        None => fluid.step(dt),
    }
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
#[allow(
    clippy::too_many_arguments,
    reason = "Bevy injects resources as independent system parameters"
)]
fn handle_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform, &OrbitCamera)>,
    config: Res<Config>,
    time: Res<Time>,
    mut fluid: ResMut<Fluid>,
    gpu: Option<ResMut<GpuFluid>>,
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
    // With the GPU solving, only the GPU knows where the particles are.
    let hit = match gpu.as_ref() {
        Some(gpu) => gpu.nearest_along_ray(&fluid, ray.origin, *ray.direction, reach),
        None => fluid.nearest_along_ray(ray.origin, *ray.direction, reach),
    };
    let point = match hit {
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
    match gpu {
        Some(mut gpu) => gpu.apply_radial_impulse(point, config.input.mouse_radius, strength),
        None => fluid.apply_radial_impulse(point, config.input.mouse_radius, strength),
    }
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
    gpu: Option<Res<GpuFluid>>,
) {
    Timings::feed(&mut timings.frame_ms, real.delta_secs() * 1000.0);
    if !timer.0.tick(time.delta()).just_finished() {
        return;
    }
    // Measured where the neighbour lists live: on the GPU when it is solving.
    let (compression, bulk, peak) = match gpu.as_ref().and_then(|gpu| gpu.readout(&fluid)) {
        Some(readout) => (readout.compression, readout.bulk, readout.peak_speed),
        None => (
            fluid.compression_error(),
            fluid.interior_density_ratio(),
            fluid.max_speed(),
        ),
    };
    // Counted geometrically rather than read off the SPH estimate, which is
    // truncated near every surface. 100% is a fluid at rest density; a pool too
    // thin to have a bulk has nothing to report.
    let bulk = if bulk.is_finite() {
        format!("{:.0}%", bulk * 100.0)
    } else {
        "n/a".to_string()
    };
    window.title = format!(
        "{} | {:.2}s / {:.2}x | {} particles | {:.1} ms ({:.1} {} solve + {:.1} surface) | compression {:.1}% | bulk {bulk} | peak {:.0}{} | SPACE pause · R replay · S speed · P particles · B bounds",
        if fluid.wave.is_some() {
            "Swell / reef break"
        } else {
            "Fluids / dam break"
        },
        fluid.elapsed,
        playback.scale,
        fluid.len(),
        timings.frame_ms,
        timings.solver_ms,
        if gpu.is_some() { "GPU" } else { "CPU" },
        surface.0,
        compression * 100.0,
        peak,
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
    if capture.requested || !capture.reached(fluid.elapsed) {
        return;
    }
    // The first frame waits for the renderer to warm up; later ones only need
    // the frozen surface to reach the screen.
    capture.warmup += 1;
    if capture.warmup < if capture.frame == 0 { 30 } else { 2 } {
        return;
    }
    capture.warmup = 0;
    let (path, last) = match capture.until {
        None => (capture.path.clone(), true),
        Some(until) => {
            let path = std::path::Path::new(&capture.path)
                .join(format!("frame-{:05}.png", capture.frame))
                .to_string_lossy()
                .into_owned();
            capture.frame += 1;
            capture.at = capture.frame_time();
            (path, capture.at > until + Capture::SLACK)
        }
    };
    if capture.frame <= 1 || last {
        println!(
            "Capture at {:.3}s: {:.2} ms solver, {:.2} ms surface",
            fluid.elapsed, timings.solver_ms, surface.0
        );
    }
    let mut screenshot = commands.spawn(Screenshot::primary_window());
    screenshot.observe(save_to_disk(path));
    if last {
        capture.requested = true;
        screenshot.observe(
            |_: On<bevy::render::view::screenshot::ScreenshotCaptured>,
             mut exit: MessageWriter<AppExit>| {
                exit.write(AppExit::Success);
            },
        );
    }
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
