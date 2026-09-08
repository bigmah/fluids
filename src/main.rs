//! A 2D water simulation: Position Based Fluids, drawn with Bevy sprites.
//!
//! Controls:
//!   left mouse   push the water away from the cursor
//!   right mouse  pull the water towards the cursor
//!   space        pause / resume
//!   R            reset to the starting dam break
//!   G            flip gravity

mod render;
mod sim;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use render::FluidRenderPlugin;
use sim::{Fluid, FluidParams, START_BLOCK};

/// Solver rate. The solver is stable here; see the tests in `sim`.
const SIM_HZ: f64 = 60.0;

/// Radius of the mouse's influence, in world units.
const MOUSE_RADIUS: f32 = 130.0;
/// Velocity change applied at the centre of the mouse's influence, per second.
const MOUSE_STRENGTH: f32 = 5200.0;

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
    let params = FluidParams::default();
    let window_size = params.bounds.size();

    let mut fluid = Fluid::new(params);
    fluid.fill_block(START_BLOCK.0, START_BLOCK.1);

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Fluids".into(),
                        resolution: (window_size.x as u32, window_size.y as u32).into(),
                        resizable: false,
                        // Without this the window can open on whichever
                        // monitor the WM feels like, including off-screen.
                        position: WindowPosition::Centered(MonitorSelection::Primary),
                        ..default()
                    }),
                    ..default()
                })
                // Nearest sampling would show the particle texture's pixels
                // when it is scaled up.
                .set(ImagePlugin::default_linear()),
        )
        .insert_resource(Time::<Fixed>::from_hz(SIM_HZ))
        .insert_resource(params)
        .insert_resource(fluid)
        .init_resource::<Paused>()
        .init_resource::<HudTimer>()
        .add_plugins(FluidRenderPlugin)
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
    mut fluid: ResMut<Fluid>,
    mut paused: ResMut<Paused>,
) {
    if keys.just_pressed(KeyCode::Space) {
        paused.0 = !paused.0;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        fluid.fill_block(START_BLOCK.0, START_BLOCK.1);
    }
    if keys.just_pressed(KeyCode::KeyG) {
        fluid.params.gravity = -fluid.params.gravity;
    }
}

fn handle_mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    window: Single<&Window, With<PrimaryWindow>>,
    camera: Single<(&Camera, &GlobalTransform), With<Camera2d>>,
    time: Res<Time>,
    mut fluid: ResMut<Fluid>,
) {
    let push = buttons.pressed(MouseButton::Left);
    let pull = buttons.pressed(MouseButton::Right);
    if !push && !pull {
        return;
    }

    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let (camera, camera_transform) = *camera;
    let Ok(world) = camera.viewport_to_world_2d(camera_transform, cursor) else {
        return;
    };

    // Scaled by frame time so the push feels the same regardless of frame rate.
    let strength = MOUSE_STRENGTH * time.delta_secs() * if push { 1.0 } else { -1.0 };
    fluid.apply_radial_impulse(world, MOUSE_RADIUS, strength);
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
        "Fluids - {} particles - compression {:.1}% - peak {:.0} u/s{}",
        fluid.len(),
        fluid.compression_error() * 100.0,
        fluid.max_speed(),
        if paused.0 { " - PAUSED" } else { "" },
    );
}
