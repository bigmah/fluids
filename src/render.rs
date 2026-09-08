//! Draws the fluid as a cloud of soft round sprites.
//!
//! One sprite per particle, all sharing a single procedurally generated texture
//! so the renderer can batch the whole fluid into one draw call. The soft alpha
//! edge is what makes overlapping particles read as a continuous surface rather
//! than as a bag of discs.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::config::Config;
use crate::sim::Fluid;

/// Links a sprite back to its index in the solver's arrays.
#[derive(Component)]
pub struct ParticleIndex(pub usize);

pub struct FluidRenderPlugin;

impl Plugin for FluidRenderPlugin {
    fn build(&self, app: &mut App) {
        let bg = app.world().resource::<Config>().render.background;
        app.insert_resource(ClearColor(Color::srgb(bg[0], bg[1], bg[2])))
            .add_systems(Startup, spawn_particles)
            // Runs in Update, not FixedUpdate: the solver ticks at a fixed 60 Hz
            // but the sprites should follow the latest state at whatever rate
            // the window actually redraws.
            .add_systems(Update, sync_particles);
    }
}

fn spawn_particles(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    fluid: Res<Fluid>,
    config: Res<Config>,
) {
    commands.spawn(Camera2d);

    let texture = circle_texture(&mut images, 64);
    // Drawn well over the rest spacing. The opaque core has to be more than
    // half a spacing in radius or neighbouring particles leave visible gaps and
    // the fluid reads as a dot screen instead of a body of water.
    let diameter = config.fluid.spacing * config.render.particle_scale;

    for i in 0..fluid.len() {
        commands.spawn((
            Sprite {
                image: texture.clone(),
                custom_size: Some(Vec2::splat(diameter)),
                ..default()
            },
            Transform::from_xyz(fluid.pos[i].x, fluid.pos[i].y, 0.0),
            ParticleIndex(i),
        ));
    }
}

/// Builds a white disc with a soft edge, to be tinted per particle.
fn circle_texture(images: &mut Assets<Image>, size: u32) -> Handle<Image> {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let radius = size as f32 * 0.5;
    for y in 0..size {
        for x in 0..size {
            let offset = Vec2::new(x as f32 + 0.5 - radius, y as f32 + 0.5 - radius);
            let d = offset.length() / radius;
            // Opaque out to half the radius, then a smoothstep to nothing.
            // A linear falloff here looks like fog; the plateau keeps particles
            // solid enough to read as liquid, and wide enough that neighbours
            // at rest spacing overlap without a gap.
            let t = ((1.0 - d) / 0.5).clamp(0.0, 1.0);
            let alpha = t * t * (3.0 - 2.0 * t);
            let i = ((y * size + x) * 4) as usize;
            data[i] = 255;
            data[i + 1] = 255;
            data[i + 2] = 255;
            data[i + 3] = (alpha * 255.0) as u8;
        }
    }
    images.add(Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    ))
}

/// Deep blue at rest, through cyan, to white foam at speed.
fn speed_color(speed: f32, palette: &crate::config::Render) -> Color {
    // Square root, so the slow end of the range -- where most of the fluid
    // lives -- still shows variation instead of flattening to one blue.
    let t = (speed / palette.foam_speed).clamp(0.0, 1.0).sqrt();
    let deep = Vec3::from(palette.deep_color);
    let mid = Vec3::from(palette.mid_color);
    let foam = Vec3::from(palette.foam_color);
    let c = if t < 0.5 {
        deep.lerp(mid, t * 2.0)
    } else {
        mid.lerp(foam, (t - 0.5) * 2.0)
    };
    Color::srgb(c.x, c.y, c.z)
}

fn sync_particles(
    fluid: Res<Fluid>,
    config: Res<Config>,
    mut particles: Query<(&ParticleIndex, &mut Transform, &mut Sprite)>,
) {
    for (index, mut transform, mut sprite) in &mut particles {
        let i = index.0;
        transform.translation.x = fluid.pos[i].x;
        transform.translation.y = fluid.pos[i].y;
        sprite.color = speed_color(fluid.vel[i].length(), &config.render);
    }
}
