//! Draws the fluid as a cloud of lit spheres, inside a wireframe box.
//!
//! Colouring particles by speed needs per-particle colour, and a
//! `StandardMaterial` carries one colour for every mesh that shares it. Rather
//! than a material per particle -- which would be one draw call per particle --
//! the speed range is quantised into [`PALETTE_STEPS`] materials and each
//! particle is pointed at the closest one. The renderer then batches the fluid
//! into that many instanced draws regardless of how many particles there are.
//!
//! This draws the fluid as what it is, a pile of particles. Making it read as a
//! continuous liquid surface is a rendering problem rather than a simulation
//! one, and the usual answer -- screen-space fluid rendering: splat depth,
//! blur it, rebuild normals from the smoothed depth, then shade with refraction
//! -- is a much larger piece of work than the solver it would be sitting on.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::PrimitiveTopology;
use bevy::prelude::*;

use crate::config::Config;
use crate::sim::Fluid;

/// Links a sprite back to its index in the solver's arrays.
#[derive(Component)]
pub struct ParticleIndex(pub usize);

/// How finely the speed gradient is quantised. Each step is one material and
/// so one draw call; 24 is smooth enough that the banding is invisible against
/// moving water.
const PALETTE_STEPS: usize = 24;

/// The materials the speed gradient was baked into, coolest first.
#[derive(Resource)]
struct Palette(Vec<Handle<StandardMaterial>>);

pub struct FluidRenderPlugin;

impl Plugin for FluidRenderPlugin {
    fn build(&self, app: &mut App) {
        let bg = app.world().resource::<Config>().render.background;
        app.insert_resource(ClearColor(Color::srgb(bg[0], bg[1], bg[2])))
            .insert_resource(GlobalAmbientLight {
                color: Color::srgb(0.6, 0.75, 1.0),
                brightness: 260.0,
                ..default()
            })
            .add_systems(Startup, spawn_scene)
            // Runs in Update, not FixedUpdate: the solver ticks at a fixed 60 Hz
            // but the spheres should follow the latest state at whatever rate
            // the window actually redraws.
            .add_systems(Update, sync_particles);
    }
}

fn spawn_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    fluid: Res<Fluid>,
    config: Res<Config>,
) {
    // Two lights: a key from above and in front to shape the surface, and a
    // dimmer fill from behind so the far side of the fluid isn't a silhouette.
    commands.spawn((
        DirectionalLight {
            illuminance: 9000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(1.0, 2.4, 1.6).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 2600.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-1.4, 0.6, -1.2).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // The floor of the tank, so the water has something to read against, and
    // the edges of the tank as wireframe. Without them the fluid reads as a
    // slab floating in the void: in 3D there is nothing else to judge its
    // depth, scale or orientation against.
    let bounds = config.bounds();
    let size = bounds.size();
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size.x, size.z))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.19, 0.26),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, bounds.min.y + 0.5, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(box_wireframe(bounds.min, bounds.max))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.30, 0.36, 0.48),
            // Unlit, or the lines pick up shading and fade out on the far side.
            unlit: true,
            ..default()
        })),
    ));

    // Drawn a little over the rest spacing so neighbouring particles touch and
    // the body of the fluid reads as a mass rather than a lattice of beads.
    // Low subdivision: at this size a sphere is a few pixels across, and the
    // triangle count is multiplied by every particle.
    let radius = config.fluid.spacing * config.render.particle_scale * 0.5;
    let sphere = meshes.add(Sphere::new(radius).mesh().ico(1).unwrap());

    let palette: Vec<_> = (0..PALETTE_STEPS)
        .map(|i| {
            let t = i as f32 / (PALETTE_STEPS - 1) as f32;
            materials.add(StandardMaterial {
                base_color: gradient(t, &config.render),
                // Wet, but not a mirror: a fully smooth material turns the
                // fluid into a field of specular dots.
                perceptual_roughness: 0.25,
                metallic: 0.0,
                reflectance: 0.6,
                ..default()
            })
        })
        .collect();

    for i in 0..fluid.len() {
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(palette[0].clone()),
            Transform::from_translation(fluid.pos[i]),
            ParticleIndex(i),
        ));
    }
    commands.insert_resource(Palette(palette));
}

/// The twelve edges of a box, as a line list.
fn box_wireframe(min: Vec3, max: Vec3) -> Mesh {
    let corner = |i: usize| {
        Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        )
    };
    // Corner indices differing in exactly one bit are the ones joined by an
    // edge, which is all twelve of them and no diagonals.
    let mut points = Vec::with_capacity(24);
    for a in 0..8usize {
        for bit in [1, 2, 4] {
            let b = a ^ bit;
            if b > a {
                points.push(corner(a));
                points.push(corner(b));
            }
        }
    }
    Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, points)
}

/// Deep blue at rest, through cyan, to white foam at speed.
fn gradient(t: f32, palette: &crate::config::Render) -> Color {
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
    palette: Res<Palette>,
    mut particles: Query<(
        &ParticleIndex,
        &mut Transform,
        &mut MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let foam_speed = config.render.foam_speed;
    for (index, mut transform, mut material) in &mut particles {
        let i = index.0;
        transform.translation = fluid.pos[i];
        // Square root, so the slow end of the range -- where most of the fluid
        // lives -- still shows variation instead of flattening to one blue.
        let t = (fluid.vel[i].length() / foam_speed).clamp(0.0, 1.0).sqrt();
        let step = ((t * (PALETTE_STEPS - 1) as f32).round() as usize).min(PALETTE_STEPS - 1);
        let wanted = &palette.0[step];
        // Only touch the component when the bucket actually changes: writing it
        // every frame would mark all of them changed and defeat the batching.
        if material.0.id() != wanted.id() {
            material.0 = wanted.clone();
        }
    }
}
