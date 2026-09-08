//! Continuous, volumetrically shaded water with a particle diagnostic view.

use crate::{config::Config, sim::Fluid, surface::Surface};
use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    camera::visibility::NoFrustumCulling,
    image::ImageSampler,
    mesh::PrimitiveTopology,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, Extent3d, TextureDimension, TextureFormat},
    shader::ShaderRef,
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct WaterMaterial {
    #[uniform(0)]
    origin_cell: Vec4,
    #[uniform(0)]
    extent_level: Vec4,
    #[uniform(0)]
    deep: LinearRgba,
    #[uniform(0)]
    shallow: LinearRgba,
    #[uniform(0)]
    foam: LinearRgba,
    #[uniform(0)]
    clock_floor: Vec4,
    #[texture(1, dimension = "3d")]
    #[sampler(2)]
    density: Handle<Image>,
}

impl Material for WaterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://fluids/water.wgsl".into()
    }
    fn enable_prepass() -> bool {
        false
    }
    fn enable_shadows() -> bool {
        false
    }
}

#[derive(Resource)]
struct WaterAssets {
    mesh: Handle<Mesh>,
    density: Handle<Image>,
    material: Handle<WaterMaterial>,
}
#[derive(Component)]
enum ScenePart {
    Water,
    Bounds,
}
#[derive(Component)]
struct ParticleIndex(usize);
#[derive(Resource, Default)]
struct DiagnosticView(bool);
#[derive(Resource, Default)]
pub struct SurfaceTiming(pub f32);

pub struct FluidRenderPlugin;
impl Plugin for FluidRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "water.wgsl");
        let bg = app.world().resource::<Config>().render.background;
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .insert_resource(ClearColor(Color::srgb(bg[0], bg[1], bg[2])))
            .insert_resource(GlobalAmbientLight {
                color: Color::srgb(0.72, 0.85, 1.0),
                brightness: 180.0,
                ..default()
            })
            .init_resource::<DiagnosticView>()
            .init_resource::<SurfaceTiming>()
            .add_systems(Startup, spawn_scene)
            .add_systems(
                Update,
                (toggle_diagnostics, sync_surface, sync_particles)
                    .chain()
                    .after(crate::handle_keys),
            );
    }
}

fn spawn_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut water_materials: ResMut<Assets<WaterMaterial>>,
    mut images: ResMut<Assets<Image>>,
    fluid: Res<Fluid>,
    config: Res<Config>,
) {
    commands.spawn((
        DirectionalLight {
            illuminance: 12000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-0.5, 0.8, -0.3).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    let b = config.bounds();
    let size = b.size();
    commands.spawn((
        Mesh3d(meshes.add(reef_mesh(&fluid))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.39, 0.39, 0.30),
            perceptual_roughness: 0.95,
            ..default()
        })),
    ));
    commands.spawn((
        Mesh3d(meshes.add(box_wireframe(b.min, b.max))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.29, 0.39, 0.42),
            unlit: true,
            ..default()
        })),
        if config.render.show_bounds {
            Visibility::Visible
        } else {
            Visibility::Hidden
        },
        ScenePart::Bounds,
    ));

    let mut surface = Surface::new(b, config.fluid.spacing, config.render.surface_resolution);
    let mesh = meshes.add(surface.rebuild(&fluid));
    let [x, y, z] = surface.dims.map(|v| v as u32);
    let mut volume = Image::new(
        Extent3d {
            width: x,
            height: y,
            depth_or_array_layers: z,
        },
        TextureDimension::D3,
        surface.texture_bytes(),
        TextureFormat::R8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    volume.sampler = ImageSampler::linear();
    let density = images.add(volume);
    let rgb = |v: [f32; 3]| Color::srgb(v[0], v[1], v[2]).to_linear();
    let extent = Vec3::new(x as f32, y as f32, z as f32) * surface.cell;
    let level = fluid.wave.map_or(
        b.min.y + config.particle_count() as f32 * config.fluid.spacing.powi(3) / (size.x * size.z),
        |w| w.water_level(b),
    );
    let material = water_materials.add(WaterMaterial {
        origin_cell: surface.origin.extend(surface.cell),
        extent_level: extent.extend(level),
        deep: rgb(config.render.deep_color),
        shallow: rgb(config.render.mid_color),
        foam: rgb(config.render.foam_color),
        clock_floor: Vec4::new(0.0, b.min.y, config.fluid.spacing, config.render.foam_speed),
        density: density.clone(),
    });
    commands.spawn((
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material.clone()),
        ScenePart::Water,
        NoFrustumCulling,
    ));
    commands.insert_resource(WaterAssets {
        mesh,
        density,
        material,
    });
    commands.insert_resource(surface);

    let sphere = meshes.add(
        Sphere::new(config.fluid.spacing * 0.3)
            .mesh()
            .ico(1)
            .unwrap(),
    );
    let particle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.83, 0.94, 0.94),
        perceptual_roughness: 0.3,
        ..default()
    });
    for (i, p) in fluid.pos.iter().enumerate() {
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(particle_material.clone()),
            Transform::from_translation(*p),
            Visibility::Hidden,
            ParticleIndex(i),
        ));
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Bevy injects resources as independent system parameters"
)]
fn sync_surface(
    fluid: Res<Fluid>,
    assets: Res<WaterAssets>,
    mut surface: ResMut<Surface>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut timing: ResMut<SurfaceTiming>,
    fixed: Res<Time<Fixed>>,
    paused: Res<crate::Paused>,
    capture: Option<Res<crate::Capture>>,
) {
    let frozen = paused.0
        || capture
            .as_ref()
            .is_some_and(|c| fluid.elapsed + 1e-5 >= c.at);
    if !fluid.is_changed() && frozen {
        return;
    }
    let start = std::time::Instant::now();
    if let Some(mut mesh) = meshes.get_mut(&assets.mesh) {
        *mesh = surface.rebuild_interpolated(
            &fluid,
            if frozen {
                1.0
            } else {
                fixed.overstep_fraction()
            },
        );
    }
    if let Some(mut image) = images.get_mut(&assets.density) {
        image.data = Some(surface.texture_bytes());
    }
    if let Some(mut material) = materials.get_mut(&assets.material) {
        material.clock_floor.x = fluid.elapsed;
    }
    crate::Timings::feed(&mut timing.0, start.elapsed().as_secs_f32() * 1000.0);
}

fn toggle_diagnostics(
    keys: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<DiagnosticView>,
    mut parts: Query<(&ScenePart, &mut Visibility)>,
) {
    let toggle_water = keys.just_pressed(KeyCode::KeyP);
    let toggle_bounds = keys.just_pressed(KeyCode::KeyB);
    if toggle_water {
        mode.0 = !mode.0;
    }
    for (part, mut visibility) in &mut parts {
        match part {
            ScenePart::Water if toggle_water => {
                *visibility = if mode.0 {
                    Visibility::Hidden
                } else {
                    Visibility::Visible
                };
            }
            ScenePart::Bounds if toggle_bounds => {
                *visibility = if *visibility == Visibility::Hidden {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
            _ => {}
        }
    }
}

fn sync_particles(
    fluid: Res<Fluid>,
    mode: Res<DiagnosticView>,
    config: Res<Config>,
    fixed: Res<Time<Fixed>>,
    paused: Res<crate::Paused>,
    capture: Option<Res<crate::Capture>>,
    mut particles: Query<(&ParticleIndex, &mut Transform, &mut Visibility)>,
) {
    let frozen = paused.0
        || capture
            .as_ref()
            .is_some_and(|c| fluid.elapsed + 1e-5 >= c.at);
    if !fluid.is_changed() && !mode.is_changed() && frozen {
        return;
    }
    for (i, mut t, mut visibility) in &mut particles {
        if let Some(p) = fluid.pos.get(i.0) {
            let spray = fluid.spray[i.0];
            let visible = mode.0 || spray > 0.10;
            let wanted = if visible {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
            if *visibility != wanted {
                *visibility = wanted;
            }
            if visible {
                t.translation = fluid.previous_pos[i.0].lerp(
                    *p,
                    if frozen {
                        1.0
                    } else {
                        fixed.overstep_fraction()
                    },
                );
                if mode.0 {
                    t.scale = Vec3::ONE;
                    t.rotation = Quat::IDENTITY;
                } else {
                    let size = (0.3 + spray * 0.45) * config.render.particle_scale / 2.4;
                    let stretch = (fluid.vel[i.0].length() / 300.0).clamp(1.0, 2.8);
                    t.scale = Vec3::new(size, size * stretch, size);
                    t.rotation =
                        Quat::from_rotation_arc(Vec3::Y, fluid.vel[i.0].normalize_or_zero());
                }
            }
        }
    }
}

fn reef_mesh(fluid: &Fluid) -> Mesh {
    let b = fluid.params.bounds;
    let mut positions = Vec::new();
    let mut colors = Vec::new();
    let n = 48;
    for z in 0..n {
        for x in 0..n {
            for (dx, dz) in [(0, 0), (0, 1), (1, 0), (1, 0), (0, 1), (1, 1)] {
                let mut p = Vec3::new(
                    b.min.x + (x + dx) as f32 / n as f32 * b.size().x,
                    b.min.y,
                    b.min.z + (z + dz) as f32 / n as f32 * b.size().z,
                );
                p.y = fluid.wave.map_or(b.min.y, |w| w.floor(p, b)) - 0.3;
                positions.push(p.to_array());
                let mottling = 0.78 + 0.15 * (p.x * 0.09).sin() * (p.z * 0.11).cos();
                colors.push([mottling, mottling, mottling * 0.9, 1.0]);
            }
        }
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.compute_flat_normals();
    mesh
}

fn box_wireframe(min: Vec3, max: Vec3) -> Mesh {
    let corner = |i: usize| {
        Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        )
    };
    let mut points = Vec::with_capacity(24);
    for a in 0..8 {
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
