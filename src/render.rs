//! Continuous, volumetrically shaded water with a particle diagnostic view.
//!
//! With the CPU solver, the surface is rebuilt on the CPU and uploaded as a
//! mesh and a density texture every frame, and spray is an entity per particle.
//! With the GPU solver, the surface, its texture and the spray are written in
//! place on the GPU during extraction (see [`rebuild_on_gpu`]); their meshes
//! only cross to the GPU empty, when they are made or outgrown.

use crate::{
    config::Config,
    sim::Fluid,
    sim_gpu::GpuFluid,
    spray_gpu::{DROPLET_VERTICES, GpuSpray, SprayStyle},
    surface::{Surface, SurfaceLayout},
    surface_gpu::{DENSITY_FORMAT, GpuSurface, SurfaceTarget},
};
use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    camera::visibility::NoFrustumCulling,
    image::ImageSampler,
    mesh::PrimitiveTopology,
    prelude::*,
    reflect::TypePath,
    render::{
        Extract, ExtractSchedule, RenderApp,
        mesh::allocator::{MeshAllocator, MeshAllocatorSettings},
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, BufferUsages, Extent3d, TextureDimension, TextureFormat, TextureUsages,
        },
        texture::GpuImage,
    },
    shader::ShaderRef,
};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
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

/// What the last GPU surface rebuild needed and cost, shared between the
/// render world that rebuilds it and the main world that sizes the mesh.
#[derive(Resource, Clone, Default)]
struct SurfaceStatus(Arc<SurfaceCounters>);

#[derive(Default)]
struct SurfaceCounters {
    /// Triangles the surface needed, which can exceed what the mesh holds.
    triangles: AtomicU32,
    /// Droplets the spray needed, likewise.
    droplets: AtomicU32,
    milliseconds: AtomicU32,
    /// Bumped per rebuild, so the main world only reads fresh figures.
    rebuilds: AtomicU32,
}

/// Triangles the GPU-written water mesh has room for.
#[derive(Resource)]
struct WaterCapacity(u32);

/// The GPU-written spray mesh, and the droplets it has room for.
#[derive(Resource)]
struct SprayMesh {
    mesh: Handle<Mesh>,
    droplets: u32,
}

/// The most droplets the spray mesh grows to, about 75 MB of vertices. Beyond
/// it the particle diagnostic shows every few particles rather than all.
const MAX_DROPLETS: u32 = 1 << 17;

pub struct FluidRenderPlugin;
impl Plugin for FluidRenderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "water.wgsl");
        let bg = app.world().resource::<Config>().render.background;
        let status = SurfaceStatus::default();
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .insert_resource(status.clone())
                .add_systems(ExtractSchedule, rebuild_on_gpu);
        }
        app.insert_resource(status)
            .add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .insert_resource(ClearColor(Color::srgb(bg[0], bg[1], bg[2])))
            .insert_resource(GlobalAmbientLight {
                color: Color::srgb(0.72, 0.85, 1.0),
                brightness: 180.0,
                ..default()
            })
            .init_resource::<DiagnosticView>()
            .init_resource::<SurfaceTiming>()
            .add_systems(Startup, spawn_scene.after(crate::start_gpu_solver))
            .add_systems(
                Update,
                (
                    toggle_diagnostics,
                    sync_surface,
                    sync_particles,
                    grow_water_mesh,
                    grow_spray_mesh,
                )
                    .chain()
                    .after(crate::handle_keys),
            );
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            // Lets the GPU surface write straight into the water mesh's vertex
            // buffer, as Bevy's `compute_mesh` example does.
            render_app
                .world_mut()
                .resource_mut::<MeshAllocatorSettings>()
                .extra_buffer_usages |= BufferUsages::STORAGE;
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "Bevy injects resources as independent system parameters"
)]
fn spawn_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut water_materials: ResMut<Assets<WaterMaterial>>,
    mut images: ResMut<Assets<Image>>,
    fluid: Res<Fluid>,
    config: Res<Config>,
    gpu: Option<Res<GpuFluid>>,
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
    let first = surface.rebuild(&fluid);
    let [x, y, z] = surface.dims.map(|v| v as u32);
    let extent = Extent3d {
        width: x,
        height: y,
        depth_or_array_layers: z,
    };
    let (mesh, mut volume) = if gpu.is_some() {
        // Empty, and never touched again from here: the GPU fills both. The
        // first CPU rebuild only sizes the mesh.
        let capacity = (surface.triangles as u32 * 5 / 4).max(1 << 14);
        commands.insert_resource(WaterCapacity(capacity));
        let mut volume = Image::new_fill(
            extent,
            TextureDimension::D3,
            &[0; 4],
            DENSITY_FORMAT,
            RenderAssetUsages::RENDER_WORLD,
        );
        volume.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING;
        (meshes.add(empty_water_mesh(capacity)), volume)
    } else {
        let volume = Image::new(
            extent,
            TextureDimension::D3,
            surface.texture_bytes(),
            TextureFormat::R8Unorm,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        (meshes.add(first), volume)
    };
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
    // The GPU solver keeps its particles on the GPU, and draws them as one mesh
    // written there; entities here would read positions that never change.
    let cpu_particles = if gpu.is_some() {
        let droplets = 1 << 12;
        let mesh = meshes.add(empty_spray_mesh(droplets));
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(particle_material.clone()),
            NoFrustumCulling,
        ));
        commands.insert_resource(SprayMesh { mesh, droplets });
        0
    } else {
        fluid.len()
    };
    for (i, p) in fluid.pos.iter().take(cpu_particles).enumerate() {
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
    gpu: Option<Res<GpuFluid>>,
    status: Res<SurfaceStatus>,
    mut rebuilds: Local<u32>,
) {
    let frozen = paused.0 || capture.as_ref().is_some_and(|c| c.reached(fluid.elapsed));
    if !fluid.is_changed() && frozen {
        return;
    }
    if gpu.is_some() {
        // Rebuilt during extraction; only the clock and the timing live here.
        if let Some(mut material) = materials.get_mut(&assets.material) {
            material.clock_floor.x = fluid.elapsed;
        }
        let latest = status.0.rebuilds.load(Ordering::Relaxed);
        if latest != *rebuilds {
            *rebuilds = latest;
            let ms = f32::from_bits(status.0.milliseconds.load(Ordering::Relaxed));
            crate::Timings::feed(&mut timing.0, ms);
        }
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

/// A water mesh with room for `triangles`, all collapsed to the origin.
fn empty_water_mesh(triangles: u32) -> Mesh {
    let vertices = 3 * triangles as usize;
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; vertices])
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32; 3]; vertices])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0f32; 4]; vertices]);
    // `surface.wgsl` writes vertices in exactly this layout.
    assert_eq!(mesh.get_vertex_size(), crate::surface_gpu::VERTEX_BYTES);
    mesh
}

/// A spray mesh with room for `droplets`, all collapsed to the origin.
fn empty_spray_mesh(droplets: u32) -> Mesh {
    let vertices = (DROPLET_VERTICES * droplets) as usize;
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; vertices])
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32; 3]; vertices]);
    // `spray.wgsl` writes vertices in exactly this layout.
    assert_eq!(mesh.get_vertex_size(), crate::spray_gpu::VERTEX_BYTES);
    mesh
}

fn grow_spray_mesh(
    status: Res<SurfaceStatus>,
    spray: Option<ResMut<SprayMesh>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Some(mut spray) = spray else {
        return;
    };
    let needed = status.0.droplets.load(Ordering::Relaxed).min(MAX_DROPLETS);
    if needed > spray.droplets {
        spray.droplets = (needed + needed / 2).min(MAX_DROPLETS);
        let mesh = empty_spray_mesh(spray.droplets);
        meshes
            .insert(&spray.mesh, mesh)
            .expect("the spray mesh handle is live");
    }
}

/// Replaces the GPU-written water mesh with a larger one when the surface has
/// outgrown it. Until then the triangles past its end are not drawn.
fn grow_water_mesh(
    status: Res<SurfaceStatus>,
    capacity: Option<ResMut<WaterCapacity>>,
    assets: Res<WaterAssets>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Some(mut capacity) = capacity else {
        return;
    };
    let needed = status.0.triangles.load(Ordering::Relaxed);
    if needed > capacity.0 {
        capacity.0 = needed + needed / 2;
        info!("water surface needs {needed} triangles; growing its mesh to {}", capacity.0);
        meshes
            .insert(&assets.mesh, empty_water_mesh(capacity.0))
            .expect("the water mesh handle is live");
    }
}

/// What the last GPU rebuild was built from: fluid generation, clock, the point
/// between steps, both meshes' capacities, and the diagnostic view.
type RebuildKey = (u64, u32, u32, u32, u32, bool);

/// The GPU surface and spray, rebuilt during extraction. Extraction runs after
/// this frame's solver steps were submitted and before anything draws the
/// frame, and the render thread cannot start the next frame's extraction until
/// this frame is drawn, so what is drawn is always built from the steps before
/// it, whatever the solver does in the meantime.
#[allow(
    clippy::too_many_arguments,
    reason = "Bevy injects resources as independent system parameters"
)]
fn rebuild_on_gpu(
    fluid: Extract<Res<Fluid>>,
    solver: Extract<Option<Res<GpuFluid>>>,
    assets: Extract<Option<Res<WaterAssets>>>,
    spray_mesh: Extract<Option<Res<SprayMesh>>>,
    mode: Extract<Res<DiagnosticView>>,
    config: Extract<Res<Config>>,
    fixed: Extract<Res<Time<Fixed>>>,
    paused: Extract<Res<crate::Paused>>,
    capture: Extract<Option<Res<crate::Capture>>>,
    allocator: Res<MeshAllocator>,
    images: Res<RenderAssets<GpuImage>>,
    status: Res<SurfaceStatus>,
    mut builders: Local<Option<(GpuSurface, GpuSpray)>>,
    mut built: Local<Option<RebuildKey>>,
) {
    let (Some(solver), Some(assets), Some(spray_mesh)) =
        (solver.as_ref(), assets.as_ref(), spray_mesh.as_ref())
    else {
        return;
    };
    let (Some(water), Some(droplets), Some(image)) = (
        allocator.mesh_vertex_slice(&assets.mesh.id()),
        allocator.mesh_vertex_slice(&spray_mesh.mesh.id()),
        images.get(&assets.density),
    ) else {
        // Not allocated or uploaded yet: the first frame or two, or a mesh
        // that has just grown.
        return;
    };
    let frozen = paused.0 || capture.as_ref().is_some_and(|c| c.reached(fluid.elapsed));
    let alpha = if frozen { 1.0 } else { fixed.overstep_fraction() };
    let water_capacity = water.range.end - water.range.start;
    let droplet_capacity = (droplets.range.end - droplets.range.start) / DROPLET_VERTICES;
    let key = (
        fluid.generation,
        fluid.elapsed.to_bits(),
        alpha.to_bits(),
        water_capacity,
        droplet_capacity,
        mode.0,
    );
    if *built == Some(key) {
        return;
    }
    if builders
        .as_ref()
        .is_none_or(|(surface, spray)| !surface.matches(solver) || !spray.matches(solver))
    {
        let layout = SurfaceLayout::new(
            config.bounds(),
            config.fluid.spacing,
            config.render.surface_resolution,
        );
        *builders = Some((
            GpuSurface::new(solver, &layout, config.bounds()),
            GpuSpray::new(solver),
        ));
    }
    let (surface, spray) = builders.as_ref().unwrap();
    let start = std::time::Instant::now();
    let triangles = surface.rebuild(
        solver,
        alpha,
        &SurfaceTarget {
            vertices: water.buffer,
            start: water.range.start,
            capacity: water_capacity,
            density: &image.texture_view,
        },
    );
    let style = SprayStyle {
        radius: config.fluid.spacing * 0.3,
        scale: config.render.particle_scale / 2.4,
        diagnostic: mode
            .0
            .then(|| (solver.len() as u32).div_ceil(MAX_DROPLETS).max(1)),
    };
    let wanted = spray.rebuild(
        solver,
        alpha,
        style,
        droplets.buffer,
        droplets.range.start,
        droplet_capacity,
    );
    let counters = &status.0;
    counters.triangles.store(triangles, Ordering::Relaxed);
    counters.droplets.store(wanted, Ordering::Relaxed);
    counters
        .milliseconds
        .store((start.elapsed().as_secs_f32() * 1000.0).to_bits(), Ordering::Relaxed);
    counters.rebuilds.fetch_add(1, Ordering::Relaxed);
    *built = Some(key);
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
    let frozen = paused.0 || capture.as_ref().is_some_and(|c| c.reached(fluid.elapsed));
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
