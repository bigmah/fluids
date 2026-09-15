// Spray, and the particle diagnostic, written on the GPU as one mesh: a small
// smooth-shaded octahedron per visible particle, placed, sized and stretched
// along its velocity as `render.rs` places the CPU solver's spray spheres.

const WORKGROUP: u32 = 256u;
const ROW: u32 = 65535u * WORKGROUP;
// Vertices per droplet, and floats per vertex: position, normal.
const VERTICES: u32 = 24u;
const STRIDE: u32 = 6u;

// Mirrors `SprayParams` in `spray_gpu.rs`.
struct Params {
    n: u32,
    alpha: f32,
    // The droplet sphere's radius, and `render.particle_scale` over its reference.
    radius: f32,
    scale: f32,
    // Nonzero draws every `stride`th particle at rest size: the diagnostic view.
    all: u32,
    stride: u32,
    // Droplets the mesh has room for, and where its vertices start.
    capacity: u32,
    vertex_start: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> spray: array<f32>;
// 1 per visible particle, summed in place into each one's droplet index.
@group(0) @binding(5) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> vertices: array<f32>;
// Droplets the previous rebuild wrote.
@group(0) @binding(7) var<storage, read_write> drawn: array<u32>;

fn invocation(id: vec3<u32>) -> u32 {
    return id.x + id.y * ROW;
}

fn visible(i: u32) -> bool {
    if params.all != 0u {
        return i % params.stride == 0u;
    }
    return spray[i] > 0.10;
}

@compute @workgroup_size(256)
fn select_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    offsets[i] = select(0u, 1u, visible(i));
}

fn put_vertex(index: u32, p: vec3<f32>, n: vec3<f32>) {
    let base = (params.vertex_start + index) * STRIDE;
    vertices[base] = p.x;
    vertices[base + 1u] = p.y;
    vertices[base + 2u] = p.z;
    vertices[base + 3u] = n.x;
    vertices[base + 4u] = n.y;
    vertices[base + 5u] = n.z;
}

@compute @workgroup_size(256)
fn place_droplets(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n || !visible(i) {
        return;
    }
    let k = offsets[i];
    if k >= params.capacity {
        return;
    }
    let centre = previous[i].xyz * (1.0 - params.alpha) + positions[i].xyz * params.alpha;
    var size = 1.0;
    var stretch = 1.0;
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if params.all == 0u {
        let v = velocities[i].xyz;
        let speed = length(v);
        size = (0.3 + spray[i] * 0.45) * params.scale;
        stretch = clamp(speed / 300.0, 1.0, 2.8);
        if speed > 0.0 {
            up = v / speed;
        }
    }
    // Any frame with `up` as its y axis: an octahedron looks the same turned
    // a quarter about it.
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.y) > 0.9);
    let side = normalize(cross(helper, up));
    let front = cross(side, up);
    let axes = vec3<f32>(size, size * stretch, size) * params.radius;
    let at = 24u * k;
    var corner = 0u;
    for (var face = 0u; face < 8u; face++) {
        let signs = vec3<f32>(
            select(1.0, -1.0, (face & 1u) != 0u),
            select(1.0, -1.0, (face & 2u) != 0u),
            select(1.0, -1.0, (face & 4u) != 0u),
        );
        var local = array<vec3<f32>, 3>(
            vec3<f32>(signs.x, 0.0, 0.0),
            vec3<f32>(0.0, signs.y, 0.0),
            vec3<f32>(0.0, 0.0, signs.z),
        );
        // Flipping an odd number of axes turns the winding inside out.
        if signs.x * signs.y * signs.z < 0.0 {
            let swap = local[1];
            local[1] = local[2];
            local[2] = swap;
        }
        for (var m = 0u; m < 3u; m++) {
            let u = local[m];
            let offset = side * (u.x * axes.x) + up * (u.y * axes.y) + front * (u.z * axes.z);
            // The ellipsoid's normal at an axis tip, for rounded shading.
            let normal = normalize(side * (u.x / axes.x) + up * (u.y / axes.y) + front * (u.z / axes.z));
            put_vertex(at + corner, centre + offset, normal);
            corner++;
        }
    }
}

@compute @workgroup_size(256)
fn clear_droplets(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    let written = min(offsets[params.n], params.capacity);
    if k < written || k >= min(drawn[0], params.capacity) {
        return;
    }
    for (var v = 0u; v < VERTICES * STRIDE; v++) {
        vertices[(params.vertex_start + VERTICES * k) * STRIDE + v] = 0.0;
    }
}
