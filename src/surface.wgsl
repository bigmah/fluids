// The isosurface reconstruction in `surface.rs`, on the GPU: the particles the
// solver leaves on the GPU are sampled into a density grid, which fills the
// water shader's volume texture and is polygonised by marching tetrahedra
// straight into the water mesh's vertex buffer. Same kernel, same cubes, same
// tetrahedra, same triangles, in the same order as the CPU builds them.

const WORKGROUP: u32 = 256u;
const ROW: u32 = 65535u * WORKGROUP;
const ISO: f32 = 0.45;
// The water mesh's vertex layout: position, normal, colour.
const STRIDE: u32 = 10u;

// Mirrors `SurfaceParams` in `surface_gpu.rs`.
struct Params {
    // xyz: the first voxel's centre; w: voxel size.
    origin: vec4<f32>,
    // xyz: the particle grid's origin; w: its cell size, the kernel radius.
    grid_origin: vec4<f32>,
    n: u32,
    alpha: f32,
    radius2: f32,
    reference: f32,
    voxels_x: u32,
    voxels_y: u32,
    voxels_z: u32,
    voxel_count: u32,
    dims_x: i32,
    dims_y: i32,
    dims_z: i32,
    cube_count: u32,
    vertex_start: u32,
    vertex_capacity: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> foam: array<f32>;
// xyz: position at this frame's point between steps; w: foam.
@group(0) @binding(4) var<storage, read_write> interp: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> grid_start: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted: array<u32>;
// xyz: unnormalised normal; w: density.
@group(0) @binding(7) var<storage, read_write> samples: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> sample_foam: array<f32>;
@group(0) @binding(9) var density_texture: texture_storage_3d<rgba8unorm, write>;
// Triangles per cube, summed in place into each cube's first triangle.
@group(0) @binding(10) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(11) var<storage, read_write> vertices: array<f32>;
// How many triangles the previous rebuild wrote.
@group(0) @binding(12) var<storage, read_write> drawn: array<u32>;

fn invocation(id: vec3<u32>) -> u32 {
    return id.x + id.y * ROW;
}

@compute @workgroup_size(256)
fn interpolate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let p = previous[i].xyz * (1.0 - params.alpha) + positions[i].xyz * params.alpha;
    interp[i] = vec4<f32>(p, foam[i]);
}

// ---------------------------------------------------------------------------
// Density, normal and foam at every voxel
// ---------------------------------------------------------------------------

fn dims() -> vec3<i32> {
    return vec3<i32>(params.dims_x, params.dims_y, params.dims_z);
}

fn cell_index(c: vec3<i32>) -> u32 {
    return u32((c.z * params.dims_y + c.y) * params.dims_x + c.x);
}

@compute @workgroup_size(256)
fn sample_density(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = invocation(id);
    if v >= params.voxel_count {
        return;
    }
    let x = v % params.voxels_x;
    let y = (v / params.voxels_x) % params.voxels_y;
    let z = v / (params.voxels_x * params.voxels_y);
    let p = params.origin.xyz + vec3<f32>(f32(x), f32(y), f32(z)) * params.origin.w;

    let local = (p - params.grid_origin.xyz) / params.grid_origin.w;
    let c = clamp(vec3<i32>(local), vec3<i32>(0), dims() - vec3<i32>(1));
    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), dims() - vec3<i32>(1));
    var density = 0.0;
    var normal = vec3<f32>(0.0);
    var foam_sum = 0.0;
    for (var gz = lo.z; gz <= hi.z; gz++) {
        for (var gy = lo.y; gy <= hi.y; gy++) {
            let first = grid_start[cell_index(vec3<i32>(lo.x, gy, gz))];
            let last = min(grid_start[cell_index(vec3<i32>(hi.x, gy, gz)) + 1u], params.n);
            for (var k = first; k < last; k++) {
                let particle = interp[sorted[k]];
                let r = p - particle.xyz;
                let q = 1.0 - dot(r, r) / params.radius2;
                if q > 0.0 {
                    let w = q * q * q;
                    density += w;
                    normal += r * (q * q);
                    foam_sum += particle.w * w;
                }
            }
        }
    }
    let level = density / params.reference;
    samples[v] = vec4<f32>(normal, level);
    sample_foam[v] = foam_sum / max(density, 1e-6);
    textureStore(
        density_texture,
        vec3<i32>(i32(x), i32(y), i32(z)),
        vec4<f32>(clamp(level, 0.0, 1.0), 0.0, 0.0, 1.0),
    );
}

// ---------------------------------------------------------------------------
// Marching tetrahedra
// ---------------------------------------------------------------------------

// A consistent body diagonal makes neighbouring cube faces agree.
const TETS = array<array<u32, 4>, 6>(
    array<u32, 4>(0u, 1u, 3u, 7u),
    array<u32, 4>(0u, 3u, 2u, 7u),
    array<u32, 4>(0u, 2u, 6u, 7u),
    array<u32, 4>(0u, 6u, 4u, 7u),
    array<u32, 4>(0u, 4u, 5u, 7u),
    array<u32, 4>(0u, 5u, 1u, 7u),
);

fn cube_of(c: u32) -> vec3<u32> {
    let nx = params.voxels_x - 1u;
    let ny = params.voxels_y - 1u;
    return vec3<u32>(c % nx, (c / nx) % ny, c / (nx * ny));
}

fn corner_voxel(cube: vec3<u32>, corner: u32) -> u32 {
    let x = cube.x + (corner & 1u);
    let y = cube.y + ((corner >> 1u) & 1u);
    let z = cube.z + ((corner >> 2u) & 1u);
    return x + params.voxels_x * (y + params.voxels_y * z);
}

fn tet_triangles(inside: u32) -> u32 {
    switch inside {
        case 1u, 3u: {
            return 1u;
        }
        case 2u: {
            return 2u;
        }
        default: {
            return 0u;
        }
    }
}

@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = invocation(id);
    if c >= params.cube_count {
        return;
    }
    let cube = cube_of(c);
    var mask = 0u;
    for (var corner = 0u; corner < 8u; corner++) {
        if samples[corner_voxel(cube, corner)].w >= ISO {
            mask |= 1u << corner;
        }
    }
    if mask == 0u || mask == 255u {
        offsets[c] = 0u;
        return;
    }
    let tets = TETS;
    var count = 0u;
    for (var t = 0u; t < 6u; t++) {
        let tet = tets[t];
        var inside = 0u;
        for (var k = 0u; k < 4u; k++) {
            inside += (mask >> tet[k]) & 1u;
        }
        count += tet_triangles(inside);
    }
    offsets[c] = count;
}

struct Vertex {
    p: vec3<f32>,
    n: vec3<f32>,
    foam: f32,
}

fn put_vertex(index: u32, v: Vertex) {
    let base = (params.vertex_start + index) * STRIDE;
    let len = length(v.n);
    let n = select(vec3<f32>(0.0), v.n / len, len > 0.0);
    vertices[base] = v.p.x;
    vertices[base + 1u] = v.p.y;
    vertices[base + 2u] = v.p.z;
    vertices[base + 3u] = n.x;
    vertices[base + 4u] = n.y;
    vertices[base + 5u] = n.z;
    vertices[base + 6u] = v.foam;
    vertices[base + 7u] = 0.0;
    vertices[base + 8u] = 0.0;
    vertices[base + 9u] = 1.0;
}

// Triangle `t`, wound so its face agrees with the density gradient. The CPU
// drops a degenerate triangle; here its slot was already counted, so it is
// written collapsed to a point instead, which draws nothing.
fn put_triangle(t: u32, a: Vertex, b: Vertex, c: Vertex) {
    let face = cross(b.p - a.p, c.p - a.p);
    if dot(face, face) < 1e-10 {
        put_vertex(3u * t, a);
        put_vertex(3u * t + 1u, a);
        put_vertex(3u * t + 2u, a);
        return;
    }
    put_vertex(3u * t, a);
    if dot(face, a.n + b.n + c.n) < 0.0 {
        put_vertex(3u * t + 1u, c);
        put_vertex(3u * t + 2u, b);
    } else {
        put_vertex(3u * t + 1u, b);
        put_vertex(3u * t + 2u, c);
    }
}

@compute @workgroup_size(256)
fn emit(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = invocation(id);
    if c >= params.cube_count {
        return;
    }
    let first = offsets[c];
    let last = offsets[c + 1u];
    // Past the mesh's capacity: dropped until the mesh grows.
    if first == last || 3u * last > params.vertex_capacity {
        return;
    }
    let cube = cube_of(c);
    var p: array<vec3<f32>, 8>;
    var s: array<vec4<f32>, 8>;
    var f: array<f32, 8>;
    for (var corner = 0u; corner < 8u; corner++) {
        let voxel = corner_voxel(cube, corner);
        let at = vec3<u32>(
            cube.x + (corner & 1u),
            cube.y + ((corner >> 1u) & 1u),
            cube.z + ((corner >> 2u) & 1u),
        );
        p[corner] = params.origin.xyz + vec3<f32>(at) * params.origin.w;
        s[corner] = samples[voxel];
        f[corner] = sample_foam[voxel];
    }
    let tets = TETS;
    var t = first;
    for (var k = 0u; k < 6u; k++) {
        let tet = tets[k];
        var inside: array<u32, 4>;
        var outside: array<u32, 4>;
        var ni = 0u;
        var no = 0u;
        for (var m = 0u; m < 4u; m++) {
            let corner = tet[m];
            if s[corner].w >= ISO {
                inside[ni] = corner;
                ni++;
            } else {
                outside[no] = corner;
                no++;
            }
        }
        if ni == 1u {
            put_triangle(
                t,
                edge(p, s, f, inside[0], outside[0]),
                edge(p, s, f, inside[0], outside[1]),
                edge(p, s, f, inside[0], outside[2]),
            );
            t++;
        } else if ni == 3u {
            put_triangle(
                t,
                edge(p, s, f, outside[0], inside[0]),
                edge(p, s, f, outside[0], inside[1]),
                edge(p, s, f, outside[0], inside[2]),
            );
            t++;
        } else if ni == 2u {
            let a = edge(p, s, f, inside[0], outside[0]);
            let b = edge(p, s, f, inside[0], outside[1]);
            let cc = edge(p, s, f, inside[1], outside[0]);
            let d = edge(p, s, f, inside[1], outside[1]);
            put_triangle(t, a, b, cc);
            put_triangle(t + 1u, b, d, cc);
            t += 2u;
        }
    }
}

fn edge(
    p: array<vec3<f32>, 8>,
    s: array<vec4<f32>, 8>,
    f: array<f32, 8>,
    a: u32,
    b: u32,
) -> Vertex {
    let t = clamp((ISO - s[a].w) / (s[b].w - s[a].w), 0.0, 1.0);
    return Vertex(
        p[a] * (1.0 - t) + p[b] * t,
        s[a].xyz * (1.0 - t) + s[b].xyz * t,
        f[a] + (f[b] - f[a]) * t,
    );
}

// The mesh draws every vertex it has room for, so whatever the last rebuild
// wrote past this one's end is collapsed away.
@compute @workgroup_size(256)
fn clear_tail(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = invocation(id);
    let written = min(3u * offsets[params.cube_count], params.vertex_capacity);
    if v < written || v >= min(3u * drawn[0], params.vertex_capacity) {
        return;
    }
    let base = (params.vertex_start + v) * STRIDE;
    for (var k = 0u; k < STRIDE; k++) {
        vertices[base + k] = 0.0;
    }
}
