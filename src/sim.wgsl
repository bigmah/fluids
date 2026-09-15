// The Position Based Fluids solver in `sim.rs`, ported pass for pass to compute
// shaders. The physics, and the reasons behind each step, are documented there;
// this file keeps its order and its arithmetic so the two can be checked against
// each other (see the tests in `sim_gpu.rs`).
//
// Every buffer is declared once, at a fixed binding. Each pass is its own
// pipeline whose layout names only the bindings that pass touches, which is what
// lets one buffer be an atomic counter in one pass and a plain array in the next.

const WORKGROUP: u32 = 256u;
// Dispatches wider than the per-dimension workgroup limit spill into y.
const ROW: u32 = 65535u * WORKGROUP;
const PI: f32 = 3.14159265358979;

// No cell holds anywhere near this many particles unless the solver has
// already blown up; the insertion sort refuses anything larger.
const MAX_CELL: u32 = 4096u;

const WAVE: u32 = 1u;
const SINGLE: u32 = 2u;
const CLAMP: u32 = 4u;
const MAKER: u32 = 8u;
const CHEBYSHEV: u32 = 16u;

// Mirrors `GpuParams` in `sim_gpu.rs`: the vec4s first, then scalars only, so the
// uniform layout and the Rust struct agree without implicit padding.
struct Params {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    bounds_size: vec4<f32>,
    wall_min: vec4<f32>,
    wall_max: vec4<f32>,
    gravity: vec4<f32>,
    grid_origin: vec4<f32>,
    direction: vec4<f32>,
    // The mouse. xyz centre (or ray origin), w radius.
    impulse: vec4<f32>,
    // xyz direction of the mouse ray.
    ray: vec4<f32>,
    n: u32,
    capacity: u32,
    dims_x: i32,
    dims_y: i32,
    dims_z: i32,
    cells: u32,
    flags: u32,
    tensile_n: i32,
    reduce_size: u32,
    dt: f32,
    time: f32,
    h: f32,
    h2: f32,
    poly6: f32,
    spiky: f32,
    inv_rho0: f32,
    epsilon: f32,
    tensile_scale: f32,
    tensile_w: f32,
    relax: f32,
    margin: f32,
    touch: f32,
    friction: f32,
    viscosity: f32,
    speed_scale: f32,
    interior_scale: f32,
    strength: f32,
    reef_height: f32,
    reef_start: f32,
    reef_width: f32,
    reef_skew: f32,
    origin_x: f32,
    amplitude: f32,
    omega: f32,
    wavenumber: f32,
    depth: f32,
    level: f32,
    generation_width: f32,
    period: f32,
    beach_start: f32,
    beach_width: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// One level of a parallel prefix sum or reduction: pairs `stride` apart, `count`
// of them. `top` is the first down-sweep stride, where the root is cleared.
struct Level {
    stride: u32,
    count: u32,
    top: u32,
    down: u32,
}

// One Jacobi iteration's Chebyshev weight; see `chebyshev_weights` in `sim.rs`.
struct Sweep {
    omega: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> level: Level;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> sweep: Sweep;
@group(0) @binding(4) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> predicted: array<vec4<f32>>;
// xyz: the positions λ was computed at; w: λ. A neighbour's position and
// multiplier are then one load, not two; see `solve_delta`.
@group(0) @binding(6) var<storage, read_write> lambdas: array<vec4<f32>>;
// xyz: gradient of the solid's kernel volume; w: the volume itself.
@group(0) @binding(8) var<storage, read_write> boundary: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> scratch: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> foam: array<f32>;
@group(0) @binding(11) var<storage, read_write> spray: array<f32>;
@group(0) @binding(12) var<storage, read_write> cells: array<u32>;
// `grid_count` and `grid_start` are the same buffer: counted atomically, then
// summed in place into start offsets.
@group(0) @binding(13) var<storage, read_write> grid_count: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> grid_start: array<u32>;
@group(0) @binding(15) var<storage, read_write> grid_cursor: array<atomic<u32>>;
@group(0) @binding(16) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(17) var<storage, read_write> neighbors: array<u32>;
@group(0) @binding(18) var<storage, read_write> neighbor_count: array<u32>;
@group(0) @binding(19) var<storage, read_write> max_neighbors: array<atomic<u32>>;
@group(0) @binding(20) var<storage, read_write> stats: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read_write> top: array<vec4<f32>>;
// The predictions one Jacobi iteration back, which Chebyshev extrapolates from.
@group(0) @binding(22) var<storage, read_write> earlier: array<vec4<f32>>;
// A per-particle buffer being renumbered into grid order: a vec4 one into
// `scratch`, or a scalar one, copied as bits, into `gathered`.
@group(0) @binding(23) var<storage, read_write> vectors: array<vec4<f32>>;
@group(0) @binding(24) var<storage, read_write> scalars: array<u32>;
@group(0) @binding(25) var<storage, read_write> gathered: array<u32>;

fn invocation(id: vec3<u32>) -> u32 {
    return id.x + id.y * ROW;
}

fn has(flag: u32) -> bool {
    return (params.flags & flag) != 0u;
}

// Rust's `f32::powi` with a runtime exponent, as compiler-rt computes it.
fn powi(base: f32, exponent: i32) -> f32 {
    var a = base;
    var b = exponent;
    var r = 1.0;
    loop {
        if (b & 1) != 0 {
            r *= a;
        }
        b /= 2;
        if b == 0 {
            break;
        }
        a *= a;
    }
    return select(r, 1.0 / r, exponent < 0);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

fn poly6(r2: f32) -> f32 {
    if r2 >= params.h2 {
        return 0.0;
    }
    let d = params.h2 - r2;
    return params.poly6 * d * d * d;
}

fn spiky_grad(r: vec3<f32>) -> vec3<f32> {
    let len = length(r);
    if len <= 1e-6 || len >= params.h {
        return vec3<f32>(0.0);
    }
    let d = params.h - len;
    return r * (params.spiky * d * d / len);
}

// ---------------------------------------------------------------------------
// Reef and wavemaker, from `wave.rs`
// ---------------------------------------------------------------------------

fn reef_t(p: vec3<f32>) -> f32 {
    let start = params.bounds_min.x + params.bounds_size.x * params.reef_start + params.reef_skew * p.z;
    return clamp((p.x - start) / (params.bounds_size.x * params.reef_width), 0.0, 1.0);
}

fn reef_floor(p: vec3<f32>) -> f32 {
    let t = reef_t(p);
    return params.bounds_min.y + params.reef_height * t * t * (3.0 - 2.0 * t);
}

fn reef_slope(p: vec3<f32>) -> f32 {
    let t = reef_t(p);
    return params.reef_height * 6.0 * t * (1.0 - t) / (params.bounds_size.x * params.reef_width);
}

fn reef_normal(p: vec3<f32>) -> vec3<f32> {
    let slope = reef_slope(p);
    return normalize(vec3<f32>(-slope, 1.0, slope * params.reef_skew));
}

fn project(point: vec3<f32>) -> vec3<f32> {
    var p = point;
    for (var k = 0; k < 8; k++) {
        let penetration = reef_floor(p) + params.margin - p.y;
        if penetration <= 1e-4 {
            break;
        }
        let slope = reef_slope(p);
        let normal = vec3<f32>(-slope, 1.0, slope * params.reef_skew);
        p += normal * (penetration / dot(normal, normal));
        p = clamp(p, params.wall_min.xyz, params.wall_max.xyz);
    }
    p.y = max(p.y, reef_floor(p) + params.margin);
    return p;
}

fn ramp(time: f32) -> f32 {
    let t = clamp((time - params.period) / (2.0 * params.period), 0.0, 1.0);
    return 0.5 - 0.5 * cos(PI * t);
}

fn orbital_velocity(p: vec3<f32>, time: f32) -> vec3<f32> {
    let k = params.wavenumber;
    let z = clamp(p.y - params.level, -params.depth, 0.0);
    let a = exp(k * z);
    let b = exp(-k * (z + 2.0 * params.depth));
    let denominator = 1.0 - exp(-2.0 * k * params.depth);
    let phase = k * dot(p - vec3<f32>(params.origin_x, 0.0, 0.0), params.direction.xyz)
        - params.omega * (time - params.period);
    let scale = params.amplitude * params.omega * ramp(time) / denominator;
    return params.direction.xyz * (scale * (a + b) * cos(phase))
        + vec3<f32>(0.0, 1.0, 0.0) * (scale * (a - b) * sin(phase));
}

fn generation_weight(p: vec3<f32>) -> f32 {
    if has(SINGLE) {
        return 0.0;
    }
    let q = clamp((p.x - params.origin_x) / params.generation_width, 0.0, 1.0);
    let s = sin(PI * q);
    return s * s;
}

fn drive(p: vec3<f32>, velocity: vec3<f32>, time: f32, dt: f32) -> vec3<f32> {
    let weight = generation_weight(p);
    if weight == 0.0 {
        return velocity;
    }
    let blend = 1.0 - exp(-12.0 * weight * dt / params.period);
    return velocity * (1.0 - blend) + orbital_velocity(p, time) * blend;
}

fn damping(p: vec3<f32>, dt: f32) -> f32 {
    let q = clamp((p.x - params.beach_start) / params.beach_width, 0.0, 1.0);
    return exp(-6.0 * q * q * dt / params.period);
}

// Kernel volume inside one solid half-space `distance` away, and its gradient.
fn half_space(distance: f32, normal: vec3<f32>) -> vec4<f32> {
    let q = clamp(distance / params.h, 0.0, 1.0);
    if q >= 1.0 {
        return vec4<f32>(0.0);
    }
    let q2 = q * q;
    let q3 = q2 * q;
    let q5 = q3 * q2;
    let q7 = q5 * q2;
    let q9 = q7 * q2;
    let integral = q - 4.0 * q3 / 3.0 + 6.0 * q5 / 5.0 - 4.0 * q7 / 7.0 + q9 / 9.0;
    let s = 1.0 - q2;
    let s2 = s * s;
    return vec4<f32>(
        -normal * (315.0 / 256.0 / params.h * (s2 * s2)),
        max(0.5 - 315.0 / 256.0 * integral, 0.0),
    );
}

fn solid_support(p: vec3<f32>) -> vec4<f32> {
    let normal = reef_normal(p);
    let lo = params.bounds_min.xyz;
    let hi = params.bounds_max.xyz;
    var out = vec4<f32>(0.0);
    out += half_space((p.y - reef_floor(p)) * normal.y, normal);
    out += half_space(p.x - lo.x, vec3<f32>(1.0, 0.0, 0.0));
    out += half_space(hi.x - p.x, vec3<f32>(-1.0, 0.0, 0.0));
    out += half_space(p.z - lo.z, vec3<f32>(0.0, 0.0, 1.0));
    out += half_space(hi.z - p.z, vec3<f32>(0.0, 0.0, -1.0));
    out += half_space(hi.y - p.y, vec3<f32>(0.0, -1.0, 0.0));
    return out;
}

// ---------------------------------------------------------------------------
// The mouse
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn apply_impulse(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let offset = positions[i].xyz - params.impulse.xyz;
    let d2 = dot(offset, offset);
    let r2 = params.impulse.w * params.impulse.w;
    if d2 < r2 && d2 > 1e-6 {
        let falloff = 1.0 - d2 / r2;
        velocities[i] += vec4<f32>(normalize(offset) * (params.strength * falloff), 0.0);
    }
}

// ---------------------------------------------------------------------------
// 1. External forces and prediction
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pos = positions[i].xyz;
    var v = velocities[i].xyz;
    if has(MAKER) {
        v = drive(pos, v, params.time, params.dt) * damping(pos, params.dt);
    }
    v += params.gravity.xyz * params.dt;
    var p = clamp(pos + v * params.dt, params.wall_min.xyz, params.wall_max.xyz);
    if has(WAVE) {
        p = project(p);
    }
    velocities[i] = vec4<f32>(v, 0.0);
    predicted[i] = vec4<f32>(p, 0.0);
}

// ---------------------------------------------------------------------------
// 2. Neighbourhoods: counting sort into a dense grid, then fixed-capacity lists
// ---------------------------------------------------------------------------

fn dims() -> vec3<i32> {
    return vec3<i32>(params.dims_x, params.dims_y, params.dims_z);
}

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    let local = (p - params.grid_origin.xyz) / params.grid_origin.w;
    return clamp(vec3<i32>(local), vec3<i32>(0), dims() - vec3<i32>(1));
}

fn cell_index(c: vec3<i32>) -> u32 {
    return u32((c.z * params.dims_y + c.y) * params.dims_x + c.x);
}

@compute @workgroup_size(256)
fn bin_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    cells[i] = cell_index(cell_of(predicted[i].xyz));
}

@compute @workgroup_size(256)
fn count_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    atomicAdd(&grid_count[cells[i]], 1u);
}

// Blelloch scan over a power-of-two buffer. The up-sweep leaves partial sums at
// the right of each pair; the down-sweep turns them into exclusive prefix sums,
// so a cell's entry becomes the number of particles in every cell before it.
@compute @workgroup_size(256)
fn prefix_sum(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    if k >= level.count {
        return;
    }
    let left = k * 2u * level.stride + level.stride - 1u;
    let right = left + level.stride;
    if level.down == 0u {
        grid_start[right] += grid_start[left];
    } else {
        let partial = grid_start[left];
        let root = select(grid_start[right], 0u, level.stride == level.top);
        grid_start[left] = root;
        grid_start[right] = root + partial;
    }
}

@compute @workgroup_size(256)
fn scatter_particles(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let slot = atomicAdd(&grid_cursor[cells[i]], 1u);
    sorted[slot] = i;
}

// The scatter lands particles in their cells in whatever order the threads ran.
// Sorting each cell restores ascending index, which makes the solver
// deterministic and gives neighbour lists the same order the CPU builds.
@compute @workgroup_size(256)
fn sort_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = invocation(id);
    if c >= params.cells {
        return;
    }
    let first = grid_start[c];
    let last = min(grid_start[c + 1u], params.n);
    // Bounded, so a corrupted grid costs a bad step rather than a GPU that
    // stays busy long after the process that submitted the work has gone.
    if first >= last || last - first > MAX_CELL {
        return;
    }
    for (var a = first + 1u; a < last; a++) {
        let value = sorted[a];
        var b = a;
        while b > first && sorted[b - 1u] > value {
            sorted[b] = sorted[b - 1u];
            b--;
        }
        sorted[b] = value;
    }
}

fn record_max_neighbors(count: u32) {
    var seen = atomicLoad(&max_neighbors[0]);
    loop {
        if count <= seen {
            break;
        }
        let swap = atomicCompareExchangeWeak(&max_neighbors[0], seen, count);
        if swap.exchanged {
            break;
        }
        seen = swap.old_value;
    }
}

@compute @workgroup_size(256)
fn find_neighbors(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pi = predicted[i].xyz;
    let c = cell_of(pi);
    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), dims() - vec3<i32>(1));
    let base = i * params.capacity;
    var count = 0u;
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            // The x run is contiguous in the sorted order, as on the CPU.
            let first = grid_start[cell_index(vec3<i32>(lo.x, y, z))];
            let last = min(grid_start[cell_index(vec3<i32>(hi.x, y, z)) + 1u], params.n);
            for (var k = first; k < last; k++) {
                let j = sorted[k];
                let r = pi - predicted[j].xyz;
                if j != i && dot(r, r) < params.h2 {
                    if count < params.capacity {
                        neighbors[base + count] = j;
                    }
                    count++;
                }
            }
        }
    }
    neighbor_count[i] = min(count, params.capacity);
    record_max_neighbors(count);
}

// Renumbering. The grid's cell order from the last substep is a permutation of
// the particles that puts neighbours next to each other.
@compute @workgroup_size(256)
fn gather_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    if k >= params.n {
        return;
    }
    scratch[k] = vectors[min(sorted[k], params.n - 1u)];
}

@compute @workgroup_size(256)
fn gather_scalars(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    if k >= params.n {
        return;
    }
    gathered[k] = scalars[min(sorted[k], params.n - 1u)];
}

// ---------------------------------------------------------------------------
// 3. Jacobi-solve the density constraint
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn solve_lambda(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pi = predicted[i].xyz;
    var support = vec4<f32>(0.0);
    if has(WAVE) {
        support = solid_support(pi);
    }
    var rho = poly6(0.0);
    var grad_self = support.xyz;
    var sum_sq = 0.0;
    let base = i * params.capacity;
    let count = neighbor_count[i];
    for (var m = 0u; m < count; m++) {
        let r = pi - predicted[neighbors[base + m]].xyz;
        rho += poly6(dot(r, r));
        let g = spiky_grad(r) * params.inv_rho0;
        grad_self += g;
        sum_sq += dot(g, g);
    }
    sum_sq += dot(grad_self, grad_self);
    let c = rho * params.inv_rho0 + support.w - 1.0;
    boundary[i] = support;
    var lambda = 0.0;
    if !(has(CLAMP) && c <= 0.0) {
        lambda = -c / (sum_sq + params.epsilon);
    }
    lambdas[i] = vec4<f32>(pi, lambda);
}

// The correction, and `sim.rs`'s third loop, which applies it, in one pass. It
// reads the positions `solve_lambda` copied out alongside each multiplier and
// writes `predicted`, so no invocation reads what another writes. The loop is
// bound by memory rather than arithmetic: reading a neighbour's multiplier from
// a buffer of its own cost more than the whole kernel evaluation.
@compute @workgroup_size(256)
fn solve_delta(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pi = lambdas[i].xyz;
    let li = lambdas[i].w;
    var d = vec3<f32>(0.0);
    let base = i * params.capacity;
    let count = neighbor_count[i];
    for (var m = 0u; m < count; m++) {
        let pj = lambdas[neighbors[base + m]];
        let r = pi - pj.xyz;
        // With artificial pressure off the term is -0, which changes no sum, so
        // the wave presets skip its kernel and power entirely.
        var s_corr = 0.0;
        if params.tensile_scale != 0.0 {
            let ratio = poly6(dot(r, r)) / params.tensile_w;
            s_corr = -params.tensile_scale * powi(ratio, params.tensile_n);
        }
        d += spiky_grad(r) * (li + pj.w + s_corr);
    }
    let delta = (d * params.inv_rho0 + boundary[i].xyz * li) * params.relax;
    var p = clamp(pi + delta, params.wall_min.xyz, params.wall_max.xyz);
    if has(WAVE) {
        p = project(p);
    }
    if has(CHEBYSHEV) {
        if sweep.omega != 1.0 {
            let e = earlier[i].xyz;
            p = clamp(e + (p - e) * sweep.omega, params.wall_min.xyz, params.wall_max.xyz);
            if has(WAVE) {
                p = project(p);
            }
        }
        earlier[i] = vec4<f32>(pi, 0.0);
    }
    predicted[i] = vec4<f32>(p, 0.0);
}

// ---------------------------------------------------------------------------
// 4. Velocity from the positions reached, and 5. XSPH viscosity
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn update_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let p = predicted[i].xyz;
    var v = (p - positions[i].xyz) * (1.0 / params.dt);
    positions[i] = vec4<f32>(p, 0.0);
    let lo = params.wall_min.xyz;
    let hi = params.wall_max.xyz;
    let f = params.friction;
    if p.x <= lo.x + params.touch || p.x >= hi.x - params.touch {
        v.y *= f;
        v.z *= f;
    }
    if p.y <= lo.y + params.touch || p.y >= hi.y - params.touch {
        v.x *= f;
        v.z *= f;
    }
    if p.z <= lo.z + params.touch || p.z >= hi.z - params.touch {
        v.x *= f;
        v.y *= f;
    }
    velocities[i] = vec4<f32>(v, 0.0);
}

@compute @workgroup_size(256)
fn apply_viscosity(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pi = positions[i].xyz;
    let vi = velocities[i].xyz;
    var dv = vec3<f32>(0.0);
    let base = i * params.capacity;
    for (var m = 0u; m < neighbor_count[i]; m++) {
        let j = neighbors[base + m];
        let r = pi - positions[j].xyz;
        dv += (velocities[j].xyz - vi) * poly6(dot(r, r));
    }
    scratch[i] = vec4<f32>(vi + dv * (params.viscosity * params.inv_rho0), 0.0);
}

// ---------------------------------------------------------------------------
// Foam and spray, once per step
// ---------------------------------------------------------------------------

@compute @workgroup_size(256)
fn update_foam(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.n {
        return;
    }
    let pi = positions[i].xyz;
    let vi = velocities[i].xyz;
    var count = 0.0;
    var disorder = 0.0;
    let base = i * params.capacity;
    for (var m = 0u; m < neighbor_count[i]; m++) {
        let j = neighbors[base + m];
        let r = pi - positions[j].xyz;
        if dot(r, r) < params.h2 {
            count += 1.0;
            let dv = vi - velocities[j].xyz;
            disorder += dot(dv, dv);
        }
    }
    let agitation = sqrt(disorder / max(count, 1.0)) / params.speed_scale;
    let exposed = clamp((28.0 - count) / 18.0, 0.0, 1.0);
    let source = clamp((agitation - 0.30) * 1.8, 0.0, 1.0) * exposed;
    foam[i] = clamp(foam[i] * exp(-params.dt / 1.8) + source * params.dt * 3.0, 0.0, 1.0);
    spray[i] = clamp((10.0 - count) / 10.0, 0.0, 1.0)
        * clamp(length(vi) / params.speed_scale - 0.6, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Readouts: per-particle values folded down to one by `reduce`
// ---------------------------------------------------------------------------

const LOWEST: f32 = -3.4e38;

@compute @workgroup_size(256)
fn measure_top(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.reduce_size {
        return;
    }
    if i < params.n {
        stats[i] = vec4<f32>(0.0, 0.0, 0.0, positions[i].y);
    } else {
        stats[i] = vec4<f32>(0.0, 0.0, 0.0, LOWEST);
    }
}

// xyz: compression, bulk density sample, bulk sample count (summed); w: speed (max).
@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.reduce_size {
        return;
    }
    if i >= params.n {
        stats[i] = vec4<f32>(0.0);
        return;
    }
    let pi = positions[i].xyz;
    var rho = poly6(0.0);
    let base = i * params.capacity;
    for (var m = 0u; m < neighbor_count[i]; m++) {
        let r = pi - positions[neighbors[base + m]].xyz;
        rho += poly6(dot(r, r));
    }
    let compression = max(rho * params.inv_rho0 + boundary[i].w - 1.0, 0.0);

    let h = params.h;
    let lo = params.bounds_min.xyz;
    let hi = params.bounds_max.xyz;
    var interior = pi.x > lo.x + h && pi.x < hi.x - h && pi.z > lo.z + h && pi.z < hi.z - h
        && pi.y > lo.y + h && pi.y < top[0].w - h;
    if has(WAVE) {
        interior = interior && (pi.y - reef_floor(pi)) * reef_normal(pi).y > h;
    }
    let sample = (f32(neighbor_count[i]) + 1.0) * params.interior_scale;
    stats[i] = vec4<f32>(
        compression,
        select(0.0, sample, interior),
        select(0.0, 1.0, interior),
        length(velocities[i].xyz),
    );
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    if k >= level.count {
        return;
    }
    let left = k * 2u * level.stride + level.stride - 1u;
    let right = left + level.stride;
    let a = stats[left];
    let b = stats[right];
    stats[right] = vec4<f32>(a.xyz + b.xyz, max(a.w, b.w));
}

// The frontmost particle within `impulse.w` of the mouse ray, as
// `Fluid::nearest_along_ray`: xyz position, w minus its distance along the ray.
@compute @workgroup_size(256)
fn ray_hits(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = invocation(id);
    if i >= params.reduce_size {
        return;
    }
    stats[i] = vec4<f32>(0.0, 0.0, 0.0, LOWEST);
    if i >= params.n {
        return;
    }
    let rel = positions[i].xyz - params.impulse.xyz;
    let along = dot(rel, params.ray.xyz);
    if along <= 0.0 {
        return;
    }
    if dot(rel, rel) - along * along < params.impulse.w * params.impulse.w {
        stats[i] = vec4<f32>(positions[i].xyz, -along);
    }
}

@compute @workgroup_size(256)
fn reduce_nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = invocation(id);
    if k >= level.count {
        return;
    }
    let left = k * 2u * level.stride + level.stride - 1u;
    let right = left + level.stride;
    if stats[left].w > stats[right].w {
        stats[right] = stats[left];
    }
}
