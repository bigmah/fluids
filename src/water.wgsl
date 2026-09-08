#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::view

struct Water {
    origin_cell: vec4<f32>,
    extent_level: vec4<f32>,
    deep: vec4<f32>,
    shallow: vec4<f32>,
    foam: vec4<f32>,
    clock_floor: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> water: Water;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var density: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var density_sampler: sampler;

fn field(p: vec3<f32>) -> f32 {
    let uv = (p - water.origin_cell.xyz + vec3<f32>(water.origin_cell.w * 0.5)) / water.extent_level.xyz;
    if any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0)) { return 0.0; }
    return textureSampleLevel(density, density_sampler, uv, 0.0).r;
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let elevation = pow(clamp(dir.y, 0.0, 1.0), 0.45);
    var color = mix(vec3<f32>(0.65, 0.79, 0.84), vec3<f32>(0.16, 0.36, 0.57), elevation);
    let cloud = smoothstep(0.32, 0.72, sin(dir.x * 8.0 + dir.z * 4.0) * sin(dir.z * 11.0 - dir.y * 8.0));
    color = mix(color, vec3<f32>(0.90, 0.91, 0.86), cloud * smoothstep(0.0, 0.25, dir.y) * 0.48);
    let sun = normalize(vec3<f32>(-0.5, 0.8, -0.3));
    color += vec3<f32>(1.0, 0.88, 0.66) * pow(max(dot(dir, sun), 0.0), 600.0) * 5.0;
    return mix(vec3<f32>(0.026, 0.10, 0.12), color, smoothstep(-0.15, 0.06, dir.y));
}

fn hash(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.1031);
    q += dot(q, q.yzx + vec3<f32>(33.33));
    return fract((q.x + q.y) * q.z);
}

fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash(i), hash(i + vec3<f32>(1.,0.,0.)),u.x),
                   mix(hash(i+vec3<f32>(0.,1.,0.)),hash(i+vec3<f32>(1.,1.,0.)),u.x),u.y),
               mix(mix(hash(i+vec3<f32>(0.,0.,1.)),hash(i+vec3<f32>(1.,0.,1.)),u.x),
                   mix(hash(i+vec3<f32>(0.,1.,1.)),hash(i+vec3<f32>(1.,1.,1.)),u.x),u.y),u.z);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.world_position.xyz;
    let v = normalize(view.world_position - p);
    let spacing = water.clock_floor.z;
    let time = water.clock_floor.x;
    var n = normalize(in.world_normal);
    let ripple = vec3<f32>(
        sin(p.x * 0.19 + p.z * 0.12 - time * 2.0) + 0.45 * sin(p.z * 0.49 + p.y * 0.21 + time),
        0.0,
        cos(p.z * 0.17 - p.x * 0.11 + time * 1.8) + 0.4 * sin(p.x * 0.42 - time));
    n = normalize(n + (ripple - n * dot(ripple,n)) * 0.022);
    let facing = clamp(dot(n,v), 0.0, 1.0);
    // Schlick Fresnel with the air/water index of refraction, 1.333.
    let fresnel = 0.0204 + 0.9796 * pow(1.0 - facing, 5.0);
    let refracted = refract(-v, n, 1.0 / 1.333);
    let step_length = spacing * 0.85;
    var thickness = 0.0;
    var exit_point = p;
    var previous_density = field(p - n * spacing * 0.25);
    // Trace the first continuous water volume along the refracted ray.
    for (var i = 1; i <= 28; i += 1) {
        let q = p - n * spacing * 0.25 + refracted * (f32(i) * step_length);
        let d = field(q);
        if d < 0.45 {
            let fraction = clamp((previous_density - 0.45) / max(previous_density - d, 0.0001), 0.0, 1.0);
            thickness += step_length * fraction;
            exit_point = mix(exit_point, q, fraction);
            break;
        }
        thickness += step_length;
        exit_point = q;
        previous_density = d;
    }
    let transmission = exp(-vec3<f32>(0.22, 0.055, 0.032) * thickness / spacing);
    let sun = normalize(vec3<f32>(-0.5, 0.8, -0.3));
    var occlusion = 0.0;
    for (var i = 1; i <= 10; i += 1) {
        let q = p + sun * (f32(i) * spacing * 1.4) + n * spacing * 0.3;
        occlusion += smoothstep(0.35, 0.65, field(q));
    }
    let sunlight = exp(-occlusion * 0.26);
    let bed_noise = 0.86 + 0.12 * sin(exit_point.x * 0.12) * cos(exit_point.z * 0.13);
    let bed = vec3<f32>(0.30, 0.29, 0.20) * bed_noise;
    let through = mix(bed, sky(refracted), smoothstep(0.02, 0.30, refracted.y));
    let scatter = mix(water.deep.rgb, water.shallow.rgb, sunlight * 0.62) * (0.4 + sunlight * 0.7);
    var color = through * transmission + scatter * (vec3<f32>(1.0) - transmission);
    color += water.shallow.rgb * pow(max(dot(-v, sun), 0.0), 4.0) * transmission.g * sunlight * 0.65;
    color = mix(color, sky(reflect(-v,n)) * (0.5 + 0.5 * sunlight), fresnel);
    let half_vector = normalize(v + sun);
    color += vec3<f32>(1.0, 0.93, 0.79) * pow(max(dot(n, half_vector), 0.0), 420.0) * sunlight * 0.9;
    let patches = noise(p * 0.14) * 0.65 + noise(p * 0.37) * 0.35;
    let grain = 0.80 + 0.20 * noise(p * 1.2);
    let foam_amount = in.color.r * 700.0 / water.clock_floor.w;
    let foam = smoothstep(0.18, 0.55, foam_amount + (patches - 0.5) * 0.25) * grain;
    color = mix(color, water.foam.rgb * (0.5 + 0.5 * sunlight), foam);
    return vec4<f32>(color, 1.0);
}
