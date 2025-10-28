struct BrushParams {
  gridRadiusIntensity: vec4<f32>, // x: width, y: height, z: radius, w: intensity
  centerMode: vec4<f32> // x: centerX, y: centerY, z: mode, w: unused
};

@group(0) @binding(0)
var<storage, read_write> soilHeight: array<f32>;
@group(0) @binding(1)
var<uniform> params: BrushParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = i32(params.gridRadiusIntensity.x);
  let height = i32(params.gridRadiusIntensity.y);
  if (i32(gid.x) >= width || i32(gid.y) >= height) {
    return;
  }

  let cell = vec2<f32>(f32(gid.x), f32(gid.y));
  let center = params.centerMode.xy;
  let radius = params.gridRadiusIntensity.z;
  let dist = distance(cell, center);
  if (dist > radius) {
    return;
  }

  let idx = i32(gid.y) * width + i32(gid.x);
  let falloff = 1.0 - dist / radius;
  let intensity = params.gridRadiusIntensity.w;
  let mode = params.centerMode.z;
  let delta = intensity * falloff * select(-1.0, 1.0, mode > 0.0);
  soilHeight[idx] = max(0.0, soilHeight[idx] + delta);
}
