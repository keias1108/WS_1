struct SimParams {
  gridSizeFlags: vec4<f32>,
  dtCellGravity: vec4<f32>,
  waterParams: vec4<f32>,
  soilParams: vec4<f32>,
  sourceParams: vec4<f32>,
  waterSource: vec4<f32>,
  soilSource: vec4<f32>
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>
};

@group(0) @binding(0)
var<storage, read> waterHeight: array<f32>;
@group(0) @binding(1)
var<storage, read> soilHeight: array<f32>;
@group(0) @binding(2)
var<storage, read> waterPath: array<f32>;
@group(0) @binding(3)
var<storage, read> soilPath: array<f32>;
@group(0) @binding(4)
var<uniform> params: SimParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex : u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0)
  );
  var vsOut : VertexOut;
  let pos = positions[vertexIndex];
  vsOut.position = vec4<f32>(pos, 0.0, 1.0);
  var uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
  uv.y = 1.0 - uv.y;
  vsOut.uv = uv;
  return vsOut;
}

fn clamp_i32(value : i32, min_value : i32, max_value : i32) -> i32 {
  return min(max(value, min_value), max_value);
}

fn readSoilHeight(x : i32, y : i32, width : i32, height : i32) -> f32 {
  let cx = clamp_i32(x, 0, width - 1);
  let cy = clamp_i32(y, 0, height - 1);
  return soilHeight[cy * width + cx];
}

@fragment
fn fsMain(input : VertexOut) -> @location(0) vec4<f32> {
  let width = i32(params.gridSizeFlags.x);
  let height = i32(params.gridSizeFlags.y);
  let fx = clamp_i32(i32(input.uv.x * params.gridSizeFlags.x), 0, width - 1);
  let fy = clamp_i32(i32(input.uv.y * params.gridSizeFlags.y), 0, height - 1);
  let idx = fy * width + fx;

  let soil = soilHeight[idx];
  let water = waterHeight[idx];

  let totalHeight = soil + water;

  let east = readSoilHeight(fx + 1, fy, width, height);
  let west = readSoilHeight(fx - 1, fy, width, height);
  let north = readSoilHeight(fx, fy - 1, width, height);
  let south = readSoilHeight(fx, fy + 1, width, height);

  let normal = normalize(vec3<f32>(
    (west - east) * 0.5,
    (north - south) * 0.5,
    1.5
  ));

  let lightDir = normalize(vec3<f32>(0.4, 0.5, 1.0));
  var soilColor = vec3<f32>(0.36, 0.28, 0.22) + soil * 0.08;
  soilColor = soilColor * (0.4 + 0.6 * max(dot(normal, lightDir), 0.0));

  let waterFactor = clamp(water * 4.0, 0.0, 1.0);
  let waterColor = vec3<f32>(0.1, 0.25, 0.6) * (0.5 + 0.5 * max(dot(normal, lightDir), 0.0)) + water * 0.2;

  var color = mix(soilColor, waterColor, waterFactor);

  if (params.gridSizeFlags.z > 0.5) {
    let flowHighlight = clamp(waterPath[idx] * 1.2, 0.0, 1.0);
    let sedimentHighlight = clamp(soilPath[idx] * 1.4, 0.0, 1.0);
    color = color + vec3<f32>(0.07, 0.18, 0.42) * flowHighlight;
    color = color + vec3<f32>(0.32, 0.21, 0.08) * sedimentHighlight;
  }

  color = mix(color, vec3<f32>(0.95, 0.98, 1.0), clamp(totalHeight * 0.01, 0.0, 0.15));

  return vec4<f32>(color, 1.0);
}
