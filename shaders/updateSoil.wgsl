struct SimParams {
  gridSizeFlags: vec4<f32>,
  dtCellGravity: vec4<f32>,
  waterParams: vec4<f32>,
  soilParams: vec4<f32>,
  sourceParams: vec4<f32>,
  waterSource: vec4<f32>,
  soilSource: vec4<f32>
};

@group(0) @binding(0)
var<storage, read> soilHeightSrc: array<f32>;
@group(0) @binding(1)
var<storage, read> soilVelocitySrc: array<vec2<f32>>;
@group(0) @binding(2)
var<storage, read> waterVelocity: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read> waterHeight: array<f32>;
@group(0) @binding(4)
var<storage, read_write> soilHeightDst: array<f32>;
@group(0) @binding(5)
var<storage, read_write> soilVelocityDst: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> soilPath: array<f32>;
@group(0) @binding(7)
var<uniform> params: SimParams;

fn clamp_i32(value : i32, min_value : i32, max_value : i32) -> i32 {
  return min(max(value, min_value), max_value);
}

fn readSoilHeight(x : i32, y : i32, width : i32, height : i32) -> f32 {
  let clampedX = clamp_i32(x, 0, width - 1);
  let clampedY = clamp_i32(y, 0, height - 1);
  return soilHeightSrc[clampedY * width + clampedX];
}

fn readSoilVelocity(x : i32, y : i32, width : i32, height : i32) -> vec2<f32> {
  let clampedX = clamp_i32(x, 0, width - 1);
  let clampedY = clamp_i32(y, 0, height - 1);
  return soilVelocitySrc[clampedY * width + clampedX];
}

fn readWaterDepth(x : i32, y : i32, width : i32, height : i32) -> f32 {
  let clampedX = clamp_i32(x, 0, width - 1);
  let clampedY = clamp_i32(y, 0, height - 1);
  return waterHeight[clampedY * width + clampedX];
}

fn sourceContribution(cell : vec2<i32>, source : vec2<i32>, strength : f32) -> f32 {
  let dist = vec2<f32>(f32(cell.x - source.x), f32(cell.y - source.y));
  let distSq = dot(dist, dist);
  if (distSq >= 9.0) {
    return 0.0;
  }
  return strength * (1.0 - distSq / 9.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let width = i32(params.gridSizeFlags.x);
  let height = i32(params.gridSizeFlags.y);
  if (i32(gid.x) >= width || i32(gid.y) >= height) {
    return;
  }

  let x = i32(gid.x);
  let y = i32(gid.y);
  let idx = y * width + x;

  let soilHeight = soilHeightSrc[idx];
  var soilVel = soilVelocitySrc[idx];
  let flowVelocity = waterVelocity[idx];
  let waterDepth = waterHeight[idx];

  let dtSoil = params.dtCellGravity.y;
  let invCell = 1.0 / params.dtCellGravity.z;

  let soilAdvection = params.soilParams.x;
  let soilSlope = params.soilParams.y;
  let soilDamping = params.soilParams.z;
  let soilDiffusion = params.soilParams.w;

  let soilSourceStrength = params.sourceParams.x;
  let erosionRate = params.sourceParams.y;
  let depositionRate = params.sourceParams.z;
  let erosionThreshold = params.sourceParams.w;

  let north = readSoilHeight(x, y - 1, width, height);
  let south = readSoilHeight(x, y + 1, width, height);
  let east = readSoilHeight(x + 1, y, width, height);
  let west = readSoilHeight(x - 1, y, width, height);

  let gradX = (east - west) * 0.5 * invCell;
  let gradY = (south - north) * 0.5 * invCell;

  let advectionScale = min(1.5, waterDepth * 4.0 + 0.25);
  soilVel = soilVel + dtSoil * ((soilAdvection * advectionScale) * flowVelocity - soilSlope * vec2<f32>(gradX, gradY));
  soilVel = soilVel * max(0.0, 1.0 - soilDamping);
  soilVel = clamp(soilVel, vec2<f32>(-0.8, -0.8), vec2<f32>(0.8, 0.8));

  let velLeft = readSoilVelocity(x - 1, y, width, height);
  let velDown = readSoilVelocity(x, y - 1, width, height);

  var divergence = (soilVel.x - velLeft.x) * invCell;
  divergence = divergence + (soilVel.y - velDown.y) * invCell;

  var newSoil = soilHeight - dtSoil * divergence;

  let neighborSum = east + west + north + south;
  newSoil = newSoil + soilDiffusion * (neighborSum - 4.0 * soilHeight);

  let flowMagnitude = length(flowVelocity) * (waterDepth + 0.02);
  let erosion = max(0.0, flowMagnitude - erosionThreshold) * erosionRate;
  let depositionBase = max(0.0, erosionThreshold - flowMagnitude);
  let deposition = depositionBase * depositionRate * max(0.0, 1.0 - waterDepth * 12.0);
  newSoil = max(0.0, newSoil - erosion + deposition);

  let sourcePosA = vec2<i32>(i32(params.soilSource.x * params.gridSizeFlags.x), i32(params.soilSource.y * params.gridSizeFlags.y));
  let sourcePosB = vec2<i32>(i32(params.soilSource.z * params.gridSizeFlags.x), i32(params.soilSource.w * params.gridSizeFlags.y));
  let cellPos = vec2<i32>(x, y);
  let sourceSum = sourceContribution(cellPos, sourcePosA, soilSourceStrength) +
    sourceContribution(cellPos, sourcePosB, soilSourceStrength * 0.5);
  newSoil = max(0.0, newSoil + dtSoil * sourceSum);

  let previousTrail = soilPath[idx];
  let trailBlend = 0.08;
  soilPath[idx] = mix(previousTrail, flowMagnitude, trailBlend);

  soilHeightDst[idx] = newSoil;
  soilVelocityDst[idx] = soilVel;
}
