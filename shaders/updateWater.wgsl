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
var<storage, read> waterHeightSrc: array<f32>;
@group(0) @binding(1)
var<storage, read> soilHeightSrc: array<f32>;
@group(0) @binding(2)
var<storage, read> waterFluxSrc: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read_write> waterHeightDst: array<f32>;
@group(0) @binding(4)
var<storage, read_write> waterFluxDst: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read_write> waterVelocityDst: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> waterPath: array<f32>;
@group(0) @binding(7)
var<uniform> params: SimParams;

fn readTotalHeight(x : i32, y : i32, width : i32, height : i32, fallback : f32) -> f32 {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return fallback;
  }
  let idx = y * width + x;
  return soilHeightSrc[idx] + waterHeightSrc[idx];
}

fn readFlux(x : i32, y : i32, width : i32, height : i32) -> vec4<f32> {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  return waterFluxSrc[y * width + x];
}

fn sourceContribution(cell : vec2<f32>, source : vec2<f32>, radius : f32, strength : f32) -> f32 {
  let offset = cell - source;
  let distSq = dot(offset, offset);
  let radiusSq = radius * radius;
  if (distSq >= radiusSq) {
    return 0.0;
  }
  let falloff = 1.0 - sqrt(distSq) / radius;
  return strength * falloff;
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

  let dt = params.dtCellGravity.x;
  let cellSize = params.dtCellGravity.z;
  let gravity = params.dtCellGravity.w;

  let waterFloor = params.waterParams.z;
  let sourceStrength = params.waterParams.w;
  let damping = clamp(params.waterParams.y, 0.0, 0.95);
  let flowScale = params.waterParams.x * gravity / max(cellSize, 1e-4);

  let water = waterHeightSrc[idx];
  let soil = soilHeightSrc[idx];
  let totalHeight = water + soil;
  let fluxPrev = waterFluxSrc[idx];

  let totalNorth = readTotalHeight(x, y - 1, width, height, totalHeight);
  let totalEast = readTotalHeight(x + 1, y, width, height, totalHeight);
  let totalSouth = readTotalHeight(x, y + 1, width, height, totalHeight);
  let totalWest = readTotalHeight(x - 1, y, width, height, totalHeight);

  let diffNorth = max(0.0, totalHeight - totalNorth);
  let diffEast = max(0.0, totalHeight - totalEast);
  let diffSouth = max(0.0, totalHeight - totalSouth);
  let diffWest = max(0.0, totalHeight - totalWest);

  var fluxNorth = max(0.0, fluxPrev.x * (1.0 - damping) + dt * flowScale * diffNorth);
  var fluxEast = max(0.0, fluxPrev.y * (1.0 - damping) + dt * flowScale * diffEast);
  var fluxSouth = max(0.0, fluxPrev.z * (1.0 - damping) + dt * flowScale * diffSouth);
  var fluxWest = max(0.0, fluxPrev.w * (1.0 - damping) + dt * flowScale * diffWest);

  var totalOut = fluxNorth + fluxEast + fluxSouth + fluxWest;
  if (totalOut > 0.0) {
    let safeDt = max(dt, 1e-4);
    let capacity = max(0.0, water) / safeDt;
    let limiter = min(1.0, capacity / (totalOut + 1e-6));
    fluxNorth = fluxNorth * limiter;
    fluxEast = fluxEast * limiter;
    fluxSouth = fluxSouth * limiter;
    fluxWest = fluxWest * limiter;
    totalOut = totalOut * limiter;
  }

  let neighborNorth = readFlux(x, y - 1, width, height);
  let neighborEast = readFlux(x + 1, y, width, height);
  let neighborSouth = readFlux(x, y + 1, width, height);
  let neighborWest = readFlux(x - 1, y, width, height);

  let inflow = neighborNorth.z + neighborEast.w + neighborSouth.x + neighborWest.y;

  var newWater = water + dt * (inflow - totalOut);

  let grid = params.gridSizeFlags.xy;
  let sourceA = vec2<f32>(params.waterSource.x * (grid.x - 1.0), params.waterSource.y * (grid.y - 1.0));
  let sourceB = vec2<f32>(params.waterSource.z * (grid.x - 1.0), params.waterSource.w * (grid.y - 1.0));
  let cellPosition = vec2<f32>(f32(x), f32(y));
  let addedWater = sourceContribution(cellPosition, sourceA, 4.0, sourceStrength) +
    sourceContribution(cellPosition, sourceB, 3.5, sourceStrength * 0.65);
  newWater = newWater + dt * addedWater;

  if (newWater < waterFloor) {
    newWater = waterFloor;
    fluxNorth = 0.0;
    fluxEast = 0.0;
    fluxSouth = 0.0;
    fluxWest = 0.0;
  }

  waterHeightDst[idx] = newWater;
  waterFluxDst[idx] = vec4<f32>(fluxNorth, fluxEast, fluxSouth, fluxWest);

  let invCell = 1.0 / max(cellSize, 1e-4);
  let volume = max(newWater, 0.02);
  let velocity = vec2<f32>(
    (fluxEast - fluxWest) * invCell,
    (fluxSouth - fluxNorth) * invCell
  ) / volume;
  waterVelocityDst[idx] = velocity;

  let speed = length(velocity) * volume;
  let previousPath = waterPath[idx];
  let decay = 0.92;
  waterPath[idx] = mix(previousPath, speed, 1.0 - decay);
}
