const GRID_WIDTH = 256;
const GRID_HEIGHT = 256;
const WORKGROUP_SIZE = 16;
const BASE_DT_WATER = 0.018;
const BASE_DT_SOIL = 0.045;
const CELL_SIZE = 1.0;
const GRAVITY = 9.8 * 0.35;

const shaderPaths = {
  water: "shaders/updateWater.wgsl",
  soil: "shaders/updateSoil.wgsl",
  render: "shaders/render.wgsl",
  brush: "shaders/brush.wgsl",
};

const ui = {
  canvas: document.getElementById("canvas"),
  toggle: document.getElementById("toggle"),
  timeScale: document.getElementById("timeScale"),
  timeValue: document.getElementById("timeScaleValue"),
  waterInput: document.getElementById("waterInput"),
  waterValue: document.getElementById("waterInputValue"),
  soilViscosity: document.getElementById("soilViscosity"),
  soilValue: document.getElementById("soilViscosityValue"),
  showPaths: document.getElementById("showPaths"),
  status: document.getElementById("status"),
};

let device;
let context;
let presentationFormat;

let pipelines = {};
let buffers = {};
let uniformData = new Float32Array(32);
let uniformBuffer;
let brushUniformData = new Float32Array(8);
let brushUniformBuffer;

let waterIndex = 0;
let soilIndex = 0;
let running = true;
let timeScale = parseFloat(ui.timeScale.value);
let waterInputStrength = parseFloat(ui.waterInput.value);
let soilViscosityFactor = parseFloat(ui.soilViscosity.value);
let showPaths = ui.showPaths.checked;

let frameTime = performance.now();

async function fetchShader(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`셰이더 로드 실패: ${path}`);
  }
  return response.text();
}

function createStorageBuffer(size, label) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
  });
}

function createUniformBuffer(size, label) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label,
  });
}

function initTerrain() {
  const cellCount = GRID_WIDTH * GRID_HEIGHT;
  const soil = new Float32Array(cellCount);
  const water = new Float32Array(cellCount);
  const waterVel = new Float32Array(cellCount * 2);
  const waterFlux = new Float32Array(cellCount * 4);
  const soilVel = new Float32Array(cellCount * 2);
  const zeroPath = new Float32Array(cellCount);

  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      const idx = y * GRID_WIDTH + x;
      const nx = x / GRID_WIDTH;
      const ny = y / GRID_HEIGHT;
      const ridge = 0.08 * Math.sin(nx * Math.PI * 3.4) + 0.05 * Math.cos(ny * Math.PI * 4.1);
      const slope = 0.25 * ny;
      const valley = 0.12 * Math.exp(-Math.pow((nx - 0.45) * 3.2, 2.0));
      soil[idx] = Math.max(0.02, 0.25 + slope + ridge - valley);
      water[idx] = 0.0;
      waterVel[idx * 2] = 0.0;
      waterVel[idx * 2 + 1] = 0.0;
      soilVel[idx * 2] = 0.0;
      soilVel[idx * 2 + 1] = 0.0;
      zeroPath[idx] = 0.0;
    }
  }

  const soilHeightBuffers = [createStorageBuffer(soil.byteLength, "soilHeightA"), createStorageBuffer(soil.byteLength, "soilHeightB")];
  const soilVelocityBuffers = [createStorageBuffer(soilVel.byteLength, "soilVelA"), createStorageBuffer(soilVel.byteLength, "soilVelB")];
  const waterHeightBuffers = [createStorageBuffer(water.byteLength, "waterHeightA"), createStorageBuffer(water.byteLength, "waterHeightB")];
  const waterFluxBuffers = [createStorageBuffer(waterFlux.byteLength, "waterFluxA"), createStorageBuffer(waterFlux.byteLength, "waterFluxB")];
  const waterVelocityBuffers = [createStorageBuffer(waterVel.byteLength, "waterVelA"), createStorageBuffer(waterVel.byteLength, "waterVelB")];
  const waterPathBuffer = createStorageBuffer(zeroPath.byteLength, "waterPath");
  const soilPathBuffer = createStorageBuffer(zeroPath.byteLength, "soilPath");

  device.queue.writeBuffer(soilHeightBuffers[0], 0, soil);
  device.queue.writeBuffer(soilHeightBuffers[1], 0, soil);
  device.queue.writeBuffer(soilVelocityBuffers[0], 0, soilVel);
  device.queue.writeBuffer(soilVelocityBuffers[1], 0, soilVel);
  device.queue.writeBuffer(waterHeightBuffers[0], 0, water);
  device.queue.writeBuffer(waterHeightBuffers[1], 0, water);
  device.queue.writeBuffer(waterFluxBuffers[0], 0, waterFlux);
  device.queue.writeBuffer(waterFluxBuffers[1], 0, waterFlux);
  device.queue.writeBuffer(waterVelocityBuffers[0], 0, waterVel);
  device.queue.writeBuffer(waterVelocityBuffers[1], 0, waterVel);
  device.queue.writeBuffer(waterPathBuffer, 0, zeroPath);
  device.queue.writeBuffer(soilPathBuffer, 0, zeroPath);

  buffers = {
    soilHeight: soilHeightBuffers,
    soilVelocity: soilVelocityBuffers,
    waterHeight: waterHeightBuffers,
    waterFlux: waterFluxBuffers,
    waterVelocity: waterVelocityBuffers,
    waterPath: waterPathBuffer,
    soilPath: soilPathBuffer,
  };
}

function updateUniformData(deltaSeconds) {
  const deltaFactor = Math.min(deltaSeconds * 60.0, 4.0);
  const dtWater = BASE_DT_WATER * timeScale * deltaFactor;
  const dtSoil = BASE_DT_SOIL * timeScale * deltaFactor;

  const fluxAcceleration = 55.0;
  const fluxDamping = 0.075 * deltaFactor;
  const waterHeightFloor = 0.00005;
  const waterSourceStrength = 0.18 * waterInputStrength;

  const soilAdvection = 0.16;
  const soilSlope = 0.09;
  const soilDamping = 0.05 * soilViscosityFactor;
  const soilDiffusion = 0.006 / (soilViscosityFactor + 0.25);

  const soilSourceStrength = 0.0065;
  const erosionRate = 0.02 * (1.0 + 0.4 * timeScale);
  const depositionRate = 0.01 * (1.0 + 0.2 * soilViscosityFactor);
  const erosionThreshold = 0.09;

  // gridSizeFlags
  uniformData[0] = GRID_WIDTH;
  uniformData[1] = GRID_HEIGHT;
  uniformData[2] = showPaths ? 1.0 : 0.0;
  uniformData[3] = 0.0;

  // dtCellGravity
  uniformData[4] = dtWater;
  uniformData[5] = dtSoil;
  uniformData[6] = CELL_SIZE;
  uniformData[7] = GRAVITY;

  // waterParams
  uniformData[8] = fluxAcceleration;
  uniformData[9] = fluxDamping;
  uniformData[10] = waterHeightFloor;
  uniformData[11] = waterSourceStrength;

  // soilParams
  uniformData[12] = soilAdvection;
  uniformData[13] = soilSlope;
  uniformData[14] = soilDamping;
  uniformData[15] = soilDiffusion;

  // sourceParams
  uniformData[16] = soilSourceStrength;
  uniformData[17] = erosionRate;
  uniformData[18] = depositionRate;
  uniformData[19] = erosionThreshold;

  // waterSource positions (normalized)
  uniformData[20] = 0.22;
  uniformData[21] = 0.08;
  uniformData[22] = 0.78;
  uniformData[23] = 0.12;

  // soilSource positions
  uniformData[24] = 0.5;
  uniformData[25] = 0.92;
  uniformData[26] = 0.55;
  uniformData[27] = 0.95;

  device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);
}

function bindGroupForWater(srcIndex, dstIndex, soilIndexForWater) {
  return device.createBindGroup({
    layout: pipelines.water.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.waterHeight[srcIndex] } },
      { binding: 1, resource: { buffer: buffers.soilHeight[soilIndexForWater] } },
      { binding: 2, resource: { buffer: buffers.waterFlux[srcIndex] } },
      { binding: 3, resource: { buffer: buffers.waterHeight[dstIndex] } },
      { binding: 4, resource: { buffer: buffers.waterFlux[dstIndex] } },
      { binding: 5, resource: { buffer: buffers.waterVelocity[dstIndex] } },
      { binding: 6, resource: { buffer: buffers.waterPath } },
      { binding: 7, resource: { buffer: uniformBuffer } },
    ],
  });
}

function bindGroupForSoil(srcIndex, dstIndex, waterIndexForSoil) {
  return device.createBindGroup({
    layout: pipelines.soil.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.soilHeight[srcIndex] } },
      { binding: 1, resource: { buffer: buffers.soilVelocity[srcIndex] } },
      { binding: 2, resource: { buffer: buffers.waterVelocity[waterIndexForSoil] } },
      { binding: 3, resource: { buffer: buffers.waterHeight[waterIndexForSoil] } },
      { binding: 4, resource: { buffer: buffers.soilHeight[dstIndex] } },
      { binding: 5, resource: { buffer: buffers.soilVelocity[dstIndex] } },
      { binding: 6, resource: { buffer: buffers.soilPath } },
      { binding: 7, resource: { buffer: uniformBuffer } },
    ],
  });
}

function bindGroupForRender() {
  return device.createBindGroup({
    layout: pipelines.render.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.waterHeight[waterIndex] } },
      { binding: 1, resource: { buffer: buffers.soilHeight[soilIndex] } },
      { binding: 2, resource: { buffer: buffers.waterPath } },
      { binding: 3, resource: { buffer: buffers.soilPath } },
      { binding: 4, resource: { buffer: uniformBuffer } },
    ],
  });
}

function dispatchBrush(centerX, centerY, radius, intensity, mode) {
  brushUniformData[0] = GRID_WIDTH;
  brushUniformData[1] = GRID_HEIGHT;
  brushUniformData[2] = radius;
  brushUniformData[3] = intensity;
  brushUniformData[4] = centerX;
  brushUniformData[5] = centerY;
  brushUniformData[6] = mode;
  brushUniformData[7] = 0.0;
  device.queue.writeBuffer(brushUniformBuffer, 0, brushUniformData.buffer);

  const workgroupsX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const workgroupsY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipelines.brush);
  const bindGroupA = device.createBindGroup({
    layout: pipelines.brush.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.soilHeight[soilIndex] } },
      { binding: 1, resource: { buffer: brushUniformBuffer } },
    ],
  });
  pass.setBindGroup(0, bindGroupA);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);

  const bindGroupB = device.createBindGroup({
    layout: pipelines.brush.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.soilHeight[1 - soilIndex] } },
      { binding: 1, resource: { buffer: brushUniformBuffer } },
    ],
  });
  pass.setBindGroup(0, bindGroupB);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

function setupInteraction() {
  ui.toggle.addEventListener("click", () => {
    running = !running;
    ui.toggle.textContent = running ? "일시정지" : "재생";
  });

  ui.timeScale.addEventListener("input", () => {
    timeScale = parseFloat(ui.timeScale.value);
    ui.timeValue.textContent = `${timeScale.toFixed(1)}x`;
  });

  ui.waterInput.addEventListener("input", () => {
    waterInputStrength = parseFloat(ui.waterInput.value);
    ui.waterValue.textContent = waterInputStrength.toFixed(1);
  });

  ui.soilViscosity.addEventListener("input", () => {
    soilViscosityFactor = parseFloat(ui.soilViscosity.value);
    ui.soilValue.textContent = soilViscosityFactor.toFixed(1);
  });

  ui.showPaths.addEventListener("change", () => {
    showPaths = ui.showPaths.checked;
  });

  let painting = false;
  let paintMode = 1.0;

  ui.canvas.addEventListener("pointerdown", (event) => {
    painting = true;
    paintMode = event.shiftKey ? -1.0 : 1.0;
    applyBrushEvent(event, paintMode);
    ui.canvas.setPointerCapture(event.pointerId);
  });

  ui.canvas.addEventListener("pointermove", (event) => {
    if (!painting) return;
    applyBrushEvent(event, paintMode);
  });

  ui.canvas.addEventListener("pointerup", (event) => {
    painting = false;
    ui.canvas.releasePointerCapture(event.pointerId);
  });

  ui.canvas.addEventListener("pointerleave", () => {
    painting = false;
  });
}

function applyBrushEvent(event, mode) {
  const rect = ui.canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * GRID_WIDTH;
  const y = ((event.clientY - rect.top) / rect.height) * GRID_HEIGHT;
  const clampX = Math.max(0, Math.min(GRID_WIDTH - 1, x));
  const clampY = Math.max(0, Math.min(GRID_HEIGHT - 1, y));
  const radius = 6.0;
  const intensity = 0.05;
  dispatchBrush(clampX, clampY, radius, intensity, mode);
}

function stepSimulation() {
  const waterSrc = waterIndex;
  const waterDst = 1 - waterSrc;
  const soilSrc = soilIndex;
  const soilDst = 1 - soilSrc;

  const workgroupsX = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
  const workgroupsY = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);

  const encoder = device.createCommandEncoder();

  const waterPass = encoder.beginComputePass();
  waterPass.setPipeline(pipelines.water);
  waterPass.setBindGroup(0, bindGroupForWater(waterSrc, waterDst, soilSrc));
  waterPass.dispatchWorkgroups(workgroupsX, workgroupsY);
  waterPass.end();

  const soilPass = encoder.beginComputePass();
  soilPass.setPipeline(pipelines.soil);
  soilPass.setBindGroup(0, bindGroupForSoil(soilSrc, soilDst, waterDst));
  soilPass.dispatchWorkgroups(workgroupsX, workgroupsY);
  soilPass.end();

  waterIndex = waterDst;
  soilIndex = soilDst;

  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.04, g: 0.06, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  };

  const renderPass = encoder.beginRenderPass(renderPassDescriptor);
  renderPass.setPipeline(pipelines.render);
  renderPass.setBindGroup(0, bindGroupForRender());
  renderPass.draw(6, 1, 0, 0);
  renderPass.end();

  device.queue.submit([encoder.finish()]);
}

function renderOnly() {
  const encoder = device.createCommandEncoder();
  const renderPassDescriptor = {
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0.04, g: 0.06, b: 0.1, a: 1.0 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  };

  const renderPass = encoder.beginRenderPass(renderPassDescriptor);
  renderPass.setPipeline(pipelines.render);
  renderPass.setBindGroup(0, bindGroupForRender());
  renderPass.draw(6, 1, 0, 0);
  renderPass.end();
  device.queue.submit([encoder.finish()]);
}

function loop(now) {
  const delta = Math.max(0.001, (now - frameTime) * 0.001);
  frameTime = now;
  if (running) {
    updateUniformData(delta);
    stepSimulation();
  } else {
    updateUniformData(1 / 60);
    renderOnly();
  }
  requestAnimationFrame(loop);
}

async function init() {
  if (!navigator.gpu) {
    ui.status.textContent = "이 브라우저는 WebGPU를 지원하지 않습니다. 크롬 최신 버전을 사용해주세요.";
    throw new Error("WebGPU unsupported");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    ui.status.textContent = "GPU 어댑터를 찾을 수 없습니다.";
    throw new Error("No adapter");
  }

  device = await adapter.requestDevice();
  context = ui.canvas.getContext("webgpu");
  presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  const [waterCode, soilCode, renderCode, brushCode] = await Promise.all([
    fetchShader(shaderPaths.water),
    fetchShader(shaderPaths.soil),
    fetchShader(shaderPaths.render),
    fetchShader(shaderPaths.brush),
  ]);

  pipelines.water = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: waterCode }),
      entryPoint: "main",
    },
  });

  pipelines.soil = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: soilCode }),
      entryPoint: "main",
    },
  });

  pipelines.brush = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: brushCode }),
      entryPoint: "main",
    },
  });

  pipelines.render = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: renderCode }),
      entryPoint: "vsMain",
    },
    fragment: {
      module: device.createShaderModule({ code: renderCode }),
      entryPoint: "fsMain",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list" },
  });

  uniformBuffer = createUniformBuffer(uniformData.byteLength, "simUniforms");
  brushUniformBuffer = createUniformBuffer(brushUniformData.byteLength, "brushUniform");

  initTerrain();
  updateUniformData(1 / 60);
  setupInteraction();
  ui.status.textContent = "시뮬레이션이 실행 중입니다.";
  requestAnimationFrame(loop);
}

init().catch((error) => {
  console.error(error);
  ui.status.textContent = `초기화 오류: ${error.message}`;
});
