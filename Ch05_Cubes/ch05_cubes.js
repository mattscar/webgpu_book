import { mat4 } from 'https://wgpu-matrix.org/dist/2.x/wgpu-matrix.module.js'; 

const shaderCode = `
struct MVP_Matrices {
    modelMatrix : mat4x4f,
    viewMatrix : mat4x4f,
    projectionMatrix : mat4x4f
}
@group(0) @binding(0) var<uniform> mats: MVP_Matrices;

struct DataStruct {
    @builtin(position) pos: vec4f,
    @location(0) colors: vec3f
}

@vertex
fn vertexMain(@location(0) coords: vec3f, @location(1) colors: vec3f, @builtin(instance_index) instance: u32) -> DataStruct {
    var outData: DataStruct;

    /* Apply the model transformation */
    var world_coords = mats.modelMatrix * vec4f(coords, 1.0);
    
    /* Translate the second and third instances */
    world_coords.z = world_coords.z - f32(instance) * 5.0;
    
    /* Apply the view transformation */
    var eye_coords = mats.viewMatrix * world_coords;
    
    /* Apply the projection transformation */
    var clip_coords = mats.projectionMatrix * eye_coords;
    
    /* Create output structure */
    outData.pos = clip_coords;
    outData.colors = colors;
    return outData;
}

@fragment
fn fragmentMain(fragData: DataStruct) -> @location(0) vec4f {
    return vec4f(fragData.colors, 1.0);
}
`;

// Create top-level asynchronous function
async function runExample() {

// Check if WebGPU is supported
if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
}

// Access the GPUAdapter
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No GPUAdapter found");
}

// Access the client"s GPU
const device = await adapter.requestDevice();
if (!device) {
    throw new Error("Failed to create a GPUDevice");
}

// Access the canvas
const canvas = document.getElementById("canvas_example");
if (!canvas) {
    throw new Error("Could not access canvas in page");
}

// Obtain a WebGPU context for the canvas
const context = canvas.getContext("webgpu");
if (!context) {
    throw new Error("Could not obtain WebGPU context for canvas");
}

// Configure the context with the device and format
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

// Create the command encoder
const encoder = device.createCommandEncoder();
if (!encoder) {
    throw new Error("Failed to create a GPUCommandEncoder");
}

// Create the render pass encoder
const renderPass = encoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0.9, g: 0.9, b: 0.9, a: 1.0 },
        storeOp: "store"
    }]
});

// Define vertex data
const vertexData = new Float32Array([
    // Left face
    -1.0, -1.0, -1.0,  0.0,  0.0,  0.8,
    -1.0,  1.0, -1.0,  0.0,  0.0,  0.8,
    -1.0,  1.0,  1.0,  0.0,  0.0,  0.8,
    -1.0, -1.0,  1.0,  0.0,  0.0,  0.8,
    // Right face
     1.0, -1.0, -1.0,  0.8,  0.8,  0.0,
     1.0,  1.0, -1.0,  0.8,  0.8,  0.0,
     1.0,  1.0,  1.0,  0.8,  0.8,  0.0,
     1.0, -1.0,  1.0,  0.8,  0.8,  0.0,
    // Top face
    -1.0, -1.0, -1.0,  0.0,  0.8,  0.0,
     1.0, -1.0, -1.0,  0.0,  0.8,  0.0,
     1.0, -1.0,  1.0,  0.0,  0.8,  0.0,
    -1.0, -1.0,  1.0,  0.0,  0.8,  0.0,
    // Bottom face
    -1.0,  1.0, -1.0,  0.8,  0.0,  0.8,
     1.0,  1.0, -1.0,  0.8,  0.0,  0.8,
     1.0,  1.0,  1.0,  0.8,  0.0,  0.8,
    -1.0,  1.0,  1.0,  0.8,  0.0,  0.8,
    // Front face
    -1.0, -1.0,  1.0,  0.8,  0.0,  0.0,
     1.0, -1.0,  1.0,  0.8,  0.0,  0.0,
     1.0,  1.0,  1.0,  0.8,  0.0,  0.0,
    -1.0,  1.0,  1.0,  0.8,  0.0,  0.0,
    // Rear face
    -1.0, -1.0, -1.0,  0.0,  0.8,  0.8,
     1.0, -1.0, -1.0,  0.0,  0.8,  0.8,
     1.0,  1.0, -1.0,  0.0,  0.8,  0.8,
    -1.0,  1.0, -1.0,  0.0,  0.8,  0.8,
]);

// Create vertex buffer
const vertexBuffer = device.createBuffer({
    label: "Vertex Buffer 0",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
renderPass.setVertexBuffer(0, vertexBuffer);

// Define layout of vertex buffer
const bufferLayout = {
    arrayStride: 24,
    attributes: [
       { format: "float32x3", offset: 0, shaderLocation: 0 }, 
       { format: "float32x3", offset: 12, shaderLocation: 1 }
    ],
};

// Define index data
const indexData = new Uint16Array([
  0, 1, 3, 2, 0xffff,
  7, 6, 4, 5, 0xffff,
  11, 10, 8, 9, 0xffff,
  12, 13, 15, 14, 0xffff,
  19, 18, 16, 17, 0xffff,
  20, 21, 23, 22, 0xffff
]);

// Create index buffer
const indexBuffer = device.createBuffer({
    label: "Index Buffer 0",
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(indexBuffer, 0, indexData);
renderPass.setIndexBuffer(indexBuffer, "uint16");

// Define uniform data
const modelMat = mat4.translation([0.0, 0.0, -10.0]);
const viewMat = mat4.lookAt([12.0, 4.0, -3.0], [0.0, 0.0, -14.0], [0.0, 1.0, 0.0]);
const projMat = mat4.perspective(30.0 * Math.PI / 180.0, 1.33, 5.0, 25.0);
const uniformData = Float32Array.of(...modelMat, ...viewMat, ...projMat);

// Create uniform buffer
const uniformBuffer = device.createBuffer({
    label: "Uniform Buffer 0",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(uniformBuffer, 0, uniformData); 

// Create the shader module
const shaderModule = device.createShaderModule({
    label: "Shader module 0",
    code: shaderCode
});

// Define the rendering procedure
const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vertexMain",
      buffers: [bufferLayout]
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fragmentMain",
      targets: [{
        format: canvasFormat
      }]
    },
    primitive: {
        topology: "triangle-strip",
        stripIndexFormat: "uint16",
        frontFace: "cw",
        cullMode: "back"
    }
});
renderPass.setPipeline(renderPipeline);

// Access the bind group layout
const bindGroupLayout = renderPipeline.getBindGroupLayout(0);

// Create the bind group
let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: uniformBuffer }
    }]
});

// Associate bind group with render pass encoder
renderPass.setBindGroup(0, bindGroup);

// Draw vertices and complete rendering
renderPass.drawIndexed(30, 3);
renderPass.end();

// Submit the render commands to the GPU
device.queue.submit([encoder.finish()]);
}

// Run example function
runExample();