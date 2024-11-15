const shaderCode = `

struct DataStruct {
    @builtin(position) pos: vec4f,
    @location(0) uvPos: vec2f,
}

@group(0) @binding(0) var sam : sampler;
@group(0) @binding(1) var tex : texture_2d<f32>;

@vertex
fn vertexMain(@location(0) coords: vec2f, @location(1) uvCoords: vec2f) -> DataStruct {
    var outData: DataStruct;
    outData.pos = vec4f(coords, 0.0, 1.0);
    outData.uvPos = uvCoords;
    return outData;
}

@fragment
fn fragmentMain(fragData: DataStruct) -> @location(0) vec4f {
    return textureSample(tex, sam, fragData.uvPos);
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

// Access the GPU
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

// Define vertex data (vertex coordinates and UV coordinates)
const vertexData = new Float32Array([
   -1.0,  1.0, 0.0, 0.0,   // First vertex
   -1.0, -1.0, 0.0, 1.0,   // Second vertex
    1.0,  1.0, 1.0, 0.0,   // Third vertex
    1.0, -1.0, 1.0, 1.0    // Fourth vertex
]);

// Create vertex buffer
const vertexBuffer = device.createBuffer({
    label: "Example vertex buffer",
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});

// Write data to buffer
device.queue.writeBuffer(vertexBuffer, 0, vertexData);

// Define layout of buffer data
const bufferLayout = {
    arrayStride: 16,
    attributes: [
       { format: "float32x2", offset: 0, shaderLocation: 0 },
       { format: "float32x2", offset: 8, shaderLocation: 1 }
    ],
};

// Create ImageBitmap from image file
const response = await fetch("smiley.png");
const imageBitmap = await createImageBitmap(await response.blob());

// Create texture object
const texture = device.createTexture({
    size: [imageBitmap.width, imageBitmap.height, 1],
    format: "rgba8unorm",
    usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT
});

// Write data to texture
device.queue.copyExternalImageToTexture(
    { source: imageBitmap },
    { texture: texture },
    [imageBitmap.width, imageBitmap.height]
);

// Create sampler
const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
});

// Create the shader module
const shaderModule = device.createShaderModule({
    label: "Example shader module",
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
        topology: "triangle-strip"
    }
});

// Access the bind group layout
const bindGroupLayout = renderPipeline.getBindGroupLayout(0);

// Create the bind group
let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: sampler
    },
    {
        binding: 1,
        resource: texture.createView({
            dimension: "2d",
        })
   }]
});

// Called just before the window is repainted
function newFrame(currentTime) {

    // Create the command encoder and the render pass encoder
    const encoder = device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.9, g: 0.9, b: 0.9, a: 1.0 },
            storeOp: "store"
        }]
    });

    // Set the vertex buffer and pipeline
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setPipeline(renderPipeline);

    // Associate bind group with render pass encoder
    renderPass.setBindGroup(0, bindGroup);

    // Draw vertices
    renderPass.draw(4);
    renderPass.end();

    // Submit the render commands to the GPU
    device.queue.submit([encoder.finish()]);
    window.requestAnimationFrame(newFrame);
}

window.requestAnimationFrame(newFrame);
}

// Run example function
runExample();