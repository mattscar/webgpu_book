const shaderCode = `

struct DataStruct {
    @builtin(position) pos: vec4f,
    @location(0) uvPos: vec2f,
}

@group(0) @binding(0) var sam : sampler;
@group(0) @binding(1) var tex : texture_external;

@vertex
fn vertexMain(@location(0) coords: vec2f, @location(1) uvCoords: vec2f) -> DataStruct {
    var outData: DataStruct;
    outData.pos = vec4f(coords, 0.0, 1.0);
    outData.uvPos = uvCoords;
    return outData;
}

@fragment
fn fragmentMain(fragData: DataStruct) -> @location(0) vec4f {
    
    // Read clamped texel value
    var texel = textureSampleBaseClampToEdge(tex, sam, fragData.uvPos);
    
    // Convert to grayscale
    var gray = 0.299 * texel.r + 0.587 * texel.g + 0.114 * texel.b; 

    // Return grayscale texel
    return vec4f(gray, gray, gray, 1.0);
}
`;

// Create top-level asynchronous function
async function runExample() {

// Create video element
const video = document.createElement('video');

// Configure video
video.loop = true;
video.autoplay = true;
video.muted = true;
video.src = "example.webm";

// Play video
await video.play();

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

// Create sampler
const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
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

function newframe() {

    // Access new frame
    const videoFrame = new VideoFrame(video);

    // Access the bind group layout
    const bindGroupLayout = renderPipeline.getBindGroupLayout(0);

    // Create the bind group
    let bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
        {
            binding: 0,
            resource: sampler
        },
        {
            binding: 1,
            resource: device.importExternalTexture({
                source: videoFrame,
            })
       }] 
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
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            storeOp: "store"
        }]
    });

    // Associate bind group with render pass encoder
    renderPass.setBindGroup(0, bindGroup);

    // Associate vertex buffer with render pass encoder
    renderPass.setVertexBuffer(0, vertexBuffer);

    // Associate render pipeline with render pass encoder
    renderPass.setPipeline(renderPipeline);

    // Draw vertices and complete rendering
    renderPass.draw(4);
    renderPass.end();

    // Submit the render commands to the GPU
    device.queue.submit([encoder.finish()]);

    // Close the VideoFrame
    videoFrame.close();

    // Set callback for the next frame
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
        video.requestVideoFrameCallback(newframe);
    } else {
        requestAnimationFrame(newframe);
    }
}

// Set callback for the next frame
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    video.requestVideoFrameCallback(newframe);
} else {
    requestAnimationFrame(newframe);
}

}

// Run example function
runExample();