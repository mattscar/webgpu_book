const shaderCode = `

struct DataStruct {
    @builtin(position) pos: vec4f,
    @location(0) colors: vec3f,
}

@vertex
fn vertexMain(@location(0) coords: vec2f, @location(1) colors: vec3f) -> DataStruct {
    var outData: DataStruct;
    outData.pos = vec4f(coords, 0.0, 1.0);
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
const canvas = document.getElementById("canvas_example") as HTMLCanvasElement;
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

// Define vertex data (coordinates and colors)
const vertexData = new Float32Array([
    0.0, 0.5, 0.0, 1.0, 0.0,    // First vertex
    -0.5, -0.5, 1.0, 0.0, 0.0,  // Second vertex
    0.5, -0.5, 0.0, 0.0, 1.0    // Third vertex
]);

// Create vertex buffer
const vertexBuffer = device.createBuffer({
    label: "Example vertex buffer",
    size: vertexData.byteLength,
    usage: 
        GPUBufferUsage.VERTEX | 
        GPUBufferUsage.COPY_DST
});

// Write data to buffer
device.queue.writeBuffer(vertexBuffer, 0, vertexData);
renderPass.setVertexBuffer(0, vertexBuffer);

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
        buffers: [{
            arrayStride: 20,
            attributes: [
                { format: "float32x2", offset: 0, shaderLocation: 0 }, 
                { format: "float32x3", offset: 8, shaderLocation: 1 }
            ]      
        }]
    },
    fragment: {
        module: shaderModule,
        entryPoint: "fragmentMain",
        targets: [{
            format: canvasFormat
        }]
    }    
});
renderPass.setPipeline(renderPipeline);

// Draw vertices and complete rendering
renderPass.draw(3);
renderPass.end();

// Submit the render commands to the GPU
device.queue.submit([encoder.finish()]);
}

// Run example function
runExample();