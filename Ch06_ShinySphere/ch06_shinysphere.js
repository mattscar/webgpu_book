import { mat4, vec4 } from 'https://wgpu-matrix.org/dist/2.x/wgpu-matrix.module.js'; 

const shaderCode = `

/* Input to vertex shader */
struct InputData {
    mvpMatrix: mat4x4f,
    centerPos: vec4f,
    viewerPos: vec4f,
    lightPos: vec4f,    
    ambient: vec4f,
    diffuse: vec4f,
    specular: vec3f,
    shininess: f32
}

/* Access the uniform buffer */
@group(0) @binding(0) var<uniform> input: InputData;

/* Output to fragment shader */
struct OutputData {
    @builtin(position) pos: vec4f,
    @location(0) normalVec: vec4f,    
    @location(1) viewerVec: vec4f,
    @location(2) lightVec: vec4f,    
    @location(3) ambient: vec4f,
    @location(4) diffuse: vec4f,
    @location(5) specular: vec3f,
    @location(6) shininess: f32
}

@vertex
fn vertexMain(@location(0) coords: vec3f) -> OutputData {
    
    var outData: OutputData;
    
    /* Transform coordinates */
    outData.pos = input.mvpMatrix * vec4f(coords, 1.0);
        
    /* Compute normal vector */
    outData.normalVec = normalize(outData.pos - input.centerPos);

    /* Compute direction to viewer */
    outData.viewerVec = normalize(input.viewerPos - outData.pos);
    
    /* Compute direction to light source */
    outData.lightVec = normalize(input.lightPos - outData.pos);

    /* Set data for fragment shader */
    outData.ambient = input.ambient;
    outData.diffuse = input.diffuse;
    outData.specular = input.specular;
    outData.shininess = input.shininess;

    return outData;
}

@fragment
fn fragmentMain(fragData: OutputData) -> @location(0) vec4f {
    
    /* Set minimum and maximum vectors used in clamp */    
    let low_clamp = vec3f(0.0, 0.0, 0.0);
    let high_clamp = vec3f(1.0, 1.0, 1.0);    
    
    /* Step 1: Compute N . L */
    let n_dot_l = dot(fragData.normalVec.xyz, fragData.lightVec.xyz);
    
    /* Step 2: Compute H, the vector between L and V */
    let half_vector = normalize(fragData.lightVec.xyz + fragData.viewerVec.xyz);
    
    /* Step 3: Compute (N . H)^n' */
    var blinn = dot(fragData.normalVec.xyz, half_vector);
    blinn = clamp(blinn, 0.0, 1.0);
    blinn = pow(blinn, fragData.shininess);    
    
    /* Step 4: Compute sum of light components */
    var light_color = fragData.ambient.xyz + fragData.diffuse.xyz * n_dot_l + fragData.specular * blinn;
    light_color = clamp(light_color, low_clamp, high_clamp);
  
    /* Step 5: Blend light color and original color */
    let orig_color = vec3f(0.5, 0.6, 0.7);
    let color_sum = clamp((light_color + orig_color)/2.0, low_clamp, high_clamp);
    
    return vec4f(color_sum, 1.0);
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
        clearValue: { r: 0.4, g: 0.4, b: 0.4, a: 1.0 },
        storeOp: "store"
    }]
});

// Define vertex data (coordinates and colors)
const RAD = 1.5;
const NUM_LATITUDE = 16;
const NUM_LONGITUDE = 32;
const NUM_VERTICES = NUM_LATITUDE * NUM_LONGITUDE + 2;
const NUM_INDICES = NUM_LONGITUDE * (2 * NUM_LATITUDE + 3);
const THETA_CONVERSION = (2.0 * Math.PI)/NUM_LONGITUDE;
const PHI_CONVERSION = Math.PI/(NUM_LATITUDE + 1);

// Set coordinates of top and bottom points
const vData = new Float32Array(3 * NUM_VERTICES);
vData[0] = 0.0; vData[1] = 0.0; vData[2] = RAD;
vData[3] = 0.0; vData[4] = 0.0; vData[5] = -1.0 * RAD;

// Create data arrays
const iData = new Uint16Array(NUM_INDICES);

// Iterate through slices
let ptIndex = 0; let vertIndex = 2;
let theta = 0.0; let phi = 0.0;
let rad_cos_theta = 0.0; let rad_sin_theta = 0.0;

for (let lon = 0; lon < NUM_LONGITUDE; lon++) {

    theta = lon * THETA_CONVERSION;
    rad_sin_theta = RAD * Math.sin(theta);
    rad_cos_theta = RAD * Math.cos(theta);

    // Add top vertex
    iData[ptIndex++] = 0;

    for (let lat = 1; lat <= NUM_LATITUDE; lat++) {

        // Set index values
        iData[ptIndex++] = vertIndex;
        iData[ptIndex++] = (vertIndex + NUM_LATITUDE) % (NUM_VERTICES - 2);

        // Compute phi
        phi = Math.PI/2.0 - lat * PHI_CONVERSION;

        // Set vertex values
        vData[3 * vertIndex] = rad_cos_theta * Math.cos(phi);
        vData[3 * vertIndex + 1] = rad_sin_theta * Math.cos(phi);
        vData[3 * vertIndex++ + 2] = RAD * Math.sin(phi);
    }

    // Add bottom vertex
    iData[ptIndex++] = 1;
    
    // Add primitive restart
    if(lon != NUM_LONGITUDE - 1) {
        iData[ptIndex++] = 0xffff;
    }
}

// Create vertex buffer
const vertexBuffer = device.createBuffer({
    label: "Vertex Buffer 0",
    size: vData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(vertexBuffer, 0, vData);
renderPass.setVertexBuffer(0, vertexBuffer);

// Define layout of buffer data
const bufferLayout = {
    arrayStride: 12,
    attributes: [
       { format: "float32x3", offset: 0, shaderLocation: 0 }
    ],
};

// Create index buffer
const indexBuffer = device.createBuffer({
    label: "Index Buffer 0",
    size: iData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(indexBuffer, 0, iData);
renderPass.setIndexBuffer(indexBuffer, "uint16");

// Define transformation
const modelMat = mat4.translation([0.0, 0.0, -10.0]);
const projMat = mat4.perspective(30.0 * Math.PI / 180.0, 1.0, 5.0, 25.0);
const viewMat = mat4.identity();
const mvpMat = mat4.mul(mat4.mul(projMat, viewMat), modelMat);

// Set positions
const oldCenter = new Float32Array([0.0, 0.0, 0.0, 1.0]);
const centerPos = vec4.transformMat4(oldCenter, mvpMat);
const viewerPos = new Float32Array([0.0, 0.0, 0.0, 0.0]);
const lightPos = new Float32Array([5.0, 15.0, 9.0, 0.0]);

// Set light components
const ambient = new Float32Array([0.7, 0.7, 0.7, 0.0]);
const diffuse = new Float32Array([0.9, 0.9, 0.9, 0.0]);
const specular = new Float32Array([1.0, 1.0, 1.0]);
const shininess = new Float32Array([1.5]);

// Combine data into uniform buffer
const uniformData = Float32Array.of(...mvpMat, ...centerPos, ...viewerPos, ...lightPos, ...ambient, ...diffuse, ...specular, ...shininess);

// Create uniform buffer
const uniformBuffer = device.createBuffer({
    label: "Uniform Buffer 0",
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(uniformBuffer, 0, uniformData); 

// Create the shader module
const shaderModule = device.createShaderModule({
    label: "Shader Module 0",
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
renderPass.drawIndexed(NUM_INDICES);
renderPass.end();

// Submit the render commands to the GPU
device.queue.submit([encoder.finish()]);
}

// Run example function
runExample();