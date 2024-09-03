const shaderCode = `

@binding(0) @group(0) var<storage, read> a : array<vec4u, 256>;
@binding(1) @group(0) var<storage, read> b : array<vec4u, 256>;
@binding(2) @group(0) var<storage, read_write> res : atomic<u32>;

override group_size: u32;

@compute @workgroup_size(group_size)
fn computeMain(@builtin(global_invocation_id) id : vec3<u32>) {

    // Compute dot product of vectors
    let prod = dot(a[id.x], b[id.x]);

    // Update result atomically
    atomicAdd(&res, prod);
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

// Check for timestamp support
const timeSupport = adapter.features.has("timestamp-query");

// Access the GPU
const device = timeSupport ?
    await adapter.requestDevice({ 
        requiredFeatures: ["timestamp-query"] }) :
    await adapter.requestDevice();
if (!device) {
    throw new Error("Failed to create a GPUDevice");
}

// Create the command encoder
const encoder = device.createCommandEncoder();
if (!encoder) {
    throw new Error("Failed to create a GPUCommandEncoder");
}

// Set the number of values
const numVals = 1024;

// Create the query set
const querySet = timeSupport ?
    device.createQuerySet({
        label: "Query Set",
        count: 2,
        type: "timestamp"
    }) : None;

// Create the query buffer
const queryBuffer = timeSupport ?
    device.createBuffer({
        size: querySet.count * BigInt64Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    }) : None

// Create compute buffers
const aBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: numVals * 4,
    usage:
        GPUBufferUsage.STORAGE
});
const aRange = aBuffer.getMappedRange();

// Create compute buffers
const bBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: numVals * 4,
    usage:
        GPUBufferUsage.STORAGE
});
const bRange = bBuffer.getMappedRange();

// Create compute buffers
const resBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const resRange = resBuffer.getMappedRange();

// Create the data arrays
const aArray = new Array(numVals);
const bArray = new Array(numVals);
const res = new Array([0]);

// Initialize vectors
for (let i = 0; i < numVals; i++) {
    aArray[i] = i + 1;
    bArray[i] = numVals - i;
}

// Create arrays in buffer memory
new Uint32Array(aRange).set(aArray);
new Uint32Array(bRange).set(bArray);
new Uint32Array(resRange).set(res);

// Unmap buffers
aBuffer.unmap();
bBuffer.unmap();
resBuffer.unmap();

// Create the shader module
const shaderModule = device.createShaderModule({
    label: "Shader module 0",
    code: shaderCode
});

// Create the compute pass encoder
const computePass = timeSupport ?
    encoder.beginComputePass({
        timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1
        }}) :
    encoder.beginComputePass({});

// Define the compute procedure
const computePipeline = device.createComputePipeline({
    label: "Compute Pipeline 0",
    layout: "auto",
    compute: {
        module: shaderModule,
        entryPoint: "computeMain",
        constants: {
            group_size: 256
        }
    }
});
computePass.setPipeline(computePipeline);

// Access the bind group layout
const bindGroupLayout = computePipeline.getBindGroupLayout(0);

// Create the bind group
let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: aBuffer }
    },
    {
        binding: 1,
        resource: { buffer: bBuffer }
    },
    {
        binding: 2,
        resource: { buffer: resBuffer }
    }]
});
computePass.setBindGroup(0, bindGroup);

// Encode compute commands
computePass.dispatchWorkgroups(1);

// Complete encoding compute commands
computePass.end();

// Create buffer to hold timestamp results
const tsBuffer = timeSupport ?
    device.createBuffer({
        size: querySet.count * BigInt64Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    }) : None;

if (timeSupport) {
    
    // Encode timestamp query command
    encoder.resolveQuerySet(querySet,
        0, querySet.count, queryBuffer, 0);
    
    // Encode command to copy timestamp data
    encoder.copyBufferToBuffer(queryBuffer, 0, tsBuffer, 0, 
        querySet.count * BigInt64Array.BYTES_PER_ELEMENT);
}

// Create mappable buffer for dot product
const mappableBuffer = device.createBuffer({
  size: 4,
  usage:
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.MAP_READ
});

// Encode copy command for dot product
encoder.copyBufferToBuffer(resBuffer, 0, mappableBuffer, 0, 4);

// Submit the commands to the GPU
device.queue.submit([encoder.finish()]);

// Read data from compute buffer
await mappableBuffer.mapAsync(GPUMapMode.READ);
const procData = mappableBuffer.getMappedRange();
const resData = new Uint32Array(procData);

// Display output in page
const outputMsg = "Dot product: ";
document.getElementById("result").innerHTML = outputMsg.concat(resData.toString());

// Destroy the mapping
mappableBuffer.unmap();

if (timeSupport) {
    
    // Read data from compute buffer
    await tsBuffer.mapAsync(GPUMapMode.READ);
    const mapData = tsBuffer.getMappedRange();
    const tsData = new BigInt64Array(mapData);
    
    // Display output in page
    const t1 = Number(tsData[0]) / 1000000.0;
    const t2 = Number(tsData[1]) / 1000000.0;    
    const t = t2 - t1;
    const tsMsg = "Time: ".concat(t2.toString()).concat(" - ").concat(t1.toString()).concat(" = ");
    document.getElementById("timestamp").innerHTML = tsMsg.concat(t.toString());
    
    // Destroy the mapping
    tsBuffer.unmap();
}
}

// Run example function
runExample();