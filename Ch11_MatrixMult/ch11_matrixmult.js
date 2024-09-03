const multCode = `

override vectors_per_row: u32;
override group_size_x: u32;
override group_size_y: u32;

@binding(0) @group(0) var<storage, read_write> aMat : array<vec4f>;
@binding(1) @group(0) var<storage, read_write> bMat : array<vec4f>;
@binding(2) @group(0) var<storage, read_write> cMat : array<f32>;

@compute @workgroup_size(group_size_x, group_size_y)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {
    
    var sum = 0.0;
    let a_row = gid.x * vectors_per_row;
    let b_row = gid.y * vectors_per_row;
    
    // Multiply row of A by row of B^T
    for (var i: u32 = 0; i < vectors_per_row; i++) {
        sum += dot(aMat[a_row + i], bMat[b_row + i]);
    }   
    
    // Store the result to C
    cMat[a_row * 4 + gid.y] = sum;
}
`

const transposeCode = `

override block_dim: u32;
override group_size: u32;

@binding(0) @group(0) var<storage, read_write> aMat : array<vec4f>;
@binding(1) @group(0) var<storage, read_write> bMat : array<vec4f>;
@binding(2) @group(0) var<storage, read_write> cMat : array<vec4f>;

@compute @workgroup_size(group_size)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {

    /* Matrix variables */
    var src_mat: mat4x4f;
    var dst_mat: mat4x4f;
        
    /* Determine the row and column of the invocation's block */
    var tmp = block_dim;
    var col = gid.x;
    var row = u32(0);
    while (col >= tmp) {
        col -= tmp;
        tmp--;
        row++;
    }
    col += row;
    
    /* Read source block into source matrix */
    src_mat = mat4x4f(
        bMat[4 * row * block_dim + col], 
        bMat[(4 * row + 1) * block_dim + col],
        bMat[(4 * row + 2) * block_dim + col],
        bMat[(4 * row + 3) * block_dim + col]);

    /* Take the transpose of source matrix */
    src_mat = transpose(src_mat);

    /* Block on matrix diagonal */
    if (row == col) {
        bMat[4 * row * block_dim + col] = src_mat[0];
        bMat[(4 * row + 1) * block_dim + col] = src_mat[1];
        bMat[(4 * row + 2) * block_dim + col] = src_mat[2];
        bMat[(4 * row + 3) * block_dim + col] = src_mat[3];
    }
    /* Block off matrix diagonal */
    else {

        /* Read destination block into destination matrix */
        dst_mat = mat4x4f(
            bMat[4 * col * block_dim + row], 
            bMat[(4 * col + 1) * block_dim + row],
            bMat[(4 * col + 2) * block_dim + row],
            bMat[(4 * col + 3) * block_dim + row]);

        /* Take the transpose of source matrix */
        dst_mat = transpose(dst_mat);
    
        /* Write transposed destination matrix to source block */
        bMat[4 * row * block_dim + col] = dst_mat[0];
        bMat[(4 * row + 1) * block_dim + col] = dst_mat[1];
        bMat[(4 * row + 2) * block_dim + col] = dst_mat[2];
        bMat[(4 * row + 3) * block_dim + col] = dst_mat[3];
        
        /* Write transposed source matrix to destination block */
        bMat[4 * col * block_dim + row] = src_mat[0];
        bMat[(4 * col + 1) * block_dim + row] = src_mat[1];
        bMat[(4 * col + 2) * block_dim + row] = src_mat[2];
        bMat[(4 * col + 3) * block_dim + row] = src_mat[3];
    }
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

// Access the client's GPU
const device = await adapter.requestDevice();
if (!device) {
    throw new Error("Failed to create a GPUDevice");
}

// Set parameters
const matrixDim = 16;
const buffSize = matrixDim * matrixDim * 4;
const groupSize = [(matrixDim/4) * (matrixDim/4 + 1)]/2;

// Create buffers
const aBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});
const bBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});
const cBuffer = device.createBuffer({
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});

// Access the mapped data
const aRange = aBuffer.getMappedRange();
const bRange = bBuffer.getMappedRange();

// Set the content of the mapped data
const inputVals = new Array(matrixDim * matrixDim);
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        inputVals[i*matrixDim + j] = 1.0 * i * matrixDim + j;
    }
}
new Float32Array(aRange).set(inputVals);
new Float32Array(bRange).set(inputVals);

// Unmap buffer
aBuffer.unmap();
bBuffer.unmap();

// Create the command encoder
const transposeEncoder = device.createCommandEncoder();
if (!transposeEncoder) {
    throw new Error("Failed to create a GPUCommandEncoder");
}

// Create the shader module
const transposeModule = device.createShaderModule({
    label: "Shader module 0",
    code: transposeCode
});

// Create the compute pass encoder
const transposePass = transposeEncoder.beginComputePass({
    label: "Compute Pass 0"
});

// Create the bind group layout
const bindGroupLayout = device.createBindGroupLayout({
    entries: [{
        binding: 0, 
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
    }, {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
    }]
});

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
        resource: { buffer: cBuffer }
    }]
});
transposePass.setBindGroup(0, bindGroup);

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ bindGroupLayout ]
});

// Define the compute procedure
const transposePipeline = device.createComputePipeline({
    label: "Compute Pipeline 0",
    layout: pipelineLayout,
    compute: {
        module: transposeModule,
        entryPoint: "computeMain",
        constants: {
            block_dim: matrixDim/4,
            group_size: groupSize
        }
    }   
});
transposePass.setPipeline(transposePipeline);

// Encode compute commands
transposePass.dispatchWorkgroups(1);

// Complete encoding compute commands
transposePass.end();

// Submit the commands to the GPU
device.queue.submit([transposeEncoder.finish()]);

// Re-create the command encoder
const multEncoder = device.createCommandEncoder();
if (!multEncoder) {
    throw new Error("Failed to create a GPUCommandEncoder");
}

// Create the shader module for matrix multiplication
const multModule = device.createShaderModule({
    label: "Shader module 1",
    code: multCode
});

// Create the compute pass encoder
const multPass = multEncoder.beginComputePass({
    label: "Compute Pass 1"
});

// Define the compute procedure
const multPipeline = device.createComputePipeline({
    label: "Compute Pipeline 1",
    layout: pipelineLayout,
    compute: {
        module: multModule,
        entryPoint: "computeMain",
        constants: {
            vectors_per_row: matrixDim/4,
            group_size_x: matrixDim,
            group_size_y: matrixDim
        }
    }
});
multPass.setPipeline(multPipeline);
multPass.setBindGroup(0, bindGroup);

// Encode compute commands
multPass.dispatchWorkgroups(1, 1);

// Complete encoding compute commands
multPass.end();

// Create mappable buffer
const mappableBuffer = device.createBuffer({
  size: buffSize,
  usage: 
      GPUBufferUsage.COPY_DST | 
      GPUBufferUsage.MAP_READ
});

// Encode copy command
multEncoder.copyBufferToBuffer(cBuffer, 0, mappableBuffer, 0, buffSize);

// Submit the commands to the GPU
device.queue.submit([multEncoder.finish()]);

// Read data from compute buffer
await mappableBuffer.mapAsync(GPUMapMode.READ);
const procData = mappableBuffer.getMappedRange();
const floatData = new Float32Array(procData);

// Check multiplication result
let checkMat = true;
let sum = 0.0;
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        sum = 0.0;
        for(let k = 0; k < matrixDim; k++) {
            sum += inputVals[i*matrixDim + k] * inputVals[k*matrixDim + j];
        }
        if (floatData[i*matrixDim + j] != sum) {
            checkMat = false;
        }
    }
}

// Display output in page
const outputMsg = checkMat ? "Multiplication check passed" : "Multiplication check failed";
document.getElementById("output").innerHTML = outputMsg;

// Destroy the mapping
mappableBuffer.unmap();
}

// Run example function
runExample();