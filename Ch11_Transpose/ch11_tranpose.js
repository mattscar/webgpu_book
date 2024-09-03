const shaderCode = `

override block_dim: u32;
override group_size: u32;

@binding(0) @group(0) var<storage, read_write> matrix : array<vec4f>;

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
        matrix[4 * row * block_dim + col], 
        matrix[(4 * row + 1) * block_dim + col],
        matrix[(4 * row + 2) * block_dim + col],
        matrix[(4 * row + 3) * block_dim + col]);

    /* Take the transpose of source matrix */
    src_mat = transpose(src_mat);

    /* Block on matrix diagonal */
    if (row == col) {
        matrix[4 * row * block_dim + col] = src_mat[0];
        matrix[(4 * row + 1) * block_dim + col] = src_mat[1];
        matrix[(4 * row + 2) * block_dim + col] = src_mat[2];
        matrix[(4 * row + 3) * block_dim + col] = src_mat[3];
    }
    /* Block off matrix diagonal */
    else {

        /* Read destination block into destination matrix */
        dst_mat = mat4x4f(
            matrix[4 * col * block_dim + row], 
            matrix[(4 * col + 1) * block_dim + row],
            matrix[(4 * col + 2) * block_dim + row],
            matrix[(4 * col + 3) * block_dim + row]);

        /* Take the transpose of source matrix */
        dst_mat = transpose(dst_mat);
    
        /* Write transposed destination matrix to source block */
        matrix[4 * row * block_dim + col] = dst_mat[0];
        matrix[(4 * row + 1) * block_dim + col] = dst_mat[1];
        matrix[(4 * row + 2) * block_dim + col] = dst_mat[2];
        matrix[(4 * row + 3) * block_dim + col] = dst_mat[3];
        
        /* Write transposed source matrix to destination block */
        matrix[4 * col * block_dim + row] = src_mat[0];
        matrix[(4 * col + 1) * block_dim + row] = src_mat[1];
        matrix[(4 * col + 2) * block_dim + row] = src_mat[2];
        matrix[(4 * col + 3) * block_dim + row] = src_mat[3];
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

// Create the command encoder
const encoder = device.createCommandEncoder();
if (!encoder) {
    throw new Error("Failed to create a GPUCommandEncoder");
}

// Create compute buffer
const matrixDim = 64;
const buffSize = matrixDim * matrixDim * 4;
const groupSize = [(matrixDim/4) * (matrixDim/4 + 1)]/2;

const computeBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});

// Access the mapped data
const buff = computeBuffer.getMappedRange();

// Set the content of the mapped data
const inputVals = new Array(matrixDim * matrixDim);
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        inputVals[i*matrixDim + j] = 1.0 * i * matrixDim + j;
    }
}
new Float32Array(buff).set(inputVals);

// Unmap buffer
computeBuffer.unmap();

// Create the shader module
const shaderModule = device.createShaderModule({
    label: "Shader module 0",
    code: shaderCode
});

// Create the compute pass encoder
const computePass = encoder.beginComputePass({
    label: "Compute Pass 0"
});

// Define the compute procedure
const computePipeline = device.createComputePipeline({
    label: "Compute Pipeline 0",
    layout: "auto",
    compute: {
        module: shaderModule,
        entryPoint: "computeMain",
        constants: {
            block_dim: matrixDim/4,
            group_size: groupSize
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
        resource: { buffer: computeBuffer }
    }]
});
computePass.setBindGroup(0, bindGroup);

// Encode compute commands
computePass.dispatchWorkgroups(1);

// Complete encoding compute commands
computePass.end();

// Create mappable buffer
const mappableBuffer = device.createBuffer({
  size: buffSize,
  usage: 
      GPUBufferUsage.COPY_DST | 
      GPUBufferUsage.MAP_READ
});

// Encode copy command
encoder.copyBufferToBuffer(computeBuffer, 0, mappableBuffer, 0, buffSize);

// Submit the commands to the GPU
device.queue.submit([encoder.finish()]);

// Read data from compute buffer
await mappableBuffer.mapAsync(GPUMapMode.READ);
const procData = mappableBuffer.getMappedRange();
const floatData = new Float32Array(procData);

// Check result
let checkMat = true;
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        if (floatData[i*matrixDim + j] != 1.0 * j * matrixDim + i) {
            checkMat = false;
        }
    }
}

// Display output in page
const outputMsg = checkMat ? "Transpose check passed" : "Transpose check failed";
document.getElementById("output").innerHTML = outputMsg;

// Destroy the mapping
mappableBuffer.unmap();
}

// Run example function
runExample();