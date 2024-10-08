const shaderCode = `

@binding(0) @group(0) var<storage, read_write> a_mat : array<f32>;
@binding(1) @group(0) var<storage, read_write> q_mat : array<f32>;
@binding(2) @group(0) var<storage, read_write> p_mat : array<f32>;
@binding(3) @group(0) var<storage, read_write> prod_mat : array<f32>;

override matrix_dim: u32;

var<workgroup> u_vec: array<f32, matrix_dim>;
var<workgroup> dot: f32;
var<workgroup> u_length_squared: f32;

@compute @workgroup_size(matrix_dim)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>) {

    /* Variable declarations */
    var vec_length = 0.0;
    var prod: f32;
    var i: u32;
    var j: u32;
        
    /* Load first column into workgroup memory */
    u_vec[gid.x] = a_mat[gid.x * matrix_dim];
    storageBarrier();
    
    /* Find length of first column and u vector */
    if (gid.x == 0) {
        for (i = 1; i < matrix_dim; i++) {
            vec_length += u_vec[i] * u_vec[i];
        }
        u_length_squared = vec_length;
        vec_length = sqrt(vec_length + u_vec[0] * u_vec[0]);
        a_mat[0] = vec_length;
        u_vec[0] -= vec_length;
        u_length_squared += u_vec[0] * u_vec[0];

    }
    else {
        a_mat[gid.x * matrix_dim] = 0.0;
    }
    storageBarrier();

    /* Transform further columns of A */
    for (i = 1; i < matrix_dim; i++) {
        dot = 0.0;
        if(gid.x == 0) {
            for (j = 0; j < matrix_dim; j++) {
                dot += a_mat[j * matrix_dim + i] * u_vec[j];
            }
        }
        workgroupBarrier();
        a_mat[gid.x * matrix_dim + i] -= 2 * u_vec[gid.x] * dot/u_length_squared;
    }

    /* Update Q matrix */
    for (i = 0; i < matrix_dim; i++) {
        q_mat[gid.x * matrix_dim + i] = -2 * u_vec[i] * u_vec[gid.x]/u_length_squared;
    }
    q_mat[gid.x * matrix_dim + gid.x] += 1;
    storageBarrier();
    
    /* Loop through other columns */
    for (var col: u32 = 1; col < matrix_dim-1; col++) {
        
        /* Load new column into memory */
        u_vec[gid.x] = a_mat[gid.x * matrix_dim + col];
        workgroupBarrier();
        
        /* Find length of A column and u vector */
        if(gid.x == col) {
            vec_length = 0.0;
            for (i = col + 1; i < matrix_dim; i++) {
                vec_length += u_vec[i] * u_vec[i];
            }
            u_length_squared = vec_length;
            vec_length = sqrt(vec_length + u_vec[col] * u_vec[col]);
            u_vec[col] -= vec_length;
            u_length_squared += u_vec[col] * u_vec[col];
            a_mat[col * matrix_dim + col] = vec_length;
        }
        else if(gid.x > col) {
            a_mat[gid.x * matrix_dim + col] = 0.0;            
        }
        storageBarrier();
        
        /* Transform further columns of A */
        for (i = col+1; i < matrix_dim; i++) {
            if(gid.x == 0) {
                dot = 0.0;                
                for (j = 0; j < matrix_dim; j++) {
                    dot += a_mat[j * matrix_dim + i] * u_vec[j];
                }
            }
            workgroupBarrier();
            
            if(gid.x >= col) {
                a_mat[gid.x * matrix_dim + i] -= 2 * u_vec[gid.x] * dot/u_length_squared;
            }
            storageBarrier();
        }
        
        /* Update P matrix */
        if (gid.x >= col) {
            for (i = col; i < matrix_dim; i++) {
                p_mat[gid.x * matrix_dim + i] = -2 * u_vec[i] * u_vec[gid.x]/u_length_squared;
            }
            p_mat[gid.x * matrix_dim + gid.x] += 1;
        }
        storageBarrier();
        
        /* Multiply q_mat * p_mat = prod_mat */
        for (i = col; i < matrix_dim; i++) {
            prod = 0.0;
            for (j = col; j < matrix_dim; j++) {
                prod += q_mat[gid.x * matrix_dim + j] * p_mat[j * matrix_dim + i];
            }
            prod_mat[gid.x * matrix_dim + i] = prod;
        }
        storageBarrier();
        
        /* Place the content of prod_mat in q_mat */
        for (i = col; i < matrix_dim; i++) {
            q_mat[gid.x * matrix_dim + i] = prod_mat[gid.x * matrix_dim + i];
        }
        storageBarrier();
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

// Set parameters
const matrixDim = 32;
const buffSize = matrixDim * matrixDim * 4;

// Create buffer containing A matrix
const aBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});
const aRange = aBuffer.getMappedRange();
const aMatrix = new Array(matrixDim * matrixDim);
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        aMatrix[i*matrixDim + j] = Math.random() * 1000;
    }
}
new Float32Array(aRange).set(aMatrix);
aBuffer.unmap();

// Create buffer containing Q matrix
const qBuffer = device.createBuffer({
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE | 
        GPUBufferUsage.COPY_SRC
});

// Create buffer containing R matrix
const pBuffer = device.createBuffer({
    size: buffSize,
    usage: GPUBufferUsage.STORAGE
});

// Create buffer containing check matrix
const prodBuffer = device.createBuffer({
    size: buffSize,
    usage: 
        GPUBufferUsage.STORAGE
});

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
            matrix_dim: matrixDim
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
        resource: { buffer: qBuffer }
    },
    {
        binding: 2,
        resource: { buffer: pBuffer }
    },
    {
        binding: 3,
        resource: { buffer: prodBuffer }
    }]
});
computePass.setBindGroup(0, bindGroup);

// Encode compute commands
computePass.dispatchWorkgroups(1);

// Complete encoding compute commands
computePass.end();

// Create mappable buffer for Q matrix
const mappableBufferQ = device.createBuffer({
  size: buffSize,
  usage: 
      GPUBufferUsage.COPY_DST | 
      GPUBufferUsage.MAP_READ
});

// Create mappable buffer for R matrix
const mappableBufferR = device.createBuffer({
  size: buffSize,
  usage: 
      GPUBufferUsage.COPY_DST | 
      GPUBufferUsage.MAP_READ
});

// Encode copy commands
encoder.copyBufferToBuffer(qBuffer, 0, mappableBufferQ, 0, buffSize);
encoder.copyBufferToBuffer(aBuffer, 0, mappableBufferR, 0, buffSize);

// Submit the commands to the GPU
device.queue.submit([encoder.finish()]);

// Read data from Q buffer
await mappableBufferQ.mapAsync(GPUMapMode.READ);
const procDataQ = mappableBufferQ.getMappedRange();
const floatDataQ = new Float32Array(procDataQ);

// Read data from A buffer (now contains the R matrix)
await mappableBufferR.mapAsync(GPUMapMode.READ);
const procDataR = mappableBufferR.getMappedRange();
const floatDataR = new Float32Array(procDataR);

// Compute product of Q and R
const checkMatrix = new Array(matrixDim * matrixDim);
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        product = 0.0;
        for(let k = 0; k < matrixDim; k++) {
            product += floatDataQ[i*matrixDim + k] * floatDataR[k*matrixDim + j];
        }
        checkMatrix[i*matrixDim + j] = product;
    }
}

// Check that A = QR
let checkMat = true;
for(let i = 0; i < matrixDim; i++) {
    for(let j = 0; j < matrixDim; j++) {
        if (Math.abs(checkMatrix[i*matrixDim + j] - aMatrix[i*matrixDim + j]) > 0.01) {
            checkMat = false;
        }
    }
}

// Display output in page
const outputMsg = checkMat ? "Factorization check passed" : "Factorization check failed";
document.getElementById("output").innerHTML = outputMsg;

// Destroy the mapping
mappableBufferQ.unmap();
mappableBufferR.unmap();
}

// Run example function
runExample();