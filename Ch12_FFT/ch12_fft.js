const fftInitShader = `

// Real part of time-domain sequence
@binding(0) @group(0) var<storage, read_write> xn_real : array<f32>;

// Imaginary part of time-domain sequencee
@binding(1) @group(0) var<storage, read_write> xn_imag : array<f32>;

// Real part of frequency-domain sequence
@binding(2) @group(0) var<storage, read_write> xk_real : array<f32>;

// Imaginary part of frequency-domain sequence
@binding(3) @group(0) var<storage, read_write> xk_imag : array<f32>;

// Whether to perform the fft or ifft
@binding(4) @group(0) var<storage, read_write> factor : f32;

override group_size: u32;
override pts_per_group: u32;
override filter_start: u32;
override filter_end: u32;

var<workgroup> wg_mem: array<vec2f, pts_per_group>;

@compute @workgroup_size(group_size)
fn fft_init(@builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(num_workgroups) num_groups : vec3<u32>,
    @builtin(workgroup_id) group_id : vec3<u32>,
    @builtin(local_invocation_index) local_id : u32) {

    let pts_per_inv = pts_per_group/group_size;
    let N = pts_per_group * num_groups.x;
    var g_addr = global_id.x * pts_per_inv;
    var wg_addr = local_id * pts_per_inv;
    var x1: vec2f; var x2: vec2f;
    var x3: vec2f; var x4: vec2f;
    var sum12: vec2f; var diff12: vec2f;
    var sum34: vec2f; var diff34: vec2f;
    var i: u32; var N2: u32; var k: f32;
    var c: f32; var s: f32; var wk: vec2f;

    // Store 4-point FFTs in workgroup memory
    for (i = 0; i < pts_per_inv; i += 4) {

        var index = vec4u(g_addr, g_addr+1, g_addr+2, g_addr+3);
        var mask_left: u32 = N/2;
        var mask_left_vec = vec4u(mask_left, mask_left, mask_left, mask_left);
        var mask_right: u32 = 1;
        var mask_right_vec = vec4u(mask_right, mask_right, mask_right, mask_right);
        var shift_pos = u32(log2(f32(N))) - 1;
        var shift_vec = vec4u(shift_pos, shift_pos, shift_pos, shift_pos);
        var bitrev = (index << shift_vec) & mask_left_vec;
        bitrev |= (index >> shift_vec) & mask_right_vec;

        // Bit-reverse addresses
        while(shift_pos > 1) {
            shift_pos -= 2;
            mask_left >>= 1;
            mask_left_vec = vec4u(mask_left, mask_left, mask_left, mask_left);
            mask_right <<= 1;
            mask_right_vec = vec4u(mask_right, mask_right, mask_right, mask_right);
            shift_vec = vec4u(shift_pos, shift_pos, shift_pos, shift_pos);
            bitrev |= (index << shift_vec) & mask_left_vec;
            bitrev |= (index >> shift_vec) & mask_right_vec;
        }

        // Load storage data
        if (factor == -1.0) {
            x1 = vec2f(xn_real[bitrev.x], xn_imag[bitrev.x]);
            x2 = vec2f(xn_real[bitrev.y], xn_imag[bitrev.y]);
            x3 = vec2f(xn_real[bitrev.z], xn_imag[bitrev.z]);
            x4 = vec2f(xn_real[bitrev.w], xn_imag[bitrev.w]);
        } else {
            if((bitrev.x > filter_start) && (bitrev.x < filter_end)) {
                x1 = vec2f(0.0, 0.0);
            } else {
                x1 = vec2f(xk_real[bitrev.x], xk_imag[bitrev.x]);
            }
            if((bitrev.y > filter_start) && (bitrev.y < filter_end)) {
                x2 = vec2f(0.0, 0.0);
            } else {
                x2 = vec2f(xk_real[bitrev.y], xk_imag[bitrev.y]);
            }
            if((bitrev.z > filter_start) && (bitrev.z < filter_end)) {
                x3 = vec2f(0.0, 0.0);
            } else {
                x3 = vec2f(xk_real[bitrev.z], xk_imag[bitrev.z]);
            }
            if((bitrev.w > filter_start) && (bitrev.w < filter_end)) {
                x4 = vec2f(0.0, 0.0);
            } else {
                x4 = vec2f(xk_real[bitrev.w], xk_imag[bitrev.w]);
            }
        }

        // Perform 4-pt FFT
        sum12 = x1 + x2;
        diff12 = x1 - x2;
        sum34 = x3 + x4;
        diff34 = vec2f(x4.y - x3.y, x3.x - x4.x) * factor;

        // Store results to workgroup memory
        wg_mem[wg_addr] = sum12 + sum34;
        wg_mem[wg_addr + 1] = diff12 + diff34;
        wg_mem[wg_addr + 2] = sum12 - sum34;
        wg_mem[wg_addr + 3] = diff12 - diff34;

        // Update addresses
        g_addr += 4;
        wg_addr += 4;
    }

    // Iterate through further stages
    for(N2 = 4; N2 < pts_per_inv; N2 <<= 1) {

        // Reset local memory address
        wg_addr = local_id * pts_per_inv;

        // Trig constant
        k = -3.14159/f32(N2);

        // Iterate through FFT/IFFT chunks
        for (var fft_index: u32 = 0; fft_index < pts_per_inv; fft_index += 2*N2) {
            var first_val = wg_mem[wg_addr];
            wg_mem[wg_addr] += wg_mem[wg_addr + N2];
            wg_mem[wg_addr + N2] = first_val - wg_mem[wg_addr + N2];

            // Iterate through FFT/IFFT points
            for (i = 1; i < N2; i++) {
                c = cos(k * f32(i));
                s = sin(k * f32(i)) * factor;
                wk = vec2f(
                    c * wg_mem[wg_addr + N2 + i].x +
                    s * wg_mem[wg_addr + N2 + i].y,
                    c * wg_mem[wg_addr + N2 + i].y -
                    s * wg_mem[wg_addr + N2 + i].x);
                wg_mem[wg_addr + N2 + i] = wg_mem[wg_addr + i] - wk;
                wg_mem[wg_addr + i] += wk;
            }
            wg_addr += 2*N2;
        }
    }
    workgroupBarrier();

    // Iterate through further stages
    var stage: u32 = 2;
    var start: u32;
    for(N2 = pts_per_inv; N2 < pts_per_group; N2 <<= 1) {
        start = (local_id + (local_id/stage)*stage) * (pts_per_inv/2);
        var angle = start % (N2*2);
        k = -3.14159/f32(N2);

        // Iterate through FFT/IFFT points
        for (i = start; i < start + pts_per_inv/2; i++) {
            c = cos(k * f32(angle));
            s = sin(k * f32(angle)) * factor;
            wk = vec2f(
                c * wg_mem[N2 + i].x + s * wg_mem[N2 + i].y,
                c * wg_mem[N2 + i].y - s * wg_mem[N2 + i].x);
            wg_mem[N2 + i] = wg_mem[i] - wk;
            wg_mem[i] += wk;
            angle++;
        }
        stage <<= 1;
        workgroupBarrier();
    }

    // Store vectors in storage buffers
    g_addr = global_id.x * pts_per_inv;
    wg_addr = local_id * pts_per_inv;
    for (i = 0; i < pts_per_inv; i++) {

        if (factor == -1.0) {
            xk_real[g_addr + i] = wg_mem[wg_addr + i].x;
            xk_imag[g_addr + i] = wg_mem[wg_addr + i].y;
        } else {
            xn_real[g_addr + i] = wg_mem[wg_addr + i].x/f32(N);
            xn_imag[g_addr + i] = wg_mem[wg_addr + i].y/f32(N);
        }
    }
}
`;

const fftStageShader = `
// Real part of time-domain sequence
@binding(0) @group(0) var<storage, read_write> xn_real : array<f32>;

// Imaginary part of time-domain sequencee
@binding(1) @group(0) var<storage, read_write> xn_imag : array<f32>;

// Real part of frequency-domain sequence
@binding(2) @group(0) var<storage, read_write> xk_real : array<f32>;

// Imaginary part of frequency-domain sequence
@binding(3) @group(0) var<storage, read_write> xk_imag : array<f32>;

// Whether to perform the fft or ifft
@binding(4) @group(0) var<storage, read_write> factor : f32;

override group_size: u32;
override pts_per_group: u32;
override stage: u32;

@compute @workgroup_size(group_size)
fn fft_stage(@builtin(global_invocation_id) global_id : vec3<u32>,
    @builtin(num_workgroups) num_groups : vec3<u32>,
    @builtin(workgroup_id) group_id : vec3<u32>,
    @builtin(local_invocation_index) local_id : u32) {

    var points_per_inv = pts_per_group/group_size;
    var addr = (group_id.x + (group_id.x/stage)*stage) * (pts_per_group/2) +
        local_id * (points_per_inv/2);
    var N = pts_per_group*(stage/2);
    var angle = addr % (N*2);
    var k = -3.14159/f32(N);
    var c: f32; var s: f32;
    var x1: vec2f; var x2: vec2f; var wk: vec2f;

    for(var i: u32 = addr; i < addr + points_per_inv/2; i++) {

        // Compute trig values
        c = cos(k * f32(angle));
        s = sin(k * f32(angle)) * factor;

        // Read data from buffer
        if (factor == -1.0) {
            x1 = vec2f(xk_real[i], xk_imag[i]);
            x2 = vec2f(xk_real[i + N], xk_imag[i + N]);
        } else {
            x1 = vec2f(xn_real[i], xn_imag[i]);
            x2 = vec2f(xn_real[i + N], xn_imag[i + N]);
        }

        // Compute scaled value
        wk = vec2f(c * x2.x + s * x2.y, c * x2.y - s * x2.x);

        // Write data to buffer
        if (factor == -1.0) {
            xk_real[i] = x1.x + wk.x;
            xk_imag[i] = x1.y + wk.y;
            xk_real[i + N] = x1.x - wk.x;
            xk_imag[i + N] = x1.y - wk.y;
        } else {
            xn_real[i] = x1.x + wk.x;
            xn_imag[i] = x1.y + wk.y;
            xn_real[i + N] = x1.x - wk.x;
            xn_imag[i + N] = x1.y - wk.y;
        }
        angle++;
    }
}
`

// Play noisy audio
var noisyAudio = new Audio("noisy.wav");
const noisyButton = document.getElementById("noisy");
noisyButton.onclick = function() {
    if (noisyAudio.duration > 0 && !noisyAudio.paused) {
        noisyAudio.pause();
        noisyButton.innerHTML = "Play Noisy Audio";
    } else {
        noisyAudio.play();
        noisyButton.innerHTML = "Pause Noisy Audio";
    }
}

// Play filtered audio
var filteredAudio = new Audio();
const filteredButton = document.getElementById("filtered");
filteredButton.onclick = function() {
    if (filteredAudio.duration > 0 && !filteredAudio.paused) {
        filteredAudio.pause();
        filteredButton.innerHTML = "Play Filtered Audio";
    } else {
        filteredAudio.play();
        filteredButton.innerHTML = "Pause Filtered Audio";
    }
}
filteredButton.disabled = true;

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

// Obtain a response from the input file
const resp = await fetch("noisy.wav");
const arrayBuff = await resp.arrayBuffer();

// Create an audio context with the sample rate at 44,100 samples/second
const AudioContext = window.AudioContext || window.webkitAudioContext;
const ctx = new AudioContext( { sampleRate: 44100.0 } );

async function decodeSuccess(audioBuffer) {

    // Read samples from channel
    var samples = audioBuffer.getChannelData(0);

    // Time sequence - real values
    const realTimeBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: samples.length * 4,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC
    });
    const realTimeRange = realTimeBuffer.getMappedRange();
    new Float32Array(realTimeRange).set(samples);
    realTimeBuffer.unmap();

    // Time sequence - imaginary values
    const imagData = new Array(samples.length);
    imagData.fill(0.0);
    const imagTimeBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: samples.length * 4,
        usage: GPUBufferUsage.STORAGE
    });
    const imagTimeRange = imagTimeBuffer.getMappedRange();
    new Float32Array(imagTimeRange).set(imagData);
    imagTimeBuffer.unmap();

    // Frequency sequence - real values
    const realFreqBuffer = device.createBuffer({
        size: samples.length * 4,
        usage: GPUBufferUsage.STORAGE
    });

    // Frequency sequence - imaginary values
    const imagFreqBuffer = device.createBuffer({
        size: samples.length * 4,
        usage: GPUBufferUsage.STORAGE
    });

    // DFT/IFFT factor
    let factor = [-1.0];
    const factorBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: 4,
        usage: GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST
    });
    const factorRange = factorBuffer.getMappedRange();
    new Float32Array(factorRange).set(factor);
    factorBuffer.unmap();

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
        }, {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
        }, {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" }
        }]
    });

    // Create the bind group
    let bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: { buffer: realTimeBuffer }
        },
        {
            binding: 1,
            resource: { buffer: imagTimeBuffer }
        },
        {
            binding: 2,
            resource: { buffer: realFreqBuffer }
        },
        {
            binding: 3,
            resource: { buffer: imagFreqBuffer }
        },
        {
            binding: 4,
            resource: { buffer: factorBuffer }
        }]
    });

    // Create the pipeline layout
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayout ]
    });

    // Set execution parameters
    const groupSize = 256;
    const groupMemSize = 16384;
    const ptsPerGroup = groupMemSize/8;            // 2048
    const numGroups = samples.length/ptsPerGroup;  // 64   

    // Create the fftInit shader module
    const fftInitShaderModule = device.createShaderModule({
        code: fftInitShader
    });

    // Create the fftStage shader module
    const fftStageShaderModule = device.createShaderModule({
        code: fftStageShader
    });

    // Define the fftInit pipeline
    const fftInitPipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: fftInitShaderModule,
            entryPoint: "fft_init",
            constants: {
                group_size: groupSize,
                pts_per_group: ptsPerGroup,
                filter_start: 11000,
                filter_end: 120072
            }
        }
    });

    // Create the command encoder
    let fftEncoder = device.createCommandEncoder();

    // Create the compute pass encoder
    let fftComputePass = fftEncoder.beginComputePass();
    fftComputePass.setPipeline(fftInitPipeline);
    fftComputePass.setBindGroup(0, bindGroup);

    // Encode compute commands
    fftComputePass.dispatchWorkgroups(numGroups);

    // Complete encoding compute commands
    fftComputePass.end();

    // Submit the commands to the GPU
    device.queue.submit([fftEncoder.finish()]);

    // Perform successive stages
    if(samples.length > ptsPerGroup) {

        let i = 2;
        while (i <= numGroups) {

            // Define the fftStage pipeline
            let fftStagePipeline = device.createComputePipeline({
                layout: pipelineLayout,
                compute: {
                    module: fftStageShaderModule,
                    entryPoint: "fft_stage",
                    constants: {
                        group_size: groupSize,
                        pts_per_group: ptsPerGroup,
                        stage: i
                    }
                }
            });

            // Create the command encoder
            let fftEncoder = device.createCommandEncoder();

            // Create the compute pass encoder
            let fftComputePass = fftEncoder.beginComputePass();
            fftComputePass.setPipeline(fftStagePipeline);
            fftComputePass.setBindGroup(0, bindGroup);

            // Encode compute commands
            fftComputePass.dispatchWorkgroups(numGroups);

            // Complete encoding compute commands
            fftComputePass.end();

            // Submit the commands to the GPU
            device.queue.submit([fftEncoder.finish()]);
            i <<= 1;
        }
    }

    // Set factor to 1.0 - perform the IFFT
    device.queue.writeBuffer(factorBuffer, 0, new Float32Array([1.0]));

    // Create the command encoder
    fftEncoder = device.createCommandEncoder();

    // Create the compute pass encoder
    fftComputePass = fftEncoder.beginComputePass();
    fftComputePass.setPipeline(fftInitPipeline);
    fftComputePass.setBindGroup(0, bindGroup);

    // Encode compute commands
    fftComputePass.dispatchWorkgroups(numGroups);

    // Complete encoding compute commands
    fftComputePass.end();

    // Submit the commands to the GPU
    device.queue.submit([fftEncoder.finish()]);

    // Create mappable buffer
    const realMappableBuffer = device.createBuffer({
        size: samples.length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Perform successive stages
    if(samples.length > ptsPerGroup) {

        let i = 2;
        while (i < numGroups) {

            // Define the fftStage pipeline
            let fftStagePipeline = device.createComputePipeline({
                layout: pipelineLayout,
                compute: {
                    module: fftStageShaderModule,
                    entryPoint: "fft_stage",
                    constants: {
                        group_size: groupSize,
                        pts_per_group: ptsPerGroup,
                        stage: i
                    }
                }
            });

            // Create the command encoder
            let fftEncoder = device.createCommandEncoder();

            // Create the compute pass encoder
            let fftComputePass = fftEncoder.beginComputePass();
            fftComputePass.setPipeline(fftStagePipeline);
            fftComputePass.setBindGroup(0, bindGroup);

            // Encode compute commands
            fftComputePass.dispatchWorkgroups(numGroups);

            // Complete encoding compute commands
            fftComputePass.end();

            // Submit the commands to the GPU
            device.queue.submit([fftEncoder.finish()]);
            i <<= 1;
        }
        
        // Define the fftStage pipeline
        let fftStagePipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
                module: fftStageShaderModule,
                entryPoint: "fft_stage",
                constants: {
                    group_size: groupSize,
                    pts_per_group: ptsPerGroup,
                    stage: i
                }
            }
        });

        // Create the command encoder
        let fftEncoder = device.createCommandEncoder();

        // Create the compute pass encoder
        let fftComputePass = fftEncoder.beginComputePass();
        fftComputePass.setPipeline(fftStagePipeline);
        fftComputePass.setBindGroup(0, bindGroup);

        // Encode compute commands
        fftComputePass.dispatchWorkgroups(numGroups);

        // Complete encoding compute commands
        fftComputePass.end();
        
        // Encode copy command
        fftEncoder.copyBufferToBuffer(realTimeBuffer, 0, realMappableBuffer, 0, samples.length * 4);

        // Submit the commands to the GPU
        device.queue.submit([fftEncoder.finish()]);
    }

    // Read data from compute buffer
    await realMappableBuffer.mapAsync(GPUMapMode.READ);
    const realProcData = realMappableBuffer.getMappedRange();
    const newSamples = new Float32Array(realProcData);

    // Create ArrayBuffer and DataView;
    const fileLength = 44 + 2 * newSamples.length;
    const buffer = new ArrayBuffer(fileLength);
    let view = new DataView(buffer);

    // Define RIFF chunk
    view.setUint32(0, 0x46464952, true);
    view.setUint32(4, fileLength - 8, true);
    view.setUint32(8, 0x45564157, true);

    // Define Format chunk
    view.setUint32(12, 0x20746d66, true);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, audioBuffer.sampleRate, true);
    view.setUint32(28, audioBuffer.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);

    // Define Data chunk
    view.setUint32(36, 0x61746164, true);
    view.setUint32(40, newSamples.length * 2, true);

    // Add samples to DataView
    var offset = 44;
    for (let i = 0; i < newSamples.length; i++) {
        newSamples[i] *= 32768;
        if(newSamples[i] > 32767) {
            newSamples[i] = 32767;
        }
        if (newSamples[i] < -32768) {
            newSamples[i] = -32768;
        }
        view.setUint16(offset, newSamples[i], true);
        offset += 2;
    }

    // Set data as audio source
    const audioBlob = new Blob([buffer], {type: "audio/wav"});
    const audioURL = URL.createObjectURL(audioBlob);
    filteredAudio.src = audioURL;
    filteredButton.disabled = false;

    // Unmap buffer
    realMappableBuffer.unmap();
}

ctx.decodeAudioData(arrayBuff, decodeSuccess);
}

// Run example function
runExample();

