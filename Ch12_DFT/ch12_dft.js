const dftShader = `

// Real part of time-domain sequence
@binding(0) @group(0) var<storage, read_write> xn_real : array<vec4f>;

// Imaginary part of time-domain sequencee
@binding(1) @group(0) var<storage, read_write> xn_imag : array<vec4f>;

// Real part of frequency-domain sequence
@binding(2) @group(0) var<storage, read_write> xk_real : array<vec4f>;

// Imaginary part of frequency-domain sequence
@binding(3) @group(0) var<storage, read_write> xk_imag : array<vec4f>;

// Whether to perform the DFT or IDFT
@binding(4) @group(0) var<storage, read_write> factor : f32;

override group_size: u32;
override filter_start: u32;
override filter_end: u32;

@compute @workgroup_size(group_size)
fn computeMain(@builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(num_workgroups) num_groups : vec3<u32>) {

    let N = group_size * num_groups.x;
    let two_pi_k_over_N = (-2.0 * 3.14159 * f32(gid.x))/f32(N);
    let num_vectors = N/4;
    var real_sum = 0.0;
    var imag_sum = 0.0;
    var real_val: vec4f;
    var imag_val: vec4f;

    // Iterate through input vectors
    for (var i: u32 = 0; i < num_vectors; i++) {

        // Form vectors of wk values
        var arg = vec4f(two_pi_k_over_N * f32(4 * i),
                        two_pi_k_over_N * f32(4 * i + 1),
                        two_pi_k_over_N * f32(4 * i + 2),
                        two_pi_k_over_N * f32(4 * i + 3));

        var w_real = cos(arg);
        var w_imag = sin(arg) * factor;

        // Process input vector
        if (factor == -1.0) {        
            real_val = xn_real[i];
            imag_val = xn_imag[i];
        } else {
            real_val = xk_real[i];
            imag_val = xk_imag[i];            
        }
        real_sum += dot(real_val, w_real) - dot(imag_val, w_imag);
        imag_sum += dot(real_val, w_imag) + dot(imag_val, w_real);
    }

    // Store results to memory
    if (factor == -1.0) {
        
        // Apply filter
        if((gid.x > filter_start) && (gid.x < filter_end)) {
            xk_real[gid.x/4][gid.x % 4] = 0.0;
            xk_imag[gid.x/4][gid.x % 4] = 0.0;                
        } else {
            xk_real[gid.x/4][gid.x % 4] = real_sum;
            xk_imag[gid.x/4][gid.x % 4] = imag_sum;        
        }
    } else {
        xn_real[gid.x/4][gid.x % 4] = real_sum/f32(N);
        xn_imag[gid.x/4][gid.x % 4] = imag_sum/f32(N);
    }
}
`;

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

    // Create the shader module
    const dftShaderModule = device.createShaderModule({
        code: dftShader
    });

    // Define the compute pipeline
    const dftComputePipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: dftShaderModule,
            entryPoint: "computeMain",
            constants: {
                group_size: 256,
                filter_start: 11000,
                filter_end: 120072
            }
        }
    });

    // Create the command encoder
    const dftEncoder = device.createCommandEncoder();

    // Create the compute pass encoder
    const dftComputePass = dftEncoder.beginComputePass();
    dftComputePass.setPipeline(dftComputePipeline);
    dftComputePass.setBindGroup(0, bindGroup);

    // Encode compute commands
    dftComputePass.dispatchWorkgroups(512);

    // Complete encoding compute commands
    dftComputePass.end();

    // Submit the commands to the GPU
    device.queue.submit([dftEncoder.finish()]);

    // The IDFT should be performed
    device.queue.writeBuffer(factorBuffer, 0, new Float32Array([1.0]));

    // Create the command encoder
    const idftEncoder = device.createCommandEncoder();

    // Create the compute pass encoder
    const idftComputePass = idftEncoder.beginComputePass();
    idftComputePass.setPipeline(dftComputePipeline);
    idftComputePass.setBindGroup(0, bindGroup);

    // Encode compute commands
    idftComputePass.dispatchWorkgroups(512);

    // Complete encoding compute commands
    idftComputePass.end();

    // Create mappable buffer
    const realMappableBuffer = device.createBuffer({
        size: samples.length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    // Encode copy command
    idftEncoder.copyBufferToBuffer(realTimeBuffer, 0, realMappableBuffer, 0, samples.length * 4);

    // Submit the commands to the GPU
    device.queue.submit([idftEncoder.finish()]);

    // Read data from real buffer
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

