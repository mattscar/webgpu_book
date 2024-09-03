const computeLoss = `

// Training data
@binding(0) @group(0) var<storage, read_write> train_data : array<f32>;

// Training results
@binding(1) @group(0) var<storage, read_write> train_results : array<vec4f>;

// Weights
@binding(2) @group(0) var<storage, read_write> weights : array<f32>;

// Bias weights
@binding(3) @group(0) var<storage, read_write> bias_weights : array<f32>;

// Test data
@binding(4) @group(0) var<storage, read_write> test_data : array<f32>;

// Test results
@binding(5) @group(0) var<storage, read_write> test_results : array<f32>;

override group_size: u32;
override batch_size: u32;
override num_train_points: u32;
override num_test_points: u32;
override num_inputs: u32;
override l1_size: u32;
override l2_size: u32;
override l3_size: u32;
override num_weights: u32;
override num_bias: u32;
override num_epochs: u32;
override eta: f32;

var<workgroup> layer1: array<f32, l1_size * batch_size>;
var<workgroup> layer2: array<f32, l2_size * batch_size>;
var<workgroup> layer3: array<f32, l3_size * batch_size>;
var<workgroup> J: array<f32, num_weights * batch_size>;
var<workgroup> J_bias: array<f32, num_bias * batch_size>;

@compute @workgroup_size(group_size)
fn computeMain(@builtin(local_invocation_id) id : vec3<u32>) {

    var num_iterations = num_train_points/batch_size;
    var batch_id = id.x / num_inputs;
    var node_id = id.x % num_inputs;
    var l1_id = batch_id * l1_size;
    var l2_id = batch_id * l2_size;
    var l3_id = batch_id * l3_size;
    var sum: f32; var d: f32;
    var sm1: f32; var sm2: f32; var sm3: f32;
    var i: u32; var j: u32; var k: u32; var l: u32;
    var weight_addr: u32; var bias_addr: u32;

    // Iterate through the entire training set several times
    for (var epoch: u32 = 0; epoch < num_epochs; epoch++) {

        // Process nine minibatches per training set
        for (var iter: u32 = 0; iter < num_iterations; iter++) {

            var train_id = iter * batch_size * num_inputs + batch_id * num_inputs;
            weight_addr = node_id * num_inputs;
            bias_addr = node_id;

            // First layer
            sum = 0.0;
            for (i = 0; i < num_inputs; i++) {
                sum += train_data[train_id + i] * weights[weight_addr + i];
            }
            sum += bias_weights[bias_addr];
            layer1[l1_id + node_id] = max(0.0, sum);
            weight_addr += l1_size * num_inputs;
            bias_addr += num_inputs;
            workgroupBarrier();

            // Second layer
            sum = 0.0;
            for (i = 0; i < l1_size; i++) {
                sum += layer1[l1_id + i] * weights[weight_addr + i];
            }
            sum += bias_weights[bias_addr];
            layer2[l2_id + node_id] = max(0.0, sum);
            weight_addr += l2_size * l1_size;
            bias_addr += l2_size;
            workgroupBarrier();

            // Third layer
            if(node_id < l3_size) {
                sum = 0.0;
                for (i = 0; i < l2_size; i++) {
                    sum += layer2[l2_id + i] * weights[weight_addr + i];
                }
                sum += bias_weights[bias_addr];
                layer3[l3_id + node_id] = sum;
            }
            workgroupBarrier();

            // Compute softmax and loss gradient
            if(node_id == 0) {

                // Compute the softmax outputs
                d = exp(layer3[l3_id]) + exp(layer3[l3_id+1]) + exp(layer3[l3_id+2]);
                sm1 = exp(layer3[l3_id])/d;
                sm2 = exp(layer3[l3_id+1])/d;
                sm3 = exp(layer3[l3_id+2])/d;

                // Compute loss gradients for output nodes
                var res = train_results[iter * batch_size + batch_id];
                layer3[l3_id] = sm1 - res[0];
                layer3[l3_id+1] = sm2 - res[1];
                layer3[l3_id+2] = sm3 - res[2];

                var l2_offset = num_inputs * l1_size;
                var l3_offset = l2_offset + l1_size * l2_size;

                // Set loss gradients and biases to zero
                for (i = 0; i < num_weights; i++) {
                    J[batch_id * num_weights + i] = 0.0;
                }
                for (i = 0; i < num_bias; i++) {
                    J_bias[batch_id * num_bias + i] = 0.0;
                }

                // Compute elements of loss gradient
                for (i = 0; i < l3_size; i++) {

                    J_bias[batch_id * num_bias + l1_size + l2_size + i] = layer3[l3_id + i];

                    for (j = 0; j < l2_size; j++) {

                        // Set the Layer 3 weights
                        J[batch_id * num_weights + l3_offset + i * l2_size + j] = layer3[l3_id + i] * layer2[l2_id + j];

                        // Set Layer 2 weights
                        if (layer2[l2_id + j] > 0) {

                            J_bias[batch_id * num_bias + l1_size + j] += layer3[l3_id + i] * weights[l3_offset + i * l2_size + j];

                            for (k = 0; k < l1_size; k++) {

                                J[batch_id * num_weights + l2_offset + j * l1_size + k] +=
                                    layer3[l3_id + i] * weights[l3_offset + i * l2_size + j] * layer1[l1_id + k];

                                // Set Layer 1 weights
                                if (layer1[l1_id + k] > 0) {

                                    J_bias[batch_id * num_bias + k] += layer3[l3_id + i] * weights[l3_offset + i * l2_size + j] *
                                    weights[l2_offset + j * l1_size + k];

                                    for (l = 0; l < num_inputs; l++) {

                                        J[batch_id * num_weights + k * num_inputs + l] +=
                                            layer3[l3_id + i] * weights[l3_offset + i * l2_size + j] *
                                            weights[l2_offset + j * l1_size + k] * train_data[train_id + l];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            workgroupBarrier();

            // Update weights and biases
            if (id.x == 0) {

                // Iterate through weights
                for (i = 0; i < num_weights; i++) {
                    sum = 0.0;
                    for (j = 0; j < batch_size; j++) {
                        sum += J[j * num_weights + i];
                    }
                    weights[i] -= eta * sum;
                }

                // Iterate through biases
                for (i = 0; i < num_bias; i++) {
                    sum = 0.0;
                    for (j = 0; j < batch_size; j++) {
                        sum += J_bias[j * num_bias + i];
                    }
                    bias_weights[i] -= eta * sum;
                }
            }
            storageBarrier();
        }
    }

    // Iterate through test points
    for (var test: u32 = 0; test < num_test_points; test++) {

        var test_addr = test * num_inputs;

        // First layer
        if (batch_id == 0) {
            sum = 0.0;
            for (i = 0; i < num_inputs; i++) {
                sum += test_data[test_addr + i] * weights[node_id * num_inputs + i];
            }
            sum += bias_weights[node_id];
            layer1[node_id] = max(0.0, sum);
        }
        workgroupBarrier();

        // Second layer
        weight_addr = l1_size * num_inputs;
        bias_addr = l1_size;        
        if (batch_id == 0) {
            sum = 0.0;
            for (i = 0; i < l1_size; i++) {
                sum += layer1[i] * weights[weight_addr + node_id * l1_size + i];
            }
            sum += bias_weights[bias_addr + node_id];
            layer2[node_id] = max(0.0, sum);
        }
        workgroupBarrier();

        // Third layer
        weight_addr += l2_size * l1_size;
        bias_addr += l2_size;
        if((batch_id == 0) && (node_id < l3_size)) {
            sum = 0.0;
            for (i = 0; i < l2_size; i++) {
                sum += layer2[i] * weights[weight_addr + node_id * l2_size + i];
            }
            sum += bias_weights[bias_addr + node_id];
            layer3[node_id] = sum;
        }
        workgroupBarrier();

        // Update test result buffer
        if(id.x == 0) {
            d = exp(layer3[0]) + exp(layer3[1]) + exp(layer3[2]);
            test_results[test * 3] = exp(layer3[0])/d;
            test_results[test * 3 + 1] = exp(layer3[1])/d;
            test_results[test * 3 + 2] = exp(layer3[2])/d;
        }
        storageBarrier();        
    }
}
`;

const trainData = new Float32Array([5.1,2.5,3.0,1.1,5.8,2.7,3.9,1.2,5.4,3.0,4.5,1.5,6.2,2.8,4.8,1.8,6.1,2.6,5.6,1.4,6.7,3.0,5.2,2.3,6.0,2.9,4.5,1.5,4.3,3.0,1.1,0.1,4.7,3.2,1.3,0.2,6.5,3.0,5.2,2.0,7.2,3.2,6.0,1.8,6.3,3.3,4.7,1.6,6.9,3.1,5.4,2.1,6.0,3.4,4.5,1.6,5.5,2.4,3.8,1.1,7.4,2.8,6.1,1.9,6.3,3.4,5.6,2.4,6.5,2.8,4.6,1.5,7.3,2.9,6.3,1.8,5.1,3.4,1.5,0.2,5.4,3.9,1.7,0.4,5.2,2.7,3.9,1.4,6.9,3.2,5.7,2.3,5.5,2.3,4.0,1.3,6.0,3.0,4.8,1.8,5.6,2.7,4.2,1.3,5.6,2.8,4.9,2.0,6.8,2.8,4.8,1.4,6.9,3.1,4.9,1.5,7.2,3.6,6.1,2.5,6.3,2.5,4.9,1.5,5.9,3.0,4.2,1.5,6.7,3.3,5.7,2.1,5.7,3.0,4.2,1.2,6.4,2.9,4.3,1.3,5.0,3.2,1.2,0.2,5.0,3.4,1.5,0.2,6.2,2.2,4.5,1.5,5.9,3.2,4.8,1.8,5.6,3.0,4.1,1.3,5.4,3.9,1.3,0.4,5.0,3.0,1.6,0.2,5.9,3.0,5.1,1.8,5.0,2.3,3.3,1.0,6.4,2.8,5.6,2.2,4.8,3.0,1.4,0.1,5.4,3.7,1.5,0.2,6.4,3.2,5.3,2.3,4.6,3.6,1.0,0.2,5.0,3.5,1.3,0.3,4.8,3.4,1.9,0.2,6.3,2.7,4.9,1.8,6.3,2.8,5.1,1.5,5.2,3.5,1.5,0.2,6.1,2.8,4.0,1.3,6.7,3.1,4.7,1.5,6.0,2.7,5.1,1.6,5.1,3.5,1.4,0.2,5.5,4.2,1.4,0.2,6.5,3.0,5.5,1.8,4.4,2.9,1.4,0.2,7.9,3.8,6.4,2.0,6.4,2.8,5.6,2.1,6.9,3.1,5.1,2.3,5.0,3.4,1.6,0.4,6.0,2.2,5.0,1.5,6.1,2.9,4.7,1.4,5.6,2.9,3.6,1.3,4.5,2.3,1.3,0.3,5.7,2.8,4.1,1.3,5.2,4.1,1.5,0.1,6.1,2.8,4.7,1.2,6.8,3.0,5.5,2.1,6.1,3.0,4.9,1.8,5.8,2.8,5.1,2.4,5.5,2.6,4.4,1.2,4.9,3.1,1.5,0.1,6.5,3.0,5.8,2.2,5.8,2.7,5.1,1.9,4.6,3.2,1.4,0.2,6.6,2.9,4.6,1.3,6.3,2.3,4.4,1.3,6.3,2.9,5.6,1.8,4.9,3.0,1.4,0.2,5.7,2.9,4.2,1.3,5.0,3.6,1.4,0.2,7.7,3.0,6.1,2.3,7.2,3.0,5.8,1.6,6.2,3.4,5.4,2.3,5.1,3.8,1.6,0.2,6.0,2.2,4.0,1.0,6.4,3.2,4.5,1.5,5.5,2.5,4.0,1.3,5.6,2.5,3.9,1.1,5.0,3.5,1.6,0.6,6.7,3.1,5.6,2.4,7.0,3.2,4.7,1.4,6.7,2.5,5.8,1.8,5.4,3.4,1.7,0.2,4.9,2.5,4.5,1.7,6.4,3.1,5.5,1.8,5.8,2.7,5.1,1.9,4.8,3.0,1.4,0.3,6.6,3.0,4.4,1.4,4.8,3.4,1.6,0.2,6.2,2.9,4.3,1.3,5.5,3.5,1.3,0.2,5.7,2.8,4.5,1.3,4.8,3.1,1.6,0.2,5.8,4.0,1.2,0.2,6.3,3.3,6.0,2.5,5.4,3.4,1.5,0.4,4.9,3.1,1.5,0.1,4.4,3.0,1.3,0.2,5.1,3.8,1.9,0.4,4.7,3.2,1.6,0.2,5.0,3.3,1.4,0.2,5.1,3.5,1.4,0.3,5.8,2.7,4.1,1.0,5.3,3.7,1.5,0.2,6.8,3.2,5.9,2.3,5.1,3.7,1.5,0.4,7.7,2.6,6.9,2.3,5.7,4.4,1.5,0.4,5.7,2.6,3.5,1.0,7.7,3.8,6.7,2.2,5.7,3.8,1.7,0.3,7.1,3.0,5.9,2.1,4.9,3.1,1.5,0.1,7.6,3.0,6.6,2.1,6.3,2.5,5.0,1.9,5.7,2.5,5.0,2.0,5.0,2.0,3.5,1.0,6.5,3.2,5.1,2.0,4.9,2.4,3.3,1.0,5.6,3.0,4.5,1.5,4.4,3.2,1.3,0.2,5.1,3.8,1.5,0.3,5.8,2.6,4.0,1.2,5.2,3.4,1.4,0.2,5.5,2.4,3.7,1.0,6.7,3.3,5.7,2.5,6.4,2.7,5.3,1.9,5.1,3.3,1.7,0.5]);

const trainResults = new Float32Array([0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0]);

const testData = new Float32Array([4.6,3.4,1.4,0.3,4.6,3.1,1.5,0.2,7.7,2.8,6.7,2.0,6.1,3.0,4.6,1.4,6.7,3.1,4.4,1.4,6.7,3.0,5.0,1.7]);

const testResults = new Float32Array([1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0]);

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

// Define constants
const batchSize = 16;
const nodesPerLayer = 4;
const groupSize = batchSize * nodesPerLayer;
const numInputs = 4;
const layer1Size = 4;
const layer2Size = 4;
const layer3Size = 3;
const numTrainPoints = Math.trunc(trainData.length/numInputs);
const numTestPoints = Math.trunc(testData.length/numInputs);
const numWeights = layer1Size * numInputs + layer2Size * layer1Size + layer3Size * layer2Size;
const numBias = layer1Size + layer2Size + layer3Size;
const numEpochs = 150;
const eta = 0.001;

// Store training data
const trainDataBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: trainData.length * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const trainDataRange = trainDataBuffer.getMappedRange();
new Float32Array(trainDataRange).set(trainData);
trainDataBuffer.unmap();

// Store training results
const trainResultBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: trainResults.length * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const trainResultsRange = trainResultBuffer.getMappedRange();
new Float32Array(trainResultsRange).set(trainResults);
trainResultBuffer.unmap();

// Store node weights
const sigma = Math.sqrt(2.0/numInputs);
const weightBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: numWeights * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});

// Generate and store weights
const weightData = new Array(numWeights);
for (let i = 0; i < numWeights; i+= 2) {
    let x1 = Math.random();
    let x2 = Math.random();
    weightData[i] = Math.sqrt(-2.0 * Math.log(x1)) * Math.cos(2 * Math.PI * x2) * sigma;
    weightData[i + 1] = Math.sqrt(-2.0 * Math.log(x1)) * Math.sin(2 * Math.PI * x2) * sigma;
}
const weightRange = weightBuffer.getMappedRange();
new Float32Array(weightRange).set(weightData);
weightBuffer.unmap();

// Store bias weights
let biasWeightData =  new Array(numBias).fill(0.0);
const biasWeightBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: numBias * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const biasWeightRange = biasWeightBuffer.getMappedRange();
new Float32Array(biasWeightRange).set(biasWeightData);
biasWeightBuffer.unmap();

// Store test data
const testDataBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: testData.length * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const testDataRange = testDataBuffer.getMappedRange();
new Float32Array(testDataRange).set(testData);
testDataBuffer.unmap();

// Store test results computed by the GPU
let testResultData =  new Array(numTestPoints * 3).fill(0.0);
const testResultBuffer = device.createBuffer({
    mappedAtCreation: true,
    size: numTestPoints * 3 * 4,
    usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC
});
const testResultRange = testResultBuffer.getMappedRange();
new Float32Array(testResultRange).set(testResultData);
testResultBuffer.unmap();

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
    }, {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
    }]
});

// Create the bind group
let bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{
        binding: 0,
        resource: { buffer: trainDataBuffer }
    },
    {
        binding: 1,
        resource: { buffer: trainResultBuffer }
    },
    {
        binding: 2,
        resource: { buffer: weightBuffer }
    },
    {
        binding: 3,
        resource: { buffer: biasWeightBuffer }
    },
    {
        binding: 4,
        resource: { buffer: testDataBuffer }
    },
    {
        binding: 5,
        resource: { buffer: testResultBuffer }
    }]
});

// Create the pipeline layout
const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ bindGroupLayout ]
});

// Create the shader module for the computeLoss shader
const computeLossModule = device.createShaderModule({
    code: computeLoss
});

// Create the compute pass encoder
const computePass = encoder.beginComputePass({
    label: "Compute Pass 0"
});

// Define the compute procedure
const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
        module: computeLossModule,
        entryPoint: "computeMain",
        constants: {
            group_size: groupSize,
            batch_size: batchSize,
            num_train_points: numTrainPoints,
            num_test_points: numTestPoints,
            num_inputs: numInputs,
            l1_size: layer1Size,
            l2_size: layer2Size,
            l3_size: layer3Size,
            num_weights: numWeights,
            num_bias: numBias,
            num_epochs: numEpochs,
            eta: eta
        }
    }
});

computePass.setPipeline(computePipeline);
computePass.setBindGroup(0, bindGroup);

// Encode compute commands
computePass.dispatchWorkgroups(1);

// Complete encoding compute commands
computePass.end();

// Create mappable buffer
const mappableBuffer = device.createBuffer({
    size: numTestPoints * 3 * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
});

// Encode copy command
encoder.copyBufferToBuffer(testResultBuffer, 0, mappableBuffer, 0, numTestPoints * 3 * 4);

// Submit the commands to the GPU
device.queue.submit([encoder.finish()]);

// Read data from compute buffer
await mappableBuffer.mapAsync(GPUMapMode.READ);
const procData = mappableBuffer.getMappedRange();
const floatData = new Float32Array(procData);

let msg = "";
for (let test = 0; test < numTestPoints; test++) {
    
    // Actual values
    max_value = -999.0;
    max_actual_index = -1;
    msg = msg.concat("Actual classification:&nbsp;&nbsp;");
    for (let i = 0; i < 3; i++) {
        msg = msg.concat(parseFloat(testResults[test * 3 + i]).toFixed(3)).concat(" ");
        if(testResults[test * 3 + i] > max_value) {
            max_value = testResults[test * 3 + i];
            max_actual_index = i;
        }
    }
    msg = msg.concat("<br />");
    
    // Actual values
    max_value = -999.0;
    max_computed_index = -1;
    msg = msg.concat("Computed outputs:&nbsp;&nbsp;&nbsp;&nbsp;");
    for (let i = 0; i < 3; i++) {
        msg = msg.concat(parseFloat(floatData[test * 3 + i]).toFixed(3)).concat(" ");
        if(floatData[test * 3 + i] > max_value) {
            max_value = testResults[test * 3 + i];
            max_computed_index = i;
        }
    }
    msg = msg.concat("<br />");
    
    // Display result
    if(max_actual_index == max_computed_index) {
        msg = msg.concat("Result: SUCCESS");
    } else {
        msg = msg.concat("Result: FAILURE");
    }
    msg = msg.concat("<br /><br />");
}

// Update label in page
document.getElementById("results").innerHTML = msg;

// Destroy the mapping
mappableBuffer.unmap();
}

// Run example function
runExample();