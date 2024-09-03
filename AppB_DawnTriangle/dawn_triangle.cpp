#undef NDEBUG

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdio>
#include <memory>

static wgpu::Instance instance;

static const char shaderCode[] = R"(
@vertex
fn vertexMain(@location(0) coords: vec2f) -> @builtin(position) vec4f {
    return vec4f(coords, 0.0, 1.0);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0, 0.647, 0.0, 1.0);
}
)";

// Called when device request is complete
void deviceCallback(WGPURequestDeviceStatus status, 
    WGPUDevice cDevice, const char* message, void* data) {
    
    // Check for successful device access
    if (status != WGPURequestDeviceStatus_Success) {
        exit(0);
    }

    // Convert C type to C++ object
    wgpu::Device device = wgpu::Device::Acquire(cDevice);

    // Define vertex data to be stored in the vertex buffer
    const float vertexData[6] = { 0.0, 0.5, -0.5, -0.5, 0.5, -0.5 };

    // Create a buffer descriptor and buffer
    wgpu::BufferDescriptor buffDesc {
        .usage = wgpu::BufferUsage::Vertex | wgpu::BufferUsage::CopyDst,
        .size = sizeof(vertexData)
    };
    wgpu::Buffer vertexBuffer = device.CreateBuffer(&buffDesc);
    
    // Encode command to write data to buffer
    device.GetQueue().WriteBuffer(vertexBuffer, 0, vertexData, sizeof(vertexData));
    
    // Define the layout of the vertex buffer
    wgpu::VertexAttribute attr {
        .format = wgpu::VertexFormat::Float32x2,
        .offset = 0,
        .shaderLocation = 0
    };
    wgpu::VertexBufferLayout vertLayout {
        .arrayStride = 8,
        .attributeCount = 1,
        .attributes = &attr
    };
    
    // Create pipeline layout
    wgpu::PipelineLayoutDescriptor plDesc {
        .bindGroupLayoutCount = 0,
        .bindGroupLayouts = nullptr
    };    
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&plDesc);
    
    // Create shader module
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;
    wgpu::ShaderModuleDescriptor shaderDesc {
        .nextInChain = &wgslDesc
    };
    wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);

    // Create vertex state
    wgpu::VertexState vertexState {
        .module = shaderModule,
        .entryPoint = "vertexMain",
        .bufferCount = 1,
        .buffers = &vertLayout
    };
    
    // Define the color state
    wgpu::ColorTargetState colorTargetState {
        .format = wgpu::TextureFormat::BGRA8Unorm
    };

    // Create fragment state
    wgpu::FragmentState fragmentState {
        .module = shaderModule,
        .entryPoint = "fragmentMain",
        .targetCount = 1,
        .targets = &colorTargetState
    };
    
    // Set the primitive state
    wgpu::PrimitiveState primitiveState {
        .topology = wgpu::PrimitiveTopology::TriangleStrip
    };
    
    // Create the render pipeline
    wgpu::RenderPipelineDescriptor pipelineDesc {
        .layout = pipelineLayout,
        .vertex = vertexState,
        .primitive = primitiveState,
        .fragment = &fragmentState
    };
    wgpu::RenderPipeline pipeline = device.CreateRenderPipeline(&pipelineDesc);

    // Create the rendering surface
    wgpu::SurfaceDescriptorFromCanvasHTMLSelector canvasDesc{};
    canvasDesc.selector = "#canvas";
    wgpu::SurfaceDescriptor surfDesc {
        .nextInChain = &canvasDesc
    };
    wgpu::Surface surface = instance.CreateSurface(&surfDesc);

    // Create the swapchain
    wgpu::SwapChainDescriptor swapDesc {
        .usage = wgpu::TextureUsage::RenderAttachment,
        .format = wgpu::TextureFormat::BGRA8Unorm,
        .width = 400,
        .height = 400,
        .presentMode = wgpu::PresentMode::Fifo,
    };
    wgpu::SwapChain swapChain = device.CreateSwapChain(surface, &swapDesc);
        
    // Create the command encoder
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    // Define render pass color attachment
    wgpu::RenderPassColorAttachment attachment {
        .view = swapChain.GetCurrentTextureView(),
        .loadOp = wgpu::LoadOp::Clear,
        .storeOp = wgpu::StoreOp::Store,
        .clearValue = { 
            .r = 0.9, 
            .g = 0.9, 
            .b = 0.9, 
            .a = 1.0 }
    };

    // Create render pass encoder
    wgpu::RenderPassDescriptor renderDesc {
        .colorAttachmentCount = 1,
        .colorAttachments = &attachment
    };
    wgpu::RenderPassEncoder renderPass = encoder.BeginRenderPass(&renderDesc);
    renderPass.SetPipeline(pipeline);
    renderPass.SetVertexBuffer(0, vertexBuffer);

    // Launch draw operations
    renderPass.Draw(3);
    renderPass.End();
    
    // Submit commands to the queue
    wgpu::CommandBuffer commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

int main() {
                
    // Create instance
    instance = wgpu::CreateInstance();
    
    // Request adapter
    instance.RequestAdapter(nullptr, [](
        WGPURequestAdapterStatus adapterStatus, 
        WGPUAdapter cAdapter, const char* message, 
        void* userdata) {
            
        if (adapterStatus == WGPURequestAdapterStatus_Success) {    
            
            // Convert C type to C++ object
            wgpu::Adapter adapter = wgpu::Adapter::Acquire(cAdapter);
            
            // Request device
            adapter.RequestDevice(nullptr, deviceCallback, nullptr);
        }
    }, nullptr);
    
    return 0;
}