#include "../include/GaussianFilter.hpp"


GaussianFilter::GaussianFilter(MTL::Device* device, int width, int height): device(device), width(width), height(height),
horizontalPipelineState(nullptr), verticalPipelineState(nullptr), intermediateTexture(nullptr), kernelSize(0), kernel(std::array<float, MAX_KERNEL_SIZE>()) {
    
    NS::SharedPtr<MTL::Library> defaultLibrary = NS::TransferPtr(device->newDefaultLibrary());
    
    if (!defaultLibrary)
        throw std::runtime_error("Failed to find default library.");

    horizontalPipelineState = createPipelineState(defaultLibrary.get(), "gaussian_blur_horizontal");
    verticalPipelineState = createPipelineState(defaultLibrary.get(), "gaussian_blur_vertical");

    
    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, width, height, false);
    
    if (!desc)
        throw std::runtime_error("Failed to create texture descriptor");
    
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    intermediateTexture = NS::TransferPtr(device->newTexture(desc));
    
    if (!intermediateTexture)
        throw std::runtime_error("Failed to create texture.");
            
    
}

void GaussianFilter::encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int sigma) {
    
    generateGaussianKernel(sigma);
    
    int blockWidth = 32;
    int blockHeight = 8;
    
    MTL::Size gridSize = MTL::Size(width, height, 1);
    MTL::Size threadsPerThreadgroup = MTL::Size(blockWidth, blockHeight, 1);
    
    //Horizontal Pass
    
    MTL::ComputeCommandEncoder* horizontalEncoder = commandBuffer->computeCommandEncoder();
    horizontalEncoder->setTexture(intermediateTexture.get(), 0);
    horizontalEncoder->setComputePipelineState(horizontalPipelineState.get());
    horizontalEncoder->setBuffer(inputBuffer, 0, 0);
    horizontalEncoder->setBytes(kernel.data(), kernelSize*sizeof(float), 1);
    horizontalEncoder->setBytes(&kernelSize, sizeof(int), 2);
    horizontalEncoder->setThreadgroupMemoryLength((blockWidth + (kernelSize-1)*2)*blockHeight*sizeof(float)*4, 0);

    //Dispatch
    
    horizontalEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    horizontalEncoder->endEncoding();
    
    blockWidth = 8;
    blockHeight = 32;
    
    threadsPerThreadgroup = MTL::Size(blockWidth, blockHeight, 1);
    
    //Vertical Pass
    
    MTL::ComputeCommandEncoder *verticalEncoder = commandBuffer->computeCommandEncoder();
    verticalEncoder->setComputePipelineState(verticalPipelineState.get());
    verticalEncoder->setTexture(intermediateTexture.get(), 0);
    verticalEncoder->setBuffer(inputBuffer, 0, 0);
    verticalEncoder->setBytes(kernel.data(), kernelSize*sizeof(float), 1);
    verticalEncoder->setBytes(&kernelSize, sizeof(int), 2);
    verticalEncoder->setThreadgroupMemoryLength(blockWidth * (blockHeight + (kernelSize - 1) * 2) * sizeof(float) * 4, 0);
    
    //Dispatch
    
    verticalEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
    verticalEncoder->endEncoding();
}

NS::SharedPtr<MTL::ComputePipelineState> GaussianFilter::createPipelineState(MTL::Library* library, const char* name) {
    NS::Error* error = nullptr;
    auto strName = NS::String::string(name, NS::UTF8StringEncoding);
    
    NS::SharedPtr<MTL::Function> func = NS::TransferPtr(library->newFunction(strName));
    
    if (!func) {
        std::cerr << "Shader function not found: " << name << std::endl;
        return nullptr;
    }
    
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState = NS::TransferPtr(device->newComputePipelineState(func.get(), &error));
    
    if (!pipelineState) {
        if (error){
            throw std::runtime_error(error->localizedDescription()->utf8String());
        }
    }
    
    return pipelineState;
}

void GaussianFilter::generateGaussianKernel(int sigma) {
    
    int radius = 3 * sigma;
    if (sigma < 1 || radius >= MAX_KERNEL_SIZE) throw std::invalid_argument("Invalid standard deviation");

    
    float sum = 0;
    kernelSize = radius + 1;
    
    for (int i = 0; i < kernelSize; i++) {
        float val = std::expf(-(i * i) / (2.0f * sigma * sigma));
        kernel[i] = val;
        sum += val;
        if (i > 0) sum += val; //sum twice due to symmetry
    }
    
    //norm
    for (int i = 0; i < kernelSize; i++) kernel[i] /= sum;
}
