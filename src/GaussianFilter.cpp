#include "../include/GaussianFilter.hpp"


GaussianFilter::GaussianFilter(MTL::Device* device, int width, int height): device(device), width(width), height(height),
horizontalPipelineState(nullptr), verticalPipelineState(nullptr), intermediateTexture(nullptr) {
    
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
    
    GaussianKernel gaus;
    ::generateGaussianKernel(sigma, gaus);
    
    int blockWidth = 32;
    int blockHeight = 8;
    
    MTL::Size gridSize = MTL::Size(width, height, 1);
    MTL::Size threadsPerThreadgroup = MTL::Size(blockWidth, blockHeight, 1);
    
    //Horizontal Pass
    
    MTL::ComputeCommandEncoder* horizontalEncoder = commandBuffer->computeCommandEncoder();
    horizontalEncoder->setTexture(intermediateTexture.get(), 0);
    horizontalEncoder->setComputePipelineState(horizontalPipelineState.get());
    horizontalEncoder->setBuffer(inputBuffer, 0, 0);
    horizontalEncoder->setBytes(gaus.kernel.data(), gaus.size*sizeof(float), 1);
    horizontalEncoder->setBytes(&gaus.size, sizeof(int), 2);
    horizontalEncoder->setThreadgroupMemoryLength((blockWidth + (gaus.size-1)*2)*blockHeight*sizeof(float)*4, 0);

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
    verticalEncoder->setBytes(gaus.kernel.data(), gaus.size*sizeof(float), 1);
    verticalEncoder->setBytes(&gaus.size, sizeof(int), 2);
    verticalEncoder->setThreadgroupMemoryLength(blockWidth * (blockHeight + (gaus.size - 1) * 2) * sizeof(float) * 4, 0);
    
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
