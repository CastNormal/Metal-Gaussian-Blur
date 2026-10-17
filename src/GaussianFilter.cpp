#include "../include/GaussianFilter.hpp"
#include <iostream>
#include <numbers>

GaussianFilter::GaussianFilter(MTL::Device* device, int width, int height): device(device), width(width), height(height),
horizontalPipelineState(nullptr), verticalPipelineState(nullptr), intermediateTexture(nullptr), kernelSize(0), kernel(std::array<float, MAX_KERNEL_SIZE>()) {
    MTL::Library* defaultLibrary = device->newDefaultLibrary();
    if (!defaultLibrary) {
        std::cerr << "Failed to find default library." << std::endl;
        return;
    }
    
    
    horizontalPipelineState = createPipelineState(defaultLibrary, "gaussian_blur_horizontal");
    verticalPipelineState = createPipelineState(defaultLibrary, "gaussian_blur_vertical");
    
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float, this->width, this->height, false);
    
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    this->intermediateTexture = device->newTexture(desc);
    
    desc->release();
    defaultLibrary->release();
}

void GaussianFilter::cleanup() {
    if (this->horizontalPipelineState != nullptr) this->horizontalPipelineState->release();
    if (this->verticalPipelineState != nullptr) this->horizontalPipelineState->release();
    if (this->intermediateTexture != nullptr) this->intermediateTexture->release();
}

void GaussianFilter::encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int stddev) {
    
    half_gaussean1d(stddev);
    
    int w = 32;
    int h = 8;
    
    MTL::Size gridSize = MTL::Size(this->width, this->height, 1);
    MTL::Size threadsPerThreadgroup_horizontal = MTL::Size(w, h, 1);
    MTL::Size threadsPerThreadgroup_vertical = MTL::Size(w, h, 1);
    
    //Horizontal Pass
    
    MTL::ComputeCommandEncoder* horizontalEncoder = commandBuffer->computeCommandEncoder();
    horizontalEncoder->setTexture(this->intermediateTexture, 0);
    horizontalEncoder->setComputePipelineState(this->horizontalPipelineState);
    horizontalEncoder->setBuffer(inputBuffer, 0, 0);
    horizontalEncoder->setBytes(this->kernel.data(), this->kernelSize*sizeof(float), 1);
    horizontalEncoder->setBytes(&this->kernelSize, sizeof(int), 2);
    horizontalEncoder->setThreadgroupMemoryLength((w + (this->kernelSize-1)*2)*h*sizeof(float)*4, 0);

    //Dispatch
    
    horizontalEncoder->dispatchThreads(gridSize, threadsPerThreadgroup_horizontal);
    horizontalEncoder->endEncoding();
    
    w = 8;
    h = 32;
    
    //Vertical Pass
    
    MTL::ComputeCommandEncoder *verticalEncoder = commandBuffer->computeCommandEncoder();
    verticalEncoder->setComputePipelineState(this->verticalPipelineState);
    verticalEncoder->setTexture(this->intermediateTexture, 0);
    verticalEncoder->setBuffer(inputBuffer, 0, 0);
    verticalEncoder->setBytes(this->kernel.data(), this->kernelSize*sizeof(float), 1);
    verticalEncoder->setBytes(&this->kernelSize, sizeof(int), 0, 2);
    verticalEncoder->setThreadgroupMemoryLength((w + (this->kernelSize-1)*2)*h*sizeof(float)*4, 0);
    
    //Dispatch
    
    verticalEncoder->dispatchThreadgroups(gridSize, threadsPerThreadgroup_vertical);
    verticalEncoder->endEncoding();
}

MTL::ComputePipelineState* GaussianFilter::createPipelineState(MTL::Library* library, const char* name) {
    NS::Error* error = nullptr;
    
    auto strName = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* func = library->newFunction(strName);
    
    MTL::ComputePipelineState *pipelineState = this->device->newComputePipelineState(func, &error);
    if (!pipelineState) { std::cerr << "Failed to create pipeline state." << std::endl; return nullptr; }
    
    return pipelineState;
}

void GaussianFilter::half_gaussean1d(int stddev) {
    this->kernelSize = stddev + 1;
    float sum = 0;
    
    for (int i = 0; i <= stddev; i++) {
        float val = std::expf(-(i * i) / (2.0f * stddev * stddev));
        this->kernel[i] = val;
        sum += val;
        if (i > 0) sum += val; //sum twice due to symmetry
    }
    
    //norm
    for (int i = 0; i <= stddev; i++) this->kernel[i] /= sum;
}
