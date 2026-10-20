#include "../include/GaussianFilter.hpp"


GaussianFilter::GaussianFilter(MTL::Device* device, int width, int height): device(device), width(width), height(height),
horizontalPipelineState(nullptr), verticalPipelineState(nullptr), intermediateTexture(nullptr), kernelSize(0), kernel(std::array<float, MAX_KERNEL_SIZE>()) {
    MTL::Library* defaultLibrary = device->newDefaultLibrary();
    
    if (!defaultLibrary)
        throw std::runtime_error("Failed to find default library.");
    
    try {
        horizontalPipelineState = createPipelineState(defaultLibrary, "gaussian_blur_horizontal");
        
        if (!horizontalPipelineState)
            throw std::runtime_error("Failed to create horizontal pipeline.");
        
        verticalPipelineState = createPipelineState(defaultLibrary, "gaussian_blur_vertical");
        
        if (!horizontalPipelineState)
            throw std::runtime_error("Failed to create horizontal pipeline.");
        
        MTL::TextureDescriptor* desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float,     this->width, this->height, false);
        
        if (!desc)
            throw std::runtime_error("Failed to create texture descriptor");
        
        desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        this->intermediateTexture = device->newTexture(desc);
        
        if (!intermediateTexture)
            throw std::runtime_error("Failed to create texture.");
        
        desc->release();
        defaultLibrary->release();
    } catch(...) {
        if (horizontalPipelineState) horizontalPipelineState->release();
        if (verticalPipelineState) verticalPipelineState->release();
        if (intermediateTexture) intermediateTexture->release();
        
        defaultLibrary->release();
        throw;
    }
    
    defaultLibrary->release();
}

GaussianFilter::~GaussianFilter() {
    if (this->horizontalPipelineState) this->horizontalPipelineState->release();
    if (this->verticalPipelineState) this->verticalPipelineState->release();
    if (this->intermediateTexture) this->intermediateTexture->release();
}

void GaussianFilter::encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int stddev) {
    
    half_gaussean1d(stddev);
    
    int w = 32;
    int h = 8;
    
    MTL::Size gridSize = MTL::Size(this->width, this->height, 1);
    MTL::Size threadBlock = MTL::Size(w, h, 1);
    
    //Horizontal Pass
    
    MTL::ComputeCommandEncoder* horizontalEncoder = commandBuffer->computeCommandEncoder();
    horizontalEncoder->setTexture(this->intermediateTexture, 0);
    horizontalEncoder->setComputePipelineState(this->horizontalPipelineState);
    horizontalEncoder->setBuffer(inputBuffer, 0, 0);
    horizontalEncoder->setBytes(this->kernel.data(), this->kernelSize*sizeof(float), 1);
    horizontalEncoder->setBytes(&this->kernelSize, sizeof(int), 2);
    horizontalEncoder->setThreadgroupMemoryLength((w + (this->kernelSize-1)*2)*h*sizeof(float)*4, 0);

    //Dispatch
    
    horizontalEncoder->dispatchThreads(gridSize, threadBlock);
    horizontalEncoder->endEncoding();
    
    w = 8;
    h = 32;
    
    threadBlock = MTL::Size(w, h, 1);
    
    //Vertical Pass
    
    MTL::ComputeCommandEncoder *verticalEncoder = commandBuffer->computeCommandEncoder();
    verticalEncoder->setComputePipelineState(this->verticalPipelineState);
    verticalEncoder->setTexture(this->intermediateTexture, 0);
    verticalEncoder->setBuffer(inputBuffer, 0, 0);
    verticalEncoder->setBytes(this->kernel.data(), this->kernelSize*sizeof(float), 1);
    verticalEncoder->setBytes(&this->kernelSize, sizeof(int), 2);
    verticalEncoder->setThreadgroupMemoryLength(w * (h + (this->kernelSize - 1) * 2) * sizeof(float) * 4, 0);
    
    //Dispatch
    
    verticalEncoder->dispatchThreads(gridSize, threadBlock);
    verticalEncoder->endEncoding();
}

MTL::ComputePipelineState* GaussianFilter::createPipelineState(MTL::Library* library, const char* name) {
    NS::Error* error = nullptr;
    auto strName = NS::String::string(name, NS::UTF8StringEncoding);
    MTL::ComputePipelineState *pipelineState = nullptr;
    
    try {
        MTL::Function* func = library->newFunction(strName);
        
        pipelineState = this->device->newComputePipelineState(func, &error);
        if (!pipelineState) {
            if (error){
                std::cerr << error->localizedDescription()->utf8String() << std::endl;
            }
        }
    } catch(...) {
        if (pipelineState) pipelineState->release();
        throw;
    }
    
    return pipelineState;
}

void GaussianFilter::half_gaussean1d(int sigma) {
    
    if (sigma <= 0 || sigma >= MAX_KERNEL_SIZE) throw std::invalid_argument("Invalid standard deviation");

    this->kernelSize = sigma + 1;
    float sum = 0;
    
    for (int i = 0; i <= sigma; i++) {
        float val = std::expf(-(i * i) / (2.0f * sigma * sigma));
        this->kernel[i] = val;
        sum += val;
        if (i > 0) sum += val; //sum twice due to symmetry
    }
    
    //norm
    for (int i = 0; i <= sigma; i++) this->kernel[i] /= sum;
}
