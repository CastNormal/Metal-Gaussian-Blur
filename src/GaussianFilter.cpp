#include "../include/GaussianFilter.hpp"


GaussianFilter::GaussianFilter(MTL::Device* device, int width, int height): device(device), width(width), height(height),
horizontalPipelineState(nullptr), verticalPipelineState(nullptr), intermediateTexture(nullptr), kernelSize(0), kernel(std::array<float, MAX_KERNEL_SIZE>()) {
    
    NS::SharedPtr<MTL::Library> defaultLibrary = NS::TransferPtr(device->newDefaultLibrary());
    
    if (!defaultLibrary)
        throw std::runtime_error("Failed to find default library.");

    horizontalPipelineState = createPipelineState(defaultLibrary.get(), "gaussian_blur_horizontal");
    verticalPipelineState = createPipelineState(defaultLibrary.get(), "gaussian_blur_vertical");

    
    MTL::TextureDescriptor *desc = MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA32Float,    this->width, this->height, false);
    
    if (!desc)
        throw std::runtime_error("Failed to create texture descriptor");
    
    desc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
    this->intermediateTexture = NS::TransferPtr(device->newTexture(desc));
    
    if (!intermediateTexture)
        throw std::runtime_error("Failed to create texture.");
            
    
}
/*
GaussianFilter::~GaussianFilter() {
    if (this->horizontalPipelineState) this->horizontalPipelineState->release();
    if (this->verticalPipelineState) this->verticalPipelineState->release();
    if (this->intermediateTexture) this->intermediateTexture->release();
}
*/
void GaussianFilter::encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int stddev) {
    
    half_gaussean1d(stddev);
    
    int w = 32;
    int h = 8;
    
    MTL::Size gridSize = MTL::Size(this->width, this->height, 1);
    MTL::Size threadBlock = MTL::Size(w, h, 1);
    
    //Horizontal Pass
    
    MTL::ComputeCommandEncoder* horizontalEncoder = commandBuffer->computeCommandEncoder();
    horizontalEncoder->setTexture(this->intermediateTexture.get(), 0);
    horizontalEncoder->setComputePipelineState(this->horizontalPipelineState.get());
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
    verticalEncoder->setComputePipelineState(this->verticalPipelineState.get());
    verticalEncoder->setTexture(this->intermediateTexture.get(), 0);
    verticalEncoder->setBuffer(inputBuffer, 0, 0);
    verticalEncoder->setBytes(this->kernel.data(), this->kernelSize*sizeof(float), 1);
    verticalEncoder->setBytes(&this->kernelSize, sizeof(int), 2);
    verticalEncoder->setThreadgroupMemoryLength(w * (h + (this->kernelSize - 1) * 2) * sizeof(float) * 4, 0);
    
    //Dispatch
    
    verticalEncoder->dispatchThreads(gridSize, threadBlock);
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
    
    NS::SharedPtr<MTL::ComputePipelineState> pipelineState = NS::TransferPtr(this->device->newComputePipelineState(func.get(), &error));
    
    if (!pipelineState) {
        if (error){
            throw std::runtime_error(error->localizedDescription()->utf8String());
        }
    }
    
    return pipelineState;
}

void GaussianFilter::half_gaussean1d(int sigma) {
    
    int radius = 3 * sigma;
    if (sigma < 1 || radius >= MAX_KERNEL_SIZE) throw std::invalid_argument("Invalid standard deviation");

    
    float sum = 0;
    this->kernelSize = radius + 1;
    
    for (int i = 0; i <= this->kernelSize; i++) {
        float val = std::expf(-(i * i) / (2.0f * sigma * sigma));
        this->kernel[i] = val;
        sum += val;
        if (i > 0) sum += val; //sum twice due to symmetry
    }
    
    //norm
    for (int i = 0; i <= sigma; i++) this->kernel[i] /= sum;
}
