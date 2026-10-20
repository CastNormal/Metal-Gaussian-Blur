#ifndef GAUSSIAN_FILTER_HPP_
#define GAUSSIAN_FILTER_HPP_

#include <Metal/Metal.hpp>
#include <iostream>
#include <numbers>
#include <stdexcept>
#define MAX_KERNEL_SIZE 16

class GaussianFilter {
public:
    GaussianFilter(MTL::Device* device, int width, int height);
    
    //prevent accidental copying and double releases
    GaussianFilter(const GaussianFilter&) = delete;
    GaussianFilter& operator=(const GaussianFilter&) = delete;
    
    // Execute horizontal and vertical passes
    void encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int sigma);

private:
    MTL::Device* device;
    int width;
    int height;
    std::array<float, MAX_KERNEL_SIZE> kernel;
    int kernelSize;
    
    NS::SharedPtr<MTL::ComputePipelineState> horizontalPipelineState;
    NS::SharedPtr<MTL::ComputePipelineState> verticalPipelineState;
    NS::SharedPtr<MTL::Texture> intermediateTexture;
    
    NS::SharedPtr<MTL::ComputePipelineState> createPipelineState(MTL::Library* library, const char* functionName);
    
    void half_gaussean1d(int sigma);
};

#endif
