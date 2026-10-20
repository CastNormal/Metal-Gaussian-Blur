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
    
    ~GaussianFilter();
    
    //prevent accidental copying and double releases
    GaussianFilter(const GaussianFilter&) = delete;
    GaussianFilter& operator=(const GaussianFilter&) = delete;
    
    // Execute horizontal and vertical passes
    void encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int stddev);

private:
    MTL::Device* device;
    int width;
    int height;
    std::array<float, MAX_KERNEL_SIZE> kernel;
    int kernelSize;
    
    MTL::ComputePipelineState* horizontalPipelineState;
    MTL::ComputePipelineState* verticalPipelineState;
    MTL::Texture* intermediateTexture;
    
    MTL::ComputePipelineState* createPipelineState(MTL::Library* library, const char* functionName);
    
    void half_gaussean1d(int stddev);
};

#endif
