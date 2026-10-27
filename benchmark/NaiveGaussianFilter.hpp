#ifndef NAIVE_GAUSSIAN_FILTER_HPP_
#define NAIVE_GAUSSIAN_FILTER_HPP_

#include <Metal/Metal.hpp>
#include <iostream>
#include <numbers>
#include <stdexcept>

#include "GaussianKernel.hpp"

class NaiveGaussianFilter {
public:
    NaiveGaussianFilter(MTL::Device* device, int width, int height);
    
    NaiveGaussianFilter(const NaiveGaussianFilter&) = delete;
    NaiveGaussianFilter& operator=(const NaiveGaussianFilter&) = delete;
    
    void encode(MTL::CommandBuffer* commandBuffer, MTL::Buffer* inputBuffer, int sigma);

private:
    MTL::Device* device;
    int width;
    int height;
    
    NS::SharedPtr<MTL::ComputePipelineState> horizontalPipelineState;
    NS::SharedPtr<MTL::ComputePipelineState> verticalPipelineState;
    NS::SharedPtr<MTL::Texture> intermediateTexture;
    
    NS::SharedPtr<MTL::ComputePipelineState> createPipelineState(MTL::Library* library, const char* functionName);
    
    void generateGaussianKernel(int sigma);
};

#endif
