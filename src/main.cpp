#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <numbers>
#include <memory>
#include <chrono>
#include <limits>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include "../include/QOI.h"
#include "../include/GaussianFilter.hpp"
#include "../include/MetalContext.hpp"
#include "../include/ImageBuffer.hpp"
#include "../include/CPUReference.hpp"

void CPUImplementation(const int sigma) {
    MetalContext context;
    
    ImageBuffer imgContent = loadQOIImage("./assets/test3.qoi");
    auto buffer = createWrappedMetalBuffer(context.device(), imgContent);
    
    const auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imgContent, sigma);
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU time: " << elapsed.count() << " ms\n";
    
    qoi_encoder(static_cast<uint8_t*>(output.data()), imgContent.headerInfo, "output.qoi");
}

void GPUImplementation(const int sigma) {
    MetalContext context;
    
    ImageBuffer imgContent = loadQOIImage("./assets/test3.qoi");
    auto buffer = createWrappedMetalBuffer(context.device(), imgContent);
    
    GaussianFilter filter(context.device(), static_cast<int>(imgContent.headerInfo.width), static_cast<int>(imgContent.headerInfo.height));
    
    MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
    
    if (!commandBuffer)
        throw std::runtime_error("failed to create command buffer.");
    
    filter.encode(commandBuffer, buffer.get(), sigma);
    
    const auto start = std::chrono::steady_clock::now();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    
    const auto end = std::chrono::steady_clock::now();
    
    
    if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
        auto* error = commandBuffer->error();
        
        if (error && error->localizedDescription())
            throw std::runtime_error(error->localizedDescription()->utf8String());
        
        throw std::runtime_error("GPU execution failed");
    }
    
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    
    std::cout << "CPU submission + wait: " << elapsed.count() << " ms\n";
    
    // Actual GPU execution timing
    const double gpuStart = commandBuffer->GPUStartTime();
    const double gpuEnd = commandBuffer->GPUEndTime();
    
    if (gpuStart > 0.0 && gpuEnd > gpuStart) {
        const double gpuTimeMs = (gpuEnd - gpuStart) * 1000.0;
        std::cout << "GPU execution: " << gpuTimeMs << " ms\n";
    } else std::cout << "GPU timing unavailable.\n";
    
    qoi_encoder(static_cast<uint8_t*>(buffer->contents()), imgContent.headerInfo, "output.qoi");
}

int main() {
    auto autoReleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    try {
        constexpr int sigma = 2;
      
        GPUImplementation(sigma);
    
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
