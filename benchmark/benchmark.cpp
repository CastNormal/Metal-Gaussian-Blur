#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <array>
#include <filesystem>
#include "../include/GaussianKernel.hpp"
#include "../include/GaussianFilter.hpp"
#include "../include/ImageBuffer.hpp"
#include "../include/MetalContext.hpp"
#include "../include/CPUReference.hpp"
#include "../include/QOI.h"

void GPUBenchmark(const int sigma, const int iterations) {
    MetalContext context;
    
    ImageBuffer imageBuffer = loadQOIImage("./assets/test.qoi");
    const size_t size = imageBuffer.headerInfo.width * imageBuffer.headerInfo.height * imageBuffer.headerInfo.channels;

    GaussianFilter filter(context.device(), static_cast<int>(imageBuffer.headerInfo.width), static_cast<int>(imageBuffer.headerInfo.height));
    
    for (int i = 0; i < iterations; i++) {
        auto buffer = NS::TransferPtr(context.device()->newBuffer(imageBuffer.image.get(), size, MTL::ResourceStorageModeShared));
        
        MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
        
        if (!commandBuffer)
            throw std::runtime_error("failed to create command buffer.");
        
        filter.encode(commandBuffer, buffer.get(), sigma);
        
        const auto start = std::chrono::high_resolution_clock::now();
        
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        
        const auto end = std::chrono::high_resolution_clock::now();
        
        
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
    }
}


int main() {
    constexpr int iterations = 100;
    constexpr int sigma= 5;
    
    GPUBenchmark(sigma, iterations);
    
    return 0;
}
