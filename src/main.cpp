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

namespace {
    struct ImageContent {
        std::unique_ptr<uint8_t, decltype(&std::free)> image{ nullptr, &std::free };
        QOI_header_info headerInfo;
    };
    
    ImageContent loadQOIImage(const std::string &fileName) {
        
        ImageContent imgContent;
        
        std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(fileName.data(), "rb"), &std::fclose);
        
        if (!file)
            throw std::runtime_error("failed to open file: " + fileName);
        
        imgContent.headerInfo = QOI_header_info();
        
        //buffer contains raw RGBA
        imgContent.image.reset(static_cast<uint8_t*>(qoi_decoder(file.get(), &imgContent.headerInfo)));
        
        if (!imgContent.image)
            throw std::runtime_error("failed to load image.");
        
        if (imgContent.headerInfo.width == 0 || imgContent.headerInfo.height == 0 || imgContent.headerInfo.width > std::numeric_limits<int>::max() || imgContent.headerInfo.height > std::numeric_limits<int>::max())
            throw std::runtime_error("Invalid image dimensions");
        
        return imgContent;
    }
    
    NS::SharedPtr<MTL::Buffer> createWrappedMetalBuffer(MTL::Device *device, const ImageContent &content) {
        //assume rgba
        const uint64_t pixelSize = uint64_t(content.headerInfo.width) * content.headerInfo.height * 4;
        
        //wrap buffer around MTL::Buffer
        const uint64_t bufferSize = round_to_page_size(pixelSize);

        if (bufferSize < pixelSize || bufferSize > device->maxBufferLength())
            throw std::runtime_error("Invalid Metal buffer size");
        
        NS::SharedPtr<MTL::Buffer> buffer = NS::TransferPtr(device->newBuffer(content.image.get(), bufferSize, MTL::ResourceStorageModeShared, nullptr));
        
        if (!buffer)
            throw std::runtime_error("Failed to create buffer.");
        
        return buffer;
    }
};

int main(int argc, char** argv) {
    auto autoReleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    
    try {
        MetalContext context;
        
        ImageContent imgContent = loadQOIImage("./assets/test3.qoi");
        auto buffer = createWrappedMetalBuffer(context.device(), imgContent);
        
        constexpr int sigma = 2;
        
        GaussianFilter filter(context.device(), static_cast<int>(imgContent.headerInfo.width), static_cast<int>(imgContent.headerInfo.height));
        
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
        
        //write out blurred image
        qoi_encoder(static_cast<uint8_t*>(buffer->contents()), imgContent.headerInfo, "output.qoi");
        
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
