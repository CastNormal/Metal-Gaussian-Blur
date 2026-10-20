#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <stdio.h>
#include <numbers>
#include <memory>
#include <chrono>
#include "../include/QOI.h"
#include "../include/GaussianFilter.hpp"
#include "../include/MetalContext.hpp"

int main(int argc, char** argv) {
    auto autoReleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    
    try {
        MetalContext context;
        
        std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen("./assets/test2.qoi", "rb"), &std::fclose);
        
        if (!file)
            throw std::runtime_error("failed to open file\n");
        
        QOI_header_info header_info;
        
        //buffer contains raw RGBA
        std::unique_ptr<void, decltype((&std::free))> image(qoi_decoder(file.get(), &header_info), &std::free);
        
        if (!image)
            throw std::runtime_error("failed to load image.");
        
        
        uint64_t pixel_size = uint64_t(header_info.width) * header_info.height * header_info.channels;
        
        //wrap buffer around MTL::Buffer
        uint64_t buffer_size = round_to_page_size(pixel_size);
        
        
        NS::SharedPtr<MTL::Buffer> buffer = NS::TransferPtr(context.device()->newBuffer(image.get(), buffer_size, MTL::ResourceStorageModeShared, nullptr));
        
        if (!buffer)
            throw std::runtime_error("Failed to create buffer.");
        
        int sigma = 2;
        
        GaussianFilter filter(context.device(), static_cast<int>(header_info.width), static_cast<int>(header_info.height));
        
        MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
        
        filter.encode(commandBuffer, buffer.get(), sigma);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // 4. Calculate the elapsed duration (e.g., in milliseconds or nanoseconds)
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        std::cout << "Waited for: " << elapsed.count() << " ms\n";
        
        //write out blurred image
        qoi_encoder((uint8_t*)buffer->contents(), header_info, "output.qoi");
        
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
