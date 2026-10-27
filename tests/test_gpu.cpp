#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <gtest/gtest.h>

#include <array>
#include <random>
#include <utility>
#include "../include/GaussianKernel.hpp"
#include "../include/GaussianFilter.hpp"
#include "../benchmark/NaiveGaussianFilter.hpp"
#include "../include/CPUReference.hpp"
#include "../include/MetalContext.hpp"

TEST(GPUImplementation, CPUImplementationComparison) {
    constexpr std::array<int, 5> sigmas = {1, 2, 3, 4, 5};
    constexpr std::array<std::pair<int, int>, 7> dimensions = {{
        {1, 1}, {5, 7}, {32, 8}, {8, 32}, {33, 9}, {35, 37}, {64, 65}
    }};
    
    constexpr int channels = 4;
    constexpr int tolerence = 1;
    
    MetalContext context;

    for (const auto& [width, height] : dimensions) {
        ImageBuffer imageBuffer;
        imageBuffer.headerInfo.width = width;
        imageBuffer.headerInfo.height = height;
        imageBuffer.headerInfo.channels = channels;
        
        const size_t size = width * height * channels;
        
        imageBuffer.image.reset(static_cast<uint8_t*>(std::malloc(size)));
        
        std::mt19937 gen(42);

        std::generate_n(imageBuffer.image.get(), size, [&]() {
            return static_cast<uint8_t>(std::uniform_int_distribution<uint16_t>(0, 255)(gen));
        });
        
        for (const int sigma : sigmas) {
            SCOPED_TRACE("Width=" + std::to_string(width) + ", Height=" + std::to_string(height) + ", Sigma=" + std::to_string(sigma));

            auto buffer = NS::TransferPtr(context.device()->newBuffer(imageBuffer.image.get(),
                    size, MTL::ResourceStorageModeShared));
            
            GaussianFilter filter(context.device(), static_cast<int>(imageBuffer.headerInfo.width), static_cast<int>(imageBuffer.headerInfo.height));
            
            MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
            
            if (!commandBuffer)
                throw std::runtime_error("failed to create command buffer.");
            
            filter.encode(commandBuffer, buffer.get(), sigma);
            
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            
            if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
                auto* error = commandBuffer->error();
                
                if (error && error->localizedDescription())
                    throw std::runtime_error(error->localizedDescription()->utf8String());
                
                throw std::runtime_error("GPU execution failed");
            }
            
            std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, sigma);
            bool all_equal = true;
            
            for (int i = 0; i < size; i++)
                if (std::abs(output[i] - static_cast<const uint8_t*>(buffer->contents())[i]) > tolerence)
                    all_equal = false;
            
            EXPECT_TRUE(all_equal);
        }
    }
}

TEST(NaiveGPUImplementation, CPUImplementationComparison) {
    constexpr std::array<int, 5> sigmas = {1, 2, 3, 4, 5};
    constexpr std::array<std::pair<int, int>, 7> dimensions = {{
        {1, 1}, {5, 7}, {32, 8}, {8, 32}, {33, 9}, {35, 37}, {64, 65}
    }};
    
    constexpr int channels = 4;
    constexpr int tolerence = 1;
    
    MetalContext context;

    for (const auto& [width, height] : dimensions) {
        ImageBuffer imageBuffer;
        imageBuffer.headerInfo.width = width;
        imageBuffer.headerInfo.height = height;
        imageBuffer.headerInfo.channels = channels;
        
        const size_t size = width * height * channels;
        
        imageBuffer.image.reset(static_cast<uint8_t*>(std::malloc(size)));
        
        std::mt19937 gen(42);

        std::generate_n(imageBuffer.image.get(), size, [&]() {
            return static_cast<uint8_t>(std::uniform_int_distribution<uint16_t>(0, 255)(gen));
        });
        
        for (const int sigma : sigmas) {
            SCOPED_TRACE("Width=" + std::to_string(width) + ", Height=" + std::to_string(height) + ", Sigma=" + std::to_string(sigma));

            auto buffer = NS::TransferPtr(context.device()->newBuffer(imageBuffer.image.get(),
                    size, MTL::ResourceStorageModeShared));
            
            NaiveGaussianFilter filter(context.device(), static_cast<int>(imageBuffer.headerInfo.width), static_cast<int>(imageBuffer.headerInfo.height));
            
            MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
            
            if (!commandBuffer)
                throw std::runtime_error("failed to create command buffer.");
            
            filter.encode(commandBuffer, buffer.get(), sigma);
            
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
            
            if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
                auto* error = commandBuffer->error();
                
                if (error && error->localizedDescription())
                    throw std::runtime_error(error->localizedDescription()->utf8String());
                
                throw std::runtime_error("GPU execution failed");
            }
            
            std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, sigma);
            bool all_equal = true;
            
            for (int i = 0; i < size; i++)
                if (std::abs(output[i] - static_cast<const uint8_t*>(buffer->contents())[i]) > tolerence)
                    all_equal = false;
            
            EXPECT_TRUE(all_equal);
        }
    }
}
