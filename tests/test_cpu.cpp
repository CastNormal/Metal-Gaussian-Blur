#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include "../include/ImageBuffer.hpp"
#include "../include/CPUReference.hpp"

TEST(CPUReference, ConstantColor) {
    ImageBuffer imageBuffer;
    
    std::array<uint8_t, 5> colors = { 0, 25, 50, 110, 255 };
    constexpr int sigma = 2;
    
    //7x7 image
    imageBuffer.headerInfo.width = 7;
    imageBuffer.headerInfo.height = 7;
    imageBuffer.headerInfo.colorspace = 4;
    
    size_t size = imageBuffer.headerInfo.width*imageBuffer.headerInfo.height*4;
    imageBuffer.image.reset(static_cast<uint8_t*>(std::malloc(size)));
    
    
    for (const uint8_t color: colors) {
        std::fill_n(imageBuffer.image.get(), size, color);
        
        std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, sigma);
        
        EXPECT_TRUE(std::all_of(output.data(), output.data() + size, [color](uint8_t byte) {
            return byte == color;
        }));
    }
}

TEST(CPUReference, SingePixel) {
    ImageBuffer imageBuffer;
    
    std::array<uint8_t, 5> colors = { 0, 25, 50, 110, 255 };
    constexpr int sigma = 4;
    
    //7x7 image
    imageBuffer.headerInfo.width = 1;
    imageBuffer.headerInfo.height = 1;
    imageBuffer.headerInfo.colorspace = 4;
    
    size_t size = 4;
    imageBuffer.image.reset(static_cast<uint8_t*>(std::malloc(size)));
    
    for (const uint8_t color: colors) {
        std::fill_n(imageBuffer.image.get(), size, color);
        
        std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, sigma);
        
        EXPECT_TRUE(std::all_of(output.data(), output.data() + size, [color](uint8_t byte) {
            return byte == color;
        }));
    }
}

TEST(CPUReference, ImpulseResponse) {
    ImageBuffer imageBuffer;
    
    constexpr int width = 9;
    constexpr int height = 9;
    constexpr int channels = 4;
    constexpr int sigma = 1;
    
    
    //9x9 image
    imageBuffer.headerInfo.width = width;
    imageBuffer.headerInfo.height = height;
    imageBuffer.headerInfo.colorspace  = 4;
    imageBuffer.headerInfo.channels = channels;
    
    const size_t size = imageBuffer.headerInfo.width*imageBuffer.headerInfo.height*4;
    imageBuffer.image.reset(static_cast<uint8_t*>(std::calloc(size, sizeof(uint8_t))));
    
    //set middle pixel to 255
    const int middle = ((width>>1) + (height>>1) * width) * channels;
    
    //rgba so fill 4 bytes
    std::fill_n(imageBuffer.image.get() + middle, 4, 255);
    
    std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, sigma);
    
    for (int channel = 0; channel < channels; channel++) {
        int prevHor = output.data()[middle + channel];
        int prevVer = output.data()[middle + channel];
        
        for (int dist = 1; dist <= width/2; dist++) {
            const int offset = dist*channels;
            
            const int left = middle - offset + channel;
            const int right = middle + offset + channel;
            const int up = middle - offset*width + channel;
            const int down = middle + offset*width + channel;
            
            //symmetry
            EXPECT_EQ(output.data()[left], output.data()[right]);
            EXPECT_EQ(output.data()[up], output.data()[down]);
            
            //decreases as furthers from middle
            EXPECT_GE(prevHor, output.data()[left]);
            EXPECT_GE(prevVer, output.data()[up]);
            
            prevHor = output.data()[left];
            prevVer = output.data()[up];
        }
    }
}
