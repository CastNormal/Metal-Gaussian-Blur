#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <random>
#include <filesystem>
#include "../include/QOI.h"
#include "../include/ImageBuffer.hpp"

TEST(QOI, RGBAPixelPreservation) {
    constexpr int width = 10;
    constexpr int height = 10;
    constexpr int channels = 4;
    
    const auto filePath = std::filesystem::temp_directory_path() / "pixelPreservation.qoi";
    const std::string fileName = filePath.string();
    
    QOI_header_info headerInfo;
    headerInfo.width = width;
    headerInfo.height = height;
    headerInfo.colorspace = 0;
    headerInfo.channels = channels;
    
    size_t size = width*height*channels;
    std::mt19937 gen(43);
    
    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(size);

    std::generate_n(buffer.get(), size, [&]() {
        return static_cast<uint8_t>(std::uniform_int_distribution<uint16_t>(0, 255)(gen));
    });
    

    ASSERT_TRUE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str())) << "Failed to encode: " << fileName;
    
    ImageBuffer imageBuffer = loadQOIImage(fileName.c_str());
    size_t decodedSize = imageBuffer.headerInfo.width*imageBuffer.headerInfo.height*imageBuffer.headerInfo.channels;
    
    EXPECT_EQ(size, decodedSize);
    EXPECT_TRUE(std::equal(buffer.get(), buffer.get()+size, imageBuffer.image.get(), imageBuffer.image.get()+size));
}

TEST(QOI, EncoderInvalidHeader) {
    QOI_header_info headerInfo;
    std::unique_ptr<uint8_t[]> buffer = std::make_unique<uint8_t[]>(1);
    const std::string fileName = "../assets/test.qoi"; //valid filepath
    
    headerInfo.width = 0;
    headerInfo.height = 1;
    headerInfo.colorspace = 0;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = 0;
    headerInfo.colorspace = 0;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = QOI_MAX_WIDTH;
    headerInfo.height = 1;
    headerInfo.colorspace = 0;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = QOI_MAX_HEIGHT;
    headerInfo.colorspace = 0;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = 1;
    headerInfo.colorspace = -1;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = 1;
    headerInfo.colorspace = 2;
    headerInfo.channels = 3;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = 1;
    headerInfo.colorspace = 0;
    headerInfo.channels = 2;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
    
    headerInfo.width = 1;
    headerInfo.height = 1;
    headerInfo.colorspace = 0;
    headerInfo.channels = 5;
    
    EXPECT_FALSE(qoi_encoder(buffer.get(), headerInfo, fileName.c_str()));
}
