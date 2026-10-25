#pragma once
#include <Metal/Metal.hpp>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include "QOI.h"

struct ImageBuffer {
    std::unique_ptr<uint8_t, decltype(&std::free)> image{ nullptr, &std::free };
    QOI_header_info headerInfo;
};
    
namespace {
    inline ImageBuffer loadQOIImage(const std::string &fileName) {
        
        ImageBuffer imgContent;
        
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
    
    inline NS::SharedPtr<MTL::Buffer> createWrappedMetalBuffer(MTL::Device *device, const ImageBuffer &content) {
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
