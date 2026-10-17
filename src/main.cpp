#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <stdio.h>
#include <numbers>
#include <chrono> // Required header
#include "../include/QOI.h"
#include "../include/GaussianFilter.hpp"
#include "../include/MetalContext.hpp"


int main() {
    MetalContext context;
    
    FILE *file = fopen("./assets/test2.qoi", "rb");
    
    if (file == NULL) {
        perror("failed to open file\n");
        return 1;
    }
    
    QOI_header_info header_info;
    
    //buffer contains raw RGBA 
    void *image = qoi_decoder(file, &header_info);
    uint64_t pixel_size = header_info.width*header_info.height*header_info.channels;
    fclose(file);
    
    //wrap buffer around MTL::Buffer
    uint64_t buffer_size = round_to_page_size(pixel_size);
    MTL::Buffer* buffer = context.device()->newBuffer(image, buffer_size, MTL::ResourceStorageModeShared, nullptr);
    
    int stddev = 2;
    
    GaussianFilter filter(context.device(), header_info.width, header_info.height);
    
    MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
    
    filter.encode(commandBuffer, buffer, stddev);

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    //write out blurred image
    qoi_encoder((uint8_t*)buffer->contents(), header_info, "output.qoi");
    

    buffer->release();
    filter.cleanup();
    context.cleanup();
    free(image);

    return 0;
}
