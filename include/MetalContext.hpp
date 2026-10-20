#pragma once
#include <Metal/Metal.hpp>


class MetalContext {
public:
    MetalContext();
    ~MetalContext();
    
    
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    
    MTL::Device *device();
    MTL::CommandQueue *commandQueue();

private:
    MTL::Device *_device;
    MTL::CommandQueue *_commandQueue;
};
