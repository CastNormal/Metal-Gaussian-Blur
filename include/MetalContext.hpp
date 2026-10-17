#pragma once
#include <Metal/Metal.hpp>


class MetalContext {
public:
    MetalContext();
    
    void cleanup();
    
    MTL::Device *device();
    MTL::CommandQueue *commandQueue();

private:
    MTL::Device *_device;
    MTL::CommandQueue *_commandQueue;
};
