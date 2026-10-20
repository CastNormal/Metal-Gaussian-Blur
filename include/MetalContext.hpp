#pragma once
#include <Metal/Metal.hpp>
#include <iostream>
#include <stdexcept>


class MetalContext {
public:
    MetalContext();
    
    
    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    
    MTL::Device *device();
    MTL::CommandQueue *commandQueue();

private:
    NS::SharedPtr<MTL::Device> _device;
    NS::SharedPtr<MTL::CommandQueue> _commandQueue;
};
