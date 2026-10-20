#include "../include/MetalContext.hpp"
#include <iostream>
#include <stdexcept>

MetalContext::MetalContext() {
    this->_device = MTL::CreateSystemDefaultDevice();
    if (!this->_device) {
        throw std::runtime_error("Metal is not supported on this device.");
    }
    
    this->_commandQueue = this->_device->newCommandQueue();
    if (!this->_commandQueue) {
        throw std::runtime_error("Failed to create Command Queue.");
    }
}

MTL::Device *MetalContext::device() { return this->_device; }
MTL::CommandQueue *MetalContext::commandQueue() { return this->_commandQueue; }

MetalContext::~MetalContext() {
    if (this->_commandQueue) this->_commandQueue->release();
    if (this->_device) this->_device->release();
}
