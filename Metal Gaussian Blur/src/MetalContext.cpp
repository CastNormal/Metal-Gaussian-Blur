#include "../include/MetalContext.hpp"
#include <iostream>

MetalContext::MetalContext() {
    this->_device = MTL::CreateSystemDefaultDevice();
    if (!this->_device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        return;
    }
    
    this->_commandQueue = this->_device->newCommandQueue();
    if (!this->_commandQueue) {
        std::cerr << "Failed to create Command Queue," << std::endl;
        return;
    }
}

MTL::Device *MetalContext::device() { return this->_device; }
MTL::CommandQueue *MetalContext::commandQueue() { return this->_commandQueue; }

void MetalContext::cleanup() {
    if (this->_commandQueue != nullptr) this->_commandQueue->release();
    if (this->_device != nullptr) this->_device->release();
}
