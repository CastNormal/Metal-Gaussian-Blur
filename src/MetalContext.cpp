#include "../include/MetalContext.hpp"

MetalContext::MetalContext() {
    this->_device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    
    if (!this->_device)
        throw std::runtime_error("Metal is not supported on this device.");
    
    this->_commandQueue = NS::TransferPtr(this->device()->newCommandQueue());
    
    if (!this->_commandQueue)
        throw std::runtime_error("Failed to create Command Queue.");
}

MTL::Device *MetalContext::device() { return this->_device.get(); }
MTL::CommandQueue *MetalContext::commandQueue() { return this->_commandQueue.get(); }
