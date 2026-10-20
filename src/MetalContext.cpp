#include "../include/MetalContext.hpp"

MetalContext::MetalContext() {
    _device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    
    if (!_device)
        throw std::runtime_error("Metal is not supported on this device.");
    
    _commandQueue = NS::TransferPtr(device()->newCommandQueue());
    
    if (!_commandQueue)
        throw std::runtime_error("Failed to create Command Queue.");
}

MTL::Device *MetalContext::device() { return _device.get(); }
MTL::CommandQueue *MetalContext::commandQueue() { return _commandQueue.get(); }
