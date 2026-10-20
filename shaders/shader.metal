#include <metal_stdlib>

[[kernel]] void gaussian_blur_horizontal(constant uchar4 *in                    [[buffer(0)]],
                               constant float *gaussianKernel                   [[buffer(1)]],
                               constant int &gaussianKernelSize                 [[buffer(2)]],
                               metal::texture2d<float, metal::access::write> out[[texture(0)]],
                               uint2 gid                                        [[thread_position_in_grid]],
                               uint2 tid                                        [[thread_position_in_threadgroup]],
                               uint2 groupSize                                  [[threads_per_threadgroup]],
                               threadgroup float4* sharedBlock                  [[threadgroup(0)]]) {
    
    int width = out.get_width();
    int height = out.get_height();
    
    int cache_width = groupSize.x + 2*(gaussianKernelSize-1);
    
    for (int i = tid.x; i < cache_width; i+=groupSize.x) {
        //start from 0 in cache
        int offset = gid.x - tid.x + i - gaussianKernelSize + 1;
        
        int clamped_x = metal::clamp(offset, 0, width - 1);
        int clamped_y = metal::clamp(int(gid.y), 0, height - 1);
        
        int in_index = clamped_x + clamped_y * width;
        
        sharedBlock[tid.y * cache_width + i] = float4(in[in_index]) / 255.0f; //normalize while we're at it
    }
    
    //wait for all of the cache in L1 to be stored
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    //boundary con
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int pos = tid.y * cache_width + tid.x + gaussianKernelSize - 1;
    
    float4 mid = sharedBlock[pos] * gaussianKernel[0];
    float4 dot = mid;
    
    for (int i = 1; i < gaussianKernelSize; i++) {
        dot += (sharedBlock[pos - i] + sharedBlock[pos + i]) * gaussianKernel[i];
    }
    
    out.write(dot, gid);
}


[[kernel]] void gaussian_blur_vertical(metal::texture2d<float, metal::access::read> in  [[texture(0)]],
                                device uchar4 *out                                      [[buffer(0)]],
                                constant float *gaus_kernel                             [[buffer(1)]],
                                constant int &gaus_kernel_size                          [[buffer(2)]],
                                uint2 gid                                               [[thread_position_in_grid]],
                                uint2 tid                                             [[thread_position_in_threadgroup]],
                                uint2 groupSize                                         [[threads_per_threadgroup]],
                                threadgroup float4* sharedBlock                         [[threadgroup(0)]]) {
    int width = in.get_width();
    int height = in.get_height();
    
    int cache_height = groupSize.y + (gaus_kernel_size-1)*2;
    
    for (int i = tid.y; i < cache_height; i += groupSize.y) {
        int offset = gid.y - tid.y + i - (gaus_kernel_size - 1);
        
        int clamped_y = metal::clamp(offset, 0, height - 1);
        int clamped_x = metal::clamp(int(gid.x), 0, width - 1);
        
        sharedBlock[tid.x * cache_height + i] = in.read(uint2(clamped_x, clamped_y));
    }
    
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    
    //boundary con
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int pos = tid.x * cache_height + tid.y + gaus_kernel_size - 1;
    float4 mid = sharedBlock[pos] * gaus_kernel[0];
    float4 dot = mid;
    
    for (int i = 1; i < gaus_kernel_size; i++) {
        dot += (sharedBlock[pos - i] + sharedBlock[pos + i]) * gaus_kernel[i];
    }
    
    out[gid.x + gid.y * width] = uchar4(metal::saturate(dot) * 255 + 0.5f);
}
