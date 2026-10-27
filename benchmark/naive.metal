#include <metal_stdlib>

[[kernel]] void naive_gaussian_blur_horizontal(
                               const device uchar4 *in                          [[buffer(0)]],
                               constant float *gaussianKernel                   [[buffer(1)]],
                               constant int &gaussianKernelSize                 [[buffer(2)]],
                               metal::texture2d<float, metal::access::write> out[[texture(0)]],
                               uint2 gid                                        [[thread_position_in_grid]]) {
    
    int width = out.get_width();
    int height = out.get_height();
    
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    int pos = gid.y * width + gid.x;
    float4 mid = float4(in[pos]) * gaussianKernel[0];
    float4 weighted_sum = mid;
    
    for (int i = 1; i < gaussianKernelSize; i++) {
        const int clamped_left = metal::clamp(int(gid.x)-i, 0, width - 1);
        const int clamped_right = metal::clamp(int(gid.x)+i, 0, width - 1);
        weighted_sum += (float4(in[clamped_left + gid.y * width]) + float4(in[clamped_right + gid.y * width])) * gaussianKernel[i];
    }
    
    out.write(weighted_sum /255.0f, gid);
}

[[kernel]] void naive_gaussian_blur_vertical(
                                metal::texture2d<float, metal::access::read> in         [[texture(0)]],
                                device uchar4 *out                                      [[buffer(0)]],
                                constant float *gaussianKernel                          [[buffer(1)]],
                                constant int &gaussianKernelSize                        [[buffer(2)]],
                                uint2 gid                                               [[thread_position_in_grid]]) {
    int width = in.get_width();
    int height = in.get_height();
    
    if (gid.x >= uint(width) || gid.y >= uint(height)) return;
    
    float4 mid = in.read(uint2(gid.x, gid.y)) * gaussianKernel[0];
    float4 weighted_sum = mid;
    
    for (int i = 1; i < gaussianKernelSize; i++) {
        const uint clamped_up = metal::clamp(int(gid.y)-i, 0, height - 1);
        const uint clamped_down = metal::clamp(int(gid.y)+i, 0, height - 1);
        
        weighted_sum += (in.read(uint2(gid.x, clamped_up)) + in.read(uint2(gid.x, clamped_down))) * gaussianKernel[i];
    }
    
    out[gid.x + gid.y * width] = uchar4(metal::saturate(weighted_sum) * 255 + 0.5f);
}
