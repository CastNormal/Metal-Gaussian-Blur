#pragma once

#include <array>

#define MAX_KERNEL_SIZE 16

namespace {
struct GaussianKernel {
    std::array<float, MAX_KERNEL_SIZE> kernel;
    int size;
    };

inline void generateGaussianKernel(int sigma, GaussianKernel &gaus) {

    int radius = 3 * sigma;
    if (sigma < 1 || radius >= MAX_KERNEL_SIZE) throw std::invalid_argument("Invalid standard deviation");


    float sum = 0;
    gaus.size = radius + 1;

    for (int i = 0; i < gaus.size; i++) {
        float val = std::expf(-(i * i) / (2.0f * sigma * sigma));
        gaus.kernel[i] = val;
        sum += val;
        if (i > 0) sum += val; //sum twice due to symmetry
    }
    
    //norm
    for (int i = 0; i < gaus.size; i++) gaus.kernel[i] /= sum;
}

};
