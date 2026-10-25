#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <gtest/gtest.h>

#include <array>
#include "../include/GaussianKernel.hpp"

TEST(GaussianKernel, KernelSumsToOne) {
    GaussianKernel kernel;
    
    constexpr std::array<int, 5> testSigmas = { 1, 2, 3, 4, 5 };
    
    for (const int sigma : testSigmas) {
        generateGaussianKernel(sigma, kernel);

        float sum = kernel.kernel[0];
        
        for (int i = 1; i < kernel.size; ++i) {
            sum += 2.0f * kernel.kernel[i];
        }

        EXPECT_NEAR(sum, 1.0f, 1e-6f) << "sigma = " << sigma;
    }
}

TEST(GaussianKernel, CorrectProperties) {
    GaussianKernel kernel;
    
    constexpr std::array<int, 3> testSigmas = { 1, 2, 3 };
    
    for (const int sigma : testSigmas) {
        generateGaussianKernel(sigma, kernel);

        EXPECT_EQ(kernel.size, sigma*3+1);

        EXPECT_EQ(kernel.kernel.size(), MAX_KERNEL_SIZE);
        
        for (int i = 1; i < kernel.size; ++i) {
            EXPECT_TRUE(std::isfinite(kernel.kernel[i]));
            EXPECT_GE(kernel.kernel[i], 0);
            EXPECT_LT(kernel.kernel[i], kernel.kernel[i - 1]);
        }
    }
}

TEST(GaussianKernel, InvalidSigma) {
    GaussianKernel kernel;
    
    EXPECT_THROW(generateGaussianKernel(-1, kernel), std::invalid_argument);
    
    EXPECT_THROW(generateGaussianKernel(0, kernel), std::invalid_argument);
    
    EXPECT_THROW(generateGaussianKernel(6, kernel), std::invalid_argument);
}
