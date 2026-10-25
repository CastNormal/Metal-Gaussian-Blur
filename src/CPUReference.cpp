#include "../include/CPUReference.hpp"

std::vector<std::uint8_t> CPUReferenceGaussianBlur(ImageBuffer &imageBuffer, const int sigma) {
    
    const auto* input = imageBuffer.image.get();
    GaussianKernel gaus;
    
    generateGaussianKernel(sigma, gaus);
    
    int width = static_cast<int>(imageBuffer.headerInfo.width);
    int height = static_cast<int>(imageBuffer.headerInfo.height);
    
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<std::array<float, 4>> intermediate(pixels);
    std::vector<std::uint8_t> output(pixels * 4);
    
    //horizontal pass
    const int radius = gaus.size - 1;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                auto& target = intermediate[static_cast<std::size_t>(y) * width + x];
                for (int channel = 0; channel < 4; ++channel) {
                    float sum = 0.0f;
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int sx = std::clamp(x + dx, 0, width - 1);
                        const auto index = (static_cast<std::size_t>(y) * width + sx) * 4 + channel;
                        sum += (input[index] / 255.0f) * gaus.kernel[std::abs(dx)];
                    }
                    target[channel] = sum;
                }
            }
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int channel = 0; channel < 4; ++channel) {
                    float sum = 0.0f;
                    for (int dy = -radius; dy <= radius; ++dy) {
                        const int sy = std::clamp(y + dy, 0, height - 1);
                        sum += intermediate[static_cast<std::size_t>(sy) * width + x][channel] *
                               gaus.kernel[std::abs(dy)];
                    }
                    const auto index = (static_cast<std::size_t>(y) * width + x) * 4 + channel;
                    output[index] = static_cast<std::uint8_t>(
                        std::clamp(std::floor(sum * 255.0f + 0.5f), 0.0f, 255.0f));
                }
            }
        }
        return output;
}
