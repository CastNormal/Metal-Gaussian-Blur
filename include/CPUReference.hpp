#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "../include/GaussianKernel.hpp"
#include "ImageBuffer.hpp"

std::vector<std::uint8_t> CPUReferenceGaussianBlur(ImageBuffer &imageBuffer, const int sigma);
