#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <array>
#include <fstream>
#include <memory>
#include <filesystem>
#include <iostream>
#include "../include/GaussianKernel.hpp"
#include "../include/GaussianFilter.hpp"
#include "NaiveGaussianFilter.hpp"
#include "../include/ImageBuffer.hpp"
#include "../include/MetalContext.hpp"
#include "../include/CPUReference.hpp"
#include "../include/QOI.h"

struct BenchmarkConfig {
    int sigma = 5;
    int gpuWarmups = 10;
    int cpuWarmups = 2;
    int gpuIterations = 100;
    int cpuIterations = 10;
};

struct BenchmarkResult {
    int iteration;
    double gpuTime, cpuTime;
};

class Benchmark {
public:
    std::filesystem::path outputDirectory;
    
    
    Benchmark(const std::string &inputImage);
    
    std::vector<BenchmarkResult> runNaive(const BenchmarkConfig &config);
    std::vector<BenchmarkResult> runTiled(const BenchmarkConfig &config);
    std::vector<BenchmarkResult> runCPU(const BenchmarkConfig &config);

    
    ImageBuffer imageBuffer;
        
private:
    MetalContext context;
        
    template<typename Filter> std::vector<BenchmarkResult> runGPU(const BenchmarkConfig &config);
    
};

Benchmark::Benchmark(const std::string &inputImage) {
    imageBuffer = loadQOIImage(inputImage.c_str());
}

std::vector<BenchmarkResult> Benchmark::runNaive(const BenchmarkConfig &config) {
    return runGPU<NaiveGaussianFilter>(config);
}

std::vector<BenchmarkResult> Benchmark::runTiled(const BenchmarkConfig &config) {
    return runGPU<GaussianFilter>(config);
}

std::vector<BenchmarkResult> Benchmark::runCPU(const BenchmarkConfig &config) {
    std::vector<BenchmarkResult> benchmarks;
    
    for (int i = -config.cpuWarmups; i < config.cpuIterations; i++) {
        const auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::uint8_t> output = CPUReferenceGaussianBlur(imageBuffer, config.sigma);
        
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        
        BenchmarkResult result{ .iteration = i+1, .cpuTime = elapsed.count() };
        if (i >= 0) benchmarks.push_back(result);
    }
    
    return benchmarks;
}

template<typename Filter> std::vector<BenchmarkResult> Benchmark::runGPU(const BenchmarkConfig &config) {
    std::vector<BenchmarkResult> benchmarks;
    const size_t size = imageBuffer.headerInfo.width * imageBuffer.headerInfo.height * imageBuffer.headerInfo.channels;

    Filter filter(context.device(), static_cast<int>(imageBuffer.headerInfo.width), static_cast<int>(imageBuffer.headerInfo.height));
    
    auto buffer = NS::TransferPtr(context.device()->newBuffer(imageBuffer.image.get(), size, MTL::ResourceStorageModeShared));
    
    for (int i = -config.gpuWarmups; i < config.gpuIterations; i++) {
        
        std::memcpy(buffer->contents(), imageBuffer.image.get(), size);
        
        MTL::CommandBuffer* commandBuffer = context.commandQueue()->commandBuffer();
        
        if (!commandBuffer)
            throw std::runtime_error("failed to create command buffer.");
        
        filter.encode(commandBuffer, buffer.get(), config.sigma);
        
        const auto start = std::chrono::steady_clock::now();

        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();

        const auto end = std::chrono::steady_clock::now();

        if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
            auto* error = commandBuffer->error();
            
            if (error && error->localizedDescription())
                throw std::runtime_error(error->localizedDescription()->utf8String());
            
            throw std::runtime_error("GPU execution failed");
        }
        
        const double cpuTime = std::chrono::duration<double, std::milli>(end - start).count();
        
        // Actual GPU execution timing
        const double gpuStart = commandBuffer->GPUStartTime();
        const double gpuEnd = commandBuffer->GPUEndTime();
        
        if (gpuStart > 0.0 && gpuEnd > gpuStart) {
            const double gpuTime = (gpuEnd - gpuStart) * 1000.0f;
            
            BenchmarkResult result{ .iteration = i+1, .gpuTime = gpuTime, .cpuTime = cpuTime };
            if (i >= 0) benchmarks.push_back(result);
        } else std::cout << "GPU timing unavailable.\n";
    }
    
    return benchmarks;
}

void writeGPUResults(const std::string &output, const std::vector<BenchmarkResult> &results, const std::pair<int, int> dimensions, int sigma) {

    std::ofstream csv(output.c_str());

    if (!csv.is_open())
        throw std::runtime_error("Failed to open benchmark CSV");
    
    csv << std::fixed << std::setprecision(6);
    
    csv << "iteration,width,height,sigma,gpu_time_ms,cpu_wait_ms\n";

    for (BenchmarkResult result : results) {
        csv << result.iteration << ',' << dimensions.first << ',' << dimensions.second << ',' << sigma << ',' << result.gpuTime << ',' << result.cpuTime << std::endl;
    }
}

void writeCPUResults(const std::string &output, const std::vector<BenchmarkResult> &results, const std::pair<int, int> dimensions, int sigma) {

    std::ofstream csv(output.c_str());

    if (!csv.is_open())
        throw std::runtime_error("Failed to open benchmark CSV");
    
    csv << std::fixed << std::setprecision(6);
    
    csv << "iteration,width,height,sigma,cpu_time_ms\n";

    for (BenchmarkResult result : results) {
        csv << result.iteration << ',' << dimensions.first << ',' << dimensions.second << ',' << sigma << ',' << result.cpuTime << std::endl;
    }
}


std::pair<BenchmarkConfig, std::string> parseInput(int argc, char** argv) {
    BenchmarkConfig config {
        .sigma = 5,
        .gpuWarmups = 10,
        .cpuWarmups = 2,
        .gpuIterations = 100,
        .cpuIterations = 10
    };

    std::string image = "./assets/test.qoi";
    
    if (--argc > 0) image = std::string(argv[1]);
    if (!std::filesystem::exists(image)) throw std::invalid_argument(std::string("failed to open file: ") + std::filesystem::current_path().string());
    
    if (--argc > 0) config.sigma = std::atoi(argv[2]);
    if (config.sigma < 1 || config.sigma > 5) throw std::invalid_argument("Invalid sigma value");
    
    if (--argc > 0) config.gpuIterations = std::atoi(argv[3]);
    if (config.gpuIterations < 0) throw std::invalid_argument("Invalid gpu iterations");
    
    if (--argc > 0) config.cpuIterations = std::atoi(argv[4]);
    if (config.gpuIterations < 0) throw std::invalid_argument("Invalid cpu iterations");
    
    if (argc > 1) throw std::invalid_argument("Usage: benchmark [image.qoi] [sigma 1..5] [GPU runs] [CPU runs]");
    
    return std::pair<BenchmarkConfig, std::string>(config, image);
}

int main(int argc, char** argv) {
    
    const auto [config, imagePath] = parseInput(argc, argv);
    
    Benchmark benchmark(imagePath);
    
    const auto naive = benchmark.runNaive(config);
    const auto tiled = benchmark.runTiled(config);
    const auto cpu = benchmark.runCPU(config);

    writeGPUResults("benchmark/results/naive.csv", naive, std::pair<int, int>(benchmark.imageBuffer.headerInfo.width, benchmark.imageBuffer.headerInfo.height), config.sigma);
    writeGPUResults("benchmark/results/tiled.csv", tiled, std::pair<int, int>(benchmark.imageBuffer.headerInfo.width, benchmark.imageBuffer.headerInfo.height), config.sigma);
    writeCPUResults("benchmark/results/cpu.csv", cpu, std::pair<int, int>(benchmark.imageBuffer.headerInfo.width, benchmark.imageBuffer.headerInfo.height), config.sigma);
    return 0;
}
