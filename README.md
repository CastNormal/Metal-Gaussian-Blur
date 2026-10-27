# Metal Gaussian Blur

Separable Gaussian image blur on Apple silicon, implemented in **C++20**, **Metal-cpp**, and **Metal Shading Language (MSL)**. The repository contains a single-threaded CPU reference, a naïve GPU implementation, and a GPU implementation that cooperatively caches image tiles and their halo pixels in threadgroup memory.

The focus is not merely to accelerate an image filter: it is to **test when threadgroup-memory tiling outperforms direct GPU sampling**, and to record repeatable measurements for both implementations.

## How it works

Gaussian convolution is separable: instead of applying a 2D kernel directly, the image is filtered once horizontally and once vertically. A Gaussian with standard deviation σ uses a truncated **radius of 3σ** (`sigma` is an integer in `[1, 5]`). All three versions use the same normalized one-dimensional weights and **clamp-to-edge** boundary handling.

1. **CPU reference:** two scalar convolution passes, with floating-point intermediate pixels and RGBA8 output.
2. **Naïve GPU:** one Metal compute pass per axis, reading neighbors directly from a device buffer (horizontal) or RGBA32Float texture (vertical).
3. **Tiled GPU:** identical two-pass structure, but threads cooperatively load tiles and halo samples into threadgroup memory, synchronize, and reuse those cached pixels during convolution.

The GPU pipelines use an **RGBA32Float intermediate texture** and produce RGBA8 results. The horizontal dispatch uses **32 × 8** threads per threadgroup, and the vertical dispatch uses **8 × 32**. Both the naïve and tiled GPU shaders use these dimensions for the initial controlled comparison.

### Implementation comparison

| | CPU | Naïve GPU | Tiled GPU |
|---|---|---|---|
| Separable horizontal + vertical blur | Yes | Yes | Yes |
| Gaussian coefficients / edge clamping | Shared | Shared | Shared |
| Intermediate representation | Float RGBA | RGBA32Float texture | RGBA32Float texture |
| Explicit shared/threadgroup-memory tiling | No | No | Yes |
| Cooperative halo loads and barriers | No | No | Yes |

### Project structure

```text
include/            QOI image codec, image buffer, kernel generation, Metal interfaces
src/                Application, CPU reference, Metal context, tiled GPU filter
shaders/shader.metal Tiled horizontal and vertical Metal kernels
benchmark/          Naïve GPU filter, shader, benchmark runner, CSV analysis
benchmark/previous_results/  Preliminary measurements from the original archive
tests/              GoogleTest cases for CPU, GPU, kernel coefficients, and QOI
assets/             QOI test images, including 418×418 and 2528×4497 images
metal-cpp-main/     Vendored Metal-cpp C++ wrapper headers and license
```

## Requirements

- macOS with an Apple Metal-capable GPU (developed for an Apple **M1**)
- Xcode **16 or newer**, with the command-line tools installed
- C++20 and the macOS Metal, Foundation, and QuartzCore frameworks (provided by Xcode)
- For unit tests: `brew install googletest`; the Xcode test target searches the stable Homebrew `opt/googletest` paths on Apple silicon and Intel Macs.

The bundled `metal-cpp-main/` folder provides the required Apple Metal-cpp headers. Its license is in `metal-cpp-main/LICENSE.txt`.

### Build and run

1. Open `Metal Gaussian Blur.xcodeproj` in Xcode.
2. Select the **Metal Gaussian Blur** scheme and build it.
3. Select **Gaussian Tests** and run its executable (or run its tests from Xcode).
4. Select **Benchmark Gaussian Blur** to record timings. Its Run scheme is configured for **Release** rather than Debug.

The shared Run schemes use `$(SRCROOT)` as the working directory so paths such as `./assets/test.qoi` resolve from the repository root. If your version of Xcode doesn't resolve that macro, set the Run scheme's **Working Directory** to the repository root (Product → Scheme → Edit Scheme → Run → Options).

The main executable currently loads `./assets/test3.qoi` and writes `output.qoi`; edit `src/main.cpp` to change the source image or standard deviation. The output is ignored by Git.

**Build note:** The uploaded Xcode project is included and no longer Git-ignored. The obsolete CMakeLists.txt was intentionally removed because it combined multiple `main()` programs and test sources into a single target. Xcode is the supported build route.

## Correctness testing

`tests/` contains checks for:

- Gaussian coefficient normalization and invalid standard deviations
- CPU constant-color, single-pixel, and impulse-response behavior
- **Both GPU implementations versus the CPU** at seven image sizes (`1×1`, `5×7`, `32×8`, `8×32`, `33×9`, `35×37`, `64×65`) and all five σ values, allowing at most **1 RGBA8 level** of floating-point/rounding difference
- QOI lossless random RGBA round-trip and RGB channel metadata
- Malformed QOI headers, truncated streams, oversized runs, invalid termination markers, and invalid encoder metadata

The custom QOI **decoder always outputs RGBA8**, even for QOI images whose header declares three channels. The **encoder currently accepts RGBA8-stride input only**; when writing a QOI with RGB metadata, pass opaque-alpha RGBA8 pixels. This is a deliberate interface restriction, not support for packed RGB input.

QOI decoding limits images to **100 million pixels** (maximum 400 MB of decoded RGBA8 data) to avoid unreasonable allocations and integer-overflow hazards. It page-aligns the decoded buffer for an optional no-copy Metal wrapping path. The QOI reader does not close the caller-owned `FILE*`.

## Benchmarking

The benchmark runner records **raw CSV timings**, not just summary statistics. It restores the same input pixels before every GPU iteration using `memcpy()` from a separate, unchanged image allocation into a reused Metal-owned buffer. Neither buffer allocation nor input copying is included in the GPU execution timestamps.

Defaults:

- **GPU:** 10 warm-up runs + 100 measured runs per implementation
- **CPU:** 2 warm-up runs + 10 measured runs
- **Metric:** `MTL::CommandBuffer::GPUStartTime()` / `GPUEndTime()` for GPU command-buffer execution, and `std::chrono::steady_clock` for CPU-function wall time
- **Output:** `benchmark/results/naive.csv`, `tiled.csv`, and `cpu.csv`, one measurement per row (`iteration,width,height,sigma,time_ms (gpu and cpu)`)

Optional arguments for the benchmark executable:

```text
Benchmark Gaussian Blur [image.qoi] [sigma 1..5] [gpu_runs] [cpu_runs]
```

For example, to benchmark the large bundled image in Release, choose the Benchmark Gaussian Blur scheme, then supply the following Run arguments:

```text
./assets/test3.qoi 5 100 20
```

### Archived preliminary results

The original archive contained 100-run CSVs for a **418 × 418**, σ = 5 experiment. Those results are preserved under `benchmark/results/` and have the following **Debug-scheme configuration**:

| Implementation | Median time |
|---|---:|
| CPU reference (wall-clock) | 40.87 ms |
| Naïve GPU (GPU timestamp) | 0.527 ms |
| Tiled GPU (GPU timestamp) | 0.476 ms
For this small image, the naïve GPU was approximately **1.27× faster** than tiled. A separate earlier set of user-collected measurements for **2528 × 4497**, σ = 5 is preserved under `benchmark/results/`: the naïve GPU median was **0.527 ms** and the tiled GPU median was **0.476 ms** (approximately **9.7% faster** when tiled). The large-image CPU dataset contains only a single run; the build configuration for that older measurement was not independently verified.

This time, rather than using `assets/test.qoi` (418 × 418), the benchmark was ran with `assets/test3.qoi` (2528 × 4497), again with σ = 5. The improvement difference become more concrete with a bigger image. This time the naïve GPU median was **19.88 ms** and the tiled GPU median was **14.27 ms** which is approixmately **28.2% faster** when tiled.

| Implementation | Median time |
| --- | --- |
| CPU reference (wall-clock) | 2625 ms |
| Naïve GPU (GPU timestamp) | 19.88 ms |
| Tiled GPU (GPU timestamp) | 14.27 ms |

## Limitations and future improvements

- Requires Metal and macOS; GPU tests and benchmarks do not run on Linux.
- QOI codec input is deliberately RGBA8-stride only, even when encoding RGB metadata.
- The CPU baseline is a straightforward scalar reference, **not** an optimized multicore CPU implementation.
- Buffer transfer, image decoding, and file output are excluded from shader-only GPU timings. Measure end-to-end latency separately for application comparisons.
- Future work: threadgroup-size tuning, GPU profiler captures, additional image resolutions, and automated batch analysis.

## Attribution

The `metal-cpp-main` directory contains Apple's Metal-cpp headers and licensing information. All custom Gaussian blur and QOI code resides outside that vendored directory.
