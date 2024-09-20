#include "image.hpp"
#include "algorithms_gpu.hpp"
#include "algorithms_cpu.hpp"
#include "common.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
    exit(EXIT_FAILURE); \
}

int main()
{
    // Load the image for benchmarking
    Image *image_gpu = load_image_gpu("../../input.png");
    Image *image_cpu = load_image_cpu("../../input.png");
    Image *image_transfer = load_image_cpu("../../input.png");

    // Transfer image to GPU
    transfer_to_gpu(image_transfer);

    // Measure time for GPU grayscale
    auto start = std::chrono::high_resolution_clock::now();
    convert_to_grayscale_gpu(*image_gpu);  // Note the change to pass pointer
    cudaError_t err = cudaDeviceSynchronize();  // Ensure kernel finishes execution
    CUDA_CHECK_ERROR(err);  // Check if any CUDA error occurred

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time (GPU 1): " << elapsed.count() << " ms" << std::endl;

    // Measure time for CPU grayscale
    start = std::chrono::high_resolution_clock::now();
    convert_to_grayscale_cpu(*image_cpu);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time (CPU): " << elapsed.count() << " ms" << std::endl;

    // Measure time for GPU grayscale on transferred image
    start = std::chrono::high_resolution_clock::now();
    convert_to_grayscale_gpu(*image_transfer);  // Note the change to pass pointer
    err = cudaDeviceSynchronize();  // Ensure kernel finishes execution
    CUDA_CHECK_ERROR(err);  // Check for any CUDA error

    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Elapsed time (GPU 2): " << elapsed.count() << " ms" << std::endl;

    // Save the results
    save_image_gpu("output_gpu.png", image_gpu);
    save_image_cpu("output_cpu.png", image_cpu);
    save_image_gpu("output_gpu2.png", image_transfer);
    std::cout << "came here..." << std::endl;

    std::cout << "Images saved." << std::endl;

    // Free the images
    free_image_gpu(image_gpu);
    free_image_cpu(image_cpu);
    free_image_gpu(image_transfer);

    return 0;
}
