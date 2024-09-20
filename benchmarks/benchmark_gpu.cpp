#include "image.hpp"
#include "algorithms_gpu.hpp"
#include <iostream>
#include <chrono>

int main()
{
    // Load the image for benchmarking
    Image image = load_image_gpu("../../input.png");

// Run the grayscale conversion on GPU
// #ifdef BENCHMARK_MODE
//     GPUStats stats = convert_to_grayscale_gpu(image);
//     std::cout << "Host to device time (CUDA): " << stats.host_to_device_time_cuda << " ms" << std::endl;
//     std::cout << "Kernel execution time (CUDA): " << stats.kernel_execution_time_cuda << " ms" << std::endl;
//     std::cout << "Device to host time (CUDA): " << stats.device_to_host_time_cuda << " ms" << std::endl;

//     std::cout << "Host to device time (Chrono): " << stats.host_to_device_time_chrono << " ms" << std::endl;
//     std::cout << "Kernel execution time (Chrono): " << stats.kernel_execution_time_chrono << " ms" << std::endl;
//     std::cout << "Device to host time (Chrono): " << stats.device_to_host_time_chrono << " ms" << std::endl;
// #endif

    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    convert_to_grayscale_gpu(image);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms" << std::endl;


    // Save the result
    save_image_gpu("output_benchmark.png", image);

    // Free the image
    free_image_gpu(image);

    return 0;
}
