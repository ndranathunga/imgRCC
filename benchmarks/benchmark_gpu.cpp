#include "image.hpp"
#include "algorithms_gpu.hpp"
#include <iostream>

int main()
{
    // Load the image for benchmarking
    Image image = load_image("../../input.png");

// Run the grayscale conversion on GPU
#ifdef BENCHMARK_MODE
    GPUStats stats = convert_to_grayscale_gpu(image);
    std::cout << "Host to device time (CUDA): " << stats.host_to_device_time_cuda << " ms" << std::endl;
    std::cout << "Kernel execution time (CUDA): " << stats.kernel_execution_time_cuda << " ms" << std::endl;
    std::cout << "Device to host time (CUDA): " << stats.device_to_host_time_cuda << " ms" << std::endl;

    std::cout << "Host to device time (Chrono): " << stats.host_to_device_time_chrono << " ms" << std::endl;
    std::cout << "Kernel execution time (Chrono): " << stats.kernel_execution_time_chrono << " ms" << std::endl;
    std::cout << "Device to host time (Chrono): " << stats.device_to_host_time_chrono << " ms" << std::endl;
#endif
    // Save the result
    save_image("output_benchmark.png", image);

    // Free the image
    free_image(image);

    return 0;
}
