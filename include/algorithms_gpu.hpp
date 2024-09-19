#ifndef ALGORITHMS_GPU_HPP
#define ALGORITHMS_GPU_HPP

#include "image.hpp"

#ifdef BENCHMARK_MODE
extern "C" struct GPUStats
{
    float host_to_device_time_cuda;
    float kernel_execution_time_cuda;
    float device_to_host_time_cuda;

    float host_to_device_time_chrono;
    float kernel_execution_time_chrono;
    float device_to_host_time_chrono;
};
#endif


#ifdef BENCHMARK_MODE
    extern "C" GPUStats convert_to_grayscale_gpu(Image& image);
#else
    extern "C" void convert_to_grayscale_gpu(Image& image);
#endif

#endif // ALGORITHMS_GPU_HPP
