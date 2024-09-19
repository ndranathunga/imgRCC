#include "algorithms_gpu.hpp"
#include <iostream>
#include <chrono>

__global__ void grayscale_kernel(unsigned char *data, int width, int height, int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height)
        return;

    unsigned char r = data[idx * channels];
    unsigned char g = data[idx * channels + 1];
    unsigned char b = data[idx * channels + 2];

    unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

    data[idx * channels] = gray;
    data[idx * channels + 1] = gray;
    data[idx * channels + 2] = gray;
}

#ifdef BENCHMARK_MODE
GPUStats convert_to_grayscale_gpu(Image &image)
#else
void convert_to_grayscale_gpu(Image &image)
#endif
{
    if (image.channels < 3)
    {
        std::cerr << "Image is not RGB or RGBA, cannot convert to grayscale." << std::endl;
        exit(1);
    }

#ifdef BENCHMARK_MODE
    GPUStats stats = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    // Measure host-to-device transfer time
    unsigned char *d_data;

#ifdef BENCHMARK_MODE
    cudaEventRecord(start);
    auto start_ = std::chrono::high_resolution_clock::now();
#endif
    cudaError_t error = cudaMalloc(&d_data, image.width * image.height * image.channels);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (cudaMalloc): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
    error = cudaMemcpy(d_data, image.data, image.width * image.height * image.channels, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (cudaMemcpy HostToDevice): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
#ifdef BENCHMARK_MODE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaEventElapsedTime(&stats.host_to_device_time_cuda, start, stop);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        stats.host_to_device_time_cuda = -1.0f; // Assign a sentinel value if there's an error
    }
    auto stop_ = std::chrono::high_resolution_clock::now();
    stats.host_to_device_time_chrono = std::chrono::duration<float, std::milli>(stop_ - start_).count(); // Time in ms
#endif
    // Set up the grid and block dimensions
    int blockSize = 256;
    int gridSize = (image.width * image.height + blockSize - 1) / blockSize;

#ifdef BENCHMARK_MODE
    // Measure kernel execution time
    cudaEventRecord(start);
    start_ = std::chrono::high_resolution_clock::now();
#endif

    grayscale_kernel<<<gridSize, blockSize>>>(d_data, image.width, image.height, image.channels);

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();
#ifdef BENCHMARK_MODE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaEventElapsedTime(&stats.kernel_execution_time_cuda, start, stop);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        stats.host_to_device_time_cuda = -1.0f; // Assign a sentinel value if there's an error
    }
    stop_ = std::chrono::high_resolution_clock::now();
    stats.kernel_execution_time_chrono = std::chrono::duration<float, std::milli>(stop_ - start_).count(); // Time in ms

    // Measure device-to-host transfer time
    cudaEventRecord(start);
    start_ = std::chrono::high_resolution_clock::now();
#endif
    error = cudaMemcpy(image.data, d_data, image.width * image.height * image.channels, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (cudaMemcpy DeviceToHost): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
#ifdef BENCHMARK_MODE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    error = cudaEventElapsedTime(&stats.device_to_host_time_cuda, start, stop);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        stats.device_to_host_time_cuda = -1.0f; // Assign a sentinel value if there's an error
    }
    stop_ = std::chrono::high_resolution_clock::now();
    stats.device_to_host_time_chrono = std::chrono::duration<float, std::milli>(stop_ - start_).count(); // Time in ms
#endif

    // Free GPU memory
    cudaFree(d_data);

#ifdef BENCHMARK_MODE
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return stats;
#endif
}
