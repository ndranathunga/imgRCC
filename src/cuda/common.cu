#include "common.hpp"
#include <iostream>

void checkCudaError(cudaError_t error, const char *message)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void transfer_to_gpu(Image *image)
{
    if (image->data == nullptr)
    {
        std::cerr << "Error: Image data is null, cannot transfer to GPU." << std::endl;
        return;
    }

    // Allocate memory on the GPU
    unsigned char *d_data;
    size_t image_size = image->width * image->height * image->channels;
    cudaError_t error = cudaMalloc(&d_data, image_size);
    checkCudaError(error, "cudaMalloc failed during transfer_to_gpu");

    // Copy the image data from CPU to GPU
    error = cudaMemcpy(d_data, image->data, image_size, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy HostToDevice failed during transfer_to_gpu");

    // Update the image structure to point to the GPU data
    image->data = d_data;
}

void transfer_to_cpu(Image *image)
{
    if (image->data == nullptr)
    {
        std::cerr << "Error: Image data is null, cannot transfer to CPU." << std::endl;
        return;
    }

    // Allocate memory on the CPU
    unsigned char *h_data = new unsigned char[image->width * image->height * image->channels];

    // Copy the image data from GPU to CPU
    cudaError_t error = cudaMemcpy(h_data, image->data, image->width * image->height * image->channels, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy DeviceToHost failed during transfer_to_cpu");

    // Free the GPU memory
    error = cudaFree(image->data);
    checkCudaError(error, "cudaFree failed during transfer_to_cpu");

    // Update the image structure to point to the CPU data
    image->data = h_data;
}
