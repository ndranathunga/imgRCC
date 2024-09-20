#include "common.hpp"
#include <iostream>

void transfer_to_gpu(Image *image)
{
    if (image->data == nullptr)
    {
        std::cerr << "Error: Image data is null, cannot transfer to GPU." << std::endl;
        exit(1);
    }

    // Allocate memory on the GPU
    unsigned char *d_data;
    size_t image_size = image->width * image->height * image->channels;
    cudaError_t error = cudaMalloc(&d_data, image_size);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaMalloc): " + std::string(cudaGetErrorString(error)));
    }

    // Copy the image data from CPU to GPU
    error = cudaMemcpy(d_data, image->data, image_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaMemcpy HostToDevice): " + std::string(cudaGetErrorString(error)));
    }

    // Free the CPU memory
    free_image_cpu(image);

    // Update the image structure to point to the GPU data
    image->data = d_data;
}

void transfer_to_cpu(Image *image)
{
    if (image->data == nullptr)
    {
        std::cerr << "Error: Image data is null, cannot transfer to CPU." << std::endl;
        exit(1);
    }

    // Allocate memory on the CPU
    unsigned char *h_data = new unsigned char[image->width * image->height * image->channels];

    // Copy the image data from GPU to CPU
    cudaError_t error = cudaMemcpy(h_data, image->data, image->width * image->height * image->channels, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaMemcpy DeviceToHost): " + std::string(cudaGetErrorString(error)));
    }

    // Free the GPU memory
    error = cudaFree(image->data);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaFree): " + std::string(cudaGetErrorString(error)));
    }

    // Update the image structure to point to the CPU data
    image->data = h_data;
}
