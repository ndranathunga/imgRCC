#include "image.hpp"
#include <iostream>
#include <stdexcept>

Image *load_image_gpu(const char *file_path)
{
    // Load the image on CPU first
    Image *image = load_image_cpu(file_path);

    // Allocate memory on the GPU
    unsigned char *d_data;
    size_t image_size = image->width * image->height * image->channels;
    cudaError_t error = cudaMalloc(&d_data, image_size);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaMalloc): " + std::string(cudaGetErrorString(error)));
    }

    // Copy image data from CPU to GPU
    error = cudaMemcpy(d_data, image->data, image_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaMemcpy): " + std::string(cudaGetErrorString(error)));
    }

    // Free the CPU data
    free_image_cpu(image);

    // Update the image structure to point to GPU data
    image->data = d_data;
    return image;
}

void save_image_gpu(const char *file_path, const Image *image)
{
    // Allocate memory on the CPU to receive the image from GPU
    unsigned char *h_data = new unsigned char[image->width * image->height * image->channels];

    // Copy the image data from GPU to CPU
    cudaError_t error = cudaMemcpy(h_data, image->data, image->width * image->height * image->channels, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        delete[] h_data;
        throw std::runtime_error("CUDA Error (cudaMemcpy DeviceToHost): " + std::string(cudaGetErrorString(error)));
    }

    // Save the image from CPU memory
    Image temp_image = {image->width, image->height, image->channels, h_data};
    save_image_cpu(file_path, &temp_image);

    // Free the temporary CPU memory
    delete[] h_data;
}

void free_image_gpu(Image *image)
{
    if (image->data == nullptr)
    {
        std::cerr << "Error: Image data is null, cannot free GPU memory." << std::endl;
        exit(1);
    }
    cudaError_t error = cudaFree(image->data);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("CUDA Error (cudaFree): " + std::string(cudaGetErrorString(error)));
    }
}