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

void convert_to_grayscale_gpu(Image &image)
{
    // Check if image data is valid
    if (!image.data)
    {
        std::cerr << "Error: Image data pointer is null." << std::endl;
        exit(1);
    }

    // Check if dimensions are valid
    if (image.width <= 0 || image.height <= 0)
    {
        std::cerr << "Error: Invalid image dimensions. Width: " << image.width
                  << ", Height: " << image.height << std::endl;
        exit(1);
    }

    // Check if channels are valid
    if (image.channels < 3)
    {
        std::cerr << "Error: Image channels must be at least 3 (RGB). Channels: " << image.channels << std::endl;
        exit(1);
    }

    // Set up the grid and block dimensions
    int blockSize = 256;
    int gridSize = (image.width * image.height + blockSize - 1) / blockSize;

    if (gridSize <= 0)
    {
        std::cerr << "Error: Invalid grid size. Grid size: " << gridSize << std::endl;
        exit(1);
    }

    grayscale_kernel<<<gridSize, blockSize>>>(image.data, image.width, image.height, image.channels);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();
}
