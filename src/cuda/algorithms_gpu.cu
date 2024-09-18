#include "algorithms_gpu.hpp"
#include <iostream>

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
    if (image.channels < 3)
    {
        std::cerr << "Image is not RGB or RGBA, cannot convert to grayscale." << std::endl;
        return;
    }

    // Allocate memory on the GPU
    unsigned char *d_data;
    cudaMalloc(&d_data, image.width * image.height * image.channels);
    // cudaMemcpy(d_data, image.data.data(), image.width * image.height * image.channels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, image.data, image.width * image.height * image.channels, cudaMemcpyHostToDevice);

    // Set up the grid and block dimensions
    int blockSize = 256;
    int gridSize = (image.width * image.height + blockSize - 1) / blockSize;

    // Launch the kernel
    grayscale_kernel<<<gridSize, blockSize>>>(d_data, image.width, image.height, image.channels);

    // Copy result back to host
    cudaMemcpy(image.data, d_data, image.width * image.height * image.channels, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_data);

    // std::cout << "Converted image to grayscale (GPU)." << std::endl;
}
