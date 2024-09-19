#include "common.hpp"

void transfer_to_gpu(Image* image) {
    if (image->data == nullptr) return;

    // Allocate memory on the GPU
    unsigned char* d_data;
    size_t image_size = image->width * image->height * image->channels;
    cudaMalloc(&d_data, image_size);

    // Copy the image data from CPU to GPU
    cudaMemcpy(d_data, image->data, image_size, cudaMemcpyHostToDevice);

    // Update the image structure to point to the GPU data
    image->data = d_data;
}

void transfer_to_cpu(Image* image) {
    if (image->data == nullptr) return;

    // Allocate memory on the CPU
    unsigned char* h_data = new unsigned char[image->width * image->height * image->channels];

    // Copy the image data from GPU to CPU
    cudaMemcpy(h_data, image->data, image->width * image->height * image->channels, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(image->data);

    // Update the image structure to point to the CPU data
    image->data = h_data;
}