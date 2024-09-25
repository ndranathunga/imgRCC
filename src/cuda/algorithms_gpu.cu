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

__global__ void convolve_kernel(unsigned char *input, unsigned char *output, int width, int height, int channels, const float *kernel, int kernel_width, int kernel_height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int kx_offset = kernel_width / 2;
    int ky_offset = kernel_height / 2;

    // Iterate over channels
    for (int c = 0; c < channels; c++)
    {
        float pixel_sum = 0.0f;

        // Apply kernel to the pixel and its neighbors
        for (int ky = 0; ky < kernel_height; ky++)
        {
            for (int kx = 0; kx < kernel_width; kx++)
            {
                int img_x = x + kx - kx_offset;
                int img_y = y + ky - ky_offset;

                // Ensure indices are within bounds
                if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height)
                {
                    int img_idx = (img_y * width + img_x) * channels + c;
                    int kernel_idx = ky * kernel_width + kx;
                    pixel_sum += input[img_idx] * kernel[kernel_idx];
                }
            }
        }

        // Clamp result to 0-255 and assign to output image
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = min(max(static_cast<int>(pixel_sum), 0), 255);
    }
}

__device__ void rgb_to_hsv_device(unsigned char r, unsigned char g, unsigned char b, float *h, float *s, float *v)
{
    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;

    float max_val = fmaxf(fmaxf(fr, fg), fb);
    float min_val = fminf(fminf(fr, fg), fb);
    float delta = max_val - min_val;

    // Calculate Value (V)
    *v = max_val;

    // Calculate Saturation (S)
    if (max_val == 0.0f)
    {
        *s = 0.0f;
    }
    else
    {
        *s = delta / max_val;
    }

    // Calculate Hue (H)
    if (delta == 0.0f)
    {
        *h = 0.0f; // Undefined hue
    }
    else if (max_val == fr)
    {
        *h = 60.0f * fmodf((fg - fb) / delta, 6.0f);
    }
    else if (max_val == fg)
    {
        *h = 60.0f * (((fb - fr) / delta) + 2.0f);
    }
    else
    {
        *h = 60.0f * (((fr - fg) / delta) + 4.0f);
    }

    if (*h < 0.0f)
    {
        *h += 360.0f;
    }
}

__global__ void rgb_to_hsv_kernel(unsigned char *input, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    float h, s, v;
    rgb_to_hsv_device(r, g, b, &h, &s, &v);

    input[idx] = static_cast<unsigned char>(h / 360.0f * 255);
    input[idx + 1] = static_cast<unsigned char>(s * 255);
    input[idx + 2] = static_cast<unsigned char>(v * 255);
}

__device__ void rgb_to_ycbcr_device(unsigned char r, unsigned char g, unsigned char b, unsigned char *y, unsigned char *cb, unsigned char *cr)
{
    *y = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    *cb = static_cast<unsigned char>(128 + (-0.168736f * r - 0.331264f * g + 0.5f * b));
    *cr = static_cast<unsigned char>(128 + (0.5f * r - 0.418688f * g - 0.081312f * b));
}

__global__ void rgb_to_ycbcr_kernel(unsigned char *input, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = (y * width + x) * channels;

    unsigned char r = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char b = input[idx + 2];

    unsigned char y_val, cb_val, cr_val;
    rgb_to_ycbcr_device(r, g, b, &y_val, &cb_val, &cr_val);

    input[idx] = y_val;
    input[idx + 1] = cb_val;
    input[idx + 2] = cr_val;
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
        std::cerr << "CUDA Error (kernel[grayscale_kernel] launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[grayscale_kernel] execution): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void convolve_image_gpu(Image *image, const float *kernel, int kernel_width, int kernel_height)
{
    int img_size = image->width * image->height * image->channels;

    unsigned char *d_input = image->data;
    unsigned char *d_output;
    float *d_kernel;

    // Allocate memory for output and kernel on the GPU
    cudaMalloc(&d_output, img_size * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernel_width * kernel_height * sizeof(float));

    // Copy kernel data to the device
    cudaMemcpy(d_kernel, kernel, kernel_width * kernel_height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);

    convolve_kernel<<<gridDim, blockDim>>>(d_input, d_output, image->width, image->height, image->channels, d_kernel, kernel_width, kernel_height);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[convolve_kernel] launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[convolve_kernel] execution): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    // Free the original image data and update the image to point to the result in GPU memory
    cudaFree(image->data);
    image->data = d_output;

    // Free the kernel memory
    cudaFree(d_kernel);
}

void convert_image_rgb_to_hsv_gpu(Image *image)
{
    int img_size = image->width * image->height * image->channels;
    unsigned char *d_input = image->data;

    dim3 blockDim(16, 16);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);

    rgb_to_hsv_kernel<<<gridDim, blockDim>>>(d_input, image->width, image->height, image->channels);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[rgb_to_hsv_kernel] launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[rgb_to_hsv_kernel] execution): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void convert_image_rgb_to_ycbcr_gpu(Image *image)
{
    int img_size = image->width * image->height * image->channels;
    unsigned char *d_input = image->data;

    // Set the grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x, (image->height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    rgb_to_ycbcr_kernel<<<gridDim, blockDim>>>(d_input, image->width, image->height, image->channels);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[rgb_to_ycbcr_kernel] launch): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error (kernel[rgb_to_ycbcr_kernel] execution): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}
