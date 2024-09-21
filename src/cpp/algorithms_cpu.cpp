#include <iostream>
#include <cstring>
#include "algorithms_cpu.hpp"

void convert_to_grayscale_cpu(Image &image)
{
    if (image.channels < 3)
    {
        std::cerr << "Image is not RGB or RGBA, cannot convert to grayscale." << std::endl;
        return;
    }

    for (int i = 0; i < image.width * image.height; i++)
    {
        unsigned char r = image.data[i * image.channels];
        unsigned char g = image.data[i * image.channels + 1];
        unsigned char b = image.data[i * image.channels + 2];

        // Grayscale formula: 0.299 * R + 0.587 * G + 0.114 * B
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        // Set all channels to the grayscale value
        image.data[i * image.channels] = gray;
        image.data[i * image.channels + 1] = gray;
        image.data[i * image.channels + 2] = gray;
    }

    // std::cout << "Converted image to grayscale (CPU)." << std::endl;
}

void convolve_image_cpu(Image *image, const float *kernel, int kernel_width, int kernel_height)
{
    int img_width = image->width;
    int img_height = image->height;
    int channels = image->channels;

    // Allocate memory for the output image
    unsigned char *output = new unsigned char[img_width * img_height * channels];

    int kx_offset = kernel_width / 2;
    int ky_offset = kernel_height / 2;

    // Iterate over every pixel in the image
    for (int y = 0; y < img_height; y++)
    {
        for (int x = 0; x < img_width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                float pixel_sum = 0.0f;

                // Iterate over kernel
                for (int ky = 0; ky < kernel_height; ky++)
                {
                    for (int kx = 0; kx < kernel_width; kx++)
                    {
                        int img_x = x + kx - kx_offset;
                        int img_y = y + ky - ky_offset;

                        // Ensure indices are within bounds
                        if (img_x >= 0 && img_x < img_width && img_y >= 0 && img_y < img_height)
                        {
                            int img_idx = (img_y * img_width + img_x) * channels + c;
                            int kernel_idx = ky * kernel_width + kx;
                            pixel_sum += image->data[img_idx] * kernel[kernel_idx];
                        }
                    }
                }

                // Clamp result to 0-255 and assign to output image
                int output_idx = (y * img_width + x) * channels + c;
                output[output_idx] = std::min(std::max(static_cast<int>(pixel_sum), 0), 255);
            }
        }
    }

    // Copy the result back to the original image data
    std::memcpy(image->data, output, img_width * img_height * channels);
    delete[] output;
}