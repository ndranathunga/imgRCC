#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
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

// Converts the entire image from RGB to HSV
void convert_image_rgb_to_hsv_cpu(Image *image)
{
    int width = image->width;
    int height = image->height;
    int channels = image->channels;

    if (channels < 3)
        return; // Image must have at least 3 channels (RGB)

    for (int i = 0; i < width * height; ++i)
    {
        unsigned char r = image->data[i * channels];
        unsigned char g = image->data[i * channels + 1];
        unsigned char b = image->data[i * channels + 2];

        float h, s, v;
        rgb_to_hsv(r, g, b, &h, &s, &v);

        // Here we can store the HSV values in separate channels if required
        image->data[i * channels] = static_cast<unsigned char>(h / 360.0f * 255);
        image->data[i * channels + 1] = static_cast<unsigned char>(s * 255);
        image->data[i * channels + 2] = static_cast<unsigned char>(v * 255);
    }
}

// Converts a single RGB pixel to HSV
void rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b, float *h, float *s, float *v)
{
    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;

    float max_val = std::max({fr, fg, fb});
    float min_val = std::min({fr, fg, fb});
    float delta = max_val - min_val;

    // Calculate Value (V)
    *v = max_val;

    // Calculate Saturation (S)
    if (max_val == 0)
    {
        *s = 0;
    }
    else
    {
        *s = delta / max_val;
    }

    // Calculate Hue (H)
    if (delta == 0)
    {
        *h = 0; // Undefined hue
    }
    else if (max_val == fr)
    {
        *h = 60.0f * std::fmod(((fg - fb) / delta), 6.0f);
    }
    else if (max_val == fg)
    {
        *h = 60.0f * (((fb - fr) / delta) + 2.0f);
    }
    else
    {
        *h = 60.0f * (((fr - fg) / delta) + 4.0f);
    }

    if (*h < 0)
    {
        *h += 360.0f;
    }
}

// Converts the entire image from RGB to YCbCr
void convert_image_rgb_to_ycbcr_cpu(Image *image)
{
    int width = image->width;
    int height = image->height;
    int channels = image->channels;

    if (channels < 3)
        return; // Image must have at least 3 channels (RGB)

    for (int i = 0; i < width * height; ++i)
    {
        unsigned char r = image->data[i * channels];
        unsigned char g = image->data[i * channels + 1];
        unsigned char b = image->data[i * channels + 2];

        unsigned char y, cb, cr;
        rgb_to_ycbcr(r, g, b, &y, &cb, &cr);

        image->data[i * channels] = y;
        image->data[i * channels + 1] = cb;
        image->data[i * channels + 2] = cr;
    }
}

// Converts a single RGB pixel to YCbCr
void rgb_to_ycbcr(unsigned char r, unsigned char g, unsigned char b, unsigned char *y, unsigned char *cb, unsigned char *cr)
{
    *y = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    *cb = static_cast<unsigned char>(128 + (-0.168736f * r - 0.331264f * g + 0.5f * b));
    *cr = static_cast<unsigned char>(128 + (0.5f * r - 0.418688f * g - 0.081312f * b));
}

// Converts a single HSV pixel to RGB
void hsv_to_rgb(float h, float s, float v, unsigned char *r, unsigned char *g, unsigned char *b)
{
    float c = v * s;
    float x = c * (1 - std::fabs(fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r_f, g_f, b_f;

    if (h < 60)
    {
        r_f = c;
        g_f = x;
        b_f = 0;
    }
    else if (h < 120)
    {
        r_f = x;
        g_f = c;
        b_f = 0;
    }
    else if (h < 180)
    {
        r_f = 0;
        g_f = c;
        b_f = x;
    }
    else if (h < 240)
    {
        r_f = 0;
        g_f = x;
        b_f = c;
    }
    else if (h < 300)
    {
        r_f = x;
        g_f = 0;
        b_f = c;
    }
    else
    {
        r_f = c;
        g_f = 0;
        b_f = x;
    }

    *r = static_cast<unsigned char>((r_f + m) * 255);
    *g = static_cast<unsigned char>((g_f + m) * 255);
    *b = static_cast<unsigned char>((b_f + m) * 255);
}

// Converts the entire image from HSV to RGB
void convert_image_hsv_to_rgb_cpu(Image *image)
{
    int width = image->width;
    int height = image->height;
    int channels = image->channels;

    for (int i = 0; i < width * height; ++i)
    {
        float h = image->data[i * channels] / 255.0f * 360.0f;
        float s = image->data[i * channels + 1] / 255.0f;
        float v = image->data[i * channels + 2] / 255.0f;

        unsigned char r, g, b;
        hsv_to_rgb(h, s, v, &r, &g, &b);

        image->data[i * channels] = r;
        image->data[i * channels + 1] = g;
        image->data[i * channels + 2] = b;
    }
}

// Converts a single YCbCr pixel to RGB
void ycbcr_to_rgb(unsigned char y, unsigned char cb, unsigned char cr, unsigned char *r, unsigned char *g, unsigned char *b)
{
    *r = static_cast<unsigned char>(y + 1.402f * (cr - 128));
    *g = static_cast<unsigned char>(y - 0.344136f * (cb - 128) - 0.714136f * (cr - 128));
    *b = static_cast<unsigned char>(y + 1.772f * (cb - 128));
}

// Converts the entire image from YCbCr to RGB
void convert_image_ycbcr_to_rgb_cpu(Image *image)
{
    int width = image->width;
    int height = image->height;
    int channels = image->channels;

    if (channels < 3)
        return; // Image must have at least 3 channels (Y, Cb, Cr)

    for (int i = 0; i < width * height; ++i)
    {
        unsigned char y = image->data[i * channels];
        unsigned char cb = image->data[i * channels + 1];
        unsigned char cr = image->data[i * channels + 2];

        unsigned char r, g, b;
        ycbcr_to_rgb(y, cb, cr, &r, &g, &b);

        image->data[i * channels] = r;
        image->data[i * channels + 1] = g;
        image->data[i * channels + 2] = b;
    }
}
