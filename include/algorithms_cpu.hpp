#ifndef ALGORITHMS_CPU_HPP
#define ALGORITHMS_CPU_HPP

#include "image.hpp"

extern "C"
{
    void convert_to_grayscale_cpu(Image &image);

    // Converts the entire image from RGB to HSV
    void convert_image_rgb_to_hsv_cpu(Image *image);
    // Converts the entire image from RGB to YCbCr
    void convert_image_rgb_to_ycbcr_cpu(Image *image);
    // Converts the entire image from HSV to RGB
    void convert_image_ycbcr_to_rgb_cpu(Image* image);
    // Converts the entire image from HSV to RGB
    void convert_image_hsv_to_rgb_cpu(Image* image);

    void convolve_image_cpu(Image *image, const float *kernel, int kernel_width, int kernel_height);
}

// Converts a single RGB pixel to HSV
void rgb_to_hsv(unsigned char r, unsigned char g, unsigned char b, float *h, float *s, float *v);
// Converts a single RGB pixel to YCbCr
void rgb_to_ycbcr(unsigned char r, unsigned char g, unsigned char b, unsigned char *y, unsigned char *cb, unsigned char *cr);
// Converts a single HSV pixel to RGB
void hsv_to_rgb(float h, float s, float v, unsigned char* r, unsigned char* g, unsigned char* b);
// Converts a single YCbCr pixel to RGB
void ycbcr_to_rgb(unsigned char y, unsigned char cb, unsigned char cr, unsigned char* r, unsigned char* g, unsigned char* b);

#endif // ALGORITHMS_CPU_HPP
