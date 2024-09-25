#ifndef ALGORITHMS_GPU_HPP
#define ALGORITHMS_GPU_HPP

#include "image.hpp"

extern "C"
{
    void convert_to_grayscale_gpu(Image &image);
    void convolve_image_gpu(Image *image, const float *kernel, int kernel_width, int kernel_height);

    // RGB to HSV conversion on GPU
    void convert_image_rgb_to_hsv_gpu(Image *image);
    // RGB to YCbCr conversion on GPU
    void convert_image_rgb_to_ycbcr_gpu(Image *image);
}

#endif // ALGORITHMS_GPU_HPP
