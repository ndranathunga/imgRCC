#ifndef ALGORITHMS_GPU_HPP
#define ALGORITHMS_GPU_HPP

#include "image.hpp"

extern "C"
{
    void convert_to_grayscale_gpu(Image &image);
    void convolve_image_gpu(Image *image, const float *kernel, int kernel_width, int kernel_height);
}

#endif // ALGORITHMS_GPU_HPP
