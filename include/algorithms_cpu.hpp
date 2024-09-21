#ifndef ALGORITHMS_CPU_HPP
#define ALGORITHMS_CPU_HPP

#include "image.hpp"

extern "C"
{
    void convert_to_grayscale_cpu(Image &image);
    void convolve_image_cpu(Image *image, const float *kernel, int kernel_width, int kernel_height);
}

#endif // ALGORITHMS_CPU_HPP
