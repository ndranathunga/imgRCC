#ifndef COMMON_HPP
#define COMMON_HPP

#include "image.hpp"

extern "C" void transfer_to_gpu(Image* image);
extern "C" void transfer_to_cpu(Image* image);

#endif // COMMON_HPP
