#include "algorithms_cpu.hpp"
#include <iostream>

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
