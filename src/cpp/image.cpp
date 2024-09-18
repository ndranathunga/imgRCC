#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.hpp"
#include <stdexcept>
#include <iostream>

Image load_image(const char* file_path)
{
    Image image;
    image.data = stbi_load(file_path, &image.width, &image.height, &image.channels, 0);
    if (!image.data)
    {
        throw std::runtime_error("failed to load image: " + std::string(file_path));
    }
    // unsigned char *data = stbi_load(file_path.c_str(), &image.width, &image.height, &image.channels, 0);
    // if (!data)
    // {
    //     throw std::runtime_error("Failed to load image: " + file_path);
    // }

    // image.data = std::vector<unsigned char>(data, data + (image.width * image.height * image.channels));
    // stbi_image_free(data);

    return image;
}

void save_image(const char* file_path, const Image &image)
{
    // int success = stbi_write_png(file_path.c_str(), image.width, image.height, image.channels, image.data.data(), image.width * image.channels);
    // if (!success)
    // {
    //     throw std::runtime_error("Failed to save image: " + file_path);
    // }
    int success = stbi_write_png(file_path, image.width, image.height, image.channels, image.data, image.width * image.channels);
    if (!success)
    {
        throw std::runtime_error("failed to save image");
    }
}

void free_image(Image image)
{
    stbi_image_free(image.data);
}