#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image.hpp"
#include "algorithms_cpu.hpp"
#include <stdexcept>
#include <iostream>

Image *load_image_cpu(const char *file_path)
{
    Image *image = new Image();
    image->data = stbi_load(file_path, &image->width, &image->height, &image->channels, 0);
    if (!image->data)
    {
        throw std::runtime_error("failed to load image: " + std::string(file_path));
    }
    return image;
}

void save_image_rgb_cpu(const char *file_path, const Image *image)
{
    int success = stbi_write_png(file_path, image->width, image->height, image->channels, image->data, image->width * image->channels);
    if (!success)
    {
        throw std::runtime_error("failed to save image");
    }
}

void save_image_hsv_as_rgb(const char *file_path, Image *image)
{
    // Convert the HSV image back to RGB before saving
    convert_image_hsv_to_rgb_cpu(image);
    save_image_rgb_cpu(file_path, image); // Save as RGB
}

void free_image_cpu(Image *image)
{
    if (image->data)
    {
        stbi_image_free(image->data);
        image->data = nullptr; // Set pointer to null to avoid double free
    }
}

void save_image_ycbcr_as_jpeg_cpu(const char *file_path, const Image *image, int quality = 100)
{
    int success = stbi_write_jpg(file_path, image->width, image->height, image->channels, image->data, quality);
    if (!success)
    {
        throw std::runtime_error("failed to save YCbCr image as JPEG");
    }
}