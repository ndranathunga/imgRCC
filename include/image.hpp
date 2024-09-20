#ifndef IMAGE_LOADER_HPP
#define IMAGE_LOADER_HPP

#include <string>
// #include <vector>

extern "C"
{
    struct Image
    {
        int width;
        int height;
        int channels;
        unsigned char *data;
    };

    // CPU Functions
    Image *load_image_cpu(const char *file_path);
    void save_image_cpu(const char *file_path, const Image *image);
    void free_image_cpu(Image *image);

    // GPU Functions
    Image *load_image_gpu(const char *file_path);
    void save_image_gpu(const char *file_path, const Image *image);
    void free_image_gpu(Image *image);
}
#endif // IMAGE_LOADER_HPP
