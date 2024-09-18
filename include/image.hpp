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
        // std::vector<unsigned char> data;
        unsigned char *data;
    };

    Image load_image(const char* file_path);
    void save_image(const char* file_path, const Image &image);
    void free_image(Image image);
}
#endif // IMAGE_LOADER_HPP
