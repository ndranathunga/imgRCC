// #[cfg(test)]
// mod tests {
//     use img_rcc::{Device, Image}; // No need to import free_image anymore

//     #[test]
//     fn test_cpu_gpu_workflow() {
//         // Step 1: Load an image on the CPU
//         let mut img_cpu = Image::load("input.png");
//         assert!(
//             !img_cpu.image_data.data.is_null(),
//             "Failed to load image on CPU"
//         );

//         // Step 2: Move the image to the GPU
//         img_cpu.to(Device::GPU);
//         assert_eq!(img_cpu.device, Device::GPU, "Failed to move image to GPU");

//         // Step 3: Perform grayscale conversion on the GPU
//         img_cpu.grayscale();
//         // No assertions here, but you can test whether the operation was successful by examining the image's pixel data.

//         // Step 4: Move the image back to the CPU
//         img_cpu.to(Device::CPU);
//         assert_eq!(
//             img_cpu.device,
//             Device::CPU,
//             "Failed to move image back to CPU"
//         );

//         // Step 5: Save the processed image
//         img_cpu.save("output_image.png");

//         // Optional: You can add checks to ensure the file was saved properly (e.g., by checking file size, existence, etc.).
//         assert!(
//             std::path::Path::new("output_image.png").exists(),
//             "Failed to save the processed image"
//         );

//         // Memory is automatically freed when `img_cpu` goes out of scope
//     }

//     #[test]
//     fn test_equivalence_of_cpu_and_gpu_grayscale() {
//         // Step 1: Load an image on the CPU
//         let mut img_cpu = Image::load("input.png");
//         let mut img_gpu = Image::load("input.png");

//         // Step 2: Convert to grayscale on CPU
//         img_cpu.grayscale();
//         img_gpu.to(Device::GPU);
//         img_gpu.grayscale();
//         img_gpu.to(Device::CPU); // Move back to CPU for comparison

//         // Step 3: Compare the two images
//         assert_eq!(
//             img_cpu.image_data.width, img_gpu.image_data.width,
//             "Image widths do not match"
//         );
//         assert_eq!(
//             img_cpu.image_data.height, img_gpu.image_data.height,
//             "Image heights do not match"
//         );
//         assert_eq!(
//             img_cpu.image_data.channels, img_gpu.image_data.channels,
//             "Image channels do not match"
//         );

//         // Step 4: Compare pixel data (this assumes the image data format is the same on CPU and GPU)
//         for i in
//             0..(img_cpu.image_data.width * img_cpu.image_data.height * img_cpu.image_data.channels)
//         {
//             unsafe {
//                 assert_eq!(
//                     *img_cpu.image_data.data.offset(i as isize),
//                     *img_gpu.image_data.data.offset(i as isize),
//                     "Grayscale data mismatch at pixel index {}",
//                     i
//                 );
//             }
//         }

//         // Memory is automatically freed when `img_cpu` and `img_gpu` go out of scope
//     }
// }
