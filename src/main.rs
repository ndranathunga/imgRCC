#[cfg(feature = "benchmark")]
use img_rcc::benchmark::{grayscale_gpu, load_image_, GPUStats};

#[cfg(feature = "benchmark")]
fn main() {
    let image_path = "input.png";

    // Load the image
    let mut image = load_image_(image_path);

    // Run the grayscale GPU benchmark and get GPUStats
    let stats: GPUStats = grayscale_gpu(&mut image);

    // Print out the GPUStats result
    println!("GPU Stats:");
    println!(
        "Host to Device (CUDA): {} ms",
        stats.host_to_device_time_cuda
    );
    // println!("Kernel Execution (CUDA): {} ms", stats.kernel_execution_time_cuda);
    // println!("Device to Host (CUDA): {} ms", stats.device_to_host_time_cuda);

    // println!("Host to Device (Chrono): {} ms", stats.host_to_device_time_chrono);
    // println!("Kernel Execution (Chrono): {} ms", stats.kernel_execution_time_chrono);
    // println!("Device to Host (Chrono): {} ms", stats.device_to_host_time_chrono);
    println!(
        "Host to Device (CUDA): {} us",
        (stats.host_to_device_time_cuda * 1000.0) as u64
    );
}

#[cfg(not(feature = "benchmark"))]
use img_rcc::{free_image, Device, Image};

#[cfg(not(feature = "benchmark"))]
fn main() {
    let image_path = "input.png";

    let mut image = Image::load(image_path);
    image.to(Device::GPU);

    image.grayscale();

    image.save("output_gpu.png");
    free_image(image);

    let mut image = Image::load(image_path); // loads to CPU by default

    image.grayscale();

    image.save("output_cpu.png");
    free_image(image);
}
