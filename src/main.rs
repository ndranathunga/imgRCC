#[cfg(feature = "benchmark")]
use img_rcc::benchmark::{grayscale_gpu, GPUStats, load_image_};

#[cfg(feature = "benchmark")]
fn main() {
    let image_path = "input.png";
    
    // Load the image
    let mut image = load_image_(image_path);
    
    // Run the grayscale GPU benchmark and get GPUStats
    let stats: GPUStats = grayscale_gpu(&mut image);

    // Print out the GPUStats result
    println!("GPU Stats:");
    println!("Host to Device (CUDA): {} ms", stats.host_to_device_time_cuda);
    // println!("Kernel Execution (CUDA): {} ms", stats.kernel_execution_time_cuda);
    // println!("Device to Host (CUDA): {} ms", stats.device_to_host_time_cuda);

    // println!("Host to Device (Chrono): {} ms", stats.host_to_device_time_chrono);
    // println!("Kernel Execution (Chrono): {} ms", stats.kernel_execution_time_chrono);
    // println!("Device to Host (Chrono): {} ms", stats.device_to_host_time_chrono);
    println!("Host to Device (CUDA): {} us", (stats.host_to_device_time_cuda * 1000.0) as u64);
}

#[cfg(not(feature = "benchmark"))]
use std::time::Instant;
#[cfg(not(feature = "benchmark"))]
use img_rcc::{free_image_, grayscale_cpu, grayscale_gpu, load_image_, save_image_};

#[cfg(not(feature = "benchmark"))]
fn main() {
    let mut image = load_image_("input.png"); // Assuming a function load_image is bound to Rust

    // Start timer for CPU grayscale
    let start_cpu = Instant::now();

    grayscale_cpu(&mut image);

    // Measure and print CPU processing time
    let duration_cpu = start_cpu.elapsed();
    println!("Time taken for CPU grayscale: {:?}", duration_cpu);

    save_image_("output_cpu.png", &image); // Assuming a function save_image is bound to Rust
    free_image_(image);

    
    let mut image = load_image_("input.png");

    // Start timer for GPU grayscale
    let start_gpu = Instant::now();

    grayscale_gpu(&mut image);


    // Measure and print GPU processing time
    let duration_gpu = start_gpu.elapsed();
    println!("Time taken for GPU grayscale: {:?}", duration_gpu);

    save_image_("output_gpu.png", &image);
    free_image_(image);
}
