// #[cfg(not(feature = "benchmark"))]
use img_rcc::{Device, Image};
use std::time::Instant;

// #[cfg(not(feature = "benchmark"))]
fn main() {
    let image_path = "input.png";

    println!("\x1b[32mRunning the benchmark for load to device:\x1b[0m");
    let start_load_to_device = Instant::now();
    let _image = Image::load_to_device(image_path, Device::GPU);
    let duration_load_to_device = start_load_to_device.elapsed();
    println!(
        "Time taken for load to device: {:?}",
        duration_load_to_device
    );

    // free_image(image);

    println!("\x1b[32mRunning the benchmark for GPU:\x1b[0m");
    let start_load_gpu = Instant::now();
    let mut image = Image::load(image_path);
    let duration_load_gpu = start_load_gpu.elapsed();
    println!("Time taken for GPU load: {:?}", duration_load_gpu);

    let start_transfer = Instant::now();
    image.to(Device::GPU);
    let duration_transfer = start_transfer.elapsed();
    println!("Time taken for GPU transfer: {:?}", duration_transfer);

    let start_grayscale_gpu = Instant::now();
    image.grayscale();
    let duration_grayscale_gpu = start_grayscale_gpu.elapsed();
    println!("Time taken for GPU grayscale: {:?}", duration_grayscale_gpu);

    let start_save_gpu = Instant::now();
    image.save("output_gpu.png");
    let duration_save_gpu = start_save_gpu.elapsed();
    println!("Time taken for GPU save: {:?}", duration_save_gpu);

    let start_free_gpu = Instant::now();
    // free_image(image);
    let duration_free_gpu = start_free_gpu.elapsed();
    println!("Time taken for GPU free: {:?}", duration_free_gpu);

    println!("\x1b[32mRunning the benchmark for CPU:\x1b[0m");
    let start_load_cpu = Instant::now();
    let mut image = Image::load(image_path);
    let duration_load_cpu = start_load_cpu.elapsed();
    println!("Time taken for CPU load: {:?}", duration_load_cpu);

    let start_transfer = Instant::now();
    image.to(Device::CPU);
    let duration_transfer = start_transfer.elapsed();
    // println!("Channels: {}", image.channels);
    println!("Time taken for CPU transfer: {:?}", duration_transfer);

    let start_grayscale_cpu = Instant::now();
    image.grayscale();
    let duration_grayscale_cpu = start_grayscale_cpu.elapsed();
    println!("Time taken for CPU grayscale: {:?}", duration_grayscale_cpu);

    let start_save_cpu = Instant::now();
    image.save("output_cpu.png");
    let duration_save_cpu = start_save_cpu.elapsed();
    println!("Time taken for CPU save: {:?}", duration_save_cpu);

    let start_free_cpu = Instant::now();
    // free_image(image);
    let duration_free_cpu = start_free_cpu.elapsed();
    println!("Time taken for CPU free: {:?}", duration_free_cpu);
}
