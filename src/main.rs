use std::time::Instant;

use img_rcc::{free_image_, grayscale_cpu, grayscale_gpu, load_image_, save_image_};

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
