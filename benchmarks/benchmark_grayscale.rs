use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use img_rcc::{Device, Image};

fn benchmark_grayscale(c: &mut Criterion) {
    let mut group = c.benchmark_group("GPU Grayscale");

    let image_path = "input.png";

    group.bench_with_input(
        BenchmarkId::new("Load to CPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let start_time = std::time::Instant::now();
                    let _image = Image::load_to_device(image_path, Device::CPU);
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Load to GPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let start_time = std::time::Instant::now();
                    let _image = Image::load_to_device(image_path, Device::GPU);
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Transfer to CPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = Image::load_to_device(image_path, Device::GPU);
                    let start_time = std::time::Instant::now();
                    image.to(Device::CPU);
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Transfer to GPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = Image::load_to_device(image_path, Device::CPU);
                    let start_time = std::time::Instant::now();
                    image.to(Device::GPU);
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Grayscale on CPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = Image::load_to_device(image_path, Device::CPU);
                    let start_time = std::time::Instant::now();
                    image.grayscale();
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    // FIXME: This benchmark is not working as expected; fix later
    group.bench_with_input(
        BenchmarkId::new("Grayscale on GPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = Image::load_to_device(image_path, Device::GPU);
                    // image.to(Device::GPU);
                    let start_time = std::time::Instant::now();
                    image.grayscale();
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Save from CPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let image = Image::load_to_device(image_path, Device::CPU);
                    let start_time = std::time::Instant::now();
                    image.save("output_cpu.png");
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Save from GPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let image = Image::load_to_device(image_path, Device::GPU);
                    let start_time = std::time::Instant::now();
                    image.save("output_gpu.png");
                    total_duration += start_time.elapsed();
                    // free_image(image);
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Free from CPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let _image = Image::load_to_device(image_path, Device::CPU);
                    let start_time = std::time::Instant::now();
                    // free_image(image);
                    total_duration += start_time.elapsed();
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Free from GPU", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = std::time::Duration::new(0, 0);

                for _ in 0..iters {
                    let _image = Image::load_to_device(image_path, Device::GPU);
                    let start_time = std::time::Instant::now();
                    // free_image(image);
                    total_duration += start_time.elapsed();
                }

                total_duration
            });
        },
    );

    group.finish();
}

// fn configure_criterion() -> Criterion {
//     Criterion::default()
//         .configure_from_args() // Configures based on command line arguments passed
//         .warm_up_time(std::time::Duration::from_secs(3)) // Time for warming up before measuring
//         .measurement_time(std::time::Duration::from_secs(20)) // Time allocated for each benchmark run
//         .sample_size(10) // Sets the number of samples to collect per benchmark
//         .nresamples(100_000) // Number of resamples for statistical noise reduction
//         .noise_threshold(0.05) // Acceptable noise threshold before deciding the result is noisy
//         .with_plots() // Enables plotting of the benchmark results (optional)
//                       // .without_plots() // Disable plotting (optional if you want plots)
// }

// criterion_group! {
//     name = benches;
//     config = configure_criterion();
//     targets = benchmark_grayscale
// }

// criterion_main!(benches);

criterion_group!(benches, benchmark_grayscale);
criterion_main!(benches);