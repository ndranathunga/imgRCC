use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use img_rcc::benchmark::{grayscale_gpu, load_image_, GPUStats};
use std::time::Duration;

fn bench_gpu_grayscale(c: &mut Criterion) {
    // Create a benchmark group for GPU stats
    let mut group = c.benchmark_group("GPU Grayscale Stats");

    let image_path = "input.png";

    group.bench_with_input(
        BenchmarkId::new("Host to Device Transfer", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = load_image_(image_path);
                    let stats: GPUStats = grayscale_gpu(&mut image);

                    // Convert device to host time from milliseconds to microseconds
                    let iteration_duration =
                        Duration::from_micros((stats.host_to_device_time_cuda * 1000.0) as u64);

                    // Accumulate the duration over all iterations
                    total_duration += iteration_duration;
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Kernel Execution", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = load_image_(image_path);
                    let stats: GPUStats = grayscale_gpu(&mut image);

                    // Convert device to host time from milliseconds to microseconds
                    let iteration_duration =
                        Duration::from_micros((stats.kernel_execution_time_cuda * 1000.0) as u64);

                    // Accumulate the duration over all iterations
                    total_duration += iteration_duration;
                }

                total_duration
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("Device to Host Transfer", image_path),
        &image_path,
        |b, &image_path| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::new(0, 0);

                for _ in 0..iters {
                    let mut image = load_image_(image_path);
                    let stats: GPUStats = grayscale_gpu(&mut image);

                    // Convert device to host time from milliseconds to microseconds
                    let iteration_duration =
                        Duration::from_micros((stats.device_to_host_time_cuda * 1000.0) as u64);

                    // Accumulate the duration over all iterations
                    total_duration += iteration_duration;
                }

                total_duration
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_gpu_grayscale);
criterion_main!(benches);
