use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell Cargo to rerun this script if the build directory changes
    println!("cargo:rerun-if-changed=");
    println!("cargo:rerun-if-changed=build.rs");

    // Path to the build directory
    let build_dir = PathBuf::from("build");

    // Check if we're building with the `benchmark` feature
    let benchmark_enabled = env::var("CARGO_FEATURE_BENCHMARK").is_ok();

    // if !build_dir.exists() {}

    // Step 1: Run CMake to configure the project if needed
    let mut cmake_args = vec!["-S", ".", "-B", "build"];
    // Set the build type based on whether `benchmark` is enabled or not
    if benchmark_enabled {
        cmake_args.push("-DCMAKE_BUILD_TYPE=Benchmark");
    }
    //  else {
    //     cmake_args.push("-DCMAKE_BUILD_TYPE=Release");
    // }

    let status = Command::new("cmake")
        .args(&cmake_args)
        .status()
        .expect("Failed to generate CMake build files.");

    // Check if the command succeeded
    if !status.success() {
        panic!("CMake configuration failed.");
    }

    // Step 2: Run CMake to build the C++/CUDA project
    let status = Command::new("cmake")
        .args(&["--build", "build"])
        .status()
        .expect("Failed to build C++/CUDA project with CMake.");

    if !status.success() {
        panic!("CMake build failed.");
    }

    // Add the search path for the compiled C++/CUDA libraries
    let lib_dir = build_dir.join("lib");

    // Inform Rust about the library search path
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Link to the static library produced by CMake (without the lib prefix or extension)
    println!("cargo:rustc-link-lib=static=img_rcc");

    // Link against the C++ Standard Library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Optionally, link with CUDA runtime or other libraries if needed
    println!("cargo:rustc-link-lib=dylib=cudart");
}
