use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Path to the build directory
    let build_dir = PathBuf::from("build");

    // Step 1: Run CMake to configure the project if needed
    if !build_dir.exists() {
        let _status = Command::new("cmake")
            .args(&["-S", ".", "-B", "build"])
            .status()
            .expect("Failed to generate CMake build files.");
    }

    // Step 2: Run CMake to build the C++/CUDA project
    let _status = Command::new("cmake")
        .args(&["--build", "build"])
        .status()
        .expect("Failed to build C++/CUDA project with CMake.");

    // Tell Cargo to rerun this script if the build directory changes
    println!("cargo:rerun-if-changed=build/");

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
