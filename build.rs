// use std::env;
use std::path::PathBuf;

fn main() {
    // Tell Cargo to rerun this script if the build directory changes
    println!("cargo:rerun-if-changed=build/");

    // Add the search path for the compiled C++/CUDA libraries (output from CMake)
    let build_dir = PathBuf::from("build/lib");

    // Inform Rust about the library search path
    println!("cargo:rustc-link-search=native={}", build_dir.display());

    // Link to the static library produced by CMake (without the lib prefix or extension)
    println!("cargo:rustc-link-lib=static=img_rcc");

    // Link against the C++ Standard Library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Optionally, link with CUDA runtime or other libraries if needed
    println!("cargo:rustc-link-lib=dylib=cudart");
}
