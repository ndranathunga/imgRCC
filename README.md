# imgRCC Image Processing Library

## Purpose of the Project

This project is a **fun and educational venture** designed to explore the interoperability of multiple programming languages—namely **C++**, **CUDA**, and **Rust**—and to understand how we can utilize the strengths of each language in a multi-module, cross-language setting. The aim is to build a high-performance **image processing library** with GPU acceleration using **CUDA**, organized modularly in **C++**, with high-level safe bindings in **Rust**. Eventually, the library will also offer bindings for **Python** and **Go**, making it versatile and usable across various programming ecosystems.

Through this project, I'm learning:
- How to implement **parallel algorithms** using **CUDA** for GPU processing.
- The process of writing **C++** code for CPU-bound processing.
- How to build **Rust bindings** for C++/CUDA code using **FFI (Foreign Function Interface)**.
- The **modular design** and **project organization** for a multi-language project.
- Optimizing performance through **parallelism** and comparing **CPU vs. GPU** execution times.
- How to extend the library to other languages like **Python** and **Go**.

## Current Features

- **Image Loading and Saving**:
   - Implemented image loading and saving using **stb_image** and **stb_image_write** headers, allowing support for multiple image formats (PNG, JPEG, BMP).

- **Grayscale Conversion**: 
  - Implemented both as a **CPU-based** function (in **C++**) and as a **GPU-based** function (in **CUDA**).
  - Usable via **Rust** bindings.

- **Benchmarking Tools**:
   - Tools to compare the performance of **CPU vs. GPU** implementations, so users can evaluate the best option based on their use case.

## Upcoming Features (Planned Functions)

1. **Image Convolution**:
   - Implement a flexible convolution function to apply various filters (e.g., sharpening, blurring) to images. 
   - Versions will be written in both **C++** (CPU) and **CUDA** (GPU).

2. **Gaussian Blur**:
   - A popular technique for smoothing images, implemented with both CPU and GPU versions.
   - This will demonstrate how algorithms can benefit from GPU acceleration on large images.

3. **Sobel Edge Detection**:
   - Detect edges in an image using the **Sobel operator**.
   - Parallelized GPU version to significantly speed up this computation.

4. **Color Conversion (RGB to HSV/YCbCr)**:
   - Functions to convert images between color spaces, useful for more advanced image processing operations.

5. **Thresholding**:
   - Binary thresholding of images to help with segmentation tasks.
   - Fast, parallelizable version using CUDA.

## Features to Implement in Other Languages

1. **Python Bindings**:
   - Provide easy-to-use bindings for Python using **PyO3** or **cffi**, so the library can be used in Python-based applications.

2. **Go Bindings**:
   - Implement Go bindings using **cgo** to expose the image processing library to the Go programming community.

## Project Structure

The project is organized into several modules:
- **/include/**: Contains C++ header files.
- **/src/cpp/** and **/src/cuda/**: Source files for C++ and CUDA code.
- **/src/**: The Rust bindings that interface with C++/CUDA via FFI.
- **/tests/**: Unit tests for C++ and Rust components.
- **/benchmarks/**: Benchmarking code for comparing CPU vs. GPU performance.

### Example Project Structure
```
/parallel_image_processing_lib/
├── include/                        # C++ headers
├── src/                            # Rust bindings
├──── cpp/                          # C++ source files
├──── cuda/                         # CUDA source files           
├── tests/                          # Unit tests
├── benchmarks/                     # Performance benchmarks
├── cmake/                          # CMake modules
├── build.rs                        # Rust build script
├── CMakeLists.txt                  # CMake build configuration
├── Cargo.toml                      # Rust project configuration
└── README.md                       # This file
```

## How to Build and Use the Library

### Building

1. Clone the repository:

   ```bash
   https://github.com/ndranathunga/imgRCC.git
   cd imgRCC
   ```

2. Build the project using CMake for the C++/CUDA part:

   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Build the Rust bindings:

   ```bash
   cargo build
   ```

### Usage (via Rust)

You can use the library within a Rust project by adding it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
img_rcc = { path = "../path/to/imgRCC" }
```

Then, use it in your Rust code:

```rust
use img_rcc::grayscale_cpu;

fn main() {
    let image = load_image_("input.jpg"); // Function to load image (not yet implemented)
    let grayscale_image = grayscale_cpu(image);
    save_image_(grayscale_image, "output.jpg"); // Function to save image (not yet implemented)
}
```

## Testing

The project includes unit tests for both C++ and Rust components. To run the Rust tests:

```bash
cargo test
```

For C++ tests, run the `make test` command after building the project using CMake.

<!-- ## Contributions and License

Feel free to fork this project and contribute! Whether you're adding new features, fixing bugs, or improving documentation, all contributions are welcome.

This project is licensed under the **MIT License**. -->
