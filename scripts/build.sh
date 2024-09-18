#!/bin/bash

# Step 1: Build C++/CUDA code using CMake
mkdir -p build
cd build
cmake ..
make

# Step 2: Build Rust code using Cargo
cd ..
cargo build
