#!/bin/bash

# Ensure the script stops on errors
set -e

# Create the build directory if it doesn't exist
mkdir -p build

# Navigate to the build directory
cd build

# Run CMake with Benchmark mode enabled
cmake .. -DCMAKE_BUILD_TYPE=Benchmark

# Build the project using Make
make

cd bin

# Check if the benchmark executable exists
if [ -f "./img_rcc_benchmark" ]; then
    echo "Running img_rcc_benchmark..."
    # Run the benchmark executable
    ./img_rcc_benchmark
else
    echo "Error: img_rcc_benchmark executable not found."
    exit 1
fi
