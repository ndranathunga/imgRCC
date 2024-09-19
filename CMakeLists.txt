cmake_minimum_required(VERSION 3.15)
project(imgRCC LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files (C++ and CUDA files)
set(SOURCE_FILES
    src/cpp/common.cpp
    src/cpp/image.cpp
    src/cpp/algorithms_cpu.cpp
    src/cuda/algorithms_gpu.cu
)

# Add a library target for the project (C++/CUDA integration)
add_library(img_rcc STATIC ${SOURCE_FILES})

# Set library output directory to be under build/lib
set_target_properties(img_rcc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Ensure that CUDA is enabled and added correctly
find_package(CUDA REQUIRED)

# Compile the C++/CUDA code with appropriate flags
if(CUDA_FOUND)
    enable_language(CUDA)
endif()

# Set custom flags for Benchmark mode
set(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DBENCHMARK_MODE")

# Add custom build types
set(CMAKE_CONFIGURATION_TYPES Debug Release Benchmark CACHE STRING "Choose the type of build" FORCE)

# If Benchmark mode is set, apply the Benchmark flags
if(CMAKE_BUILD_TYPE STREQUAL "Benchmark")
    add_definitions(-DBENCHMARK_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BENCHMARK}")
endif()

# Add the benchmark executable only in Benchmark mode
add_executable(img_rcc_benchmark benchmarks/benchmark_gpu.cpp)

# Link the benchmark executable to the img_rcc library and CUDA
target_link_libraries(img_rcc_benchmark img_rcc ${CUDA_LIBRARIES})

# Only build the benchmark executable if in Benchmark mode
if(CMAKE_BUILD_TYPE STREQUAL "Benchmark")
    add_dependencies(img_rcc_benchmark img_rcc)
endif()

# Set output directories for the benchmark executable
set_target_properties(img_rcc_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
