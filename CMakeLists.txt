cmake_minimum_required(VERSION 3.15)
project(imgRCC LANGUAGES CXX CUDA)

# Include common configuration
include(${CMAKE_SOURCE_DIR}/cmake/CommonConfig.cmake)

# Include CUDA-specific configuration
include(${CMAKE_SOURCE_DIR}/cmake/CudaConfig.cmake)

# Add source files (C++ and CUDA files)
set(SOURCE_FILES
    src/cpp/common.cpp
    src/cpp/image.cpp
    src/cpp/algorithms_cpu.cpp
    src/cuda/algorithms_gpu.cu
)

# Add a library target for the project (C++/CUDA integration)
add_library(img_rcc STATIC ${SOURCE_FILES})

# Link the CUDA libraries to the img_rcc target
target_link_libraries(img_rcc ${CUDA_LIBRARIES})

# Add the benchmark executable
add_executable(img_rcc_benchmark benchmarks/benchmark_gpu.cpp)

# Link the benchmark executable to the img_rcc library and CUDA
target_link_libraries(img_rcc_benchmark img_rcc ${CUDA_LIBRARIES})

# Only build the benchmark executable in Benchmark mode
if(CMAKE_BUILD_TYPE STREQUAL "Benchmark")
    add_dependencies(img_rcc_benchmark img_rcc)
endif()

# Set output directories for the benchmark executable
set_target_properties(img_rcc_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
