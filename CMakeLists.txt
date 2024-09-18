cmake_minimum_required(VERSION 3.15)
project(imgRCC LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files (C++ and potentially CUDA files)
set(SOURCE_FILES
    src/cpp/common.cpp
)

# Add CUDA files if necessary (if you're using CUDA in the project)
# If you're using CUDA code, add those files as well
# set(SOURCE_FILES ${SOURCE_FILES} src/cuda/some_cuda_file.cu)

# Compile C++ code into a static library
add_library(img_rcc STATIC ${SOURCE_FILES})

# Optionally, if you are using CUDA, ensure proper linking with the CUDA runtime
# target_link_libraries(img_rcc ${CUDA_LIBRARIES})

# Set library output directory to be under build/lib
set_target_properties(img_rcc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Make sure that CUDA is enabled and added correctly if you're compiling any CUDA source
find_package(CUDA REQUIRED)

# Ensure that both CUDA and C++ are properly compiled together, if you need to integrate them
if (CUDA_FOUND)
    enable_language(CUDA)
endif()
