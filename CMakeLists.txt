cmake_minimum_required(VERSION 3.15)
project(imgRCC LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
set(SOURCE_FILES
    src/common.cpp
)

# Compile C++ code into a static library
add_library(img_rcc STATIC ${SOURCE_FILES})

# Set library output directory
set_target_properties(img_rcc PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
