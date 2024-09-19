# Common configuration for the imgRCC project

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the library output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add custom build types (Debug, Release, Benchmark)
set(CMAKE_CONFIGURATION_TYPES Debug Release Benchmark CACHE STRING "Choose the type of build" FORCE)

# If no build type is specified, default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default is Release)" FORCE)
endif()

# Set custom flags for Benchmark mode
set(CMAKE_CXX_FLAGS_BENCHMARK "-O3 -DBENCHMARK_MODE")

# Apply the Benchmark flags if Benchmark mode is enabled
if(CMAKE_BUILD_TYPE STREQUAL "Benchmark")
    add_definitions(-DBENCHMARK_MODE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BENCHMARK}")
endif()

# Include the project's include directory
include_directories(${CMAKE_SOURCE_DIR}/include)
