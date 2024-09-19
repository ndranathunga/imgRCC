# CUDA-specific configuration for the imgRCC project

# Find the CUDA toolkit
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

    # Set the CUDA architecture (adjust based on the target GPU)
    set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "Specify CUDA architecture")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CMAKE_CUDA_ARCHITECTURES}")

    # Enable CUDA language
    enable_language(CUDA)
else()
    message(FATAL_ERROR "CUDA not found. Please ensure CUDA is installed and added to the system PATH.")
endif()
