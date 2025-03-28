cmake_minimum_required(VERSION 3.10)
project(cuda_fft_convolution LANGUAGES CXX CUDA)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(fmt REQUIRED)

# Source files
file(GLOB_RECURSE SRC_FILES
    "src/*.cu"
    "src/*.cpp"
)

# Include directories
include_directories(
    include
)

# Add executable
add_executable(cuda_fft_convolution ${SRC_FILES})

# Link cuFFT and cuBLAS
target_link_libraries(cuda_fft_convolution 
    PRIVATE CUDA::cufft CUDA::cublas ${OpenCV_LIBS}
    ${Boost_LIBRARIES} fmt::fmt
)

# Output path
set_target_properties(cuda_fft_convolution PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


add_executable(test test/test.cpp)
target_link_libraries(test 
    PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas ${OpenCV_LIBS}
    ${Boost_LIBRARIES} fmt::fmt
)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)