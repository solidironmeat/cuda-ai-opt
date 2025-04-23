cmake_minimum_required(VERSION 3.10)
project(InferenceEngine VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

find_package(CUDAToolkit REQUIRED)
find_package(ONNX REQUIRED CONFIG)
find_package(Protobuf REQUIRED)

# Sources
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    /usr/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNX_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC 
    CUDA::cudart
    # cutensor
    # cublas
    onnx
    onnx_proto
    protobuf::libprotobuf
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CMAKE_CUDA_STANDARD 20
    CMAKE_CUDA_STANDARD_REQUIRED ON
    CMAKE_CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 75 80 86
    CMAKE_VERBOSE_MAKEFILE ON
)