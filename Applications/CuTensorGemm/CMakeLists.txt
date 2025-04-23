cmake_minimum_required(VERSION 3.10)
project(CutensorGemm VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(${PROJECT_NAME} main.cu)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    cutensor
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75 80 86
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    TILE_DIM=16
    BLOCK_ROWS=16
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
