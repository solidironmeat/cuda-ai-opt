cmake_minimum_required(VERSION 3.10)
project(CudaProjects VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_CUDA_COMPILER)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-Wno-pedantic)
endif()

include_directories(${CMAKE_SOURCE_DIR}/tools)

add_subdirectory(Libraries/InferenceEngine)
add_subdirectory(Libraries/Tests)

add_subdirectory(Applications/BenchmarkRunner)
add_subdirectory(Applications/CutensorGemm)
add_subdirectory(Applications/DeviceInfo)

