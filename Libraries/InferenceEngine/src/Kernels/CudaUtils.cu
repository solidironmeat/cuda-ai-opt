#include "CudaUtils.cuh"

#include <iostream>

int print_device_info() {
  int deviceCount = 0;
  CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::cout << "No CUDA devices found.\n";
    return -1;
  }

  std::cout << "Found " << deviceCount << " CUDA device(s):\n";

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i));

    std::cout << "\nDevice " << i << ": " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor
              << "\n";
    std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20)
              << " MB\n";
    std::cout << "  Shared memory per block: " << (prop.sharedMemPerBlock >> 10)
              << " KB\n";
    std::cout << "  Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max threads dim: [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2]
              << "]\n";
    std::cout << "  Max grid size: [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
    std::cout << "  Clock rate: " << (prop.clockRate / 1000) << " MHz\n";
    std::cout << "  Multiprocessor count: " << prop.multiProcessorCount << "\n";
    std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "  Memory clock rate: " << (prop.memoryClockRate / 1000)
              << " MHz\n";
  }

  return 0;
}