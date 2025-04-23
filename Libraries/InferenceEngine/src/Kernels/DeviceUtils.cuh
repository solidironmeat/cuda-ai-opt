#ifndef DEVICE_UTILS_CUH
#define DEVICE_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

inline void printCudaDeviceInfo() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "[CUDA] Detected " << deviceCount << " devices\n";

  for (int i = 0; i < deviceCount; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "  Device " << i << ": " << prop.name << "\n";
    std::cout << "    Compute Capability: " << prop.major << "." << prop.minor
              << "\n";
    std::cout << "    Global Mem: " << prop.totalGlobalMem / (1024 * 1024)
              << " MB\n";
    std::cout << "    Shared Mem per Block: " << prop.sharedMemPerBlock
              << " bytes\n";
    std::cout << "    Max Threads per Block: " << prop.maxThreadsPerBlock
              << "\n";
    std::cout << "    Multi-Processor Count: " << prop.multiProcessorCount
              << "\n";
    std::cout << "    Clock Rate: " << prop.clockRate / 1000 << " MHz\n";
  }
}

#endif // DEVICE_UTILS_CUH