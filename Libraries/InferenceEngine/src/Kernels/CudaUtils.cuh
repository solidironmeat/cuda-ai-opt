#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// cuBLAS/cuDNN/cuTensor/cuFFT-style API wrapper (status int returns)
#define CHECK_STATUS(status, msg)                                              \
  do {                                                                         \
    if ((status) != 0) {                                                       \
      std::cerr << "[LIB ERROR] " << (msg) << " returned status " << status    \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int print_device_info();

// Optional: Reset device on crash for cleanup
inline void safeCudaReset() { cudaDeviceReset(); }

#endif // UTILS_H