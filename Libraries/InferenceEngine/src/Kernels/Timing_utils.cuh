#ifndef TIMING_UTILS_CUH
#define TIMING_UTILS_CUH

#include <cuda_runtime.h>

#include <iostream>
#include <string>

class CudaTimer {
public:
  CudaTimer(const std::string &label = "") : label(label), started(false) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void begin() {
    started = true;
    cudaEventRecord(start, 0);
  }

  float end(bool print = true) {
    if (!started)
      return 0.0f;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    if (print)
      std::cout << "[Timer] " << label << " took " << ms << " ms\n";
    started = false;
    return ms;
  }

private:
  std::string label;
  cudaEvent_t start, stop;
  bool started;
};

#endif // TIMING_UTILS_CUH