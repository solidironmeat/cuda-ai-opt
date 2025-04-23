#include "relu.cuh"

__global__ void relu_kernel(const float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

void launch_relu_kernel(const float *input, float *output, int size) {

  int threads = 128;
  int blocks = (size + threads - 1) / threads;

  relu_kernel<<<blocks, threads>>>(input, output, size);
}
