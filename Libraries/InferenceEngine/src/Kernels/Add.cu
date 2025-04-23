#include "Add.cuh"

__global__ void add_kernel(const float *A, const float *B, float *C, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    C[idx] = A[idx] + B[idx];
}

void launch_add_kernel(const float *A, const float *B, float *C, int size) {
  int threads = 128;
  int blocks = (size + threads - 1) / threads;

  add_kernel<<<blocks, threads>>>(A, B, C, size);
}
