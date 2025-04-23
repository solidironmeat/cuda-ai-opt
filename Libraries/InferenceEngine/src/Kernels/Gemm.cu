#include "gemm.cuh"

__global__ void gemm_kernel(const float *A, const float *B, float *C, int M,
                            int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float val = 0.0f;
    for (int k = 0; k < K; ++k) {
      val += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = val;
  }
}

// Function to launch the GEMM kernel
void launch_gemm_kernel(float *d_A, float *d_B, float *d_C, int M, int N,
                        int K) {
  dim3 threads(32, 32);
  dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

  // Launch kernel
  gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
}