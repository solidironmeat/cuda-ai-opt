#include "GlobalAveragePool.cuh"

__global__ void global_avg_pool_kernel(const float *__restrict__ input,
                                       float *__restrict__ output, int N, int C,
                                       int H, int W) {
  int n = blockIdx.x;
  int c = threadIdx.x;

  if (c >= C)
    return;

  int offset = (n * C + c) * H * W;
  float sum = 0.0f;
  for (int i = 0; i < H * W; ++i)
    sum += input[offset + i];

  output[n * C + c] = sum / (H * W);
}

void launch_global_avg_pool_kernel(const float *input, float *output, int N,
                                   int C, int H, int W) {
  dim3 grid(N);
  dim3 block(C);
  global_avg_pool_kernel<<<grid, block>>>(input, output, N, C, H, W);
}
