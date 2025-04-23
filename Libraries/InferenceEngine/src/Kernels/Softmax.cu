#include "softmax.cuh"

__global__ void softmax_kernel(const float *input, float *output, int N,
                               int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  // Offset for each row
  const float *in_row = input + idx * C;
  float *out_row = output + idx * C;

  // Step 1: compute max for numerical stability
  float max_val = in_row[0];
  for (int j = 1; j < C; ++j)
    if (in_row[j] > max_val)
      max_val = in_row[j];

  // Step 2: compute sum(exp(x - max))
  float sum = 0.0f;
  for (int j = 0; j < C; ++j) {
    out_row[j] = expf(in_row[j] - max_val);
    sum += out_row[j];
  }

  // Step 3: normalize
  for (int j = 0; j < C; ++j)
    out_row[j] /= sum;
}

void launch_softmax_kernel(const float *input, float *output, int N, int C) {
  int threads = 128;
  int blocks = (N + threads - 1) / threads;
  softmax_kernel<<<blocks, threads>>>(input, output, N, C);
}
