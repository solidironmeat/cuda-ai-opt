#include "maxpool.cuh"

#include <cfloat>

__global__ void maxpool2d_kernel(const float *input, float *output, int N,
                                 int C, int H, int W, int outH, int outW,
                                 int poolH, int poolW, int strideH,
                                 int strideW) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z;

  if (oh >= outH || ow >= outW || c >= C)
    return;

  int out_idx = (c * outH + oh) * outW + ow;
  float max_val = -FLT_MAX;

  for (int i = 0; i < poolH; ++i) {
    for (int j = 0; j < poolW; ++j) {
      int h = oh * strideH + i;
      int w = ow * strideW + j;
      if (h < H && w < W) {
        int in_idx = (c * H + h) * W + w;
        max_val = fmaxf(max_val, input[in_idx]);
      }
    }
  }

  output[out_idx] = max_val;
}

void launch_maxpool2d_kernel(const float *input, float *output, int N, int C,
                             int H, int W, int outH, int outW, int poolH,
                             int poolW, int strideH, int strideW) {
  dim3 threads(16, 16);
  dim3 blocks((outW + threads.x - 1) / threads.x,
              (outH + threads.y - 1) / threads.y, C);

  maxpool2d_kernel<<<blocks, threads>>>(input, output, N, C, H, W, outH, outW,
                                        poolH, poolW, strideH, strideW);
}