#include "Conv2d.cuh"

__global__ void conv2d_kernel(const float *__restrict__ input,
                              const float *__restrict__ weights, float *output,
                              int C, int H, int W, int K, int R, int S,
                              int outH, int outW, int strideH, int strideW) {
  int ow = blockIdx.x;
  int oh = blockIdx.y;
  int nk = blockIdx.z;
  int n = nk / K;
  int k = nk % K;

  if (ow >= outW || oh >= outH)
    return;

  float val = 0.0f;
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int ih = oh * strideH + r;
        int iw = ow * strideW + s;

        if (ih < H && iw < W) {
          int in_idx = ((n * C + c) * H + ih) * W + iw;
          int wt_idx = ((k * C + c) * R + r) * S + s;
          val += input[in_idx] * weights[wt_idx];
        }
      }
    }
  }

  int out_idx = ((n * K + k) * outH + oh) * outW + ow;
  output[out_idx] = val;
}

void launch_conv2d_kernel(float *input, float *weights, float *output,
                          const std::vector<size_t> &inShape,
                          const std::vector<size_t> &outShape,
                          const std::vector<size_t> &kernelShape,
                          const std::vector<int> &strides) {
  int N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
  int K = outShape[1], outH = outShape[2], outW = outShape[3];
  int R = kernelShape[0], S = kernelShape[1];
  int strideH = strides[0], strideW = strides[1];

  dim3 grid(outW, outH, N * K); // 3D grid: (ow, oh, n*k)
  dim3 block(1);

  conv2d_kernel<<<grid, block>>>(input, weights, output, //
                                 C, H, W, K, R, S, outH, outW, strideH,
                                 strideW);
}
