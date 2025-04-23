#ifndef MAXPOOL_CUH
#define MAXPOOL_CUH

void launch_maxpool2d_kernel(const float *input, float *output, int N, int C,
                             int H, int W, int outH, int outW, int poolH,
                             int poolW, int strideH, int strideW);

#endif // MAXPOOL_CUH