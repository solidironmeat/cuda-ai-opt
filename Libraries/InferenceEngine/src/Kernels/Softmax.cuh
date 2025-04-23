#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

void launch_softmax_kernel(const float *input, float *output, int N, int C);

#endif // SOFTMAX_CUH
