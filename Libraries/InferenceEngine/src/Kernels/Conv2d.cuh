#ifndef CONV2D_CUH
#define CONV2D_CUH

#include <vector>

void launch_conv2d_kernel(float *input, float *weights, float *output,
                          const std::vector<size_t> &inShape,
                          const std::vector<size_t> &outShape,
                          const std::vector<size_t> &kernelShape,
                          const std::vector<int> &strides);

#endif // CONV2D_CUH