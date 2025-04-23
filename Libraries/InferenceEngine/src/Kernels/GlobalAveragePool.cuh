#ifndef GLOBAL_AVG_POOL_LAYER_CUH
#define GLOBAL_AVG_POOL_LAYER_CUH

void launch_global_avg_pool_kernel(const float *input, float *output, int N,
                                   int C, int H, int W);

#endif // GLOBAL_AVG_POOL_LAYER_CUH