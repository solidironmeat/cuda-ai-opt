#ifndef GEMM_CUH
#define GEMM_CUH

void launch_gemm_kernel(float *A, float *B, float *C, int M, int N, int K);

#endif // GEMM_CUH
