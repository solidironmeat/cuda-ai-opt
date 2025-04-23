# Kernel Design

## conv2d.cu
- Implements 2D convolution using tiled shared memory
- Optimized for 3x3 and 7x7 kernel sizes
- Avoids bank conflicts and promotes coalesced access

## gemm.cu
- Implements matrix-matrix multiplication using tiling
- Uses shared memory and warp-level sync for performance
- Benchmarked against cuBLAS

## relu.cu / softmax.cu
- Element-wise operations using one thread per element
- Handles float32 and float16 input types
