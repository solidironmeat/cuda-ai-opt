#include "GlobalAveragePoolLayer.hpp"
#include "Kernels/CudaUtils.cuh"
#include "Kernels/GlobalAveragePool.cuh"

GlobalAveragePoolLayer::GlobalAveragePoolLayer(const LayerConfig &cfg)
    : config(cfg) {
  assert(cfg.outputShape.size() == 4); // Ensure {N,C,1,1}
  output = Tensor(cfg.outputShape);
}

void GlobalAveragePoolLayer::allocate() { output.allocateDevice(); }

void GlobalAveragePoolLayer::forward(Tensor &input) { forward({&input}); }

void GlobalAveragePoolLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];
  const auto &inShape = config.inputShape;
  assert(inShape.size() == 4); // NCHW
  int N = inShape[0];
  int C = inShape[1];
  int H = inShape[2];
  int W = inShape[3];

  launch_global_avg_pool_kernel(input.getDeviceData(), output.getDeviceData(),
                                N, C, H, W);

  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

Tensor *GlobalAveragePoolLayer::getOutput() { return &output; }
