#include "SoftmaxLayer.hpp"
#include "Kernels/CudaUtils.cuh"
#include "Kernels/Softmax.cuh"

SoftmaxLayer::SoftmaxLayer(const LayerConfig &cfg) : config(cfg) {
  output = Tensor(cfg.outputShape, "softmax_output");
}

void SoftmaxLayer::allocate() { output.allocateDevice(); }

void SoftmaxLayer::forward(Tensor &input) { forward({&input}); }

void SoftmaxLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];

  int N = config.inputShape[0];
  int C = config.inputShape[1];

  const float *in_data = input.getDeviceData();
  float *out_data = output.getDeviceData();

  launch_softmax_kernel(in_data, out_data, N, C);

  CHECK_CUDA(cudaGetLastError());
}

Tensor *SoftmaxLayer::getOutput() { return &output; }
