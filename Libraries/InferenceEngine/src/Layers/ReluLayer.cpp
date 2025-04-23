#include "ReluLayer.hpp"
#include "Kernels/CudaUtils.cuh"
#include "Kernels/Relu.cuh"

ReluLayer::ReluLayer(const LayerConfig &cfg) : config(cfg) {
  output = Tensor(config.outputShape);
}

void ReluLayer::allocate() { output.allocateDevice(); }

void ReluLayer::forward(Tensor &input) { forward({&input}); }

void ReluLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];

  int size = static_cast<int>(input.size());
  const float *in_data = input.getDeviceData();
  float *out_data = output.getDeviceData();

  launch_relu_kernel(in_data, out_data, size);
  CHECK_CUDA(cudaGetLastError());
}

Tensor *ReluLayer::getOutput() { return &output; }