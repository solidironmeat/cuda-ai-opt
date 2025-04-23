#include "FlattenLayer.hpp"
#include "Kernels/CudaUtils.cuh"

FlattenLayer::FlattenLayer(const LayerConfig &cfg) : config(cfg) {
  output = Tensor(cfg.outputShape);
}

void FlattenLayer::allocate() { output.allocateDevice(); }

void FlattenLayer::forward(Tensor &input) {
  assert(input.size() == output.size());
  CHECK_CUDA(cudaMemcpy(output.getDeviceData(), input.getDeviceData(),
                        input.size() * sizeof(float),
                        cudaMemcpyDeviceToDevice));
}

void FlattenLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  forward(*inputs[0]);
}

Tensor *FlattenLayer::getOutput() { return &output; }