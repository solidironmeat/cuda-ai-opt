#include "MaxPoolLayer.hpp"
#include "Kernels/CudaUtils.cuh"
#include "Kernels/MaxPool.cuh"

MaxPoolLayer::MaxPoolLayer(const LayerConfig &cfg) : config(cfg) {
  assert(cfg.outputShape.size() == 4);
  output = Tensor(cfg.outputShape, "maxpool_output");
}

void MaxPoolLayer::allocate() { output.allocateDevice(); }

void MaxPoolLayer::forward(Tensor &input) { forward({&input}); }

void MaxPoolLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];

  const auto &inShape = config.inputShape;
  const auto &outShape = config.outputShape;
  assert(inShape.size() == 4 && outShape.size() == 4);

  int N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
  int outH = outShape[2], outW = outShape[3];
  int poolH = config.kernelShape[0], poolW = config.kernelShape[1];
  int strideH = config.strides[0], strideW = config.strides[1];

  assert(N == 1 && "Only N==1 supported in current MaxPoolLayer");

  launch_maxpool2d_kernel(input.getDeviceData(), output.getDeviceData(), N, C,
                          H, W, outH, outW, poolH, poolW, strideH, strideW);

  CHECK_CUDA(cudaGetLastError());
}

Tensor *MaxPoolLayer::getOutput() { return &output; }