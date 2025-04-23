#include "Layers/ConvLayer.hpp"

#include "Kernels/Conv2d.cuh"
#include "Kernels/CudaUtils.cuh"

ConvLayer::ConvLayer(const LayerConfig &cfg) : config(cfg) {
  assert(cfg.outputShape.size() >= 2 && cfg.inputShape.size() >= 2);

  weights = Tensor({cfg.outputShape[1], //
                    cfg.inputShape[1],  //
                    cfg.kernelShape[0], //
                    cfg.kernelShape[1]},
                   "ConvLayer.weights");
  output = Tensor(cfg.outputShape, "ConvLayer.output");
  std::copy(cfg.weights.begin(), cfg.weights.end(), weights.getHostData());
}

void ConvLayer::allocate() {
  weights.allocateDevice();
  weights.copyToDevice();
  output.allocateDevice();
}

void ConvLayer::forward(Tensor &input) { forward({&input}); }

void ConvLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];

  input.printInfo();

  assert(input.getDeviceData());
  assert(weights.getDeviceData());
  assert(output.getDeviceData());

  launch_conv2d_kernel(input.getDeviceData(),   //
                       weights.getDeviceData(), //
                       output.getDeviceData(),  //
                       config.inputShape,       //
                       config.outputShape,      //
                       config.kernelShape,      //
                       config.strides);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

Tensor *ConvLayer::getOutput() { return &output; }
