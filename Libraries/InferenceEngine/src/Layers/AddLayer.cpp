#include "AddLayer.hpp"
#include "Kernels/Add.cuh"

#include <cassert>

AddLayer::AddLayer(const LayerConfig &cfg) : config(cfg) {
  output = Tensor(cfg.outputShape);
}

void AddLayer::allocate() { output.allocateDevice(); }

void AddLayer::forward(Tensor & /*input*/) {
  // not supported; Add requires 2 inputs
  throw std::runtime_error("AddLayer requires 2 inputs.");
}

void AddLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 2);
  Tensor &A = *inputs[0];
  Tensor &B = *inputs[1];
  assert(A.size() == B.size() && A.size() == output.size());

  launch_add_kernel(A.getDeviceData(),      //
                    B.getDeviceData(),      //
                    output.getDeviceData(), //
                    A.size());
  // CHECK_CUDA(cudaGetLastError());
}

Tensor *AddLayer::getOutput() { return &output; }