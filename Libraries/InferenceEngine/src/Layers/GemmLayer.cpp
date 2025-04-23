#include "Layers/GemmLayer.hpp"
#include "Kernels/CudaUtils.cuh"
#include "Kernels/Gemm.cuh"

GemmLayer::GemmLayer(const LayerConfig &cfg) : config(cfg) {
  assert(cfg.inputShape.size() == 2 && cfg.outputShape.size() == 2);
  size_t K = cfg.inputShape[1];
  size_t N = cfg.outputShape[1];

  if (cfg.weights.size() != K * N) {
    std::cout << "[GemmLayer] Expected weight size = " << (K * N)
              << ", actual = " << cfg.weights.size() << "\n";
  }

  weights = Tensor({K, N}, "weights");
  output = Tensor(cfg.outputShape, "output");

  std::copy(cfg.weights.begin(), cfg.weights.end(), weights.getHostData());
}

void GemmLayer::allocate() {
  weights.allocateDevice();
  weights.copyToDevice();
  output.allocateDevice();
}

void GemmLayer::forward(Tensor &input) { forward({&input}); }

void GemmLayer::forward(const std::vector<Tensor *> &inputs) {
  assert(inputs.size() == 1);
  Tensor &input = *inputs[0];

  int M = config.inputShape[0];
  int K = config.inputShape[1];
  int N = config.outputShape[1];

  assert(M <= 0 || N <= 0 || K <= 0); // Invalid matrix dimensions

  launch_gemm_kernel(input.getDeviceData(), weights.getDeviceData(),
                     output.getDeviceData(), M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}

Tensor *GemmLayer::getOutput() { return &output; }
