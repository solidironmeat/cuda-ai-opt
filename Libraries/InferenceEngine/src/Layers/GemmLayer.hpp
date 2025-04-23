#ifndef GEMM_LAYER_H
#define GEMM_LAYER_H

#include "Layer.hpp"
#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class GemmLayer : public Layer {
public:
  GemmLayer(const LayerConfig &config);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor weights;
  Tensor output;
  LayerConfig config;
};

#endif // GEMM_LAYER_H