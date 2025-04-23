#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "Layer.hpp"

#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class SoftmaxLayer : public Layer {
public:
  SoftmaxLayer(const LayerConfig &config);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor weights;
  Tensor output;
  LayerConfig config;
};

#endif // SOFTMAX_LAYER_H