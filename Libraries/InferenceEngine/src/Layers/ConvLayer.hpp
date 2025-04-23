#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "Layer.hpp"

#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class ConvLayer : public Layer {
public:
  ConvLayer(const LayerConfig &config);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor weights;
  Tensor output;
  LayerConfig config;
};

#endif // CONV_LAYER_H