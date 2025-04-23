#ifndef MAXPOOL_LAYER_HPP
#define MAXPOOL_LAYER_HPP

#include "Layer.hpp"

#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class MaxPoolLayer : public Layer {
public:
  MaxPoolLayer(const LayerConfig &config);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor weights;
  Tensor output;
  LayerConfig config;
};

#endif // MAXPOOL_LAYER_HPP