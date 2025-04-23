#ifndef ADD_LAYER_H
#define ADD_LAYER_H

#include "Layer.hpp"

#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class AddLayer : public Layer {
public:
  AddLayer(const LayerConfig &cfg);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor output;
  LayerConfig config;
};

#endif // ADD_LAYER_H