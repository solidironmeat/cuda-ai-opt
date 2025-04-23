#ifndef GLOBAL_AVG_POOL_LAYER_H
#define GLOBAL_AVG_POOL_LAYER_H

#include "Layer.hpp"

#include "Utils/LayerConfig.hpp"
#include "Utils/Tensor.hpp"

class GlobalAveragePoolLayer : public Layer {
public:
  GlobalAveragePoolLayer(const LayerConfig &config);
  void allocate() override;
  void forward(Tensor &input) override;
  void forward(const std::vector<Tensor *> &inputs) override;
  Tensor *getOutput() override;

private:
  Tensor weights;
  Tensor output;
  LayerConfig config;
};

#endif // GLOBAL_AVG_POOL_LAYER_H