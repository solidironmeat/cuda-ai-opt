#ifndef ILAYER_H
#define ILAYER_H

#include "Utils/Tensor.hpp"

class Layer {
public:
  virtual ~Layer() {}

  virtual void allocate() = 0;

  // Default: wrap single input in vector
  virtual void forward(Tensor &input) {
    std::vector<Tensor *> inputs = {&input};
    forward(inputs);
  }

  // Generic multi-input forward (overridable)
  virtual void forward(const std::vector<Tensor *> &inputs) = 0;

  virtual Tensor *getOutput() = 0;
};

#endif // ILAYER_H