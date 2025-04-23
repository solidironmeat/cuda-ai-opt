#ifndef ENGINE_H
#define ENGINE_H

#include <string>
#include <vector>

#include "Layers/Layer.hpp"
#include "Utils/Tensor.hpp"

class Engine {
public:
  Engine(const std::string &modelPath);
  ~Engine();

  void loadModel();
  void runInference(const std::string &inputPath);
  std::vector<float> getOutput();

private:
  std::string modelPath;
  std::vector<Layer *> layerGraph;
  Tensor inputTensor;
  Tensor outputTensor;

  void buildExecutionGraph(); // From parsed ONNX
  void allocateMemory();
  void freeResources();
};

#endif // ENGINE_H