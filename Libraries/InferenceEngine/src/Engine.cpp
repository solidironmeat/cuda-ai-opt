#include "Engine.hpp"

#include "Layers/AddLayer.hpp"
#include "Layers/ConvLayer.hpp"
#include "Layers/FlattenLayer.hpp"
#include "Layers/GemmLayer.hpp"
#include "Layers/GlobalAveragePoolLayer.hpp"
#include "Layers/MaxpoolLayer.hpp"
#include "Layers/ReluLayer.hpp"
#include "Layers/SoftmaxLayer.hpp"
#include "Utils/NpyLoader.hpp"
#include "Utils/OnnxParser.hpp"

#include <iostream>

Engine::Engine(const std::string &modelPath) : modelPath(modelPath) {}

Engine::~Engine() { freeResources(); }

void Engine::loadModel() {
  std::cout << "[Engine] Loading model: " << modelPath << std::endl;

  // Parse ONNX model using custom parser
  auto graph = parseONNX(modelPath); // returns vector<LayerConfig>

  std::cout << "[Engine] ONNX model parsed \n";

  // Build execution graph (layer objects)
  for (const auto &config : graph) {
    // config.printInfoCompact();

    if (config.type == "Add") {
      layerGraph.push_back(new AddLayer(config));
    } else if (config.type == "Conv") {
      layerGraph.push_back(new ConvLayer(config));
    } else if (config.type == "Flatten") {
      layerGraph.push_back(new FlattenLayer(config));
    } else if (config.type == "Gemm") {
      layerGraph.push_back(new GemmLayer(config));
    } else if (config.type == "GlobalAveragePool") {
      layerGraph.push_back(new GlobalAveragePoolLayer(config));
    } else if (config.type == "MaxPool") {
      layerGraph.push_back(new MaxPoolLayer(config));
    } else if (config.type == "Relu") {
      layerGraph.push_back(new ReluLayer(config));
    } else if (config.type == "Softmax") {
      layerGraph.push_back(new SoftmaxLayer(config));
    } else {
      std::cerr << "Unsupported layer type: " << config.type << std::endl;
    }
  }

  allocateMemory();
}

void Engine::allocateMemory() {
  // Allocate input/output tensor memory on GPU
  inputTensor.allocateDevice();
  outputTensor.allocateDevice();

  for (auto *layer : layerGraph)
    layer->allocate();
}

void Engine::runInference(const std::string &inputPath) {
  std::cout << "[Engine] Running inference: " << inputPath << std::endl;

  // Load input tensor from .npy file (CPU side)
  loadNpy(inputPath, inputTensor); // reloads host + device
  Tensor *current = &inputTensor;

  // Execute layer-by-layer
  // for (auto *layer : layerGraph) {
  //   layer->forward(*current);
  //   current = layer->getOutput();
  // }

  // Execute first layer only
  auto layer = layerGraph[0];
  layer->forward(*current);
  current = layer->getOutput();

  // Copy output back to host
  outputTensor = *current; // <== Fix: ensures host+device pointers valid
  outputTensor.copyToHost();
}

std::vector<float> Engine::getOutput() {
  float *data = outputTensor.getHostData();
  size_t size = outputTensor.size();
  return std::vector<float>(data, data + size);
}

void Engine::freeResources() {
  for (auto *layer : layerGraph) {
    delete layer;
  }
  layerGraph.clear();
  inputTensor.free();
  outputTensor.free();
}
