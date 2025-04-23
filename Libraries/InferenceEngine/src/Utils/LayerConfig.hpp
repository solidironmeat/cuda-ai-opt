#ifndef LAYER_CONFIG_H
#define LAYER_CONFIG_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct LayerConfig {
  std::string type;                // Layer type: "Conv", "Gemm", etc.
  std::vector<size_t> inputShape;  // Input tensor shape [N, C, H, W] or [N, K]
  std::vector<size_t> outputShape; // Output tensor shape
  std::vector<float> weights;      // Flattened weight array
  std::vector<float> bias;         // Optional bias (not required for all ops)

  // For convolution layers
  std::vector<size_t> kernelShape; // Kernel size [R, S] for Conv2D
  std::vector<int> strides;        // Stride [strideH, strideW]
  std::vector<int> pads;           // Padding [top, left, bottom, right]

  // Optional: for GEMM or others
  bool transA = false;
  bool transB = false;

  // Optional for debugging/logging
  std::string name;

  void printInfo() const {
    std::cout << "===== LayerConfig: " << (name.empty() ? "(unnamed)" : name)
              << " =====\n";
    std::cout << "Type       : " << type << "\n";
    std::cout << "InputShape : [" << join(inputShape) << "]\n";
    std::cout << "OutputShape: [" << join(outputShape) << "]\n";

    if (!weights.empty()) {
      std::cout << "Weights    : " << weights.size() << " values\n";
    }
    if (!bias.empty()) {
      std::cout << "Bias       : " << bias.size() << " values\n";
    }

    if (type == "Conv") {
      std::cout << "Kernel     : [" << join(kernelShape) << "]\n";
      std::cout << "Strides    : [" << join(strides) << "]\n";
      std::cout << "Pads       : [" << join(pads) << "]\n";
    }

    if (type == "Gemm") {
      std::cout << "Transpose A: " << (transA ? "true" : "false") << "\n";
      std::cout << "Transpose B: " << (transB ? "true" : "false") << "\n";
    }

    std::cout << "=========================================\n";
  }

  void printInfoCompact() const {
    std::cout << type << "\t";
    std::cout << "in:\t[" << join(inputShape) << "],\t";
    std::cout << "out:\t[" << join(outputShape) << "]\n";
  }

private:
  template <typename T> std::string join(const std::vector<T> &vec) const {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i + 1 < vec.size())
        oss << ", ";
    }
    return oss.str();
  }
};

#endif // LAYER_CONFIG_H