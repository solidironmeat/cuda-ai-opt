#ifndef ONNX_PARSER
#define ONNX_PARSER

#include "Utils/LayerConfig.hpp"

#include <string>
#include <vector>

std::vector<LayerConfig> parseONNX(const std::string &modelPath);

#endif // ONNX_PARSER