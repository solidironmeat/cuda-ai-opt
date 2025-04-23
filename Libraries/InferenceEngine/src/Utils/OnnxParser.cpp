#include "Utils/OnnxParser.hpp"

#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <fstream>
#include <iostream>

using ShapeMap = std::unordered_map<std::string, std::vector<size_t>>;

ShapeMap extractShapes(const onnx::GraphProto &graph) {
  ShapeMap shapeMap;

  auto parseShape = [](const onnx::TypeProto_Tensor &tensor_type) {
    std::vector<size_t> shape;
    for (const auto &dim : tensor_type.shape().dim()) {
      if (dim.has_dim_value())
        shape.push_back(dim.dim_value());
      else
        shape.push_back(-1); // Unknown dimension
    }
    return shape;
  };

  // Inputs
  for (const auto &vi : graph.input()) {
    const auto &shape = vi.type().tensor_type();
    shapeMap[vi.name()] = parseShape(shape);
  }

  for (const auto &vi : graph.value_info()) {
    const auto &shape = vi.type().tensor_type();
    shapeMap[vi.name()] = parseShape(shape);
  }

  // Outputs
  for (const auto &vi : graph.output()) {
    const auto &shape = vi.type().tensor_type();
    shapeMap[vi.name()] = parseShape(shape);
  }

  return shapeMap;
}

std::string shapeToString(const std::vector<size_t> &shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (i + 1 < shape.size())
      s += ", ";
  }
  return s + "]";
}

void printNodeInfo(const onnx::NodeProto &node, const ShapeMap &shapeMap) {
  std::cout << "\n[ONNX Node] ----------------------------------------\n";
  // std::cout << "  Name       : " << (node.name().empty() ? "(unnamed)" :
  // node.name()) << "\n";
  std::cout << "  Type       : " << node.op_type() << "\n";
  // std::cout << "  Domain     : " << node.domain() << "\n";

  std::cout << "  Inputs     : ";
  for (const auto &input : node.input()) {
    std::cout << input;
    if (shapeMap.count(input))
      std::cout << " " << shapeToString(shapeMap.at(input));
    std::cout << "  ";
  }
  std::cout << "\n";

  std::cout << "  Outputs    : ";
  for (const auto &output : node.output()) {
    std::cout << output;
    if (shapeMap.count(output))
      std::cout << " " << shapeToString(shapeMap.at(output));
    std::cout << "  ";
  }
  std::cout << "\n";

  // if (!node.attribute().empty()) {
  //   std::cout << "  Attributes :\n";
  //   for (const auto &attr : node.attribute()) {
  //     std::cout << "    - " << attr.name() << " (type " << attr.type() << "):
  //     "; switch (attr.type()) { case onnx::AttributeProto::INT:
  //       std::cout << attr.i();
  //       break;
  //     case onnx::AttributeProto::FLOAT:
  //       std::cout << attr.f();
  //       break;
  //     case onnx::AttributeProto::STRING:
  //       std::cout << attr.s();
  //       break;
  //     case onnx::AttributeProto::INTS:
  //       for (auto v : attr.ints())
  //         std::cout << v << " ";
  //       break;
  //     case onnx::AttributeProto::FLOATS:
  //       for (auto v : attr.floats())
  //         std::cout << v << " ";
  //       break;
  //     default:
  //       std::cout << "[unsupported type]";
  //     }
  //     std::cout << "\n";
  //   }
  // }

  std::cout << "----------------------------------------------------\n";
}

std::vector<LayerConfig> parseONNX(const std::string &modelPath) {
  onnx::ModelProto model;

  // Load model from file
  std::fstream in(modelPath, std::ios::in | std::ios::binary);
  if (!model.ParseFromIstream(&in)) {
    throw std::runtime_error("Failed to load or parse ONNX model: " +
                             modelPath);
  }

  onnx::shape_inference::InferShapes(model);

  const onnx::GraphProto &graph = model.graph();

  ShapeMap shapeMap = extractShapes(graph);

  std::cout << "[ONNX Parser] Nodes in model: " << graph.node_size()
            << std::endl;

  std::vector<LayerConfig> layers;
  for (const auto &node : graph.node()) {
    LayerConfig cfg;
    cfg.name = node.name();
    cfg.type = node.op_type();

    // Shapes
    if (!node.input().empty() && shapeMap.count(node.input(0)))
      cfg.inputShape = shapeMap[node.input(0)];
    if (!node.output().empty() && shapeMap.count(node.output(0)))
      cfg.outputShape = shapeMap[node.output(0)];

    if (cfg.type == "Conv") {
      for (const auto &init : graph.initializer()) {
        if (init.name() == node.input(1)) {
          int oc = init.dims(0);
          int ic = init.dims(1);
          size_t kH = init.dims(2);
          size_t kW = init.dims(3);
          cfg.kernelShape = {kH, kW};
          cfg.weights.reserve(ic * oc * kH * kW);
          for (float f : init.float_data())
            cfg.weights.push_back(f);
          break;
        }
      }
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "strides") {
          cfg.strides.assign(attr.ints().begin(), attr.ints().end());
        } else if (attr.name() == "pads") {
          cfg.pads.assign(attr.ints().begin(), attr.ints().end());
        }
      }

    } else if (cfg.type == "Gemm") {
      cfg.transA = false;
      cfg.transB = false;
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "transA")
          cfg.transA = attr.i();
        if (attr.name() == "transB")
          cfg.transB = attr.i();
      }

      // Default assumption: weights in input(1)
      for (const auto &init : graph.initializer()) {
        if (init.name() == node.input(1)) {
          int M = init.dims(0);
          int N = init.dims(1);
          cfg.weights.reserve(M * N);
          for (float f : init.float_data())
            cfg.weights.push_back(f);

          // Apply transpose if needed
          if (cfg.transB && !init.float_data().empty()) {
            std::vector<float> transposed(M * N);
            for (int i = 0; i < M; ++i)
              for (int j = 0; j < N; ++j)
                transposed[j * M + i] = cfg.weights[i * N + j];
            cfg.weights = std::move(transposed);
          }

          break;
        }
      }
    } else if (cfg.type == "Relu" || cfg.type == "Softmax") {
      // Stateless ops, no weights needed

    } else if (cfg.type == "MaxPool") {
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "kernel_shape") {
          cfg.kernelShape.assign(attr.ints().begin(), attr.ints().end());
        } else if (attr.name() == "strides") {
          cfg.strides.assign(attr.ints().begin(), attr.ints().end());
        } else if (attr.name() == "pads") {
          cfg.pads.assign(attr.ints().begin(), attr.ints().end());
        }
      }

    } else if (cfg.type == "Add") {
      // Add may use broadcast, but for now assume same-shape tensors
      // No weight parsing needed

    } else if (cfg.type == "GlobalAveragePool") {
      // No attributes required; shape info only

    } else if (cfg.type == "Flatten") {
      for (const auto &attr : node.attribute()) {
        if (attr.name() == "axis") {
          // Store axis if needed later
        }
      }

    } else {
      std::cerr << "[ONNX Parser] Skipping unsupported node: " << cfg.type
                << std::endl;
      continue;
    }

    // 0printNodeInfo(node, shapeMap);
    // cfg.printInfo();

    layers.push_back(std::move(cfg));
  }

  return layers;
}