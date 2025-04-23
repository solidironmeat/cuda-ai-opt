#include "Layers/ConvLayer.hpp"
#include "Layers/GemmLayer.hpp"

#include <cmath>

#include <gtest/gtest.h>

bool closeEnough(float a, float b, float tol = 1e-3f) {
  return std::fabs(a - b) < tol;
}

// === GEMM TEST === //
TEST(KernelTest, GemmLayerCorrectness) {
  LayerConfig cfg;
  cfg.type = "Gemm";
  cfg.inputShape = {1, 4};
  cfg.outputShape = {1, 3};
  cfg.weights = {
      1, 0, 1, // row 0
      0, 1, 1, // row 1
      0, 1, 1, // row 2
      1, 0, 1  // row 3
  };

  GemmLayer layer(cfg);
  layer.allocate();

  Tensor input(cfg.inputShape, "input");
  float *x = input.getHostData();
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;
  input.allocateDevice();
  input.copyToDevice();

  layer.forward(input);
  Tensor *out = layer.getOutput();
  out->copyToHost();
  float *y = out->getHostData();

  ASSERT_TRUE(closeEnough(y[0], 1 * 1 + 2 * 0 + 3 * 0 + 4 * 1)); // = 5
  ASSERT_TRUE(closeEnough(y[1], 1 * 0 + 2 * 1 + 3 * 1 + 4 * 0)); // = 5
  ASSERT_TRUE(closeEnough(y[2], 1 + 2 + 3 + 4));                 // = 10
}

// === CONV TEST === //
TEST(KernelTest, ConvLayerSimple3x3) {
  LayerConfig cfg;
  cfg.type = "Conv";
  cfg.inputShape = {1, 1, 3, 3};
  cfg.outputShape = {1, 1, 1, 1};
  cfg.kernelShape = {3, 3};
  cfg.strides = {1, 1};
  cfg.weights = {1, 0, -1, 1, 0, -1, 1, 0, -1};

  ConvLayer layer(cfg);
  layer.allocate();

  Tensor input(cfg.inputShape);
  float *x = input.getHostData();
  // Input: horizontal edge detector
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  x[3] = 4;
  x[4] = 5;
  x[5] = 6;
  x[6] = 7;
  x[7] = 8;
  x[8] = 9;
  input.allocateDevice();
  input.copyToDevice();

  layer.forward(input);
  Tensor *out = layer.getOutput();
  out->copyToHost();
  float *y = out->getHostData();

  float expected = (1 + 4 + 7) * 1 + (3 + 6 + 9) * -1; // = (12) - (18) = -6
  ASSERT_TRUE(closeEnough(y[0], expected));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}