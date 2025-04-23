#include "Engine.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

void save_latency_data(const std::vector<float> &latencies,
                       const std::string &filename) {
  std::ofstream file(filename);
  for (float time : latencies)
    file << time << "\n";
  file.close();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <input.npy>\n";
    return 1;
  }

  const char *modelPath = argv[1];
  const char *inputPath = argv[2];

  Engine engine(modelPath);
  engine.loadModel();

  std::vector<float> latencies;
  for (int i = 0; i < 100; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    engine.runInference(inputPath);
    auto end = std::chrono::high_resolution_clock::now();

    float elapsed =
        std::chrono::duration<float, std::milli>(end - start).count();
    latencies.push_back(elapsed);
  }

  save_latency_data(latencies, "models/latency.csv");
  std::cout << "Saved latency results.\n";
}
