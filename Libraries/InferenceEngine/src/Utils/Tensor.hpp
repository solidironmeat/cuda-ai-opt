#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Tensor {
public:
  Tensor(const std::string &label = "Unknown");
  Tensor(std::vector<size_t> shape, const std::string &label = "Unknown");

  void allocateDevice();
  void copyToDevice();
  void copyToHost();
  void free();

  float *getDeviceData() const;
  float *getHostData() const;
  std::vector<size_t> getShape() const;
  size_t size() const;

  void printInfo() const {
    std::cout << "==== Tensor Info";
    if (!label.empty())
      std::cout << " [" << label << "]";
    std::cout << " ====\n";
    std::cout << "Shape      : [" << join(shape) << "]\n";
    std::cout << "Total size : " << totalSize << " elements\n";
    std::cout << "Host ptr   : " << static_cast<const void *>(hostData) << "\n";
    std::cout << "Device ptr : " << static_cast<const void *>(deviceData)
              << "\n";
    std::cout << "=================================\n";
  }

  void printContent() const {
    std::cout << "==== Tensor Content";
    if (!label.empty())
      std::cout << " [" << label << "]";
    std::cout << " ====\n";

    if (!hostData) {
      std::cout << "(hostData is null — did you call copyToHost()?)\n";
      return;
    }

    size_t cols = shape.size() > 1 ? shape.back() : totalSize;
    for (size_t i = 0; i < totalSize; ++i) {
      std::cout << hostData[i] << " ";
      if ((i + 1) % cols == 0)
        std::cout << "\n";
    }
    std::cout << "=============================\n";
  }

private:
  std::vector<size_t> shape;
  size_t totalSize;
  float *hostData;
  float *deviceData;
  std::string label;

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

#endif // TENSOR_H