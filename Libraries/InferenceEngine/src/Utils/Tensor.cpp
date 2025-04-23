#include "Utils/Tensor.hpp"

#include "Kernels/CudaUtils.cuh"

Tensor::Tensor(const std::string &label)
    : hostData(nullptr), deviceData(nullptr), totalSize(0), label(label) {}

Tensor::Tensor(std::vector<size_t> shape, const std::string &label)
    : shape(shape), label(label) {
  totalSize = 1;
  for (int s : shape)
    totalSize *= s;
  hostData = new float[totalSize];
  deviceData = nullptr;
}

void Tensor::allocateDevice() {
  CHECK_CUDA(cudaMalloc(&deviceData, totalSize * sizeof(float)));
}

void Tensor::copyToDevice() {
  CHECK_CUDA(cudaMemcpy(deviceData, hostData, totalSize * sizeof(float),
                        cudaMemcpyHostToDevice));
}

void Tensor::copyToHost() {
  CHECK_CUDA(cudaMemcpy(hostData, deviceData, totalSize * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

void Tensor::free() {
  if (hostData)
    delete[] hostData;
  if (deviceData)
    CHECK_CUDA(cudaFree(deviceData));
  hostData = nullptr;
  deviceData = nullptr;
}

float *Tensor::getDeviceData() const { return deviceData; }
float *Tensor::getHostData() const { return hostData; }

std::vector<size_t> Tensor::getShape() const { return shape; }

size_t Tensor::size() const { return totalSize; }
