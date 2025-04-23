#include "Kernels/DeviceUtils.cuh"

int main(int /*argc*/, char ** /*argv*/) {
  printCudaDeviceInfo();
  return 0;
}