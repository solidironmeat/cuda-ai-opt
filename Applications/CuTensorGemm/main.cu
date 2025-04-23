#include <cassert>
#include <cuda_runtime.h>

#include <cutensor.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(err)                                                        \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUTENSOR(err)                                                    \
  do {                                                                         \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      std::cerr << "cuTENSOR error: " << cutensorGetErrorString(err) << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main() {
  // Print cuTENSOR version for debugging
  std::cout << "cuTENSOR version: " << cutensorGetVersion() << std::endl;

  // Create cuTENSOR handle
  cutensorHandle_t handle;
  CHECK_CUTENSOR(cutensorCreate(&handle));

  // Matrix dimensions (input [m,k], weights [k,n], output [m,n])
  int32_t m = 2; // batch size
  int32_t n = 3; // output features
  int32_t k = 4; // input features

  // Allocate device memory
  float *d_input, *d_weights, *d_output;
  CHECK_CUDA(cudaMalloc(&d_input, m * k * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_weights, k * n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_output, m * n * sizeof(float)));

  const uint32_t kAlignment = 128;
  assert(uintptr_t(d_input) % kAlignment == 0);
  assert(uintptr_t(d_weights) % kAlignment == 0);
  assert(uintptr_t(d_output) % kAlignment == 0);

  // TensorDescriptor
  cutensorTensorDescriptor_t descA;
  std::vector<int64_t> extentA = {m, k}; // [m,k]
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descA, 2,
                                                extentA.data(), NULL, //
                                                CUTENSOR_R_32F, kAlignment));
  cutensorTensorDescriptor_t descB;
  std::vector<int64_t> extentB = {k, n}; // [k,n]
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descB, 2,
                                                extentB.data(), NULL, //
                                                CUTENSOR_R_32F, kAlignment));
  cutensorTensorDescriptor_t descC;
  std::vector<int64_t> extentC = {m, n}; // [m,n]
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(handle, &descC, 2,
                                                extentC.data(), NULL, //
                                                CUTENSOR_R_32F, kAlignment));

  // Initialize input data (same as original)
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,  //
                              1.0f, 2.0f, 3.0f, 4.0f}; // input [2,4]
  std::vector<float> weights = {
      1.0f, 0.0f, 0.0f, 1.0f, // w0
      0.0f, 1.0f, 1.0f, 0.0f, // w1
      1.0f, 1.0f, 1.0f, 1.0f  // [4,3]
  };
  std::vector<float> output(m * n);

  // Copy to device
  CHECK_CUDA(cudaMemcpy(d_input, input.data(), m * k * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), k * n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Set up contraction plan
  cutensorOperationDescriptor_t opDesc;
  int32_t modeA[] = {0, 1}; // [m,k]
  int32_t modeB[] = {1, 2}; // [k,n]
  int32_t modeC[] = {0, 2}; // [m,n]
  const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
  CHECK_CUTENSOR(
      cutensorCreateContraction(handle, &opDesc,                    //
                                descA, modeA, CUTENSOR_OP_IDENTITY, //
                                descB, modeB, CUTENSOR_OP_IDENTITY, //
                                descC, modeC, CUTENSOR_OP_IDENTITY, //
                                descC, modeC, descCompute));

  // Create plan
  cutensorPlanPreference_t planPref;
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
  CHECK_CUTENSOR(cutensorCreatePlanPreference(handle, &planPref, algo,
                                              CUTENSOR_JIT_MODE_NONE));

  // Query workspace estimate
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  CHECK_CUTENSOR(cutensorEstimateWorkspaceSize(
      handle, opDesc, planPref, workspacePref, &workspaceSizeEstimate));

  // Allocate workspace
  void *d_workspace = nullptr;
  if (workspaceSizeEstimate > 0) {
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSizeEstimate));
  }

  cutensorPlan_t plan;
  CHECK_CUTENSOR(cutensorCreatePlan(handle, &plan, opDesc, planPref,
                                    workspaceSizeEstimate));

  // Execute contraction
  cudaStream_t stream;
  float alpha = 1.0;
  float beta = 0.0;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUTENSOR(cutensorContract(handle, plan,                       //
                                  (void *)&alpha, d_input, d_weights, //
                                  (void *)&beta, d_output, d_output,  //
                                  d_workspace, workspaceSizeEstimate, stream));

  // Synchronize stream before copying results
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Copy result back to host
  CHECK_CUDA(cudaMemcpy(output.data(), d_output, m * n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Print output
  std::cout << "Output: ";
  for (float val : output) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  // Cleanup
  CHECK_CUTENSOR(cutensorDestroyPlan(plan));
  CHECK_CUTENSOR(cutensorDestroyPlanPreference(planPref));
  CHECK_CUTENSOR(cutensorDestroyOperationDescriptor(opDesc));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(descA));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(descB));
  CHECK_CUTENSOR(cutensorDestroyTensorDescriptor(descC));
  CHECK_CUTENSOR(cutensorDestroy(handle));
  CHECK_CUDA(cudaStreamDestroy(stream));
  if (d_workspace)
    CHECK_CUDA(cudaFree(d_workspace));
  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_weights));
  CHECK_CUDA(cudaFree(d_output));

  return 0;
}