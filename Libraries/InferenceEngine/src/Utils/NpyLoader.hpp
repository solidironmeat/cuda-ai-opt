#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include "Utils/Tensor.hpp"

#include <string>

// Load .npy file into a Tensor (host memory)
void loadNpy(const std::string &path, Tensor &tensor);

// Save a Tensor (host memory) to .npy
void saveNpy(const std::string &path, const Tensor &tensor);

#endif // NPY_LOADER_H