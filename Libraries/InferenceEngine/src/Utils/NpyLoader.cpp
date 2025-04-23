#include "Utils/NpyLoader.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace {
std::vector<size_t> parseShape(const std::string &header) {
  std::regex shapeRegex(R"('shape'\s*:\s*\(([^)]*)\))");
  std::smatch match;

  if (!std::regex_search(header, match, shapeRegex)) {
    throw std::runtime_error("Shape not found in .npy header: " +
                             header.substr(0, 80));
  }

  std::string shapeStr = match[1]; // E.g., "1, 1000,"
  std::vector<size_t> shape;
  std::stringstream ss(shapeStr);
  std::string token;

  while (std::getline(ss, token, ',')) {
    try {
      int dim = std::stoi(token);
      shape.push_back(dim);
    } catch (...) {
      // ignore empty or malformed tokens
    }
  }

  return shape;
}

bool isLittleEndian(const std::string &descr) {
  return descr.find("<f4") != std::string::npos;
}
} // namespace

void loadNpy(const std::string &path, Tensor &tensor) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to open .npy file: " + path);

  char magic[6];
  file.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY")
    throw std::runtime_error("Invalid .npy file");

  uint8_t major, minor;
  file.read(reinterpret_cast<char *>(&major), 1);
  file.read(reinterpret_cast<char *>(&minor), 1);

  uint16_t header_len;
  file.read(reinterpret_cast<char *>(&header_len), 2);
  std::string header(header_len, ' ');
  file.read(&header[0], header_len);

  auto shape = parseShape(header);
  if (!isLittleEndian(header))
    throw std::runtime_error("Only little-endian float32 supported");

  size_t total = 1;
  for (int d : shape)
    total *= d;

  tensor = Tensor(shape, path);
  float *data = tensor.getHostData();
  file.read(reinterpret_cast<char *>(data), total * sizeof(float));

  if (!file)
    throw std::runtime_error("Failed to read full .npy data");

  // Allocate GPU memory + copy to device
  tensor.allocateDevice();
  tensor.copyToDevice();
}

void saveNpy(const std::string &path, const Tensor &tensor) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Failed to write .npy file: " + path);

  const char *magic = "\x93NUMPY";
  file.write(magic, 6);

  uint8_t major = 1, minor = 0;
  file.write(reinterpret_cast<const char *>(&major), 1);
  file.write(reinterpret_cast<const char *>(&minor), 1);

  std::string shapeStr = "(";
  auto shape = tensor.getShape();
  for (size_t i = 0; i < shape.size(); ++i)
    shapeStr += std::to_string(shape[i]) + (i + 1 == shape.size() ? ")" : ", ");
  if (shape.size() == 1)
    shapeStr += ","; // Python tuple syntax

  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': " + shapeStr + ", }";
  while ((header.size() + 10) % 16 != 0)
    header += ' ';
  header += '\n';

  uint16_t header_len = static_cast<uint16_t>(header.size());
  file.write(reinterpret_cast<const char *>(&header_len), 2);
  file.write(header.c_str(), header.size());

  const float *data = tensor.getHostData();
  file.write(reinterpret_cast<const char *>(data),
             tensor.size() * sizeof(float));
}
