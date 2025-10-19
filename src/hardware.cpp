#include "hardware.hpp"

#include <string>

namespace poe {

#if HAS_CUDA
#include <cuda_runtime.h>

bool hasCUDADevice() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
  return count > 0;
}

int getCUDADeviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

std::string getCUDAName() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "Unknown CUDA Device";
  return std::string(prop.name);
}

#else

bool hasCUDADevice() { return false; }
int getCUDADeviceCount() { return 0; }
std::string getCUDAName() { return "CUDA Unavailable"; }

#endif

} // namespace poe
