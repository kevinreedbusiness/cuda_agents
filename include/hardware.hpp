#pragma once
#include <string>

namespace poe {

bool hasCUDADevice();
std::string getCUDAName();
int getCUDADeviceCount();

} // namespace poe
