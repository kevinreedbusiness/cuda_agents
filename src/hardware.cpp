#include "poe/hardware.hpp"

#ifdef POE_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace poe {

bool cuda_available() {
#ifdef POE_ENABLE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return false;
    }
    return count > 0;
#else
    return false;
#endif
}

int cuda_device_count() {
#ifdef POE_ENABLE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

} // namespace poe

