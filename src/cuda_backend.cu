#include "poe/cuda_backend.hpp"

#include <cuda_runtime.h>

#include <numeric>
#include <stdexcept>
#include <vector>

namespace poe {

namespace {
__global__ void gradient_kernel(const double* coefficients, const double* offsets, double x, double* partial, std::size_t count) {
    extern __shared__ double buffer[];
    std::size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double grad = 0.0;
    if (global_idx < count) {
        grad = 2.0 * coefficients[global_idx] * (x - offsets[global_idx]);
    }
    buffer[threadIdx.x] = grad;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buffer[threadIdx.x] += buffer[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = buffer[0];
    }
}

inline void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

} // namespace

double gpu_sum_gradients(const double* coefficients, const double* offsets, std::size_t count, double x) {
    if (count == 0) {
        return 0.0;
    }

    constexpr unsigned int kBlockSize = 256;
    unsigned int blocks = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);

    double* d_coefficients = nullptr;
    double* d_offsets = nullptr;
    double* d_partial = nullptr;

    check_cuda(cudaMalloc(&d_coefficients, count * sizeof(double)));
    check_cuda(cudaMalloc(&d_offsets, count * sizeof(double)));
    check_cuda(cudaMalloc(&d_partial, blocks * sizeof(double)));

    check_cuda(cudaMemcpy(d_coefficients, coefficients, count * sizeof(double), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_offsets, offsets, count * sizeof(double), cudaMemcpyHostToDevice));

    gradient_kernel<<<blocks, kBlockSize, kBlockSize * sizeof(double)>>>(d_coefficients, d_offsets, x, d_partial, count);
    check_cuda(cudaDeviceSynchronize());

    std::vector<double> host_partial(blocks);
    check_cuda(cudaMemcpy(host_partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost));

    check_cuda(cudaFree(d_coefficients));
    check_cuda(cudaFree(d_offsets));
    check_cuda(cudaFree(d_partial));

    return std::accumulate(host_partial.begin(), host_partial.end(), 0.0);
}

} // namespace poe

