#include <cuda_runtime.h>
#include <vector>

extern "C" void launch_gradient_reduction(const double* a, const double* b, int n, double x,
                                          double* outSumGrad, double* outSumA, double* outSumAB);

__global__ void gradient_kernel(const double* __restrict__ a,
                                const double* __restrict__ b,
                                int n, double x,
                                double* __restrict__ partialGrad,
                                double* __restrict__ partialA,
                                double* __restrict__ partialAB) {
  extern __shared__ double sdata[];
  double* sgrad = sdata;
  double* sa = sgrad + blockDim.x;
  double* sab = sa + blockDim.x;

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  double g = 0.0, va = 0.0, vab = 0.0;
  if (gid < n) {
    const double ai = a[gid];
    const double bi = b[gid];
    g = 2.0 * ai * (x - bi);
    va = ai;
    vab = ai * bi;
  }
  sgrad[threadIdx.x] = g;
  sa[threadIdx.x] = va;
  sab[threadIdx.x] = vab;
  __syncthreads();

  // reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sgrad[threadIdx.x] += sgrad[threadIdx.x + stride];
      sa[threadIdx.x] += sa[threadIdx.x + stride];
      sab[threadIdx.x] += sab[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    partialGrad[blockIdx.x] = sgrad[0];
    partialA[blockIdx.x] = sa[0];
    partialAB[blockIdx.x] = sab[0];
  }
}

extern "C" void launch_gradient_reduction(const double* a, const double* b, int n, double x,
                                          double* outSumGrad, double* outSumA, double* outSumAB) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  double *d_a = nullptr, *d_b = nullptr;
  double *d_partialGrad = nullptr, *d_partialA = nullptr, *d_partialAB = nullptr;
  cudaMalloc(&d_a, sizeof(double) * n);
  cudaMalloc(&d_b, sizeof(double) * n);
  cudaMalloc(&d_partialGrad, sizeof(double) * blocks);
  cudaMalloc(&d_partialA, sizeof(double) * blocks);
  cudaMalloc(&d_partialAB, sizeof(double) * blocks);

  cudaMemcpy(d_a, a, sizeof(double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(double) * n, cudaMemcpyHostToDevice);

  size_t sharedBytes = sizeof(double) * threads * 3;
  gradient_kernel<<<blocks, threads, sharedBytes>>>(d_a, d_b, n, x, d_partialGrad, d_partialA, d_partialAB);
  cudaDeviceSynchronize();

  // host reduction of partials
  std::vector<double> h_grad(blocks), h_a(blocks), h_ab(blocks);
  cudaMemcpy(h_grad.data(), d_partialGrad, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a.data(), d_partialA, sizeof(double) * blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ab.data(), d_partialAB, sizeof(double) * blocks, cudaMemcpyDeviceToHost);

  double sumGrad = 0.0, sumA = 0.0, sumAB = 0.0;
  for (int i = 0; i < blocks; ++i) {
    sumGrad += h_grad[i];
    sumA += h_a[i];
    sumAB += h_ab[i];
  }

  *outSumGrad = sumGrad;
  *outSumA = sumA;
  *outSumAB = sumAB;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_partialGrad);
  cudaFree(d_partialA);
  cudaFree(d_partialAB);
}
