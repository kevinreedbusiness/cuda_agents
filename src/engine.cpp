#include "engine.hpp"
#include "hardware.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

#if HAS_OPENMP
#include <omp.h>
#endif

namespace poe {

OptimizationEngine::OptimizationEngine(const RunConfig& cfg) : config(cfg) {
  initializeAgents();
}

void OptimizationEngine::initializeAgents() {
  a.resize(config.numAgents);
  b.resize(config.numAgents);
  std::mt19937 rng(config.seed);
  std::uniform_real_distribution<double> distA(0.1, 5.0);
  std::normal_distribution<double> distB(0.0, 5.0);
  for (std::size_t i = 0; i < config.numAgents; ++i) {
    a[i] = distA(rng);
    b[i] = distB(rng);
  }
}

static inline double computeAnalyticalXStar(const std::vector<double>& a, const std::vector<double>& b) {
  double sumA = std::accumulate(a.begin(), a.end(), 0.0);
  double sumAB = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sumAB += a[i] * b[i];
  return sumAB / sumA;
}

double OptimizationEngine::computeAnalyticalOptimum() const {
  return computeAnalyticalXStar(a, b);
}

void OptimizationEngine::computeGradientCPU(double x, double& sumGrad, double& sumA, double& sumAB) const {
  double localSumGrad = 0.0;
  double localSumA = 0.0;
  double localSumAB = 0.0;

  const std::size_t n = a.size();

  #if HAS_OPENMP
  #pragma omp parallel for reduction(+:localSumGrad, localSumA, localSumAB)
  #endif
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
    const double gi = 2.0 * a[i] * (x - b[i]);
    localSumGrad += gi;
    localSumA += a[i];
    localSumAB += a[i] * b[i];
  }

  sumGrad = localSumGrad;
  sumA = localSumA;
  sumAB = localSumAB;
}

bool OptimizationEngine::canUseGPU() const {
#if HAS_CUDA
  if (config.device == "cpu") return false;
  if (config.device == "gpu") return true;
  // auto: prefer GPU when available and large enough
  return hasCUDADevice() && config.numAgents >= 2048; // threshold
#else
  (void)config;
  return false;
#endif
}

void OptimizationEngine::computeGradientGPU(double x, double& sumGrad, double& sumA, double& sumAB) const {
#if HAS_CUDA
  launch_gradient_reduction(a.data(), b.data(), static_cast<int>(a.size()), x, &sumGrad, &sumA, &sumAB);
#else
  (void)x; (void)sumGrad; (void)sumA; (void)sumAB;
  throw std::runtime_error("GPU unavailable at compile time");
#endif
}

RunResult OptimizationEngine::run() {
  RunResult result;
  const double xStar = computeAnalyticalOptimum();
  result.xStar = xStar;

  auto t0 = std::chrono::high_resolution_clock::now();

  if (config.mode == std::string("naive")) {
    // Each agent minimizes independently: x_i = b_i, then average
    double sumB = std::accumulate(b.begin(), b.end(), 0.0);
    result.xFinal = sumB / static_cast<double>(b.size());
    result.xTrajectory.push_back(result.xFinal);
  } else {
    // Collaborative gradient descent
    double x = 0.0;
    result.xTrajectory.reserve(config.iterations + 1);
    result.xTrajectory.push_back(x);

    const bool useGPU = canUseGPU();

    for (std::size_t t = 0; t < config.iterations; ++t) {
      double sumGrad = 0.0, sumA = 0.0, sumAB = 0.0;
      if (useGPU) {
        computeGradientGPU(x, sumGrad, sumA, sumAB);
      } else {
        computeGradientCPU(x, sumGrad, sumA, sumAB);
      }
      // Optional: ML predictor hook (simple momentum-like extrapolation)
      if (config.useMLPredictor && t > 1) {
        const double xPrev = result.xTrajectory[result.xTrajectory.size() - 1];
        const double xPrevPrev = result.xTrajectory[result.xTrajectory.size() - 2];
        const double velocity = xPrev - xPrevPrev;
        x += 0.1 * velocity; // tiny extrapolation
      }

      x = x - config.learningRate * sumGrad;
      result.xTrajectory.push_back(x);
    }
    result.xFinal = x;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  result.totalTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.computeTimeMs = result.totalTimeMs; // simple for now; could separate
  return result;
}

} // namespace poe
