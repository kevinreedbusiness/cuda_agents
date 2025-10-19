#pragma once
#include <vector>
#include <string>
#include <cstddef>

namespace poe {

struct RunConfig {
  std::size_t numAgents{100};
  std::size_t iterations{1000};
  unsigned int seed{42};
  double learningRate{1e-3};
  std::string mode{"collaborative"}; // "naive" or "collaborative"
  std::string device{"auto"};        // "auto", "cpu", "gpu"
  bool useMLPredictor{false};
};

struct RunResult {
  double xFinal{0.0};
  double xStar{0.0};
  double totalTimeMs{0.0};
  double computeTimeMs{0.0};
  std::vector<double> xTrajectory; // per-iteration x
};

class OptimizationEngine {
public:
  explicit OptimizationEngine(const RunConfig& cfg);
  RunResult run();

private:
  RunConfig config;
  std::vector<double> a;
  std::vector<double> b;

  void initializeAgents();
  double computeAnalyticalOptimum() const;

  // CPU paths
  void computeGradientCPU(double x, double& sumGrad, double& sumA, double& sumAB) const;

  // GPU paths (optionally defined)
  bool canUseGPU() const;
  void computeGradientGPU(double x, double& sumGrad, double& sumA, double& sumAB) const;
};

} // namespace poe
