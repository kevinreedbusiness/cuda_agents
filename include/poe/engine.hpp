#pragma once

#include "poe/config.hpp"

#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace poe {

struct AgentParameters {
    double a;
    double b;
};

struct OptimizationResult {
    double final_x{0.0};
    double analytical_optimum{0.0};
    double final_cost{0.0};
    std::vector<double> history;
    std::vector<double> naive_history;
    std::vector<double> gradient_history;
};

class OptimizationEngine {
  public:
    explicit OptimizationEngine(EngineConfig config);

    OptimizationResult run();

  private:
    EngineConfig config_;
    std::vector<AgentParameters> agents_;
    std::vector<double> coefficients_;
    std::vector<double> offsets_;

    void initialize_agents();
    OptimizationResult run_collaborative();
    OptimizationResult run_naive();

    double compute_cost(double x) const;
    double compute_gradient_cpu(double x) const;
    double compute_gradient_gpu(double x) const;

    bool should_use_gpu() const;
    double analytical_optimum() const;
};

} // namespace poe

