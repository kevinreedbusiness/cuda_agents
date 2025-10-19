#include "poe/engine.hpp"

#include "poe/hardware.hpp"

#ifdef POE_ENABLE_CUDA
#include "poe/cuda_backend.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace poe {

namespace {
constexpr double kDefaultInit = 0.0;
}

OptimizationEngine::OptimizationEngine(EngineConfig config) : config_(std::move(config)) {
    initialize_agents();
}

OptimizationResult OptimizationEngine::run() {
    if (config_.mode == "naive") {
        return run_naive();
    }

    auto result = run_collaborative();
    auto naive = run_naive();
    result.naive_history = std::move(naive.history);
    return result;
}

void OptimizationEngine::initialize_agents() {
    std::mt19937 rng(config_.seed);
    std::uniform_real_distribution<double> dist_a(0.5, 5.0);
    std::uniform_real_distribution<double> dist_b(-5.0, 5.0);

    agents_.resize(config_.num_agents);
    coefficients_.resize(config_.num_agents);
    offsets_.resize(config_.num_agents);

    for (std::size_t i = 0; i < config_.num_agents; ++i) {
        agents_[i].a = dist_a(rng);
        agents_[i].b = dist_b(rng);
        coefficients_[i] = agents_[i].a;
        offsets_[i] = agents_[i].b;
    }
}

OptimizationResult OptimizationEngine::run_collaborative() {
    OptimizationResult result;
    result.analytical_optimum = analytical_optimum();

    double x = kDefaultInit;
    result.history.reserve(config_.iterations);
    result.gradient_history.reserve(config_.iterations);

    for (std::size_t iter = 0; iter < config_.iterations; ++iter) {
        double grad = should_use_gpu() ? compute_gradient_gpu(x) : compute_gradient_cpu(x);
        x -= config_.learning_rate * grad;
        result.history.push_back(x);
        result.gradient_history.push_back(grad);
    }

    result.final_x = x;
    result.final_cost = compute_cost(x);
    return result;
}

OptimizationResult OptimizationEngine::run_naive() {
    OptimizationResult result;
    result.history.reserve(config_.num_agents);

    double sum = 0.0;
    for (std::size_t i = 0; i < agents_.size(); ++i) {
        double xi = agents_[i].b;
        sum += xi;
        result.history.push_back(sum / static_cast<double>(i + 1));
    }

    result.final_x = sum / static_cast<double>(agents_.size());
    result.analytical_optimum = analytical_optimum();
    result.final_cost = compute_cost(result.final_x);
    result.naive_history = result.history;
    return result;
}

double OptimizationEngine::compute_cost(double x) const {
    double total = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total)
#endif
    for (std::size_t i = 0; i < agents_.size(); ++i) {
        double diff = x - offsets_[i];
        total += coefficients_[i] * diff * diff;
    }
    return total;
}

double OptimizationEngine::compute_gradient_cpu(double x) const {
    double grad = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : grad)
#endif
    for (std::size_t i = 0; i < agents_.size(); ++i) {
        grad += 2.0 * coefficients_[i] * (x - offsets_[i]);
    }
    return grad;
}

double OptimizationEngine::compute_gradient_gpu(double x) const {
#ifdef POE_ENABLE_CUDA
    return gpu_sum_gradients(coefficients_.data(), offsets_.data(), agents_.size(), x);
#else
    (void)x;
    return compute_gradient_cpu(x);
#endif
}

bool OptimizationEngine::should_use_gpu() const {
    if (config_.device == "cpu") {
        return false;
    }
    if (config_.device == "gpu") {
        return cuda_available();
    }
    return cuda_available() && agents_.size() >= config_.gpu_threshold;
}

double OptimizationEngine::analytical_optimum() const {
    double num = 0.0;
    double denom = 0.0;
    for (const auto& agent : agents_) {
        num += agent.a * agent.b;
        denom += agent.a;
    }
    return denom == 0.0 ? 0.0 : num / denom;
}

} // namespace poe

