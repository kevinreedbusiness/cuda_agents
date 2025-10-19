# Short Report: Parallel Optimization Engine

## Problem and Optimum
We minimize F(x) = sum_i a_i (x - b_i)^2 with a_i > 0. Closed-form optimum:

x* = (Σ a_i b_i) / (Σ a_i)

## Algorithms
- Naive: each agent minimizes independently (x_i = b_i); final x = mean(b_i)
- Collaborative: gradient descent on F using shared sum of gradients g_i = 2 a_i (x - b_i)

## Parallelization Strategy
- CPU: OpenMP reduction over agents for sum of gradients, Σ a_i, and Σ a_i b_i
- GPU: CUDA kernel with per-thread (agent) partials; shared-memory block reduction then host-side finalize
- Hardware detection chooses GPU when available and N exceeds threshold

## Architecture
- `OptimizationEngine` encapsulates data, device selection, and run loop
- `hardware.cpp` abstracts CUDA presence
- Optional pybind11 exposes C++ to Python for orchestration/plots

## Performance Notes
- GPU provides benefit for large N; CPU/OpenMP is efficient for small-medium N
- Learning rate η controls convergence; F is strongly convex so GD converges for small enough η

## Trade-offs
- Simplicity of single-parameter GD vs. closed-form solution used for accuracy check
- Minimal data transfer each iteration; current GPU path transfers inputs once but recomputes partials per iteration
