# Parallel Optimization Engine for Multi-Agent Reasoning

A hybrid C++/CUDA/Python project demonstrating a parallel optimization engine where multiple agents cooperate to minimize a shared quadratic objective.

## Build

```bash
mkdir -p build && cd build
cmake -DENABLE_CUDA=ON -DENABLE_PYBIND=ON ..
cmake --build . --config Release -j
```

- If CUDA or pybind11 are not detected, the build will gracefully disable those parts.

## Run CLI

```bash
./src/poe_cli --agents 100 --iters 2000 --seed 42 --lr 1e-3 --mode collaborative --device auto
```

## Python Experiments

After building with `-DENABLE_PYBIND=ON`:

```bash
python3 python/run_experiments.py --agents 100 --iters 2000 --repeats 3 --seed 42
```

## Project Layout

- `include/` public headers
- `src/` C++ core, CLI, and optional CUDA kernels
- `python/` pybind11 module and experiment scripts

## Notes

- GPU kernels are used when `--device gpu` or `--device auto` with sufficiently many agents and a CUDA device present.
- CPU code uses OpenMP for parallel reductions when available.
