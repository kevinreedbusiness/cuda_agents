# Parallel Optimization Engine

This repository implements a hybrid C++/CUDA/Python engine for coordinating multiple agents that jointly minimise a shared quadratic objective. The system automatically selects between CPU and GPU execution and exposes a Python experimentation interface with lightweight ML-assisted prediction.

## Features

- **C++17 core** with OpenMP acceleration for CPU reductions.
- **CUDA kernel** for GPU-based gradient aggregation with shared-memory reduction.
- **Python orchestration** for running experiments, visualising convergence, and adding ML-driven predictions.
- Unified CLI with hardware auto-detection and configurable optimisation modes.

## Building

The project uses CMake and optionally depends on CUDA and pybind11.

```bash
mkdir -p build && cd build
cmake -DENABLE_CUDA=ON -DENABLE_PYBIND=ON ..
cmake --build . --config Release -j
```

Key CMake options:

- `ENABLE_CUDA` (default `ON`): build the CUDA reduction backend. Disable if CUDA is unavailable.
- `ENABLE_PYBIND` (default `ON`): build the Python extension module `poe_engine`.

### Dependencies

- A C++17 compiler (tested with GCC and Clang)
- CMake ≥ 3.18
- Optional: CUDA Toolkit ≥ 11 for GPU support
- Python 3.9+ with `numpy` and `matplotlib` for experiments

## CLI Usage

After building, the CLI executable is located at `build/poe_cli` (or `build/bin/poe_cli` depending on your generator).

```bash
./poe_cli --agents 256 --iters 2000 --seed 42 --lr 5e-3 --mode collaborative --device auto --verbose
```

Important flags:

- `--mode collaborative|naive` – select optimisation strategy.
- `--device auto|cpu|gpu` – manually force a device or allow automatic selection.
- `--threshold <N>` – minimum agent count to activate the GPU in auto mode (default 128).
- `--verbose` – print the trailing section of the convergence history.

## Python Experiments

Ensure the Python module is on your `PYTHONPATH` (e.g. `export PYTHONPATH=$PWD/build:$PYTHONPATH`). Then run:

```bash
python python/run_experiments.py --agents 256 --iters 1500 --lr 5e-3 --repeats 5 --device auto
```

The script executes both optimisation modes multiple times, reports summary statistics, predicts the next iterate with a regression-based gradient predictor, and saves a convergence plot to `python/output/convergence.png`.

## Repository Structure

```
├── CMakeLists.txt
├── include/poe
│   ├── config.hpp
│   ├── cuda_backend.hpp
│   ├── engine.hpp
│   └── hardware.hpp
├── python
│   └── run_experiments.py
├── src
│   ├── bindings.cpp
│   ├── cuda_backend.cu
│   ├── engine.cpp
│   ├── hardware.cpp
│   └── main.cpp
└── REPORT.md
```

## Testing

Basic smoke tests:

```bash
# C++
mkdir -p build && cd build
cmake ..
cmake --build . -j
./poe_cli --mode collaborative --iters 10 --agents 32

# Python
export PYTHONPATH=$PWD:$PYTHONPATH
python ../python/run_experiments.py --agents 64 --iters 200 --repeats 2
```

## License

This project is provided for educational and evaluation purposes.

