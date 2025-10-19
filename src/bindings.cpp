#include "poe/config.hpp"
#include "poe/engine.hpp"

#ifdef POE_ENABLE_PYBIND

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poe {
namespace bindings {

OptimizationResult run_engine(const EngineConfig& config) {
    OptimizationEngine engine(config);
    return engine.run();
}

} // namespace bindings
} // namespace poe

PYBIND11_MODULE(poe_engine, m) {
    py::class_<poe::EngineConfig>(m, "EngineConfig")
        .def(py::init<>())
        .def_readwrite("num_agents", &poe::EngineConfig::num_agents)
        .def_readwrite("iterations", &poe::EngineConfig::iterations)
        .def_readwrite("seed", &poe::EngineConfig::seed)
        .def_readwrite("learning_rate", &poe::EngineConfig::learning_rate)
        .def_readwrite("mode", &poe::EngineConfig::mode)
        .def_readwrite("device", &poe::EngineConfig::device)
        .def_readwrite("gpu_threshold", &poe::EngineConfig::gpu_threshold)
        .def_readwrite("verbose", &poe::EngineConfig::verbose);

    py::class_<poe::OptimizationResult>(m, "OptimizationResult")
        .def_readonly("final_x", &poe::OptimizationResult::final_x)
        .def_readonly("analytical_optimum", &poe::OptimizationResult::analytical_optimum)
        .def_readonly("final_cost", &poe::OptimizationResult::final_cost)
        .def_readonly("history", &poe::OptimizationResult::history)
        .def_readonly("naive_history", &poe::OptimizationResult::naive_history)
        .def_readonly("gradient_history", &poe::OptimizationResult::gradient_history);

    m.def("run", [](const poe::EngineConfig& config) {
        return poe::bindings::run_engine(config);
    });
}

#endif

