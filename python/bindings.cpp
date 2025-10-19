#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;
using namespace poe;

PYBIND11_MODULE(poe, m) {
  py::class_<RunConfig>(m, "RunConfig")
    .def(py::init<>())
    .def_readwrite("numAgents", &RunConfig::numAgents)
    .def_readwrite("iterations", &RunConfig::iterations)
    .def_readwrite("seed", &RunConfig::seed)
    .def_readwrite("learningRate", &RunConfig::learningRate)
    .def_readwrite("mode", &RunConfig::mode)
    .def_readwrite("device", &RunConfig::device)
    .def_readwrite("useMLPredictor", &RunConfig::useMLPredictor);

  py::class_<RunResult>(m, "RunResult")
    .def_readonly("xFinal", &RunResult::xFinal)
    .def_readonly("xStar", &RunResult::xStar)
    .def_readonly("totalTimeMs", &RunResult::totalTimeMs)
    .def_readonly("computeTimeMs", &RunResult::computeTimeMs)
    .def_readonly("xTrajectory", &RunResult::xTrajectory);

  py::class_<OptimizationEngine>(m, "OptimizationEngine")
    .def(py::init<const RunConfig&>())
    .def("run", &OptimizationEngine::run);
}
