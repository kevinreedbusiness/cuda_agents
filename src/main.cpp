#include "cli.hpp"
#include "engine.hpp"
#include "hardware.hpp"

#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
  using namespace poe;
  try {
    RunConfig cfg = parse_cli(argc, argv);

    std::cout << "Device availability: CUDA=" << (hasCUDADevice() ? "yes" : "no")
              << " (count=" << getCUDADeviceCount() << ", name=" << getCUDAName() << ")\n";

    OptimizationEngine engine(cfg);
    RunResult res = engine.run();
    print_summary(cfg, res);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
