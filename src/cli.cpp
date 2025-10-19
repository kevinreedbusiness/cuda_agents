#include "cli.hpp"
#include <getopt.h>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace poe {

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "  --agents N            Number of agents (default 100)\n"
            << "  --iters N             Iterations (default 1000)\n"
            << "  --seed S              Seed (default 42)\n"
            << "  --lr LR               Learning rate (default 1e-3)\n"
            << "  --mode M              naive|collaborative (default collaborative)\n"
            << "  --device D            auto|cpu|gpu (default auto)\n"
            << "  --ml                  Enable simple ML predictor (default off)\n"
            << std::endl;
}

RunConfig parse_cli(int argc, char** argv) {
  RunConfig cfg;

  static struct option long_options[] = {
    {"agents", required_argument, 0, 'a'},
    {"iters", required_argument, 0, 'i'},
    {"seed", required_argument, 0, 's'},
    {"lr", required_argument, 0, 'l'},
    {"mode", required_argument, 0, 'm'},
    {"device", required_argument, 0, 'd'},
    {"ml", no_argument, 0, 'p'},
    {0, 0, 0, 0}
  };

  int opt;
  int long_index = 0;
  while ((opt = getopt_long(argc, argv, "", long_options, &long_index)) != -1) {
    switch (opt) {
      case 'a': cfg.numAgents = std::stoull(optarg); break;
      case 'i': cfg.iterations = std::stoull(optarg); break;
      case 's': cfg.seed = static_cast<unsigned int>(std::stoul(optarg)); break;
      case 'l': cfg.learningRate = std::stod(optarg); break;
      case 'm': cfg.mode = std::string(optarg); break;
      case 'd': cfg.device = std::string(optarg); break;
      case 'p': cfg.useMLPredictor = true; break;
      default: usage(argv[0]); throw std::runtime_error("Invalid arguments");
    }
  }

  if (cfg.mode != "naive" && cfg.mode != "collaborative") {
    usage(argv[0]);
    throw std::runtime_error("Invalid mode: use naive|collaborative");
  }
  if (cfg.device != "auto" && cfg.device != "cpu" && cfg.device != "gpu") {
    usage(argv[0]);
    throw std::runtime_error("Invalid device: use auto|cpu|gpu");
  }
  return cfg;
}

void print_summary(const RunConfig& cfg, const RunResult& res) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Config: agents=" << cfg.numAgents
            << ", iters=" << cfg.iterations
            << ", lr=" << cfg.learningRate
            << ", mode=" << cfg.mode
            << ", device=" << cfg.device
            << ", ml=" << (cfg.useMLPredictor ? "on" : "off") << "\n";
  std::cout << "x* (analytic) = " << res.xStar << "\n";
  std::cout << "x_final       = " << res.xFinal << "\n";
  std::cout << "|x_final - x*|= " << std::abs(res.xFinal - res.xStar) << "\n";
  std::cout << "runtime (ms)  = " << res.totalTimeMs << "\n";
}

} // namespace poe
