#include "poe/config.hpp"
#include "poe/engine.hpp"
#include "poe/hardware.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --agents <N>        Number of agents (default 64)\n"
              << "  --iters <T>         Optimization iterations (default 1000)\n"
              << "  --seed <S>          RNG seed (default 42)\n"
              << "  --lr <LR>           Learning rate (default 1e-2)\n"
              << "  --mode <m>          Optimization mode: collaborative|naive (default collaborative)\n"
              << "  --device <d>        Device: auto|cpu|gpu (default auto)\n"
              << "  --threshold <N>     GPU threshold (default 128)\n"
              << "  --verbose           Enable verbose logging\n"
              << "  --help              Show this message\n";
}

poe::EngineConfig parse_arguments(int argc, char** argv) {
    poe::EngineConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--agents" && i + 1 < argc) {
            config.num_agents = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--iters" && i + 1 < argc) {
            config.iterations = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--lr" && i + 1 < argc) {
            config.learning_rate = std::stod(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            config.mode = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            config.device = argv[++i];
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.gpu_threshold = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    return config;
}

} // namespace

int main(int argc, char** argv) {
    try {
        auto config = parse_arguments(argc, argv);
        poe::OptimizationEngine engine(config);
        auto result = engine.run();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Mode: " << config.mode << "\n";
        std::cout << "Agents: " << config.num_agents << "\n";
        std::cout << "Iterations: " << config.iterations << "\n";
        std::cout << "Learning rate: " << config.learning_rate << "\n";
        std::cout << "Analytical optimum: " << result.analytical_optimum << "\n";
        std::cout << "Final estimate: " << result.final_x << "\n";
        std::cout << "Final cost: " << result.final_cost << "\n";

        if (!result.naive_history.empty()) {
            std::cout << "Naive final estimate: " << result.naive_history.back() << "\n";
        }

        if (config.verbose) {
            std::cout << "\nConvergence history (last 10 steps):\n";
            const auto& hist = result.history;
            std::size_t start = hist.size() > 10 ? hist.size() - 10 : 0;
            for (std::size_t i = start; i < hist.size(); ++i) {
                std::cout << "  t=" << i << ": " << hist[i] << "\n";
            }
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
}

