#pragma once

#include <cstddef>
#include <string>

namespace poe {

struct EngineConfig {
    std::size_t num_agents{64};
    std::size_t iterations{1000};
    unsigned int seed{42};
    double learning_rate{1e-2};
    std::string mode{"collaborative"};
    std::string device{"auto"};
    std::size_t gpu_threshold{128};
    bool verbose{false};
};

} // namespace poe

