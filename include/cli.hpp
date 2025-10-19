#pragma once
#include <string>
#include "engine.hpp"

namespace poe {

RunConfig parse_cli(int argc, char** argv);
void print_summary(const RunConfig& cfg, const RunResult& res);

} // namespace poe
