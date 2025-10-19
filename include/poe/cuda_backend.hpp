#pragma once

#include <cstddef>

namespace poe {

double gpu_sum_gradients(const double* coefficients, const double* offsets, std::size_t count, double x);

} // namespace poe

