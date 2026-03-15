#pragma once

#include <cstddef>
#include <string>

namespace pycuda_accelerate {

void gpu_radix_sort(const float* input, float* output, std::size_t n);
float gpu_parallel_reduce(const float* input, std::size_t n, const std::string& op);
void gpu_prefix_scan(const float* input, float* output, std::size_t n);
void gpu_gemm(const float* a, const float* b, float* c, int m, int n, int k);

}  // namespace pycuda_accelerate
