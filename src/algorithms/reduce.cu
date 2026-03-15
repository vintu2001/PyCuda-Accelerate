#include "algorithms.h"
#include "utils/cuda_check.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <stdexcept>

namespace pycuda_accelerate {

float gpu_parallel_reduce(const float* input, std::size_t n, const std::string& op) {
    if (op != "sum" && op != "min" && op != "max") {
        throw std::invalid_argument("op must be 'sum', 'min', or 'max'");
    }

    if (n == 0) {
        return 0.0f;
    }

    thrust::device_vector<float> d_values(input, input + n);

    if (op == "sum") {
        float result = thrust::reduce(d_values.begin(), d_values.end(), 0.0f, thrust::plus<float>());
        CUDA_CHECK(cudaDeviceSynchronize());
        return result;
    }

    if (op == "min") {
        auto it = thrust::min_element(d_values.begin(), d_values.end());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *it;
    }

    if (op == "max") {
        auto it = thrust::max_element(d_values.begin(), d_values.end());
        CUDA_CHECK(cudaDeviceSynchronize());
        return *it;
    }

    return 0.0f;
}

}  // namespace pycuda_accelerate
