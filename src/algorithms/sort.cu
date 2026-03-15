#include "algorithms.h"
#include "utils/cuda_check.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace pycuda_accelerate {

void gpu_radix_sort(const float* input, float* output, std::size_t n) {
    if (n == 0) {
        return;
    }

    thrust::device_vector<float> d_values(input, input + n);
    thrust::sort(d_values.begin(), d_values.end());
    thrust::copy(d_values.begin(), d_values.end(), output);
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace pycuda_accelerate
