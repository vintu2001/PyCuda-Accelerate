#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            throw std::runtime_error(                                                      \
                std::string("CUDA error: ") + cudaGetErrorString(err__) + " at " +       \
                __FILE__ + ":" + std::to_string(__LINE__));                                \
        }                                                                                   \
    } while (0)

#define CUDA_CHECK_LAST()                                                                   \
    do {                                                                                    \
        cudaError_t err__ = cudaGetLastError();                                            \
        if (err__ != cudaSuccess) {                                                        \
            throw std::runtime_error(                                                      \
                std::string("CUDA kernel error: ") + cudaGetErrorString(err__) + " at " + \
                __FILE__ + ":" + std::to_string(__LINE__));                                \
        }                                                                                   \
    } while (0)
