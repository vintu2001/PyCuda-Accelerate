#include "algorithms.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace pycuda_accelerate {
namespace {

template <int Tile>
__global__ void gemm_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float tile_a[Tile][Tile + 1];
    __shared__ float tile_b[Tile][Tile + 1];

    const int row = blockIdx.y * Tile + threadIdx.y;
    const int col = blockIdx.x * Tile + threadIdx.x;

    float acc = 0.0f;

    for (int t = 0; t < (k + Tile - 1) / Tile; ++t) {
        const int a_col = t * Tile + threadIdx.x;
        const int b_row = t * Tile + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int i = 0; i < Tile; ++i) {
            acc += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
        }
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace

void gpu_gemm(const float* a, const float* b, float* c, int m, int n, int k) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("matrix sizes must be non-negative");
    }

    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    const std::size_t a_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(k) * sizeof(float);
    const std::size_t b_bytes = static_cast<std::size_t>(k) * static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t c_bytes = static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(float);

    DeviceBuffer d_a(a_bytes);
    DeviceBuffer d_b(b_bytes);
    DeviceBuffer d_c(c_bytes);

    CUDA_CHECK(cudaMemcpy(d_a.get(), a, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), b, b_bytes, cudaMemcpyHostToDevice));

    constexpr int kTile = 32;
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);

    gemm_kernel<kTile><<<grid, block>>>(
        static_cast<const float*>(d_a.get()),
        static_cast<const float*>(d_b.get()),
        static_cast<float*>(d_c.get()),
        m,
        n,
        k);

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(c, d_c.get(), c_bytes, cudaMemcpyDeviceToHost));
}

}  // namespace pycuda_accelerate
