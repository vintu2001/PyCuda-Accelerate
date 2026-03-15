#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

#include "algorithms.h"

namespace py = pybind11;

namespace {

void validate_1d(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 1) {
        throw std::runtime_error(name + " must be a 1-D float32 array");
    }
}

void validate_2d(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 2) {
        throw std::runtime_error(name + " must be a 2-D float32 array");
    }
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "GPU-accelerated algorithms for NumPy arrays";

    m.def(
        "gpu_sort",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            const py::buffer_info in = input.request();
            validate_1d(in, "input");

            auto out = py::array_t<float>(in.size);
            py::buffer_info out_info = out.request();

            {
                py::gil_scoped_release release;
                pycuda_accelerate::gpu_radix_sort(
                    static_cast<const float*>(in.ptr),
                    static_cast<float*>(out_info.ptr),
                    static_cast<std::size_t>(in.size));
            }

            return out;
        },
        py::arg("input"));

    m.def(
        "gpu_reduce",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> input,
           const std::string& op) {
            const py::buffer_info in = input.request();
            validate_1d(in, "input");

            float result = 0.0f;
            {
                py::gil_scoped_release release;
                result = pycuda_accelerate::gpu_parallel_reduce(
                    static_cast<const float*>(in.ptr),
                    static_cast<std::size_t>(in.size),
                    op);
            }

            return result;
        },
        py::arg("input"),
        py::arg("op") = "sum");

    m.def(
        "gpu_prefix_scan",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> input) {
            const py::buffer_info in = input.request();
            validate_1d(in, "input");

            auto out = py::array_t<float>(in.size);
            py::buffer_info out_info = out.request();

            {
                py::gil_scoped_release release;
                pycuda_accelerate::gpu_prefix_scan(
                    static_cast<const float*>(in.ptr),
                    static_cast<float*>(out_info.ptr),
                    static_cast<std::size_t>(in.size));
            }

            return out;
        },
        py::arg("input"));

    m.def(
        "gpu_matmul",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> a,
           py::array_t<float, py::array::c_style | py::array::forcecast> b) {
            const py::buffer_info a_info = a.request();
            const py::buffer_info b_info = b.request();
            validate_2d(a_info, "a");
            validate_2d(b_info, "b");

            const int m = static_cast<int>(a_info.shape[0]);
            const int k = static_cast<int>(a_info.shape[1]);
            const int b_rows = static_cast<int>(b_info.shape[0]);
            const int n = static_cast<int>(b_info.shape[1]);

            if (k != b_rows) {
                throw std::runtime_error(
                    "shape mismatch: a is " + std::to_string(m) + "x" + std::to_string(k) +
                    ", b is " + std::to_string(b_rows) + "x" + std::to_string(n));
            }

            auto out = py::array_t<float>({m, n});

            {
                py::gil_scoped_release release;
                pycuda_accelerate::gpu_gemm(
                    static_cast<const float*>(a_info.ptr),
                    static_cast<const float*>(b_info.ptr),
                    static_cast<float*>(out.mutable_data()),
                    m,
                    n,
                    k);
            }

            return out;
        },
        py::arg("a"),
        py::arg("b"));
}
