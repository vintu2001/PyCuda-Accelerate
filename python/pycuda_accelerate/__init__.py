"""GPU-accelerated algorithms for NumPy arrays."""

from pycuda_accelerate._core import gpu_matmul, gpu_prefix_scan, gpu_reduce, gpu_sort

try:
    from pycuda_accelerate._version import version as __version__
except Exception:
    __version__ = "0.1.0"

__all__ = ["gpu_sort", "gpu_reduce", "gpu_prefix_scan", "gpu_matmul", "__version__"]
