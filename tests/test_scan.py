import numpy as np
import pytest

from pycuda_accelerate import gpu_prefix_scan


@pytest.mark.gpu
def test_gpu_prefix_scan_basic(require_gpu):
    arr = np.array([1, 2, 3, 4], dtype=np.float32)
    out = gpu_prefix_scan(arr)
    expected = np.array([0, 1, 3, 6], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


@pytest.mark.gpu
def test_gpu_prefix_scan_large(require_gpu):
    arr = np.ones(100000, dtype=np.float32)
    out = gpu_prefix_scan(arr)
    expected = np.arange(100000, dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
