import numpy as np
import pytest

from pycuda_accelerate import gpu_sort


@pytest.mark.gpu
def test_gpu_sort_basic(require_gpu):
    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
    out = gpu_sort(arr)
    np.testing.assert_array_equal(out, np.sort(arr))


@pytest.mark.gpu
def test_gpu_sort_large_random(require_gpu):
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(200000, dtype=np.float32)
    out = gpu_sort(arr)
    np.testing.assert_allclose(out, np.sort(arr), rtol=0.0, atol=0.0)


def test_gpu_sort_invalid_shape():
    arr = np.ones((3, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="1-D"):
        gpu_sort(arr)
