import numpy as np
import pytest

from pycuda_accelerate import gpu_reduce


@pytest.mark.gpu
def test_gpu_reduce_sum(require_gpu):
    arr = np.arange(1000, dtype=np.float32)
    out = gpu_reduce(arr, op="sum")
    assert np.isclose(out, float(np.sum(arr)), rtol=1e-5)


@pytest.mark.gpu
def test_gpu_reduce_min_max(require_gpu):
    arr = np.array([3.5, -1.0, 9.2, 0.0], dtype=np.float32)
    assert np.isclose(gpu_reduce(arr, op="min"), float(np.min(arr)))
    assert np.isclose(gpu_reduce(arr, op="max"), float(np.max(arr)))


def test_gpu_reduce_invalid_op():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="op"):
        gpu_reduce(arr, op="avg")
