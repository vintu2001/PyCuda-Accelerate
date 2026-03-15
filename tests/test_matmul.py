import numpy as np
import pytest

from pycuda_accelerate import gpu_matmul


@pytest.mark.gpu
def test_gpu_matmul_small(require_gpu):
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    out = gpu_matmul(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_gpu_matmul_rectangular(require_gpu):
    rng = np.random.default_rng(7)
    a = rng.standard_normal((64, 32), dtype=np.float32)
    b = rng.standard_normal((32, 48), dtype=np.float32)
    out = gpu_matmul(a, b)
    np.testing.assert_allclose(out, a @ b, rtol=1e-3, atol=1e-3)


def test_gpu_matmul_shape_mismatch():
    a = np.ones((2, 3), dtype=np.float32)
    b = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        gpu_matmul(a, b)
