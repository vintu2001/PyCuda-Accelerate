import shutil
import subprocess

import pytest


def _gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require an NVIDIA GPU")


def pytest_collection_modifyitems(config, items):
    if GPU_AVAILABLE:
        return
    marker = pytest.mark.skip(reason="No NVIDIA GPU available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(marker)


@pytest.fixture
def require_gpu():
    if not GPU_AVAILABLE:
        pytest.skip("No NVIDIA GPU available")
