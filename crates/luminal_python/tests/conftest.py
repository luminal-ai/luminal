"""Test configuration."""

import os

# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook
    from maturin_import_hook.settings import MaturinSettings

    use_cuda = os.getenv("LUMINAL_TEST_DEVICE", "cpu").lower() == "cuda"
    settings = MaturinSettings(features=["cuda"]) if use_cuda else None
    maturin_import_hook.install(settings=settings)
except ImportError:
    pass  # Hook not available, rebuilds will be manual

import pytest
import torch
import torch._dynamo

torch.set_float32_matmul_precision("highest")


@pytest.fixture
def device() -> torch.device:
    if (
        os.getenv("LUMINAL_TEST_DEVICE", "cpu").lower() == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(autouse=True, scope="function")
def reset_torch_dynamo():
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.suppress_errors = False
    yield
    torch._dynamo.reset()
