"""Test configuration."""

import os

# Enable automatic Rust rebuilds during test development
try:
    import maturin_import_hook
    from maturin_import_hook.settings import MaturinSettings

    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    settings = MaturinSettings(features=["cuda"]) if backend == "cuda" else None
    maturin_import_hook.install(settings=settings)
except ImportError:
    pass  # Hook not available, rebuilds will be manual

import pytest
import torch
import torch._dynamo

torch.set_float32_matmul_precision("highest")


@pytest.fixture
def device() -> torch.device:
    backend = os.getenv("LUMINAL_BACKEND", "native").lower()
    return torch.device("cuda") if backend == "cuda" else torch.device("cpu")


@pytest.fixture(autouse=True, scope="function")
def reset_torch_dynamo():
    # We need this for two reasons
    # 1. Some of our casts tests use the same model, but those graph have some state to them
    # and the cache will return old models
    # 2. The cache adds a large preformace hit to the test suite
    torch._dynamo.config.cache_size_limit = 1
    # Disable silent fallback to eager mode so backend errors surface as test failures
    torch._dynamo.config.suppress_errors = False
    """Reset PyTorch Dynamo state after each test to prevent state leakage.

    This fixture automatically runs after every test function to clear
    torch._dynamo's compilation cache, ensuring test isolation.
    """
    yield  # Test runs here
    torch._dynamo.reset()
