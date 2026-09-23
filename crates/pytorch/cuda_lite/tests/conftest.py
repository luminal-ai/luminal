"""Accuracy comparisons run against a deterministic, full-precision eager.

A comparison is only meaningful when both sides compute the same function.
Eager defaults to TF32 matmuls on Ampere and later, which keeps ten mantissa
bits and lands about three parts in a thousand from the true value, while the
CUDA-lite path asks cuBLASLt for strict float32 and verifies the compute type
at startup. Pinning eager to the highest precision and to deterministic
algorithms makes the reference the accurate side, so a tolerance failure means
a real divergence.

``CUBLAS_WORKSPACE_CONFIG`` must be set before cuBLAS creates its handle, so it
is set at import time rather than inside the fixture.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def deterministic_full_precision_eager():
    """Pin eager to full float32 and deterministic kernels for the session."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch 2.9 added the per-backend spelling; set it where it exists so the
    # precision is stated in both vocabularies.
    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(backend, "fp32_precision"):
            backend.fp32_precision = "ieee"
    torch.use_deterministic_algorithms(True)
    yield


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA backend tests require a CUDA GPU")
    return torch.device("cuda")


def pytest_configure(config):
    os.environ["LUMINAL_TEST_BACKEND"] = "cuda_lite"
    os.environ["LUMINAL_TEST_DEVICE"] = "cuda"
