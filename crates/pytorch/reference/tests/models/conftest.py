"""Shared model parity cases, also explicitly selected by the CUDA runner."""

import torch

from backend_test_utils import device, execution_device, selected_backend

__all__ = ["device", "execution_device"]


def pytest_configure(config):
    # Model modules may author tensor constants at import time, before the
    # per-test execution_device fixture begins.
    torch.set_default_device("cuda" if selected_backend() == "cuda_lite" else "cpu")
