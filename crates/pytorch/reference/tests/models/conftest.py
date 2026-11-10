"""Shared model parity cases, also explicitly selected by the CUDA runner.

Model tests state placement on their models and inputs. A suite-wide
``torch.device`` context also wraps HuggingFace ``from_pretrained`` and makes
Transformers infer a ``device_map``, which is a different loading contract and
requires the optional ``accelerate`` package.
"""

import gc

import pytest
import torch
from backend_test_utils import device, selected_backend

__all__ = ["device"]


@pytest.fixture(autouse=True)
def release_previous_model_memory():
    """Keep model loading explicit and release the preceding case.

    Full-model CUDA cases can otherwise retain the previous compiled model in
    a Python reference cycle and leave PyTorch's now-unused blocks reserved.
    Luminal correctly treats those blocks as unavailable device memory, but
    that makes the next independent test depend on collection timing.

    The shared operation suite sets a global default device during combined
    collection so its module-level constants use the selected backend. Model
    tests place every model and input explicitly; restore CPU here before the
    test body so HuggingFace loading cannot infer an unintended device map.
    """
    gc.collect()
    if selected_backend() == "cuda_lite" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous)
