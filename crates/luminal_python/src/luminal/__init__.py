"""Luminal Python bindings - PyTorch backend using Luminal."""

# Import Python components
from .compiled_model import CompiledModel

# Import Rust extension components (built by maturin)
# These are available directly in the package namespace
from .luminal import CompiledGraph, process_onnx, process_pt2
from .main import luminal_backend

# Register DynamicCache pytree serialization once at import time
from .cache_utils import _register_cache_serialization

_register_cache_serialization()

# Re-export everything for clean package interface
__all__ = [
    "CompiledModel",
    "luminal_backend",
    "process_onnx",
    "CompiledGraph",
    "process_pt2",
]
