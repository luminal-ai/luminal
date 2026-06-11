"""Luminal Python bindings - PyTorch backend using Luminal."""

# Import Python components
# Register DynamicCache pytree serialization once at import time
from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel

# Import Rust extension components (built by maturin)
from .luminal import CompiledGraph, process_pt2
from .main import luminal_backend, register_backend

_register_cache_serialization()

# Re-export everything for clean package interface
__all__ = [
    "CompiledModel",
    "luminal_backend",
    "register_backend",
    "CompiledGraph",
    "process_pt2",
]
