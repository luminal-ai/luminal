"""Luminal Python bindings - PyTorch backend using Luminal."""

# Import Python components
# Register DynamicCache pytree serialization once at import time
from .artifact_cache import (
    ArtifactCacheStats,
    artifact_cache_stats,
    clear_artifact_cache,
)
from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel

# Import Rust extension components (built by maturin)
from .luminal import CompiledGraph, process_pt2
from .main import luminal_backend, register_backend
from .region_compile import compile_region, load_region_artifact

_register_cache_serialization()

# Re-export everything for clean package interface
__all__ = [
    "CompiledModel",
    "ArtifactCacheStats",
    "artifact_cache_stats",
    "clear_artifact_cache",
    "luminal_backend",
    "register_backend",
    "CompiledGraph",
    "compile_region",
    "load_region_artifact",
    "process_pt2",
]
