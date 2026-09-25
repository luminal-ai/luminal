"""Cuda Lite compiler for ``torch.compile``.

from luminal_cuda_lite import Compiler
compiled = torch.compile(model, backend=Compiler())
"""

from luminal_reference import _torch_version as _torch_version

# Validate PyTorch before importing the native extension.
# isort: split
from ._luminal import PlanTemplate, search_configuration
from ._luminal import compile as _compile
from .arena_pool import arena_pool_stats, clear_arena_pool
from .backend import CompiledModel, luminal_cuda_lite, register_backend
from .compiler import Compiler
from .plan_cache import cache_stats, clear_plan_cache

__all__ = [
    "Compiler",
    "CompiledModel",
    "PlanTemplate",
    "arena_pool_stats",
    "cache_stats",
    "clear_arena_pool",
    "clear_plan_cache",
    "compile",
    "luminal_cuda_lite",
    "register_backend",
    "search_configuration",
]

# Register the backend string form (`backend="luminal_cuda_lite"`) on import.
register_backend()


def compile(*args, **kwargs):
    """Compile a saved ``.pt2`` on the CUDA-lite backend.

    Takes the path and the caller's boundary layouts: one
    ``(graph name, layout tag, element strides)`` row per graph input and one
    per user-visible graph output, each stride a sympy ``srepr`` expression
    over the exported program's own symbols (see ``boundary.layout_spec``). A
    writeback takes no row — it is bound at the layout of the input it
    mutates.
    """
    return _compile(*args, **kwargs)
