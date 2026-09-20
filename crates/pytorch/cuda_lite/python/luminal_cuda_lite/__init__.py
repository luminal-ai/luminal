"""Luminal's CUDA-lite torch.compile integration.

Usage::

    import luminal_cuda_lite
    compiled = torch.compile(model, backend=luminal_cuda_lite)

Requires a CUDA device and a wheel built with the runtime's ``device``
feature (the packaged wheel always is; a bare ``cargo build`` is not).
"""

import sys
import types

from ._luminal import compile as _compile
from .backend import CompiledModel, luminal_cuda_lite, register_backend

__all__ = [
    "CompiledModel",
    "luminal_cuda_lite",
    "register_backend",
    "compile",
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


class _CallableModule(types.ModuleType):
    """Make the module itself usable as a torch.compile backend callable."""

    def __call__(self, gm, example_inputs, **kwargs):
        return luminal_cuda_lite(gm, example_inputs, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
