"""Luminal's reference-backend torch.compile integration.

Usage::

    import luminal_reference
    compiled = torch.compile(model, backend=luminal_reference)
"""

import sys
import types

from ._luminal import compile as _compile
from .backend import CompiledModel, luminal_reference, register_backend

__all__ = [
    "CompiledModel",
    "luminal_reference",
    "register_backend",
    "compile",
]

# Register the backend string form (`backend="luminal_reference"`) on import.
register_backend()


def compile(*args, **kwargs):
    """Compile a saved ``.pt2`` file on the reference backend."""
    return _compile(*args, **kwargs)


class _CallableModule(types.ModuleType):
    """Make the module itself usable as a torch.compile backend callable."""

    def __call__(self, gm, example_inputs, **kwargs):
        return luminal_reference(gm, example_inputs, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
