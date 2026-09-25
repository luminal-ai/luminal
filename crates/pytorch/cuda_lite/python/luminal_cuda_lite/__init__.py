"""Cuda Lite compiler for ``torch.compile``.

from luminal_cuda_lite import Compiler
compiled = torch.compile(model, backend=Compiler())
"""

from luminal_reference import _torch_version as _torch_version

# Validate PyTorch before importing the native extension.
# isort: split
from .compiler import Compiler

__all__ = ["Compiler"]
