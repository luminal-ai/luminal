"""Reference compiler for ``torch.compile``.

from luminal_reference import Compiler
compiled = torch.compile(model, backend=Compiler())
"""

from . import _torch_version as _torch_version

# Validate PyTorch before importing the native extension.
# isort: split
from .compiler import Compiler
from .dimensions import DimBucket

__all__ = ["Compiler", "DimBucket"]
