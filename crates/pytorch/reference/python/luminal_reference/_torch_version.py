"""Check PyTorch compatibility before loading any native bindings."""

import torch
from packaging.version import Version

if Version(torch.__version__) < Version("2.14.0"):
    raise ImportError(
        f"Luminal requires PyTorch >= 2.14.0; found {torch.__version__}. "
        "Upgrade PyTorch in this Python environment before importing Luminal."
    )
