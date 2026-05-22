"""Shared dtype utility functions for the luminal Python bridge.

The PT2 dtype-code numbering is sourced from
``torch._export.serde.schema.ScalarType`` at import time — PyTorch is the
canonical source of truth on both sides of the FFI boundary. The Rust side
mirrors the same enum in ``luminal_python/rust/src/torch_dtype.rs`` and is
held in agreement by ``tests/test_torch_dtype_parity.py``.

``torch._export.serde.schema`` is a quasi-private API (leading underscore),
but it is the module PT2 export actually wire-serializes against; binding
to it here is the right boundary. If PyTorch reorganizes the module path,
the import below will fail loudly at module load.
"""

import torch
from torch._export.serde.schema import ScalarType

# Map each `torch.dtype` we care about to the PT2 code PyTorch itself
# would emit for it. Looking up `ScalarType.<NAME>.value` keeps the
# numbering in lockstep with PyTorch — if PyTorch renumbers, we pick
# up the new code automatically (and the Rust parity test catches the
# drift from the other side).
_TORCH_DTYPE_TO_CODE = {
    torch.uint8: ScalarType.BYTE.value,
    torch.int8: ScalarType.CHAR.value,
    torch.int16: ScalarType.SHORT.value,
    torch.int32: ScalarType.INT.value,
    torch.int64: ScalarType.LONG.value,
    torch.float16: ScalarType.HALF.value,
    torch.float32: ScalarType.FLOAT.value,
    torch.float64: ScalarType.DOUBLE.value,
    torch.bool: ScalarType.BOOL.value,
    torch.bfloat16: ScalarType.BFLOAT16.value,
}

_CODE_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_CODE.items()}


def torch_dtype_code(dtype):
    """Map torch.dtype to PT2 dtype integer code. Raises `KeyError`
    on an unsupported dtype rather than silently aliasing to FLOAT."""
    try:
        return _TORCH_DTYPE_TO_CODE[dtype]
    except KeyError:
        raise KeyError(
            f"torch_dtype_code: {dtype} isn't a supported PT2 dtype "
            f"(supported: {sorted(_TORCH_DTYPE_TO_CODE.keys(), key=str)})"
        ) from None


def code_to_torch_dtype(code):
    """Map PT2 dtype integer code to torch.dtype. Raises `KeyError`
    on an unknown code rather than silently defaulting to float32."""
    try:
        return _CODE_TO_TORCH_DTYPE[code]
    except KeyError:
        raise KeyError(
            f"code_to_torch_dtype: PT2 dtype code {code} isn't mapped "
            f"to a torch.dtype (known codes: "
            f"{sorted(_CODE_TO_TORCH_DTYPE.keys())})"
        ) from None
