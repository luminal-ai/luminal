"""Shared dtype utility functions for the luminal Python Bridge"""

import torch

_TORCH_DTYPE_TO_CODE = {
    torch.uint8: 1,
    torch.int8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bool: 12,
    torch.bfloat16: 13,
}

_CODE_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_TO_CODE.items()}


def torch_dtype_code(dtype):
    """Map torch.dtype to PT2 dtype integer code."""
    return _TORCH_DTYPE_TO_CODE.get(dtype, 7)  # default to f32


def code_to_torch_dtype(code):
    """Map PT2 dtype integer code to torch.dtype."""
    return _CODE_TO_TORCH_DTYPE.get(code, torch.float32)
