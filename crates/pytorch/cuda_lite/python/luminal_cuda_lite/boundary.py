"""Boundary layouts of torch tensors for `luminal_cuda_lite` bindings.

A boundary tensor is bound zero-copy with the layout it has. Recognition
never reinterprets: a stride pattern the runtime does not model is refused
by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Union

import torch


class UnsupportedBoundary(RuntimeError):
    """A boundary tensor the runtime cannot bind zero-copy."""


# Storage dtypes the CUDA-lite kernels read and write.
SUPPORTED_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.bfloat16: "Bf16",
    torch.int32: "Int",
    torch.int64: "I64",
    torch.bool: "Bool",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
}


@dataclass(frozen=True)
class RowMajor:
    pass


@dataclass(frozen=True)
class ColumnMajor:
    pass


@dataclass(frozen=True)
class Strided:
    """Element strides, one per axis, with no storage offset."""

    strides: tuple[int, ...]


BoundaryLayout = Union[RowMajor, ColumnMajor, Strided]


def _column_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    strides = []
    acc = 1
    for size in shape:
        strides.append(acc)
        acc *= size
    return tuple(strides)


def _strides_are_injective(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Distinct coordinates reach distinct elements: axes sorted by stride
    each start past the extent of the previous one."""
    axes = [(stride, size) for stride, size in zip(strides, shape) if size > 1]
    axes.sort()
    reach = 1
    for stride, size in axes:
        if stride < reach:
            return False
        reach = stride * size
    return True


def boundary_layout(name: str, tensor: torch.Tensor) -> BoundaryLayout:
    """Recognize the layout of a boundary tensor, or refuse it.

    Never reinterprets: a stride pattern the runtime does not model is an
    error naming the tensor and the offending fact.
    """
    if not tensor.is_cuda:
        raise UnsupportedBoundary(f"{name}: expected a CUDA tensor, got device {tensor.device}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise UnsupportedBoundary(f"{name}: dtype {tensor.dtype} has no CUDA-lite storage dtype")
    if tensor.storage_offset() != 0:
        raise UnsupportedBoundary(
            f"{name}: storage offset {tensor.storage_offset()} is not modeled; "
            "bind the base tensor instead"
        )
    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    if tensor.is_contiguous():
        return RowMajor()
    if len(shape) >= 2 and strides == _column_major_strides(shape):
        return ColumnMajor()
    if any(stride <= 0 for stride, size in zip(strides, shape) if size > 1):
        raise UnsupportedBoundary(
            f"{name}: strides {strides} contain a zero or negative stride "
            "(a broadcast or flipped view); it is not an addressable buffer"
        )
    if not _strides_are_injective(shape, strides):
        raise UnsupportedBoundary(
            f"{name}: strides {strides} over shape {shape} overlap; "
            "two coordinates share one element"
        )
    return Strided(strides)


def _nbytes(tensor: torch.Tensor) -> int:
    """Bytes the bound buffer spans: the last reachable element plus one."""
    if tensor.numel() == 0:
        return 0
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return span * tensor.element_size()


@dataclass(frozen=True)
class Binding:
    """One declared boundary: the graph name, the runtime buffer id and the
    layout fixed at compile time. Every call must present a tensor whose
    layout is exactly this one."""

    name: str
    buffer: int
    dtype: torch.dtype
    shape: tuple[Any, ...]
    layout: BoundaryLayout


def check_binding(binding: Binding, tensor: torch.Tensor) -> None:
    """Refuse a call-time tensor that does not match its declared binding."""
    if tensor.dtype != binding.dtype:
        raise UnsupportedBoundary(
            f"{binding.name}: bound as {binding.dtype}, called with {tensor.dtype}"
        )
    layout = boundary_layout(binding.name, tensor)
    if layout != binding.layout:
        raise UnsupportedBoundary(
            f"{binding.name}: bound with layout {binding.layout}, called with {layout}"
        )
