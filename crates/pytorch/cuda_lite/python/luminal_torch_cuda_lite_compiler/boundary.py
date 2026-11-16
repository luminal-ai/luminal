"""Boundary layouts of torch tensors for `luminal_torch_cuda_lite_compiler` bindings.

A boundary tensor is bound zero-copy in ONE form: the element strides it
has, one per axis, relative to ``tensor.data_ptr()``. Nothing here
classifies — whether a chain is contiguous, column-major or neither is
discovered from the chain itself in the e-graph, so this module states
what the tensor has and never interprets it. A tensor the runtime cannot
address as it lies is refused by name, never repacked.

A binding names a buffer's BASE, so a boundary tensor whose storage
offset is non-zero is refused: two bindings on one buffer share that
base, and there is no way to say "same buffer, different offset".

A stride is stated in the exported program's own vocabulary. Given a fake
example value, an axis strided by a dynamic dimension states THAT
dimension (``Symbol('s77')``) rather than the number one example call
happened to have; a concrete tensor states numbers (``Integer(4)``). Both
cross to the runtime as sympy ``srepr`` strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import sympy
import torch
from luminal_torch_cuda_lite_runtime import BoundaryRefusal

# A boundary tensor the runtime cannot bind zero-copy. One class for both
# halves: the compiler raises it for a tensor it cannot declare, and the
# runtime raises it for a tensor a call hands it. It is a ``RuntimeError``,
# so Dynamo surfaces it as ``BackendCompilerFailed`` rather than swallowing
# it into a silent graph break.
UnsupportedBoundary = BoundaryRefusal


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
class Strided:
    """Element strides relative to ``data_ptr()``, one per axis, each a
    sympy ``srepr`` expression: ``Integer(4)`` for a number, a named
    symbol for an axis strided by a dynamic dimension."""

    strides: tuple[str, ...]


def _extent(size: Any) -> Any:
    """One size or stride as either a Python ``int`` or the sympy
    expression the exported program carries for it."""
    if isinstance(size, torch.SymInt):
        expr = size.node.expr
        return int(expr) if expr.is_number else expr
    return int(size)


def _srepr(value: Any) -> str:
    """The wire spelling of one stride."""
    return sympy.srepr(sympy.Integer(value) if isinstance(value, int) else value)


def _reads_as_plain_storage(value: Any) -> bool:
    """A torch tensor whose sizes and strides describe its own storage: a
    tensor, a parameter, or the exported program's fake stand-in for one.
    A subclass carrying semantics of its own — a distributed, masked or
    nested tensor — describes something else.

    Asked by EXACT type: ``isinstance(t, torch.nn.Parameter)`` is true of
    any tensor carrying ``_is_param``, which is how a custom subclass
    becomes a parameter while staying that subclass.
    """
    # Imported here because the fake tensor lives in a private torch module.
    from torch._subclasses.fake_tensor import FakeTensor

    return type(value) in (torch.Tensor, torch.nn.Parameter, FakeTensor)


def _refuse_unreadable(name: str, value: Any, *, allow_offset: bool = False) -> None:
    """Refuse a boundary value whose storage a binding cannot state: a
    tensor subclass, a layout that is not dense strides, or storage whose
    base the binding cannot name. ``allow_offset`` admits a value whose
    traced storage offset is not zero: the layout statement of an output
    the caller ALLOCATES, whose own storage starts where eager's view did
    not."""
    if isinstance(value, torch.Tensor) and not _reads_as_plain_storage(value):
        raise UnsupportedBoundary(
            f"{name}: {type(value).__name__} is a tensor subclass with semantics of its "
            "own, not storage to bind"
        )
    if value.layout is not torch.strided:
        raise UnsupportedBoundary(
            f"{name}: layout {value.layout} is not torch.strided; a binding states "
            "element strides over dense storage"
        )
    offset = _extent(value.storage_offset())
    if offset != 0 and not allow_offset:
        raise UnsupportedBoundary(
            f"{name}: storage offset {offset} is not bound today; a binding names a "
            "buffer's base, and two bindings on one buffer share that base"
        )


def dense_strides(shape: Sequence[Any]) -> tuple[Any, ...]:
    """Row-major element strides for ``shape``: an ``int`` per literal axis,
    the program's own symbolic arithmetic per dynamic one."""
    strides: list[Any] = []
    span: Any = 1
    for size in reversed(tuple(shape)):
        strides.append(span)
        span = span * size
    return tuple(reversed(strides))


def boundary_layout(
    name: str,
    tensor: torch.Tensor,
    fake: Optional[Any] = None,
    *,
    allocated: bool = False,
    dense: bool = False,
) -> Strided:
    """The element strides a boundary tensor is bound at, or a refusal.

    `fake` is the exported program's example value for this tensor, whose
    sizes and strides carry the program's symbols; without one the real
    tensor's numbers are read directly. ONE FORM COMES OUT: the chain the
    tensor has. Which map that chain is — contiguous, column-major,
    neither — is the e-graph's to discover from the chain, so nothing is
    classified here and nothing is reinterpreted: a tensor this boundary
    cannot state is an error naming it and the offending fact.

    ``allocated`` states the layout of an output the caller allocates
    fresh at these strides: eager's value may be a view starting inside
    some intermediate's storage, and that offset is not reproduced, so it
    is not refused. A value bound AS IT LIES (an input, or an output that
    is a view of a boundary tensor) keeps the offset refusal. ``dense``
    states row-major strides over the value's extents instead of the
    strides eager gave it: the layout of an output nobody but this backend
    reads, such as an activation saved for the backward graph.
    """
    allocated = allocated or dense
    _refuse_unreadable(name, tensor, allow_offset=allocated)
    if fake is not None:
        _refuse_unreadable(name, fake, allow_offset=allocated)
    if not tensor.is_cuda:
        raise UnsupportedBoundary(
            f"{name}: expected a CUDA tensor, got device {tensor.device}"
        )
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise UnsupportedBoundary(
            f"{name}: dtype {tensor.dtype} has no CUDA-lite storage dtype"
        )
    source = fake if fake is not None else tensor
    if dense:
        strides = tuple(_extent(stride) for stride in dense_strides(source.shape))
    else:
        strides = tuple(_extent(stride) for stride in source.stride())
    for axis, stride in enumerate(strides):
        if isinstance(stride, int) and stride < 0:
            raise UnsupportedBoundary(
                f"{name}: element stride {stride} on axis {axis} runs backwards; a "
                "boundary is addressed forwards from data_ptr()"
            )
    return Strided(tuple(_srepr(stride) for stride in strides))


def layout_spec(layout: Strided) -> tuple[str, tuple[str, ...]]:
    """The wire form the runtime declares a layout in: the ``strided`` tag
    and the element strides as sympy ``srepr`` expressions."""
    if isinstance(layout, Strided):
        return "strided", layout.strides
    raise UnsupportedBoundary(f"{layout!r} is not a boundary layout")


def layout_from_spec(name: str, tag: str, strides: Sequence[str]) -> Strided:
    """The same wire form read back: the layout the runtime says a
    boundary is bound at, in the spelling it was declared in. This
    boundary declares element strides and nothing else, so any other tag
    is an error naming the boundary."""
    if tag == "strided":
        return Strided(tuple(strides))
    raise UnsupportedBoundary(
        f"{name}: boundary layout {tag!r} is not an element-strides chain"
    )


def buffer_nbytes(tensor: torch.Tensor) -> int:
    """Bytes the bound buffer spans, reachable from ``data_ptr()``: the
    last element the strides reach, plus one. A bound tensor has storage
    offset zero, so ``data_ptr()`` is the base of that span."""
    if tensor.numel() == 0:
        return 0
    span = 1 + sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    return span * tensor.element_size()
