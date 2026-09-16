"""Boundary layouts of torch tensors for `luminal_cuda_lite` bindings.

A boundary tensor is bound zero-copy with the layout it has: the binding
starts at ``tensor.data_ptr()`` — which already carries the storage offset
— and states one element stride per axis relative to that address.
Recognition never reinterprets and never repacks: a stride pattern the
runtime does not model is refused by name.

A stride is stated in the exported program's own vocabulary. Given a fake
example value, an axis strided by a dynamic dimension states THAT
dimension (``Symbol('s77')``) rather than the number one example call
happened to have; a concrete tensor states numbers (``Integer(4)``). Both
cross to the runtime as sympy ``srepr`` strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import sympy
import torch


class UnsupportedBoundary(RuntimeError):
    """A boundary tensor the runtime cannot bind zero-copy.

    A ``RuntimeError`` so Dynamo surfaces it as ``BackendCompilerFailed``
    rather than swallowing it into a silent graph break.
    """


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
    """Element strides relative to ``data_ptr()``, one per axis, each a
    sympy ``srepr`` expression: ``Integer(4)`` for a number, a named
    symbol for an axis strided by a dynamic dimension."""

    strides: tuple[str, ...]


BoundaryLayout = Union[RowMajor, ColumnMajor, Strided]


def _extent(size: Any) -> Any:
    """One size or stride as either a Python ``int`` or the sympy
    expression the exported program carries for it."""
    if isinstance(size, torch.SymInt):
        expr = size.node.expr
        return int(expr) if expr.is_number else expr
    return int(size)


def _same(lhs: Any, rhs: Any) -> bool:
    """Two extents are the same number. Concrete ones compare directly; a
    pair with a symbol in it is decided by sympy."""
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return sympy.simplify(sympy.sympify(lhs) - sympy.sympify(rhs)) == 0


def _srepr(value: Any) -> str:
    """The wire spelling of one stride."""
    return sympy.srepr(sympy.Integer(value) if isinstance(value, int) else value)


def _row_major_strides(shape: Sequence[Any]) -> tuple[Any, ...]:
    """Contiguous, last axis fastest."""
    strides: list[Any] = []
    acc: Any = 1
    for size in reversed(shape):
        strides.append(acc)
        acc = acc * size
    return tuple(reversed(strides))


def _column_major_strides(shape: Sequence[Any]) -> tuple[Any, ...]:
    """Contiguous, first axis fastest."""
    strides: list[Any] = []
    acc: Any = 1
    for size in shape:
        strides.append(acc)
        acc = acc * size
    return tuple(strides)


def _degenerate(size: Any) -> bool:
    """An axis with one coordinate or none: its stride never multiplies
    anything but zero, so it does not decide the layout."""
    return isinstance(size, int) and size <= 1


def _matches(shape: Sequence[Any], strides: Sequence[Any], expected: Sequence[Any]) -> bool:
    return all(
        _degenerate(size) or _same(stride, want)
        for size, stride, want in zip(shape, strides, expected)
    )


def _strides_are_injective(shape: Sequence[Any], strides: Sequence[Any]) -> bool:
    """Distinct coordinates reach distinct elements: axes sorted by stride
    each start past the extent of the previous one.

    Judged over the CONCRETE, non-zero-stride axes. A zero stride is a
    broadcast — every coordinate of that axis reads one element, which is
    a legitimate read map — and a stride the caller spelled symbolically
    is taken as stated, exactly as the runtime takes it: which number it
    is, is the runtime's dims to say.
    """
    axes = [
        (stride, size)
        for stride, size in zip(strides, shape)
        if isinstance(stride, int) and isinstance(size, int) and stride > 0 and size > 1
    ]
    axes.sort()
    reach = 1
    for stride, size in axes:
        if stride < reach:
            return False
        reach = stride * size
    return True


def boundary_shape(tensor: torch.Tensor, fake: Optional[Any] = None) -> tuple[Any, ...]:
    """The declared extents: a literal stays an ``int``, a dynamic one is
    the exported program's own sympy expression."""
    source = fake if fake is not None else tensor
    return tuple(_extent(size) for size in source.shape)


def boundary_layout(
    name: str, tensor: torch.Tensor, fake: Optional[Any] = None
) -> BoundaryLayout:
    """Recognize the layout of a boundary tensor, or refuse it.

    `fake` is the exported program's example value for this tensor, whose
    sizes and strides carry the program's symbols; without one the real
    tensor's numbers are read directly. Never reinterprets: a stride
    pattern the runtime does not model is an error naming the tensor and
    the offending fact.
    """
    if not tensor.is_cuda:
        raise UnsupportedBoundary(f"{name}: expected a CUDA tensor, got device {tensor.device}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise UnsupportedBoundary(f"{name}: dtype {tensor.dtype} has no CUDA-lite storage dtype")
    source = fake if fake is not None else tensor
    shape = tuple(_extent(size) for size in source.shape)
    strides = tuple(_extent(stride) for stride in source.stride())
    if _matches(shape, strides, _row_major_strides(shape)):
        return RowMajor()
    if len(shape) >= 2 and _matches(shape, strides, _column_major_strides(shape)):
        return ColumnMajor()
    for axis, (stride, size) in enumerate(zip(strides, shape)):
        if isinstance(stride, int) and stride < 0 and not _degenerate(size):
            raise UnsupportedBoundary(
                f"{name}: stride {stride} on axis {axis} runs backwards; a boundary is "
                "addressed forwards from data_ptr()"
            )
    if not _strides_are_injective(shape, strides):
        raise UnsupportedBoundary(
            f"{name}: strides {strides} over shape {shape} overlap; "
            "two coordinates share one element"
        )
    return Strided(tuple(_srepr(stride) for stride in strides))


def layout_spec(layout: BoundaryLayout) -> tuple[str, tuple[str, ...]]:
    """The wire form the runtime declares a layout in: a tag and, for a
    strided layout, its element strides as sympy ``srepr`` expressions."""
    if isinstance(layout, RowMajor):
        return "row_major", ()
    if isinstance(layout, ColumnMajor):
        return "column_major", ()
    if isinstance(layout, Strided):
        return "strided", layout.strides
    raise UnsupportedBoundary(f"{layout!r} is not a boundary layout")


def buffer_nbytes(tensor: torch.Tensor) -> int:
    """Bytes the bound buffer spans, reachable from ``data_ptr()``: the
    last element the strides reach, plus one. The storage offset is
    already inside ``data_ptr()``, so it is not counted again."""
    if tensor.numel() == 0:
        return 0
    span = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return span * tensor.element_size()


def storage_span(tensor: torch.Tensor) -> tuple[int, int]:
    """The device address range a binding on this tensor reaches:
    ``data_ptr()`` and one past its last reachable byte."""
    start = tensor.data_ptr()
    return start, start + buffer_nbytes(tensor)


@dataclass(frozen=True)
class Binding:
    """One declared boundary: the graph name, the runtime buffer id, and
    the shape and layout fixed at compile time. `shape` carries an ``int``
    per literal axis and the program's own sympy expression per dynamic
    one; `layout` states the element strides in the same vocabulary. Every
    call must present a tensor that satisfies both."""

    name: str
    buffer: int
    dtype: torch.dtype
    shape: tuple[Any, ...]
    layout: BoundaryLayout


def _dim_values(binding: Binding, tensor: torch.Tensor) -> dict[str, int]:
    """The concrete value this call gives each dynamic dimension, read off
    the axes that are a bare symbol. A compound extent (``s0*2``) is not
    inverted here; the runtime decides it from the dims these bind."""
    values: dict[str, int] = {}
    for declared, size in zip(binding.shape, tensor.shape):
        if isinstance(declared, sympy.Symbol):
            values.setdefault(declared.name, int(size))
    return values


def _declared_strides(
    binding: Binding, shape: Sequence[int], dims: dict[str, int]
) -> tuple[Optional[int], ...]:
    """The element strides the declared layout has at this call's shape: a
    number per axis the declaration decides, ``None`` where a symbol the
    call does not bind leaves it open."""
    if isinstance(binding.layout, RowMajor):
        return _row_major_strides(shape)
    if isinstance(binding.layout, ColumnMajor):
        return _column_major_strides(shape)
    resolved: list[Optional[int]] = []
    for spelling in binding.layout.strides:
        expr = sympy.sympify(spelling)
        expr = expr.subs(
            {symbol: dims[symbol.name] for symbol in expr.free_symbols if symbol.name in dims}
        )
        resolved.append(int(expr) if expr.is_number else None)
    return tuple(resolved)


def check_binding(binding: Binding, tensor: torch.Tensor) -> None:
    """Refuse a call-time tensor that does not match its declared binding:
    the dtype, the rank, every literal extent, and the element strides the
    declared layout has at this call's dims. A stride left symbolic by a
    dimension this call does not pin is the runtime's to check."""
    if tensor.dtype != binding.dtype:
        raise UnsupportedBoundary(
            f"{binding.name}: bound as {binding.dtype}, called with {tensor.dtype}"
        )
    shape = tuple(int(size) for size in tensor.shape)
    if len(shape) != len(binding.shape):
        raise UnsupportedBoundary(
            f"{binding.name}: bound at rank {len(binding.shape)} (shape {binding.shape}), "
            f"called with rank {len(shape)} (shape {shape})"
        )
    for axis, (declared, size) in enumerate(zip(binding.shape, shape)):
        if isinstance(declared, int) and declared != size:
            raise UnsupportedBoundary(
                f"{binding.name}: bound with extent {declared} on axis {axis}, "
                f"called with {size}"
            )
    expected = _declared_strides(binding, shape, _dim_values(binding, tensor))
    actual = tuple(int(stride) for stride in tensor.stride())
    for axis, (want, got, size) in enumerate(zip(expected, actual, shape)):
        if want is None or size <= 1 or want == got:
            continue
        raise UnsupportedBoundary(
            f"{binding.name}: bound with layout {binding.layout}, which at shape {shape} "
            f"is element stride {want} on axis {axis}; the tensor's is {got}"
        )
