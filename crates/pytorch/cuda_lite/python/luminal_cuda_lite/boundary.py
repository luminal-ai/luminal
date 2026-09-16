"""Boundary layouts of torch tensors for `luminal_cuda_lite` bindings.

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


def _refuse_unreadable(name: str, value: Any) -> None:
    """Refuse a boundary value whose storage a binding cannot state: a
    tensor subclass, a layout that is not dense strides, or storage whose
    base the binding cannot name."""
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
    if offset != 0:
        raise UnsupportedBoundary(
            f"{name}: storage offset {offset} is not bound today; a binding names a "
            "buffer's base, and two bindings on one buffer share that base"
        )


def boundary_shape(tensor: torch.Tensor, fake: Optional[Any] = None) -> tuple[Any, ...]:
    """The declared extents: a literal stays an ``int``, a dynamic one is
    the exported program's own sympy expression."""
    source = fake if fake is not None else tensor
    return tuple(_extent(size) for size in source.shape)


def boundary_layout(
    name: str, tensor: torch.Tensor, fake: Optional[Any] = None
) -> Strided:
    """The element strides a boundary tensor is bound at, or a refusal.

    `fake` is the exported program's example value for this tensor, whose
    sizes and strides carry the program's symbols; without one the real
    tensor's numbers are read directly. ONE FORM COMES OUT: the chain the
    tensor has. Which map that chain is — contiguous, column-major,
    neither — is the e-graph's to discover from the chain, so nothing is
    classified here and nothing is reinterpreted: a tensor this boundary
    cannot state is an error naming it and the offending fact.
    """
    _refuse_unreadable(name, tensor)
    if fake is not None:
        _refuse_unreadable(name, fake)
    if not tensor.is_cuda:
        raise UnsupportedBoundary(f"{name}: expected a CUDA tensor, got device {tensor.device}")
    if tensor.dtype not in SUPPORTED_DTYPES:
        raise UnsupportedBoundary(f"{name}: dtype {tensor.dtype} has no CUDA-lite storage dtype")
    source = fake if fake is not None else tensor
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
    layout: Strided


def _dim_values(binding: Binding, tensor: torch.Tensor) -> dict[str, int]:
    """The concrete value this call gives each dynamic dimension, read off
    the axes of THIS tensor that are a bare symbol. A compound extent
    (``s0*2``) is not inverted here; a dimension only this binding spells
    compound is pinned by whichever boundary spells it bare."""
    values: dict[str, int] = {}
    for declared, size in zip(binding.shape, tensor.shape):
        if isinstance(declared, sympy.Symbol):
            values.setdefault(declared.name, int(size))
    return values


def call_dim_values(pairs: Sequence[tuple[Binding, torch.Tensor]]) -> dict[str, int]:
    """The value one call gives each dynamic dimension, read off the
    bare-symbol axes of every boundary tensor in it. A dimension two
    boundaries give two different extents is refused by name: the declared
    strides are read against this one map, so it must be single-valued."""
    values: dict[str, int] = {}
    source: dict[str, str] = {}
    for binding, tensor in pairs:
        for symbol, value in _dim_values(binding, tensor).items():
            if symbol in values and values[symbol] != value:
                raise UnsupportedBoundary(
                    f"dimension {symbol} is {values[symbol]} in {source[symbol]!r} and "
                    f"{value} in {binding.name!r}: one dimension, two extents"
                )
            values.setdefault(symbol, value)
            source.setdefault(symbol, binding.name)
    return values


def declared_strides(
    binding: Binding, shape: Sequence[int], dims: dict[str, int]
) -> tuple[int, ...]:
    """The element strides the declared layout has at this call's shape:
    what an output is allocated with, and what a call's tensors are
    checked against. A declared stride still naming a symbol after the
    call's dimensions are substituted is refused by name: nothing
    downstream compares strides, so an unresolved one would go
    unchecked."""
    resolved: list[int] = []
    for axis, spelling in enumerate(binding.layout.strides):
        expr = sympy.sympify(spelling)
        expr = expr.subs(
            {symbol: dims[symbol.name] for symbol in expr.free_symbols if symbol.name in dims}
        )
        if not expr.is_number:
            free = ", ".join(sorted(symbol.name for symbol in expr.free_symbols))
            raise UnsupportedBoundary(
                f"{binding.name}: element stride {spelling} on axis {axis} still names "
                f"{free} once this call's dimensions are substituted; no boundary of this "
                "call gives that dimension an extent"
            )
        resolved.append(int(expr))
    return tuple(resolved)


def check_binding(
    binding: Binding, tensor: torch.Tensor, dims: Optional[dict[str, int]] = None
) -> None:
    """Refuse a call-time tensor that does not match its declared binding:
    the storage it describes, the dtype, the rank, every extent — literal,
    or compound over this call's dimensions — and the element strides the
    declared layout has at those dimensions.

    `dims` is the whole call's dimension map (`call_dim_values`); without
    one only this tensor's own bare-symbol axes pin the declared strides.
    This is the last place a stride is looked at: the runtime is handed an
    address and a byte count, so past this check only Dynamo's shape
    guards stand between a caller and a misread layout.
    """
    _refuse_unreadable(binding.name, tensor)
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
    dims = _dim_values(binding, tensor) if dims is None else dims
    for axis, (declared, size) in enumerate(zip(binding.shape, shape)):
        if isinstance(declared, int):
            if declared != size:
                raise UnsupportedBoundary(
                    f"{binding.name}: bound with extent {declared} on axis {axis}, "
                    f"called with {size}"
                )
            continue
        # A declared extent over the program's dimensions states what those
        # dimensions make it, which is checked here: past this the runtime
        # is handed an address and a byte count, and a call whose extent is
        # SHORTER than the declared one reaches inside the bytes it brought,
        # so no length check downstream catches it.
        extent = sympy.sympify(declared).subs(
            {
                symbol: dims[symbol.name]
                for symbol in declared.free_symbols
                if symbol.name in dims
            }
        )
        if not extent.is_number:
            free = ", ".join(sorted(symbol.name for symbol in extent.free_symbols))
            raise UnsupportedBoundary(
                f"{binding.name}: axis {axis} is declared {declared}, which still names "
                f"{free} once this call's dimensions are substituted; no boundary of this "
                "call gives that dimension an extent"
            )
        if int(extent) != size:
            raise UnsupportedBoundary(
                f"{binding.name}: axis {axis} is declared {declared}, which this call's "
                f"dimensions make {int(extent)}, but the tensor's extent is {size}"
            )
    expected = declared_strides(binding, shape, dims)
    actual = tuple(int(stride) for stride in tensor.stride())
    # Which strides carry information is torch's own contract, spelled in
    # inductor's `significant_strides_equal`: an axis of extent 0 or 1 has
    # no significant stride, and one empty axis makes every stride
    # insignificant, because the tensor addresses no element at all.
    if any(size == 0 for size in shape) or any(declared == 0 for declared in binding.shape):
        return
    for axis, (want, got, size) in enumerate(zip(expected, actual, shape)):
        if size <= 1 or want == got:
            continue
        raise UnsupportedBoundary(
            f"{binding.name}: bound with layout {binding.layout}, which at shape {shape} "
            f"is element stride {want} on axis {axis}; the tensor's is {got}"
        )
