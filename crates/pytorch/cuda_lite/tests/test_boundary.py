"""Facts about the per-call binding check, asked without a device.

``boundary.py`` never touches device memory — it reads a declaration and a
tensor's shape and strides — so these run on a CPU host and without the
built extension: the module is loaded straight from its source path rather
than through the package, whose ``__init__`` imports the Rust extension.
"""

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
sympy = pytest.importorskip("sympy")

_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "python"
    / "luminal_cuda_lite"
    / "boundary.py"
)
_SPEC = importlib.util.spec_from_file_location("luminal_cuda_lite_boundary", _SOURCE)
boundary = importlib.util.module_from_spec(_SPEC)
# Registered before execution: a dataclass resolves its field types through
# its own module.
sys.modules[_SPEC.name] = boundary
_SPEC.loader.exec_module(boundary)


def _binding(name, shape, layout):
    return boundary.Binding(
        name=name, buffer=0, dtype=torch.float32, shape=shape, layout=layout
    )


def test_check_binding_refuses_a_stride_no_dimension_of_the_call_pins():
    """A declared stride naming a dimension no boundary of this call gives
    an extent cannot be compared with the tensor's own. This is the last
    place a stride is looked at, so it is refused by name rather than left
    to a runtime that only ever sees an address and a byte count."""
    binding = _binding(
        "x", (sympy.Symbol("s0"), 4), boundary.Strided(("Integer(1)", "Symbol('s1')"))
    )
    with pytest.raises(boundary.UnsupportedBoundary, match="s1"):
        boundary.check_binding(binding, torch.empty(3, 4))


def test_check_binding_reads_strides_against_the_whole_calls_dimensions():
    """A dimension ANOTHER boundary of the same call spells bare is what
    pins this one's declared stride, so the map handed to the check is the
    call's and not the tensor's."""
    x = _binding("x", (sympy.Symbol("s0"), 4), boundary.RowMajor())
    w = _binding("w", (2, 4), boundary.Strided(("Integer(1)", "Symbol('s0')")))
    x_value = torch.empty(3, 4)
    # Shape (2, 4) at element strides (1, 3): two rows of a transposed
    # (4, 3) buffer, so the second axis is strided by x's dimension.
    w_value = torch.empty(4, 3).t()[:2]
    assert w_value.stride() == (1, 3)

    dims = boundary.call_dim_values([(x, x_value), (w, w_value)])
    assert dims == {"s0": 3}
    boundary.check_binding(w, w_value, dims)
    # Without the call's map w states a dimension of its own that nothing
    # pins, and the stride would go unchecked.
    with pytest.raises(boundary.UnsupportedBoundary, match="s0"):
        boundary.check_binding(w, w_value)


def test_check_binding_dimensions_refuse_one_symbol_with_two_extents():
    """One dimension, two extents: the strides of the whole call are read
    against one map, so a call whose boundaries disagree about a dimension
    is refused naming the dimension and both extents."""
    a = _binding("a", (sympy.Symbol("s0"), 4), boundary.RowMajor())
    b = _binding("b", (sympy.Symbol("s0"), 4), boundary.RowMajor())
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.call_dim_values([(a, torch.empty(3, 4)), (b, torch.empty(5, 4))])
    message = str(refusal.value)
    assert "s0" in message, message
    assert "3" in message and "5" in message, message
    assert "'a'" in message and "'b'" in message, message


def test_check_binding_checks_a_compound_extent_against_the_calls_dimensions():
    """A declared extent over the program's dimensions is checked here, not
    only its strides: a call that brings FEWER rows than the declaration
    makes still fits inside the bytes it hands over, so nothing downstream
    would catch it."""
    binding = _binding("x", (2 * sympy.Symbol("s0"), 4), boundary.RowMajor())
    boundary.check_binding(binding, torch.empty(6, 4), {"s0": 3})
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.check_binding(binding, torch.empty(4, 4), {"s0": 3})
    message = str(refusal.value)
    assert "axis 0" in message, message
    assert "s0" in message, message
    assert "6" in message and "4" in message, message


def test_check_binding_refuses_an_extent_no_dimension_of_the_call_pins():
    """The same declaration with nothing stating its dimension: refused by
    name rather than left unchecked."""
    binding = _binding("x", (2 * sympy.Symbol("s0"), 4), boundary.RowMajor())
    with pytest.raises(boundary.UnsupportedBoundary, match="s0"):
        boundary.check_binding(binding, torch.empty(6, 4), {})
