"""Facts about what the boundary declares and refuses, asked without a device.

``boundary.py`` never touches device memory — it reads a tensor's class,
layout, dtype, storage offset, shape and strides, and a declaration — so
these run on a CPU host and without the built extension: the module is
loaded straight from its source path rather than through the package,
whose ``__init__`` imports the Rust extension.
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


def _chain(strides):
    """A declared layout at literal element strides, in the wire spelling."""
    return boundary.Strided(tuple(f"Integer({stride})" for stride in strides))


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
    x = _binding("x", (sympy.Symbol("s0"), 4), _chain((4, 1)))
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
    a = _binding("a", (sympy.Symbol("s0"), 4), _chain((4, 1)))
    b = _binding("b", (sympy.Symbol("s0"), 4), _chain((4, 1)))
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
    binding = _binding("x", (2 * sympy.Symbol("s0"), 4), _chain((4, 1)))
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
    binding = _binding("x", (2 * sympy.Symbol("s0"), 4), _chain((4, 1)))
    with pytest.raises(boundary.UnsupportedBoundary, match="s0"):
        boundary.check_binding(binding, torch.empty(6, 4), {})


class _Boundary:
    """What the boundary reads off a tensor: the kind of storage it is, a
    dtype, extents, and element strides. Stated as plain attributes, so a
    fact about a layout that is refused — or about an EMPTY tensor —
    needs neither a device nor an allocation."""

    def __init__(
        self,
        shape,
        strides,
        dtype=torch.float32,
        layout=torch.strided,
        offset=0,
        device="cuda:0",
    ):
        self.dtype = dtype
        self.shape = tuple(shape)
        self.layout = layout
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self._strides = tuple(strides)
        self._offset = offset

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self._offset


def _torch_contiguous_strides(shape):
    """PyTorch's own contiguous strides (``TensorImpl::empty_tensor_restride``):
    the running product multiplies by ``max(size, 1)``, so a zero extent
    leaves the strides outside it as a full tensor's."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * max(shape[axis + 1], 1)
    return tuple(strides)


def test_a_layout_that_is_not_dense_strides_is_refused_by_name():
    """A sparse, nested or quantized tensor is not element strides over
    dense storage, so there is no chain to state: refused naming the
    boundary and the layout it arrived with."""
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("x", _Boundary((4, 4), (4, 1), layout=torch.sparse_coo))
    message = str(refusal.value)
    assert "x:" in message, message
    assert "sparse_coo" in message, message


def test_a_dtype_with_no_storage_dtype_is_refused_by_name():
    """The kernels read a fixed set of storage dtypes; a tensor of any
    other is refused naming it and the dtype, never reinterpreted as
    bytes of a dtype they do read."""
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("x", _Boundary((4, 4), (4, 1), dtype=torch.complex64))
    message = str(refusal.value)
    assert "x:" in message, message
    assert "complex64" in message, message


def test_a_negative_stride_is_refused_by_name():
    """A boundary is addressed forwards from ``data_ptr()``, so an axis
    that runs backwards has no binding: refused naming the axis and the
    stride."""
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("x", _Boundary((4, 4), (4, -1)))
    message = str(refusal.value)
    assert "axis 1" in message, message
    assert "-1" in message, message


def test_a_storage_offset_is_refused_by_name():
    """A binding names a buffer's BASE, and two bindings on one buffer
    share that base — there is no way to say "same buffer, different
    offset". The whole class is refused rather than bound alone with its
    aliasing left unsayable."""
    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("x", _Boundary((4, 4), (4, 1), offset=8))
    message = str(refusal.value)
    assert "x:" in message, message
    assert "storage offset 8" in message, message
    # The refusal is the OFFSET, not the view: a transposed or strided view
    # that starts at its storage's base is stated at the chain it has.
    assert boundary.boundary_layout("x", _Boundary((4, 4), (1, 8))) == _chain((1, 8))


def test_a_storage_offset_is_refused_at_call_time_too():
    """Dynamo guards shapes, not storage offsets, so a program compiled on
    a tensor at its storage's base can be called with one that is not:
    checked again per call."""
    binding = _binding("x", (4, 4), _chain((4, 1)))
    boundary.check_binding(binding, _Boundary((4, 4), (4, 1)))
    with pytest.raises(boundary.UnsupportedBoundary, match="storage offset 8"):
        boundary.check_binding(binding, _Boundary((4, 4), (4, 1), offset=8))


def test_a_tensor_subclass_is_refused_by_name():
    """A subclass carries semantics of its own — a distributed tensor's
    sizes are the global ones, a masked tensor's elements are not all
    there — so its sizes and strides are not a statement about storage to
    bind. A parameter is plain storage under another class and passes the
    same gate: what refuses it here is its device, not its type."""

    class _Exotic(torch.Tensor):
        pass

    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("x", torch.empty(4, 4).as_subclass(_Exotic))
    message = str(refusal.value)
    assert "x:" in message, message
    assert "_Exotic" in message, message

    with pytest.raises(boundary.UnsupportedBoundary, match="expected a CUDA tensor"):
        boundary.boundary_layout("w", torch.nn.Parameter(torch.empty(4, 4)))


def test_a_parameter_of_a_custom_tensor_type_is_refused_by_name():
    """``Parameter`` of a custom tensor type IS that custom type, flagged:
    ``isinstance(t, torch.nn.Parameter)`` answers yes for any tensor
    carrying ``_is_param`` while its sizes and strides still describe
    something other than the storage a binding would name. The gate is the
    exact type, so the subclass is refused naming itself."""

    class _Exotic(torch.Tensor):
        pass

    exotic = torch.empty(4, 4).as_subclass(_Exotic)
    exotic._is_param = True
    assert isinstance(exotic, torch.nn.Parameter)

    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout("w", exotic)
    message = str(refusal.value)
    assert "w:" in message, message
    assert "_Exotic" in message, message


@pytest.mark.parametrize(
    ("shape", "strides"),
    [((4, 0), (1, 1)), ((0, 4), (4, 1)), ((2, 0, 3), (3, 3, 1))],
)
def test_a_zero_extent_states_torchs_own_strides(shape, strides):
    """The declaration IS the tensor's chain, so torch's own convention at
    an empty axis crosses as it is: the running product multiplies by one
    there, not by zero, and the axes outside an empty one are strided past
    it. Nothing here synthesizes a chain that could disagree."""
    assert strides == _torch_contiguous_strides(shape)
    assert boundary.boundary_layout("out", _Boundary(shape, strides)) == _chain(strides)


@pytest.mark.parametrize(
    ("shape", "row_chain", "column_chain"),
    [((4, 0), (1, 1), (1, 4)), ((0, 4), (4, 1), (1, 1))],
)
def test_a_zero_extent_allocation_is_not_refused(shape, row_chain, column_chain):
    """What the wrapper allocates for an empty output — ``empty_strided``
    at the declared strides — is what the binding check must accept,
    whichever chain the output was declared at."""
    for chain in (row_chain, column_chain):
        binding = _binding("out", shape, _chain(chain))
        allocated = boundary.declared_strides(binding, shape, {})
        assert allocated == chain
        boundary.check_binding(binding, _Boundary(shape, allocated))


def test_a_zero_extent_makes_every_stride_insignificant():
    """One empty axis and the tensor addresses no element at all, so no
    stride carries information — the contract inductor states in
    ``significant_strides_equal``. The strides here are nothing torch would
    produce and are accepted all the same."""
    binding = _binding("out", (4, 0, 3), _chain((3, 3, 1)))
    boundary.check_binding(binding, _Boundary((4, 0, 3), (99, 7, 5)))
    # An extent that is not zero is still held to the declared stride.
    full = _binding("out", (4, 2, 3), _chain((6, 3, 1)))
    with pytest.raises(boundary.UnsupportedBoundary, match="axis 0"):
        boundary.check_binding(full, _Boundary((4, 2, 3), (99, 3, 1)))
