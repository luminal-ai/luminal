"""Facts about what the boundary declares and refuses at COMPILE time, asked
without a device.

What a CALL's tensors must satisfy is checked by the runtime when it reads
the arguments; those facts are the unit tests of ``call.rs`` in the
extension, and ``test_backend.py`` exercises them through compiled calls.

``boundary.py`` never touches device memory — it reads a tensor's class,
layout, dtype, storage offset, shape and strides, and a declaration — so
these run on a CPU host and without the compiler's extension: the module
is loaded straight from its source path rather than through the package,
whose ``__init__`` imports the compiler's Rust extension. It does import
the runtime package, for the refusal class the two share.

A real tensor built here names its device. The shared suite's fixtures set
torch's default device to the backend's for every test of a session they
are loaded into, and that includes these tests whenever both directories
run together; a tensor that must be a CPU one says so.
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
    / "luminal_torch_cuda_lite_compiler"
    / "boundary.py"
)
_SPEC = importlib.util.spec_from_file_location("luminal_cuda_lite_boundary", _SOURCE)
boundary = importlib.util.module_from_spec(_SPEC)
# Registered before execution: a dataclass resolves its field types through
# its own module.
sys.modules[_SPEC.name] = boundary
_SPEC.loader.exec_module(boundary)


def _chain(strides):
    """A declared layout at literal element strides, in the wire spelling."""
    return boundary.Strided(tuple(f"Integer({stride})" for stride in strides))


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
        boundary.boundary_layout(
            "x", _Boundary((4, 4), (4, 1), layout=torch.sparse_coo)
        )
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


def test_a_tensor_subclass_is_refused_by_name():
    """A subclass carries semantics of its own — a distributed tensor's
    sizes are the global ones, a masked tensor's elements are not all
    there — so its sizes and strides are not a statement about storage to
    bind. A parameter is plain storage under another class and passes the
    same gate: what refuses it here is its device, not its type."""

    class _Exotic(torch.Tensor):
        pass

    with pytest.raises(boundary.UnsupportedBoundary) as refusal:
        boundary.boundary_layout(
            "x", torch.empty(4, 4, device="cpu").as_subclass(_Exotic)
        )
    message = str(refusal.value)
    assert "x:" in message, message
    assert "_Exotic" in message, message

    with pytest.raises(boundary.UnsupportedBoundary, match="expected a CUDA tensor"):
        boundary.boundary_layout(
            "w", torch.nn.Parameter(torch.empty(4, 4, device="cpu"))
        )


def test_a_parameter_of_a_custom_tensor_type_is_refused_by_name():
    """``Parameter`` of a custom tensor type IS that custom type, flagged:
    ``isinstance(t, torch.nn.Parameter)`` answers yes for any tensor
    carrying ``_is_param`` while its sizes and strides still describe
    something other than the storage a binding would name. The gate is the
    exact type, so the subclass is refused naming itself."""

    class _Exotic(torch.Tensor):
        pass

    exotic = torch.empty(4, 4, device="cpu").as_subclass(_Exotic)
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
