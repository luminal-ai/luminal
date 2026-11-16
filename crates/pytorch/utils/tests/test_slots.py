"""The shared slot readers and writers (`luminal_torch_slots`).

Every backend reads torch's arguments and creates its outputs through these
Rust functions, which call torch's stable C entry points. The tests reach
them through the `_slots` hooks each backend's extension exposes, on the
selected backend's device.
"""

import importlib

import pytest
import torch
from backend_test_utils import selected_backend


@pytest.fixture
def slots():
    return importlib.import_module(
        f"luminal_torch_{selected_backend()}_compiler"
    )._luminal._slots


def _index(device):
    """The device index as torch's C side states it: the ordinal, or -1 for
    a device kind that has none."""
    if device.type == "cuda":
        return device.index if device.index is not None else torch.cuda.current_device()
    return -1


def _as_torch_states_it(tensor, device):
    return (
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        str(tensor.dtype).removeprefix("torch."),
        device.type,
        _index(device),
        tensor.storage_offset(),
    )


_TENSORS = {
    "contiguous": lambda: torch.arange(24, dtype=torch.float32).reshape(4, 6),
    "transposed": lambda: torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
    "offset_view": lambda: torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:, 2:],
    "every_other": lambda: torch.arange(24, dtype=torch.int64)[::2],
    "expanded": lambda: torch.ones(1, 3).expand(5, 3),
    "requires_grad": lambda: torch.ones(2, 3, requires_grad=True),
    "parameter": lambda: torch.nn.Parameter(torch.ones(2, 3)),
    "zero_dim": lambda: torch.tensor(3.5),
}


@pytest.mark.parametrize("case", sorted(_TENSORS))
def test_a_tensor_slot_reads_as_torch_states_it(slots, device, case):
    tensor = _TENSORS[case]()
    assert tuple(slots.read_tensor(tensor)) == _as_torch_states_it(tensor, device)


def test_a_tensor_with_no_elements_has_no_address(slots, device):
    tensor = torch.empty(0, 4)
    data, shape, strides, *_ = slots.read_tensor(tensor)
    assert (data, shape, strides) == (0, [0, 4], [4, 1])


def test_every_dtype_torch_can_build_reads_back_by_name(slots):
    read = []
    for name in slots.dtype_names():
        dtype = getattr(torch, name, None)
        if dtype is None:
            continue
        try:
            tensor = torch.empty(2, dtype=dtype)
        except (RuntimeError, TypeError):
            continue  # torch cannot build this dtype on this device
        assert slots.read_tensor(tensor)[3] == name
        read.append(name)
    # The ones every backend relies on must have been exercised.
    assert {
        "bool",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    } <= set(read)


@pytest.mark.parametrize("value", [5, 2.5, None, "x", [1, 2]])
def test_a_value_that_is_not_a_tensor_is_refused_by_type(slots, value):
    with pytest.raises(TypeError, match=type(value).__name__):
        slots.read_tensor(value)


def test_a_sparse_tensor_is_refused_as_not_strided(slots):
    with pytest.raises(TypeError, match="not strided"):
        slots.read_tensor(torch.eye(3).to_sparse())


def test_scalar_slots_accept_exactly_their_own_type(slots):
    assert slots.read_int(7) == 7
    assert slots.read_int(-(2**40)) == -(2**40)
    assert slots.read_float(1.5) == 1.5
    assert slots.read_bool(True) is True

    # A bool is an int to Python; a slot's kind was fixed at compile time.
    for reader, wrong in [
        (slots.read_int, True),
        (slots.read_int, 2.0),
        (slots.read_int, "3"),
        (slots.read_float, 1),
        (slots.read_float, True),
        (slots.read_bool, 1),
        (slots.read_bool, 0.0),
    ]:
        with pytest.raises(TypeError, match=type(wrong).__name__):
            reader(wrong)
    with pytest.raises(OverflowError):
        slots.read_int(2**70)


@pytest.mark.parametrize(
    "shape, strides",
    [
        ([3, 4], [4, 1]),  # row-major
        ([3, 4], [1, 3]),  # column-major
        ([2, 3, 4], [1, 8, 2]),  # a permuted layout
        ([], []),  # zero-dim
        ([0, 4], [4, 1]),  # no elements
    ],
)
def test_an_output_is_allocated_by_torch_at_the_strides_asked_for(
    slots, device, shape, strides
):
    before = torch.cuda.memory_allocated() if device.type == "cuda" else None
    out = slots.empty_strided(shape, strides, "float32", device.type, _index(device))

    assert type(out) is torch.Tensor
    assert list(out.shape) == shape
    assert list(out.stride()) == strides
    assert out.dtype == torch.float32
    assert out.device.type == device.type
    assert not out.requires_grad
    # It is ordinary torch memory: writable, and torch accounts for it.
    out.fill_(2.0)
    assert out.sum().item() == 2.0 * out.numel()
    if before is not None and out.numel():
        assert torch.cuda.memory_allocated() > before


def test_an_output_is_allocated_in_the_dtype_asked_for(slots, device):
    for name in ["bool", "int64", "float16", "bfloat16", "float64"]:
        out = slots.empty_strided([2, 2], [2, 1], name, device.type, _index(device))
        assert out.dtype == getattr(torch, name)


def test_an_output_whose_rank_disagrees_with_its_strides_is_refused(slots, device):
    with pytest.raises(ValueError, match="rank 2 was given 1 strides"):
        slots.empty_strided([2, 3], [1], "float32", device.type, _index(device))
    with pytest.raises(ValueError, match="not a torch dtype name"):
        slots.empty_strided([2], [1], "float33", device.type, _index(device))


def test_a_view_shares_its_bases_storage_at_an_absolute_offset(slots):
    storage = torch.arange(12, dtype=torch.float32)
    base = storage[2:]  # its own storage offset is 2

    view = slots.view_of(base, [2, 3], [3, 1], 5)

    # The offset counts from the start of the storage, as `as_strided`'s does.
    assert view.storage_offset() == 5
    assert torch.equal(view, base.as_strided((2, 3), (3, 1), 5))
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    view[0, 0] = -1.0
    assert storage[5].item() == -1.0


def test_a_view_of_a_tensor_that_requires_grad_can_be_made(slots):
    base = torch.ones(2, 3, requires_grad=True)
    with torch.no_grad():
        view = slots.view_of(base, [3, 2], [1, 3], 0)
    assert list(view.shape) == [3, 2]
    assert view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


def test_the_slot_kinds_are_torchs_value_kinds(slots):
    assert slots.slot_kinds() == ["tensor", "int", "float", "bool", "object"]


def test_the_current_cuda_stream_is_torchs(slots, device):
    if device.type != "cuda":
        pytest.skip("the stream reader is exercised on a CUDA device")
    index = _index(device)
    assert slots.current_cuda_stream(index) == torch.cuda.current_stream().cuda_stream
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        assert slots.current_cuda_stream(index) == side.cuda_stream
