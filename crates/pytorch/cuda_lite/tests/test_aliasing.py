"""Mutation and aliasing: the compiled program does to the caller's storage
exactly what eager does.

Every test states facts about what the caller can observe — values, which
tensors share storage (``data_ptr``), strides, what a later in-place write
through a returned tensor reaches — never how the plan was built. These
need the maturin-built extension and a CUDA device; without either they
skip cleanly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

import luminal_cuda_lite  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _x():
    return torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)


def _same_as_eager(fn, *make_inputs):
    """Run ``fn`` eagerly and compiled on fresh inputs; the returns, the
    inputs' final contents and the returns' strides must agree. Hands back
    the compiled call's returns and inputs for storage-identity facts."""
    eager_inputs = [make() for make in make_inputs]
    compiled_inputs = [make() for make in make_inputs]
    eager = fn(*eager_inputs)
    got = torch.compile(fn, backend=luminal_cuda_lite.Compiler())(*compiled_inputs)
    eager_list = list(eager) if isinstance(eager, (tuple, list)) else [eager]
    got_list = list(got) if isinstance(got, (tuple, list)) else [got]
    assert len(got_list) == len(eager_list)
    for g, e in zip(got_list, eager_list):
        torch.testing.assert_close(g, e)
        assert g.stride() == e.stride()
    for c, e in zip(compiled_inputs, eager_inputs):
        torch.testing.assert_close(
            c, e, msg="the caller's tensor holds what eager leaves in it"
        )
    return got_list, compiled_inputs


@cuda
def test_mutation_through_a_view_reaches_the_callers_storage():
    def fn(x):
        x[0].add_(1)
        return x

    (out,), (x,) = _same_as_eager(fn, _x)
    assert out.data_ptr() == x.data_ptr(), (
        "returning the mutated input returns its storage"
    )


@cuda
def test_chained_and_overlapping_slice_mutations():
    def fn(x):
        x[0:1].add_(1)
        x[:, 0].mul_(2)
        x[1].sub_(3)
        return x

    _same_as_eager(fn, _x)


@cuda
def test_mutation_through_a_reshape_view():
    def fn(x):
        x.view(-1).add_(1)
        return x

    _same_as_eager(fn, _x)


@cuda
def test_returned_mutated_input_is_the_callers_tensor_every_time_it_is_returned():
    def fn(x):
        x.add_(1)
        return x, x

    (a, b), (x,) = _same_as_eager(fn, _x)
    assert a.data_ptr() == x.data_ptr() and b.data_ptr() == x.data_ptr()


@cuda
def test_two_names_for_one_value_are_two_storages():
    def fn(x):
        y = x * 2
        return y, y.clone()

    (a, b), _ = _same_as_eager(fn, _x)
    assert a.data_ptr() != b.data_ptr()
    b.zero_()
    torch.testing.assert_close(a, _x() * 2, msg="the clone is not the value's storage")


@cuda
def test_returned_view_of_an_output_shares_its_storage():
    def fn(x):
        y = x * 2
        return y, y.view(-1)

    (a, b), _ = _same_as_eager(fn, _x)
    assert b.data_ptr() == a.data_ptr()
    b.zero_()
    assert a.abs().sum().item() == 0.0, "writing the view writes the output"


@cuda
def test_returned_view_of_an_input_is_a_view_of_the_callers_tensor():
    def fn(x):
        return x[0]

    (row,), (x,) = _same_as_eager(fn, _x)
    assert row.data_ptr() == x.data_ptr()
    row.zero_()
    assert x[0].abs().sum().item() == 0.0 and x[1].abs().sum().item() != 0.0


@cuda
def test_returned_view_of_a_writeback_aliases_the_input():
    def fn(x):
        x.add_(1)
        return x.t()

    (out,), (x,) = _same_as_eager(fn, _x)
    assert out.data_ptr() == x.data_ptr()
    out.mul_(2)
    torch.testing.assert_close(x, (_x() + 1) * 2)


@cuda
def test_view_taken_before_a_mutation_sees_the_mutation():
    def fn(x):
        v = x[0]
        x.add_(100)
        return v, x

    (v, out), (x,) = _same_as_eager(fn, _x)
    assert v.data_ptr() == x.data_ptr() and out.data_ptr() == x.data_ptr()


@cuda
def test_clone_of_a_view_is_fresh_storage():
    def fn(x):
        return x[0].clone()

    (out,), (x,) = _same_as_eager(fn, _x)
    assert out.data_ptr() != x.data_ptr()
    out.zero_()
    torch.testing.assert_close(x, _x(), msg="the clone is not the input's storage")


@cuda
def test_detach_of_a_mutated_input_shares_its_storage():
    def fn(x):
        x.add_(1)
        return x.detach()

    (out,), (x,) = _same_as_eager(fn, _x)
    assert out.data_ptr() == x.data_ptr()


@cuda
def test_module_buffer_slice_write():
    class Cache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("kv", torch.zeros(4, 4, device="cuda"))

        def forward(self, x):
            self.kv[1:3].copy_(x)
            return self.kv.sum() + x

    eager_model, model = Cache(), Cache()
    eager = eager_model(_x())
    got = torch.compile(model, backend=luminal_cuda_lite.Compiler())(_x())
    torch.testing.assert_close(got, eager)
    torch.testing.assert_close(
        model.kv, eager_model.kv, msg="the buffer's rows 1:3 hold x"
    )


@cuda
def test_writes_through_storage_sharing_inputs_match_eager():
    """Two inputs on one storage, one written back into. AOTAutograd
    resolves the aliasing before the backend sees the graph: the shared
    storage is ONE input, both views are read off it, and the write lands
    once, so the returned view shows the mutation exactly as eager's does."""

    def fn(a, b):
        a.add_(1)
        return b

    eager_x, x = _x(), _x()
    want = fn(eager_x, eager_x[0])
    got = torch.compile(fn, backend=luminal_cuda_lite.Compiler())(x, x[0])
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(x, eager_x)
    assert got.data_ptr() == x.data_ptr(), (
        "the returned row is a view of the caller's tensor"
    )


@cuda
def test_returned_view_at_a_storage_offset_is_refused_by_name():
    """Until the boundary can state a base offset (LUM-852) this is a named
    refusal, never a tensor holding the wrong row."""

    def fn(x):
        return x[1]

    with pytest.raises(Exception, match="storage offset"):
        torch.compile(fn, backend=luminal_cuda_lite.Compiler())(_x())
