"""Smoke tests for the CUDA-lite torch.compile backend.

These need the maturin-built extension and a CUDA device. On a host without a
GPU (or without the extension built) they skip cleanly rather than fail: the
runtime itself builds device-free and is covered by its Rust suites, so a
CUDA-free CI job should not go red over a missing device.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

import luminal_cuda_lite  # noqa: E402


def test_backend_registered():
    assert "luminal_cuda_lite" in torch._dynamo.list_backends()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_linear_roundtrip_repeated_calls():
    """Fresh inputs each call: the per-execution arena is alloc/free'd each time
    and outputs are separate tensors, so every call must still be exact."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(4, 16, device="cuda")
            torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_dynamic_batch():
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite, dynamic=True)
    with torch.no_grad():
        for n in (1, 3, 7, 2):
            x = torch.randn(n, 16, device="cuda")
            torch.testing.assert_close(compiled(x), model(x), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    "activation",
    [torch.nn.ReLU(), torch.nn.GELU()],
    ids=["relu", "gelu"],
)
def test_select_backed_activations(activation):
    """ReLU/GELU lower through the native ternary select op; before it existed
    their graphs dead-ended extraction and the backend refused to compile."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), activation, torch.nn.Linear(32, 8)
    ).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        x = torch.randn(4, 16, device="cuda")
        torch.testing.assert_close(compiled(x), model(x), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (torch.float16, 1e-2),
        (torch.bfloat16, 1e-1),
        (torch.float64, 1e-6),
    ],
    ids=["f16", "bf16", "f64"],
)
def test_half_and_double_dtypes(dtype, atol):
    torch.manual_seed(0)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8)
        )
        .to("cuda", dtype)
        .eval()
    )
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    with torch.no_grad():
        x = torch.randn(4, 16, device="cuda", dtype=dtype)
        got = compiled(x)
        assert got.dtype == dtype
        torch.testing.assert_close(got, model(x), atol=atol, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_repeated_calls_reuse_input_tensors():
    """The same input tensor every call: its buffer is re-addressed on each
    execute, so a stale pointer or a dropped binding shows up as a wrong
    result."""
    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite)
    x = torch.randn(4, 16, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            torch.testing.assert_close(compiled(x), model(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_after_read():
    """WAR: the returned value reads the input's pre-mutation contents, and
    the writeback lands in the caller's own tensor (one buffer, one pointer)."""

    def fn(x):
        y = x * 2
        x.add_(1)
        return y

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    expected_y = x * 2
    expected_x = x + 1
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected_y)
    torch.testing.assert_close(x, expected_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_consumed_downstream():
    """The mutated input is read again after the writeback: the sink and the
    reader are the same storage."""

    def fn(x):
        x.add_(1)
        return x * 2

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    expected_x = x + 1
    expected = expected_x * 2
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(x, expected_x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_transposed_input_binds_zero_copy():
    """A transposed input is bound in the layout it has (column-major), never
    copied and never reinterpreted."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda").t()
    assert not x.is_contiguous()
    expected = fn(x)
    got = torch.compile(fn, backend=luminal_cuda_lite)(x)
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_writeback_into_transposed_input_is_refused():
    """A writeback writes its target's storage, and a kernel destination must
    be row-major (the only destination layout the kernels write): any other
    target is refused by name rather than written as though it were."""

    def fn(x):
        x.add_(1)
        return x * 2

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda").t()
    with pytest.raises(Exception, match="must be row-major"):
        torch.compile(fn, backend=luminal_cuda_lite)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_read_only_aliasing_binds_two_buffers_on_one_address():
    """Two boundary tensors on one storage, neither written through, are two
    External buffers carrying one address: a fact about the caller's memory,
    not a hazard. ``x`` and ``x[:]`` are distinct objects, so Dynamo hands the
    backend two placeholders rather than deduplicating them."""

    def fn(a, b):
        return a * b

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    torch.testing.assert_close(
        torch.compile(fn, backend=luminal_cuda_lite)(x, x[:]), x * x
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_aliased_inputs_with_a_writeback_are_refused():
    """The hazardous variant: one binding of the overlapping pair is written
    back into, and one buffer per binding cannot order that write against the
    other's reads. Refused by name."""

    def fn(a, b):
        a.add_(1)
        return a * b

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(Exception, match="share device storage"):
        torch.compile(fn, backend=luminal_cuda_lite)(x, x[:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_int32_and_bool_inputs_bind_external():
    """Integer and boolean boundary tensors cross as the caller's own device
    bytes like any other: no host staging, no reinterpretation. A torch
    boolean is already the byte Bool8 reads."""

    def fn(values, mask):
        return torch.where(mask, values * 2, values)

    values = torch.arange(32, device="cuda", dtype=torch.int32).reshape(4, 8)
    mask = (torch.arange(32, device="cuda") % 2 == 0).reshape(4, 8)
    expected = fn(values, mask)
    got = torch.compile(fn, backend=luminal_cuda_lite)(values, mask)
    assert got.dtype == torch.int32
    torch.testing.assert_close(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_transposed_dynamic_batch_input():
    """The transposed input's element stride IS the dynamic batch dimension,
    so the binding states that dimension rather than the number one example
    call had: one compile serves every extent in the bucket."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    compiled = torch.compile(fn, backend=luminal_cuda_lite, dynamic=True)
    for n in (3, 7, 2):
        x = torch.randn(16, n, device="cuda").t()
        assert not x.is_contiguous()
        torch.testing.assert_close(compiled(x), fn(x), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_strided_dynamic_view_states_its_symbolic_strides():
    """Neither row- nor column-major, and strided by a symbol: the binding
    spells the stride as the program's own expression (``2*n``), which is
    what makes one searched plan serve every extent."""

    def fn(x):
        return x * 2 + 1

    torch.manual_seed(0)
    compiled = torch.compile(fn, backend=luminal_cuda_lite, dynamic=True)
    for n in (6, 10, 4):
        x = torch.randn(16, n, device="cuda").t()[:, ::2]
        assert x.stride() == (1, 2 * n)
        torch.testing.assert_close(compiled(x), fn(x), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_offset_input_binds_at_its_own_data_ptr():
    """A slice's storage offset is already inside ``data_ptr()``. The binding
    starts there and states the layout relative to it, so the base tensor is
    never needed and nothing is repacked."""

    def fn(x):
        return x * 2

    torch.manual_seed(0)
    base = torch.randn(5, 8, device="cuda")
    x = base[1:]
    assert x.storage_offset() == 8
    torch.testing.assert_close(torch.compile(fn, backend=luminal_cuda_lite)(x), fn(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_expanded_input_is_a_broadcast_read_map():
    """A zero stride is a legitimate READ map: every coordinate of the
    broadcast axis reads one element. The expanded input binds as it is."""

    def fn(x, y):
        return x + y

    torch.manual_seed(0)
    x = torch.randn(1, 8, device="cuda").expand(4, 8)
    assert x.stride() == (0, 1)
    y = torch.randn(4, 8, device="cuda")
    torch.testing.assert_close(
        torch.compile(fn, backend=luminal_cuda_lite)(x, y), fn(x, y)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_stride_zero_mutation_target_is_refused_by_name():
    """The same broadcast view as a WRITE target: every coordinate of the
    zero-stride axis would land on one element. Recognized as the strided
    read map it is, and refused by name as a writeback destination."""
    from luminal_cuda_lite.boundary import Strided, boundary_layout

    def fn(t):
        t.add_(1)
        return t * 2

    torch.manual_seed(0)
    x = torch.randn(1, 8, device="cuda").expand(4, 8)
    assert boundary_layout("x", x) == Strided(("Integer(0)", "Integer(1)"))
    with pytest.raises(Exception, match="must be row-major"):
        torch.compile(fn, backend=luminal_cuda_lite)(x)
