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
    be dense: a non-row-major target is refused by name rather than written as
    though it were dense."""

    def fn(x):
        x.add_(1)
        return x * 2

    torch.manual_seed(0)
    x = torch.randn(8, 4, device="cuda").t()
    with pytest.raises(Exception, match="must be row-major"):
        torch.compile(fn, backend=luminal_cuda_lite)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_two_inputs_on_one_storage_are_refused():
    """Two boundary tensors on one storage are one buffer carrying two
    declarations; that statement is not made yet, so it is refused by name."""

    def fn(a, b):
        return a * b

    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda")
    with pytest.raises(Exception, match="share device storage"):
        torch.compile(fn, backend=luminal_cuda_lite)(x, x)
