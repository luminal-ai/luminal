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

