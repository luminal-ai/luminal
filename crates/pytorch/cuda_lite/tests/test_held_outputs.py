"""Outputs the caller keeps: every call's output is its own live storage.

An output is allocated at the byte span the installed plan discloses for
it and written in place, so a caller may hold every call's result at once
and read it later. These need the maturin-built extension and a CUDA
device; without either they skip cleanly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_cuda_lite")

import luminal_cuda_lite  # noqa: E402

# Full float32 on both sides (see conftest), over a three-matmul stack.


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """Every test compiles the same module classes; without a reset the
    recompile limit on their shared forward is reached and Dynamo falls back
    to eager without calling the backend."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


ATOL = 1e-3
RTOL = 1e-3


def _linear_stack():
    """Three matmuls: the cuBLASLt path, whose output the plan writes
    through an escape cell of its own."""
    return (
        torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 64),
        )
        .cuda()
        .eval()
    )


def _elementwise():
    """A small pointwise chain: one elected kernel writing the output."""

    class Chain(torch.nn.Module):
        def forward(self, x):
            return (x * 2 + 1) * x - 0.5

    return Chain().cuda().eval()


MODELS = {"linear_stack": _linear_stack, "elementwise": _elementwise}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("build", list(MODELS.values()), ids=list(MODELS))
def test_outputs_held_across_calls_stay_correct(build):
    """Twelve calls, every output kept: each holds its own call's numbers,
    so no call wrote into storage an earlier one still owns."""
    torch.manual_seed(0)
    model = build()
    compiled = torch.compile(model, backend=luminal_cuda_lite.Compiler())
    calls = []
    with torch.no_grad():
        for _ in range(12):
            x = torch.randn(8, 64, device="cuda")
            calls.append((x, compiled(x)))
        pointers = [got.data_ptr() for _, got in calls]
        assert len(set(pointers)) == len(pointers), "two held outputs share an address"
        for x, got in calls:
            torch.testing.assert_close(got, model(x), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("build", list(MODELS.values()), ids=list(MODELS))
def test_rebinding_loop_matches_eager(build):
    """The other half: each call's output is released before the next, so
    the allocator hands the block straight back. Reuse must be as correct
    as holding."""
    torch.manual_seed(0)
    model = build()
    compiled = torch.compile(model, backend=luminal_cuda_lite.Compiler())
    with torch.no_grad():
        for _ in range(12):
            x = torch.randn(8, 64, device="cuda")
            y = compiled(x)
            torch.testing.assert_close(y, model(x), atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_matmul_output_is_written_in_place():
    """One matmul, one call: the returned tensor IS what the plan wrote,
    at eager's layout."""
    torch.manual_seed(0)
    model = torch.nn.Linear(64, 64).cuda().eval()
    compiled = torch.compile(model, backend=luminal_cuda_lite.Compiler())
    x = torch.randn(8, 64, device="cuda")
    with torch.no_grad():
        expected = model(x)
        got = compiled(x)
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_no_reinstantiation_when_only_pointers_move():
    """A held output moves the address the plan writes to, and nothing
    else: the captured graph is updated, never rebuilt."""
    torch.manual_seed(0)
    models = []

    def capture(gm, example_inputs, **kwargs):
        compiled = luminal_cuda_lite(gm, example_inputs, **kwargs)
        models.append(compiled)
        return compiled

    model = _linear_stack()
    compiled = torch.compile(model, backend=capture)
    with torch.no_grad():
        for _ in range(3):
            compiled(torch.randn(8, 64, device="cuda"))
        assert models, "the backend compiled no graph"
        graph = models[-1]._graph
        warm = graph.graph_stats()
        if warm is None:
            pytest.skip("no device counters (the runtime has not run on a device)")
        # Held, so every call writes an address the one before did not.
        held = []
        for _ in range(8):
            held.append(compiled(torch.randn(8, 64, device="cuda")))
        after = graph.graph_stats()
        assert len({tensor.data_ptr() for tensor in held}) == len(held)
    assert after["instantiations"] == warm["instantiations"], (
        f"the plan was re-instantiated: {warm['instantiations']} -> "
        f"{after['instantiations']}"
    )
    assert after["address_rebinds"] > warm["address_rebinds"], (
        "no captured node was re-addressed across eight calls"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_elected_layout_mismatch_is_refused_by_name():
    """Eager lays this output out column-major, so the declared layout is
    column-major too. Whether the plan elects that layout is the search's
    answer, not this test's: either the call returns eager's layout, or it
    is refused by name. What it must never do is hand back a tensor whose
    strides are not the ones declared for it."""

    def fn(a, b):
        return (a @ b).t()

    torch.manual_seed(0)
    a = torch.randn(32, 64, device="cuda")
    b = torch.randn(64, 32, device="cuda")
    expected = fn(a, b)
    try:
        got = torch.compile(fn, backend=luminal_cuda_lite.Compiler())(a, b)
    except RuntimeError as refusal:
        message = str(refusal)
        assert (
            "returns outputs in eager's layout" in message
            or "no plan writes the bound outputs" in message
        ), message
    else:
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)
