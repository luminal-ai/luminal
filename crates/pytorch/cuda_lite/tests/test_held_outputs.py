"""The memory of a CUDA call: its scratch memory, and the layout its outputs
are written at.

That every call's output is its own live storage, held or released, is a
fact of every backend and is in ``utils/tests/test_held_outputs.py``. These
need the maturin-built extension and a CUDA device; without either they
skip cleanly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("luminal_torch_cuda_lite_compiler")

import luminal_torch_cuda_lite_compiler  # noqa: E402

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_a_call_holds_no_scratch_memory_after_it_returns():
    """The scratch memory a program uses is a block from torch's allocator
    that the call takes and gives back before it returns. torch accounts for
    it while a call runs, none of it is allocated once the calls are over
    however many there were, and torch can hand its cached block back to the
    device, which it could not do if the executable still held it.

    Held outputs move the addresses the program writes to on every call,
    and that changes nothing."""
    from luminal_torch_cuda_lite_runtime import describe

    torch.manual_seed(0)
    compiler = luminal_torch_cuda_lite_compiler.Compiler()
    model = _linear_stack()
    compiled = torch.compile(model, backend=compiler)
    with torch.no_grad():
        for _ in range(3):
            compiled(torch.randn(8, 64, device="cuda"))
        assert compiler.graphs, "the backend compiled no graph"
        program = compiler.graphs[-1].program
        arena = describe(program)["program"]["arena_bytes"]
        assert arena >= 1 << 20, "the program needs too little scratch to measure"

        x = torch.randn(8, 64, device="cuda")
        expected = model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        compiled_graphs = len(compiler.graphs)
        # Held, so every call writes an address the one before did not.
        held = [compiled(x) for _ in range(8)]
        assert len({tensor.data_ptr() for tensor in held}) == len(held)
        outputs = sum(tensor.untyped_storage().nbytes() for tensor in held)
        torch.cuda.synchronize()
        allocated_after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        reserved_cached = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        reserved_released = torch.cuda.memory_reserved()

    assert len(compiler.graphs) == compiled_graphs, "a call compiled a new graph"
    for tensor in held:
        torch.testing.assert_close(tensor, expected, atol=ATOL, rtol=RTOL)
    # While a call ran, torch counted its scratch memory.
    assert peak - allocated_before >= arena, (
        f"torch saw at most {peak - allocated_before} bytes allocated during a call "
        f"whose scratch arena is {arena} bytes"
    )
    # What is allocated now that was not before is the held outputs, in
    # whatever blocks torch gave them. A retained arena would add `arena`.
    grown = allocated_after - allocated_before
    assert outputs <= grown < arena, (
        f"{grown} more bytes are allocated after 8 calls that hold {outputs} bytes "
        f"of output; the program's scratch arena is {arena} bytes and must not be kept"
    )
    # The block sat in torch's cache, free: emptying the cache released it.
    assert reserved_cached - reserved_released >= arena, (
        f"emptying torch's cache released {reserved_cached - reserved_released} "
        f"bytes; the scratch arena is {arena} bytes and something still holds it"
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
        got = torch.compile(fn, backend=luminal_torch_cuda_lite_compiler.Compiler())(
            a, b
        )
    except RuntimeError as refusal:
        message = str(refusal)
        assert (
            "returns outputs in eager's layout" in message
            or "no plan writes the bound outputs" in message
        ), message
    else:
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got, expected, atol=ATOL, rtol=RTOL)
