"""End-to-end tests for dynamic-shape support through ``torch.compile``.

These exercise the path that the standard PyTorch user hits — i.e. wrapping a
model with ``torch.compile(model, backend=luminal_backend)`` and calling it
with varying input shapes. The luminal backend is expected to recognise
Dynamo-emitted SymInt placeholders, propagate the symbolic dims through the
PT2 export, and reuse a single compiled graph across shape changes.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo

from luminal.main import luminal_backend


def _compile(model, count_holder):
    def wrapper(gm, example_inputs):
        out = luminal_backend(gm, example_inputs)
        count_holder.append(1)
        return out

    return torch.compile(model, backend=wrapper)


def _compile_with_dynamic_true(model, count_holder):
    def wrapper(gm, example_inputs):
        out = luminal_backend(gm, example_inputs)
        count_holder.append(1)
        return out

    return torch.compile(model, backend=wrapper, dynamic=True)


@pytest.fixture(autouse=True)
def _enable_automatic_dynamic():
    """Make sure the tests run with Dynamo's automatic-dynamic detection on.

    Other tests in the suite flip this off; reset state between tests so the
    cache that backs the previous suppression doesn't carry over. We also
    raise the recompile limit because Dynamo defaults to 1 (which trips
    before automatic-dynamic kicks in) and have to do an extra reset to
    drop any cached frames from prior tests in the suite.
    """
    torch._dynamo.reset()
    prev_auto = torch._dynamo.config.automatic_dynamic_shapes
    prev_limit = torch._dynamo.config.recompile_limit
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.recompile_limit = 16
    try:
        yield
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = prev_auto
        torch._dynamo.config.recompile_limit = prev_limit
        torch._dynamo.reset()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — the dynamic-shape backend wiring is exercised end to end against the cuda_lite runtime",
)
def test_dynamic_seq_via_torch_compile_reuses_compile(device: torch.device):
    """A varying seq dim should produce two backend invocations total.

    First call: Dynamo emits a static-shape graph (no SymInt placeholders).
    Second call: Dynamo detects the size mismatch and re-traces with the dim
    marked dynamic. From that point on, every subsequent shape variation
    must be served by the same compiled graph — no further backend calls.
    """

    class Mdl(torch.nn.Module):
        def forward(self, x):
            s = x.shape[0]
            return x.reshape(s, -1).sum(-1)

    model = Mdl().to(device)
    counts: list[int] = []
    compiled = _compile(model, counts)

    for shp in [4, 5, 6, 7, 5]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        assert out.shape == ref.shape, (
            f"shape={shp}: got {out.shape} expected {ref.shape}"
        )
        assert torch.allclose(out, ref, atol=1e-5), (
            f"shape={shp}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
        )

    assert len(counts) == 2, (
        f"expected exactly 2 backend invocations (one static, one dynamic), got {len(counts)}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_dynamic_via_torch_compile_with_lifted_weights(device: torch.device):
    """Combines lifted-weight re-internalization with the SymInt strip.

    Most real models hit both paths simultaneously (Dynamo lifts every
    `nn.Parameter` AND emits SymInt placeholders for any dim that varies
    between calls), so the two filters need to compose without losing
    track of input positions.
    """

    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.lin(x).sum(-1)

    model = Mdl().eval().to(device)
    counts: list[int] = []
    compiled = _compile(model, counts)

    for shp in [3, 4, 5, 6, 4]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        assert out.shape == ref.shape, (
            f"shape={shp}: got {out.shape} expected {ref.shape}"
        )
        assert torch.allclose(out, ref, atol=1e-5), (
            f"shape={shp}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
        )

    assert len(counts) == 2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_compound_shape_expression_auto_resolves(device: torch.device):
    """Affine shape expressions (`2*s` etc.) should still let auto-detect work.

    The `auto_set_dims_from_input_shapes` Rust path used to only handle bare
    `Term::Var(c)` shape expressions and silently skip anything else, leaving
    affine dims unresolved on the CompiledGraph and the corresponding output
    sizes stale. We now invert single-variable affine forms `a*x + b` by
    sampling two probe points; this test exercises that path by constructing
    a model whose first axis evolves into `2*s` after a `cat` along it.
    """

    class Mdl(torch.nn.Module):
        def forward(self, x):
            # `cat([x, x], dim=0)` doubles the leading dim — torch.export
            # encodes the resulting shape as `2*s` rather than `s`.
            return torch.cat([x, x], dim=0).sum(-1)

    model = Mdl().to(device)
    counts: list[int] = []
    compiled = _compile(model, counts)

    for shp in [4, 5, 6, 7, 5]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        assert out.shape == ref.shape, (
            f"shape={shp}: got {out.shape} expected {ref.shape}"
        )
        assert torch.allclose(out, ref, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_torch_compile_dynamic_true_single_compile(device: torch.device):
    """`torch.compile(model, backend=luminal_backend, dynamic=True)` works.

    `dynamic=True` skips Dynamo's specialise-then-promote dance and emits a
    fully-symbolic graph from the first call. The luminal backend must
    handle the SymInt placeholders Dynamo passes alongside the tensor
    inputs and reuse a single compiled graph across all shape variations —
    one backend invocation total, in contrast to the 2 we'd see under
    automatic-dynamic mode (which burns a static compile on call 1 before
    promoting to dynamic on call 2).
    """

    class Mdl(torch.nn.Module):
        def forward(self, x):
            s = x.shape[0]
            return x.reshape(s, -1).sum(-1)

    model = Mdl().to(device)
    counts: list[int] = []
    compiled = _compile_with_dynamic_true(model, counts)

    for shp in [4, 5, 6, 7, 5]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)

    assert len(counts) == 1, (
        f"dynamic=True should produce a single backend invocation, got {len(counts)}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_explicit_compile_float_input_dynamic(device: torch.device):
    """`luminal.pt2.compile(model, example, dynamic_dim=...)` with a float input.

    The previous version of `compile()` silently fell back to a static export
    for floating-point inputs (the `"auto"` heuristic was integer-only). The
    new spec accepts an explicit `int` or `Iterable[int]` regardless of dtype,
    and `"auto"` now picks every non-trivial axis.
    """
    from luminal.pt2 import compile as luminal_compile

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return (x * 2.0).sum(-1)

    model = Mdl().eval().to(device)
    example = torch.randn(4, 8, device=device)
    compiled = luminal_compile(model, example, search_iterations=3, dynamic_dim=0)

    assert compiled.has_dynamic_dims, "compile() should have produced a dynamic graph"

    for shp in [4, 5, 6, 7]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        # `compile()` returns a tuple of outputs; extract the first.
        out_t = out[0] if isinstance(out, tuple) else out
        assert out_t.shape == ref.shape, (
            f"shape={shp}: got {out_t.shape}, expected {ref.shape}"
        )
        assert torch.allclose(out_t, ref, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_explicit_compile_dynamic_shapes_passthrough(device: torch.device):
    """`luminal.pt2.compile(... , dynamic_shapes=...)` accepts a full spec.

    Lets the caller specify named `Dim` objects with ranges — the previous
    API hardcoded `Dim("seq", min=2)` for any single dynamic dim.
    """
    from torch.export import Dim
    from luminal.pt2 import compile as luminal_compile

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return x.mean(-1)

    model = Mdl().eval().to(device)
    example = torch.randn(4, 8, device=device)
    seq = Dim("seq_len", min=2, max=64)
    compiled = luminal_compile(
        model, example, search_iterations=3, dynamic_shapes=({0: seq},)
    )
    assert compiled.has_dynamic_dims
    # torch.export rewrites user-supplied Dim names to its internal s77/s33
    # convention before saving — what we actually need to verify is that a
    # symbolic dim was registered, not what label it ended up with.
    assert len(compiled.dim_params) == 1, (
        f"expected exactly one dynamic dim, got {compiled.dim_params}"
    )

    for shp in [3, 5, 16]:
        x = torch.randn(shp, 8, device=device)
        ref = model(x)
        out = compiled(x)
        out_t = out[0] if isinstance(out, tuple) else out
        assert out_t.shape == ref.shape
        assert torch.allclose(out_t, ref, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only — exercises the cuda_lite dynamic-dim runtime",
)
def test_dynamic_two_dim_via_torch_compile(device: torch.device):
    """Both batch and seq dynamic — should still reuse a single compile."""

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return x.sum(-1)

    model = Mdl().to(device)
    counts: list[int] = []
    compiled = _compile(model, counts)

    # Vary batch and seq together so Dynamo marks both as dynamic.
    for batch, seq in [(2, 8), (3, 9), (4, 10), (5, 11), (3, 12)]:
        x = torch.randn(batch, seq, device=device)
        ref = model(x)
        out = compiled(x)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5)

    # Allow at most a small number of compiles — two shape transitions can
    # legitimately take Dynamo two retraces (one per newly-dynamic dim).
    assert len(counts) <= 3, (
        f"expected ≤3 compiles for two-dim dynamic, got {len(counts)}"
    )
