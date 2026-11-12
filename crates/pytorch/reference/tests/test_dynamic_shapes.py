"""End-to-end tests for dynamic-shape support through ``torch.compile``.

These exercise the path that the standard PyTorch user hits — i.e. wrapping a
model with ``torch.compile(model, backend=luminal_backend)`` and calling it
with varying input shapes. The luminal backend is expected to recognise
Dynamo-emitted SymInt placeholders, propagate the symbolic dims through the
PT2 export, and reuse a single compiled graph across shape changes.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import pytest
import torch
import torch._dynamo
from luminal_reference import Compiler

luminal_backend = Compiler()


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


def _compile_with_capture(model, count_holder, capture_holder):
    def wrapper(gm, example_inputs):
        out = luminal_backend(gm, example_inputs)
        count_holder.append(1)
        if "gm" not in capture_holder:
            capture_holder["gm"] = copy.deepcopy(gm).eval()
            capture_holder["example_inputs"] = example_inputs
            capture_holder["compiled_impl"] = out
        return out

    return torch.compile(model, backend=wrapper)


@contextmanager
def _explicit_mark_dynamic_mode():
    prev_auto = torch._dynamo.config.automatic_dynamic_shapes
    prev_cache_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 8
    try:
        yield
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = prev_auto
        torch._dynamo.config.cache_size_limit = prev_cache_limit
        torch._dynamo.reset()


def _first_trace_dynamic_axes(capture_holder):
    """The symbolic axes of each tensor argument of the first graph Dynamo
    traced, read off the graph itself: one set per argument that has any."""
    axes = []
    for node in capture_holder["gm"].graph.nodes:
        value = node.meta.get("example_value")
        if node.op != "placeholder" or not isinstance(value, torch.Tensor):
            continue
        symbolic = {
            axis for axis, size in enumerate(value.shape) if not isinstance(size, int)
        }
        if symbolic:
            axes.append(symbolic)
    return axes


def _assert_single_dynamic_input(dynamic_axes, expected_dims):
    """Exactly one argument is dynamic, on the expected axes."""
    assert dynamic_axes == [set(expected_dims)], (
        f"expected one dynamic input on axes {set(expected_dims)}, got {dynamic_axes}"
    )


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


def test_mark_dynamic_seq_via_torch_compile_starts_dynamic(device: torch.device):
    """Explicit `mark_dynamic` should skip the static-then-promote compile dance."""

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return (x.sin() + x.square()).sum(-1)

    with _explicit_mark_dynamic_mode():
        model = Mdl().eval().to(device)
        counts: list[int] = []
        capture: dict[str, object] = {}
        compiled = _compile_with_capture(model, counts, capture)

        first = torch.randn(2, 4, device=device)
        torch._dynamo.mark_dynamic(first, 1, min=2, max=16)

        inputs = {
            4: first,
            6: torch.randn(2, 6, device=device),
            9: torch.randn(2, 9, device=device),
        }

        for seq_len, x in inputs.items():
            ref = model(x)
            out = compiled(x)
            assert out.shape == ref.shape == (2,), (
                f"seq_len={seq_len}: got {out.shape}, expected {ref.shape}"
            )
            assert torch.allclose(out, ref, atol=1e-5), (
                f"seq_len={seq_len}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )

        _assert_single_dynamic_input(_first_trace_dynamic_axes(capture), {1})

        assert len(counts) == 1, (
            "Explicit mark_dynamic should produce one dynamic backend trace from the start, "
            f"got {len(counts)} backend invocations"
        )


def test_mark_dynamic_seq_with_lifted_weights_single_compile(device: torch.device):
    """Lifted parameters should compose with an explicitly dynamic token axis."""

    class Mdl(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(128, 16)
            self.proj = torch.nn.Linear(16, 8)

        def forward(self, input_ids):
            return self.proj(self.embed(input_ids)).sum(-1)

    with _explicit_mark_dynamic_mode():
        model = Mdl().eval().to(device)
        counts: list[int] = []
        capture: dict[str, object] = {}
        compiled = _compile_with_capture(model, counts, capture)

        first = torch.tensor([[1, 2, 3, 4]], device=device)
        torch._dynamo.mark_dynamic(first, 1, min=2, max=32)

        inputs = {
            4: first,
            6: torch.arange(1, 7, device=device).unsqueeze(0),
            9: torch.arange(1, 10, device=device).unsqueeze(0),
        }

        with torch.no_grad():
            for seq_len, input_ids in inputs.items():
                ref = model(input_ids)
                out = compiled(input_ids)
                assert out.shape == ref.shape == (1, seq_len), (
                    f"seq_len={seq_len}: got {out.shape}, expected {ref.shape}"
                )
                assert torch.allclose(out, ref, atol=1e-5), (
                    "seq_len="
                    f"{seq_len}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
                )

        _assert_single_dynamic_input(_first_trace_dynamic_axes(capture), {1})

        assert len(counts) == 1, (
            "Explicit mark_dynamic should avoid a second compile for lifted-weight models, "
            f"got {len(counts)} backend invocations"
        )


def test_mark_dynamic_seq_preserves_affine_output_shape(device: torch.device):
    """Output-shape expressions like `2 * seq` should stay dynamic from call 1."""

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x, x], dim=1)

    with _explicit_mark_dynamic_mode():
        model = Mdl().eval().to(device)
        counts: list[int] = []
        capture: dict[str, object] = {}
        compiled = _compile_with_capture(model, counts, capture)

        first = torch.randn(2, 4, 3, device=device)
        torch._dynamo.mark_dynamic(first, 1, min=2, max=16)

        inputs = {
            4: first,
            5: torch.randn(2, 5, 3, device=device),
            7: torch.randn(2, 7, 3, device=device),
        }

        for seq_len, x in inputs.items():
            ref = model(x)
            out = compiled(x)
            assert out.shape == ref.shape == (2, 2 * seq_len, 3), (
                f"seq_len={seq_len}: got {out.shape}, expected {ref.shape}"
            )
            assert torch.allclose(out, ref, atol=1e-5), (
                f"seq_len={seq_len}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )

        _assert_single_dynamic_input(_first_trace_dynamic_axes(capture), {1})

        assert len(counts) == 1, (
            "Explicit mark_dynamic should keep affine output-shape models on one compile, "
            f"got {len(counts)} backend invocations"
        )


def test_mark_dynamic_two_dim_via_torch_compile_starts_dynamic(device: torch.device):
    """Marking both batch and seq dynamic should still compile only once."""

    class Mdl(torch.nn.Module):
        def forward(self, x):
            return x.mean(-1)

    with _explicit_mark_dynamic_mode():
        model = Mdl().eval().to(device)
        counts: list[int] = []
        capture: dict[str, object] = {}
        compiled = _compile_with_capture(model, counts, capture)

        first = torch.randn(2, 8, 4, device=device)
        torch._dynamo.mark_dynamic(first, 0, min=1, max=8)
        torch._dynamo.mark_dynamic(first, 1, min=2, max=16)

        inputs = {
            (2, 8): first,
            (3, 9): torch.randn(3, 9, 4, device=device),
            (5, 11): torch.randn(5, 11, 4, device=device),
        }

        for shape, x in inputs.items():
            ref = model(x)
            out = compiled(x)
            assert out.shape == ref.shape == shape, (
                f"shape={shape}: got {out.shape}, expected {ref.shape}"
            )
            assert torch.allclose(out, ref, atol=1e-5), (
                f"shape={shape}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
            )

        _assert_single_dynamic_input(_first_trace_dynamic_axes(capture), {0, 1})

        assert len(counts) == 1, (
            "Explicitly marked batch+seq dims should compile once from the first call, "
            f"got {len(counts)} backend invocations"
        )


@pytest.mark.parametrize("bounded", [False, True])
def test_compiler_explicit_dynamic_float_input(device, bounded):
    from luminal_reference import Compiler

    class Model(torch.nn.Module):
        def forward(self, x):
            return (x * 2.0).sum(-1)

    model = Model().eval().to(device)
    example = torch.randn(4, 8, device=device)
    torch._dynamo.mark_dynamic(example, 0, **({"min": 2, "max": 64} if bounded else {}))
    compiler = Compiler()
    compiled = torch.compile(model, backend=compiler, fullgraph=True)
    torch.testing.assert_close(compiled(example), model(example))
    count = len(compiler.graphs)
    for rows in (3, 5, 16):
        x = torch.randn(rows, 8, device=device)
        torch.testing.assert_close(compiled(x), model(x))
    assert len(compiler.graphs) == count


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
