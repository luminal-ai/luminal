"""ShapeVar identity and declared domains through Dynamo, AOT, and native search."""

import pytest
import torch
from luminal_reference import Compiler
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec


@pytest.fixture(autouse=True)
def reset():
    torch._dynamo.reset()
    with torch._dynamo.config.patch(recompile_limit=8):
        yield
    torch._dynamo.reset()


def test_shared_shapevar_varies_sizes_without_recompile():
    s = ShapeVar("s", min=2, max=16, optimization_hint=4)
    compiler = Compiler()

    def model(x, y):
        return y @ x

    compiled = torch.compile(
        model,
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(
            params={"x": TensorSpec([s, s]), "y": TensorSpec([3, s])}
        ),
    )
    for size in (4, 12, 8, 9, 2, 16, 5):
        x, y = torch.randn(size, size), torch.randn(3, size)
        torch.testing.assert_close(compiled(x, y), model(x, y))
    assert len(compiler.graphs) == 1
    assert list(compiler.regions[0].bounds.values()) == [(2, 16)]
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        compiled(torch.randn(4, 5), torch.randn(3, 4))
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        compiled(torch.randn(17, 17), torch.randn(3, 17))


def test_same_name_shapevars_are_independent():
    a = ShapeVar("s", min=2, max=12, optimization_hint=3)
    b = ShapeVar("s", min=2, max=12, optimization_hint=5)
    compiler = Compiler()
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([a, b])}),
    )
    for shape in ((3, 5), (7, 9)):
        x = torch.randn(shape)
        torch.testing.assert_close(compiled(x), x.sin())
    assert len(compiler.graphs) == 1
    assert list(compiler.regions[0].bounds.values()) == [(2, 12), (2, 12)]


def test_default_uses_pytorch_bounds_above_old_limit():
    s = ShapeVar("large", min=2, max=6000, optimization_hint=8)
    compiler = Compiler()
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    for size in (8, 5000):
        x = torch.randn(size)
        torch.testing.assert_close(compiled(x), x.sin())
    assert len(compiler.graphs) == 1
    assert list(compiler.regions[0].bounds.values()) == [(2, 6000)]


def test_mark_dynamic_bounds_are_preserved():
    compiler = Compiler()
    x = torch.randn(4, 3)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=20)
    compiled = torch.compile(lambda x: x.sin(), backend=compiler, fullgraph=True)
    torch.testing.assert_close(compiled(x), x.sin())
    assert list(compiler.regions[0].bounds.values()) == [(2, 20)]


def test_backward_preserves_shapevar_bounds():
    s = ShapeVar("s", min=2, max=12, optimization_hint=4)
    compiler = Compiler()
    compiled = torch.compile(
        lambda x: x.square().sum(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    for size in (4, 9):
        x = torch.randn(size, requires_grad=True)
        compiled(x).backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())
    assert {g.phase for g in compiler.graphs} == {"forward", "backward"}
    assert all(list(r.bounds.values()) == [(2, 12)] for r in compiler.regions)


def test_whisper_shapespec_mask():
    seq = ShapeVar("seq", min=2, max=448, optimization_hint=2)
    compiler = Compiler()

    def model(tokens):
        size = tokens.shape[0]
        return torch.triu(torch.full((size, size), -1e10), diagonal=1)

    compiled = torch.compile(
        model,
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"tokens": TensorSpec([seq])}),
    )
    for size in (2, 3, 33, 7):
        tokens = torch.zeros(size, dtype=torch.int64)
        torch.testing.assert_close(compiled(tokens), model(tokens))
    assert len(compiler.graphs) == 1


def test_shapevar_without_hint_and_zero_one():
    s = ShapeVar("s", min=0, max=8)
    compiler = Compiler()
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    for size in (4, 0, 1, 8):
        x = torch.randn(size)
        torch.testing.assert_close(compiled(x), x.sin())
    assert len(compiler.graphs) == 1


def test_derived_shapevar_preserves_root_bounds():
    s = ShapeVar("s", min=2, max=12, optimization_hint=3)
    compiler = Compiler()

    def model(x, y):
        return y + x.sum()

    compiled = torch.compile(
        model,
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(
            params={"x": TensorSpec([s]), "y": TensorSpec([3 * s])}
        ),
    )
    for size in (3, 9, 5):
        x, y = torch.randn(size), torch.randn(3 * size)
        torch.testing.assert_close(compiled(x, y), model(x, y))
    assert len(compiler.graphs) == 1


def test_application_dispatches_independently_bounded_callables():
    def model(x):
        return x.sin() + x

    entries = []
    for lo, hi, hint, budget in [(2, 4, 3, 1024), (8, 12, 9, 2048)]:
        size = ShapeVar("s", min=lo, max=hi, optimization_hint=hint)
        compiler = Compiler(memory_budget_bytes=budget)
        compiled = torch.compile(
            model,
            backend=compiler,
            fullgraph=True,
            isolate_recompiles=True,
            dynamic_shapes=ShapesSpec(params={"x": TensorSpec([size])}),
        )
        entries.append((lo, hi, compiler, compiled))

    def dispatch(x):
        for lo, hi, _, compiled in entries:
            if lo <= x.numel() <= hi:
                return compiled(x)
        raise ValueError("no application bucket covers input")

    for n in (3, 9, 2, 12, 4, 8, 3):
        x = torch.randn(n)
        torch.testing.assert_close(dispatch(x), model(x))
    for lo, hi, compiler, _ in entries:
        assert len(compiler.graphs) == 1
        assert list(compiler.regions[0].bounds.values()) == [(lo, hi)]
    with pytest.raises(ValueError, match="no application bucket"):
        dispatch(torch.ones(6))


def test_broad_domain_uses_small_profile_and_checks_live_budget():
    size = ShapeVar("s", min=2, max=100000000, optimization_hint=2)
    compiler = Compiler(memory_budget_bytes=1024)
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([size])}),
    )
    x = torch.ones(2)
    torch.testing.assert_close(compiled(x), x.sin())
    assert list(compiler.regions[0].bounds.values()) == [(2, 100000000)]
    with pytest.raises(RuntimeError, match="budget"):
        compiled(torch.ones(400))
    assert len(compiler.graphs) == 1
