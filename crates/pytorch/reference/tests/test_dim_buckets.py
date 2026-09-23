"""ShapeVar identity and bucket dispatch through Dynamo, AOT, and native search."""

import pytest
import torch
from luminal_reference import Compiler, DimBucket
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec


@pytest.fixture(autouse=True)
def reset():
    torch._dynamo.reset()
    with torch._dynamo.config.patch(recompile_limit=8):
        yield
    torch._dynamo.reset()


def test_bucket_default_and_validation():
    assert DimBucket(min=2, max=16).representative == 9
    for kwargs in (
        {"min": -1, "max": 2},
        {"min": 3, "max": 2},
        {"min": 2, "max": 8, "representative": 9},
    ):
        with pytest.raises(ValueError):
            DimBucket(**kwargs)
    s = ShapeVar("s", min=2, max=16)
    with pytest.raises(TypeError, match="ShapeVar"):
        Compiler(dim_buckets={"s": [DimBucket(min=2, max=8)]})
    with pytest.raises(ValueError, match="non-overlapping"):
        Compiler(dim_buckets={s: [DimBucket(min=2, max=8), DimBucket(min=8, max=16)]})
    with pytest.raises(ValueError, match="outside"):
        Compiler(dim_buckets={s: [DimBucket(min=2, max=17)]})


def test_shared_shapevar_switches_buckets_without_recompile():
    s = ShapeVar("s", min=2, max=16, optimization_hint=4)
    buckets = [DimBucket(min=2, max=8), DimBucket(min=9, max=16, representative=12)]
    compiler = Compiler(dim_buckets={s: buckets})

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
    assert list(compiler.regions[0].buckets.values()) == [
        [b.as_tuple() for b in buckets]
    ]
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        compiled(torch.randn(4, 5), torch.randn(3, 4))
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        compiled(torch.randn(17, 17), torch.randn(3, 17))


def test_same_name_shapevars_are_independent():
    a = ShapeVar("s", min=2, max=12, optimization_hint=3)
    b = ShapeVar("s", min=2, max=12, optimization_hint=5)
    compiler = Compiler(
        dim_buckets={
            a: [DimBucket(min=2, max=12, representative=3)],
            b: [DimBucket(min=2, max=6), DimBucket(min=7, max=12)],
        }
    )
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
    assert sorted(len(v) for v in compiler.regions[0].buckets.values()) == [1, 2]


def test_bucket_shapevar_identity_must_match_spec():
    s = ShapeVar("s", min=2, max=16)
    other = ShapeVar("s", min=2, max=16)
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=Compiler(dim_buckets={s: [DimBucket(min=2, max=16)]}),
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([other])}),
    )
    with pytest.raises(RuntimeError, match="not present"):
        compiled(torch.randn(4))


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
    assert list(compiler.regions[0].buckets.values()) == [[(2, 6000, 8)]]


def test_mark_dynamic_bounds_are_preserved():
    compiler = Compiler()
    x = torch.randn(4, 3)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=20)
    compiled = torch.compile(lambda x: x.sin(), backend=compiler, fullgraph=True)
    torch.testing.assert_close(compiled(x), x.sin())
    assert list(compiler.regions[0].buckets.values()) == [[(2, 20, 4)]]


def test_backward_preserves_shapevar_buckets():
    s = ShapeVar("s", min=2, max=12, optimization_hint=4)
    compiler = Compiler(
        dim_buckets={s: [DimBucket(min=2, max=6), DimBucket(min=7, max=12)]}
    )
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
    assert all(len(next(iter(r.buckets.values()))) == 2 for r in compiler.regions)


def test_whisper_shapespec_mask():
    seq = ShapeVar("seq", min=2, max=448, optimization_hint=2)
    compiler = Compiler(
        dim_buckets={
            seq: [
                DimBucket(min=2, max=32, representative=2),
                DimBucket(min=33, max=448, representative=64),
            ]
        }
    )

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


def test_gap_rejected_before_execution():
    s = ShapeVar("s", min=2, max=12, optimization_hint=3)
    compiler = Compiler(
        dim_buckets={s: [DimBucket(min=2, max=4), DimBucket(min=8, max=12)]}
    )
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    compiled(torch.ones(3))
    with pytest.raises((ValueError, RuntimeError), match="bucket"):
        compiled(torch.ones(6))


def test_bucket_profile_memory_checked_before_allocation():
    s = ShapeVar("s", min=2, max=100000000, optimization_hint=2)
    compiler = Compiler(
        memory_budget_bytes=1024, dim_buckets={s: [DimBucket(min=2, max=100000000)]}
    )
    compiled = torch.compile(
        lambda x: x.sin(),
        backend=compiler,
        fullgraph=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    with pytest.raises(RuntimeError, match="bucket profiling inputs.*memory budget"):
        compiled(torch.ones(2))


def test_shapevar_without_hint_and_zero_one():
    s = ShapeVar("s", min=0, max=8)
    compiler = Compiler(dim_buckets={s: [DimBucket(min=0, max=8)]})
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


def test_derived_shapevar_preserves_root_buckets():
    s = ShapeVar("s", min=2, max=12, optimization_hint=3)
    compiler = Compiler(
        dim_buckets={s: [DimBucket(min=2, max=6), DimBucket(min=7, max=12)]}
    )

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
