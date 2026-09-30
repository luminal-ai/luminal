"""CUDA export preserves declared domains before any device compilation."""

import pytest
import torch
from luminal_cuda_lite.backend import _dynamic_export
from luminal_reference.dimensions import profile_value
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("upper", [12, 6000])
def test_export_preserves_shapespec_bounds_and_shared_symbols(upper):
    exported = []

    def backend(gm, inputs):
        ep, tensors = _dynamic_export(gm, inputs)
        exported.append(ep)
        # The CUDA bridge needs concrete profiling shapes without pinning the
        # unbacked ShapesSpec symbols to their optimization hints.
        assert sorted(tuple(profile_value(d) for d in t.shape) for t in tensors) == [
            (4,),
            (12,),
        ]
        return gm.forward

    def model(x, y):
        return x + y.sum()

    size = ShapeVar("size", min=2, max=upper, optimization_hint=4)
    compiled = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
        isolate_recompiles=True,
        dynamic_shapes=ShapesSpec(
            params={"x": TensorSpec([size]), "y": TensorSpec([3 * size])}
        ),
    )
    for n in (4, 9):
        x, y = torch.randn(n), torch.randn(3 * n)
        torch.testing.assert_close(compiled(x, y), model(x, y))
    assert len(exported) == 1
    ep = exported[0]
    ranges = ep.range_constraints
    roots = [s for s in ranges if s.is_Symbol]
    assert len(roots) == 1
    root = roots[0]
    assert (int(ranges[root].lower), int(ranges[root].upper)) == (2, upper)
    assert 3 * root in ranges


def test_export_preserves_mark_dynamic_bounds():
    exported = []

    def backend(gm, inputs):
        ep, _ = _dynamic_export(gm, inputs)
        exported.append(ep)
        return gm.forward

    x = torch.randn(4, 3)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=20)
    compiled = torch.compile(lambda x: x.sin(), backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(x), x.sin())
    assert [
        (int(r.lower), int(r.upper)) for r in exported[0].range_constraints.values()
    ] == [(2, 20)]
