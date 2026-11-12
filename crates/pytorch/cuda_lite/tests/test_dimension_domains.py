"""ShapeVar identity and declared domains through Dynamo, AOT, and CUDA search."""

import pytest
import torch
from luminal_cuda_lite import Compiler
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@cuda
@pytest.mark.parametrize("upper", [12, 6000])
def test_compile_preserves_shapespec_bounds_and_shared_symbols(upper):
    """One root symbol shared by two inputs, one spelled as ``3 * size``:
    the compiled program carries the declared domain, not the hint, and
    serves every extent inside it from one compile."""
    compiler = Compiler()

    def model(x, y):
        return x + y.sum()

    size = ShapeVar("size", min=2, max=upper, optimization_hint=4)
    compiled = torch.compile(
        model,
        backend=compiler,
        fullgraph=True,
        isolate_recompiles=True,
        dynamic_shapes=ShapesSpec(
            params={"x": TensorSpec([size]), "y": TensorSpec([3 * size])}
        ),
    )
    for n in (4, 9):
        x, y = torch.randn(n, device="cuda"), torch.randn(3 * n, device="cuda")
        torch.testing.assert_close(compiled(x, y), model(x, y))
    assert len(compiler.graphs) == 1
    assert list(compiler.graphs[0].bounds.values()) == [(2, upper)]


@cuda
def test_mark_dynamic_bounds_are_preserved():
    compiler = Compiler()
    x = torch.randn(4, 3, device="cuda")
    torch._dynamo.mark_dynamic(x, 0, min=2, max=20)
    compiled = torch.compile(lambda x: x.sin(), backend=compiler, fullgraph=True)
    torch.testing.assert_close(compiled(x), x.sin())
    assert list(compiler.graphs[0].bounds.values()) == [(2, 20)]
