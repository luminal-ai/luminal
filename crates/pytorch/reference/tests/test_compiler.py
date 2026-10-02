"""Public compiler configuration reaches AOT and the native reference search."""

import pytest
import torch
from luminal_reference import Compiler
from luminal_reference.artifact_cache import clear_artifact_cache


class Model(torch.nn.Module):
    def forward(self, x):
        return x.sin() + x


@pytest.fixture(autouse=True)
def isolated_compiler(monkeypatch):
    monkeypatch.delenv("SEARCH_LOG", raising=False)
    monkeypatch.delenv("LUMINAL_LOG", raising=False)
    torch._dynamo.reset()
    clear_artifact_cache()
    yield
    torch._dynamo.reset()
    clear_artifact_cache()


@pytest.mark.parametrize("logging", [None, False, True])
def test_compiler_logging_and_aot(logging, capfd):
    options = {} if logging is None else {"log": logging}
    compiler = Compiler(**options)
    model = Model().eval()
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    compiled = torch.compile(model, backend=compiler, fullgraph=True)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    assert compiler.graphs
    assert compiler.regions
    assert ("Start" in capfd.readouterr().err) is (logging is True)


def test_only_compiler_is_public():
    import luminal_reference

    assert luminal_reference.__all__ == ["Compiler", "DimBucket"]
    assert not callable(luminal_reference)
    for removed in (
        "compile_model",
        "compile",
        "register_backend",
        "ReferenceAOTBackend",
    ):
        assert not hasattr(luminal_reference, removed)


def test_compiler_memory_budget_reaches_search():
    compiled = torch.compile(
        Model(), backend=Compiler(memory_budget_bytes=0), fullgraph=True
    )
    with pytest.raises(RuntimeError, match="live memory budget exceeded"):
        compiled(torch.ones(2, 2))


@pytest.mark.parametrize(
    "options",
    [
        {"search_iterations": 0},
        {"memory_budget_bytes": -1},
        {"max_intermediate_bytes": -1},
    ],
)
def test_invalid_compiler_options(options):
    with pytest.raises(ValueError):
        Compiler(**options)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("shape", [(8, 32), (2, 4, 32), (2, 2, 2, 32)])
def test_matrix_product_does_not_allocate_broadcast_intermediate(dtype, shape):
    class MatrixProduct(torch.nn.Module):
        def forward(self, a, b):
            return a @ b

    a = torch.linspace(-1, 1, 8 * 32, dtype=dtype).reshape(shape)
    b = torch.linspace(-2, 2, 32 * 128, dtype=dtype).reshape(128, 32).t()
    # The dense output fits; the 8x128x32 broadcast product cannot.
    compiled = torch.compile(
        MatrixProduct(),
        backend=Compiler(max_intermediate_bytes=8 * 128 * max(4, dtype.itemsize)),
        fullgraph=True,
    )
    torch.testing.assert_close(compiled(a, b), a @ b)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_abs_checks_runtime_values_and_preserves_signed_minimum(dtype):
    class Absolute(torch.nn.Module):
        def forward(self, x):
            return x.abs()

    limits = torch.iinfo(dtype)
    x = torch.tensor([limits.min, -limits.max, -19, 0, 19, limits.max], dtype=dtype)
    compiled = torch.compile(Absolute(), backend=Compiler(), fullgraph=True)
    torch.testing.assert_close(compiled(x), x.abs())
