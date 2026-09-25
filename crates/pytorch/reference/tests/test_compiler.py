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
