"""Translator regressions for inferred reshape extents and shape arithmetic."""

import pytest
import torch
from backend_test_utils import compile_for_test


@pytest.mark.parametrize("slice_output", [False, True])
def test_symbolic_shape_expressions(slice_output):
    class Model(torch.nn.Module):
        def forward(self, x):
            if slice_output:
                return x[: x.shape[0] // 2] + 1
            return x.reshape(x.shape[0], -1).sum(-1)

    model = Model()
    compiled = compile_for_test(
        model,
        torch.ones(8, 4),
        search_iterations=1,
        dynamic_shapes=({0: torch.export.Dim("rows", min=6, max=24)},),
    )
    for rows in (8, 13, 18):
        x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
        (actual,) = compiled(x)
        torch.testing.assert_close(actual, model(x))


def test_unknown_symbolic_expression_does_not_freeze_hint(device, monkeypatch):
    """A dimension written with a sympy function the translator does not
    read must fail translation. Falling back to the size the example call
    happened to have would freeze a dynamic dimension."""
    import importlib
    import json

    from backend_test_utils import (
        luminal_backend,
        record_lowered_graphs,
        selected_backend,
    )
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    def model(x):
        return x + 1

    lowered = record_lowered_graphs(monkeypatch)
    rows = ShapeVar("rows", min=6, max=24, optimization_hint=8)
    compiled = torch.compile(
        model,
        backend=luminal_backend,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([rows, 4])}),
    )
    x = torch.ones(8, 4, device=device)
    torch.testing.assert_close(compiled(x), model(x))

    # The program as it was handed to the translator, with the dynamic
    # extent rewritten into a function the translator does not know.
    (program,) = lowered
    payload = json.loads(program.program_json())
    changed = False
    for meta in payload["graph_module"]["graph"]["tensor_values"].values():
        for size in meta["sizes"]:
            if "as_expr" in size:
                expression = size["as_expr"]
                expression["expr_str"] = (
                    "UnsupportedShapeFunction(" + expression["expr_str"] + ")"
                )
                changed = True
    assert changed, "the program states no dynamic extent"
    altered = json.dumps(payload).encode()

    native = importlib.import_module(f"luminal_{selected_backend()}._luminal")
    # The CUDA entry also takes the boundary layouts, which translation
    # never reaches here.
    layouts = ([], [], []) if selected_backend() == "cuda_lite" else ()
    with pytest.raises(RuntimeError, match="cannot resolve"):
        native.compile_graph(altered, *layouts)
