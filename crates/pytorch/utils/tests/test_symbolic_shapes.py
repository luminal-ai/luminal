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


def test_unknown_symbolic_expression_does_not_freeze_hint(tmp_path):
    import importlib
    import json
    import zipfile

    from backend_test_utils import selected_backend

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    program = torch.export.export(
        Model(),
        (torch.ones(8, 4),),
        dynamic_shapes=({0: torch.export.Dim("rows", min=6, max=24)},),
    )
    original, altered = tmp_path / "original.pt2", tmp_path / "altered.pt2"
    torch.export.save(program, original)
    changed = False
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(altered, "w") as target:
        for entry in source.infolist():
            data = source.read(entry.filename)
            if entry.filename.endswith(".json"):
                payload = json.loads(data)
                if isinstance(payload, dict) and "graph_module" in payload:
                    metadata = payload["graph_module"]["graph"]["tensor_values"]
                    expression = metadata["x"]["sizes"][0]["as_expr"]
                    expression["expr_str"] = (
                        "UnsupportedShapeFunction(" + expression["expr_str"] + ")"
                    )
                    changed = True
                    data = json.dumps(payload).encode()
            target.writestr(entry, data)
    assert changed
    native = importlib.import_module(f"luminal_{selected_backend()}._luminal")
    with pytest.raises(RuntimeError, match="cannot resolve"):
        native.compile(str(altered))
