"""Reference-owned host-byte staging and empty-buffer tests."""

import pytest
import torch
from luminal_reference import Compiler
from test_dtype_boundary import BoundaryNoopModel, EmptyWeightModel

luminal_backend = Compiler()


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bool, id="bool"),
        pytest.param(torch.uint8, id="uint8"),
        pytest.param(torch.int8, id="int8"),
        pytest.param(torch.int16, id="int16"),
        pytest.param(torch.int32, id="int32"),
        pytest.param(torch.int64, id="int64"),
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ],
)
def test_empty_cpu_input_preserves_shape_and_dtype(dtype: torch.dtype) -> None:
    """A zero-byte CPU tensor may have a null data pointer and must still
    cross the compiled boundary with its shape and dtype intact."""
    model = BoundaryNoopModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    x = torch.empty((1, 0, 3), dtype=dtype)
    expected = model(x)
    actual = compiled(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_empty_cpu_weight_preserves_shape_and_dtype() -> None:
    """Zero-byte registered weights use the same null-pointer contract as
    runtime inputs."""
    model = EmptyWeightModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    x = torch.empty((0, 3), dtype=torch.float32)
    expected = model(x)
    actual = compiled(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_cpu_input_rejects_non_bytes_payload():
    """The native boundary accepts owned bytes, never arbitrary host pointers."""
    from luminal_reference.backend import compile_exported

    example = torch.ones(1)
    model = compile_exported(
        torch.export.export(BoundaryNoopModel(), (example,)), [example]
    )
    with pytest.raises(TypeError):
        model._graph.set_input(model._graph.input_names[0], None, [1])
