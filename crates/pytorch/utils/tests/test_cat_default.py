"""Export may omit cat's default dimension from its serialized arguments."""

import pytest
import torch
from backend_test_utils import compile_for_test, selected_backend


@pytest.mark.parametrize("sizes", [(3, 2), (0, 2), (3, 0), (0, 0)])
def test_cat_omitted_default_dimension(sizes):
    class Model(torch.nn.Module):
        def forward(self, left, right):
            return torch.cat([left, right])

    inputs = tuple(torch.arange(float(size * 2)).reshape(size, 2) for size in sizes)
    model = Model()
    compiled = compile_for_test(model, inputs, search_iterations=1)
    (actual,) = compiled(*inputs)
    torch.testing.assert_close(actual, model(*inputs))


def _compile_cat(model, inputs, *, dynamic_shapes=None):
    # Default decompositions can remove empty operands and insert promotion casts.
    # Keep aten.cat intact so these cases actually reach our translator.
    import importlib

    from backend_test_utils import selected_backend
    from luminal_reference.export_utils import _drop_input_guards

    program = torch.export.export(
        model, inputs, dynamic_shapes=dynamic_shapes, strict=False
    )
    _drop_input_guards(program)
    program = program.run_decompositions({})
    assert any(
        node.target == torch.ops.aten.cat.default for node in program.graph.nodes
    )
    backend = importlib.import_module(f"luminal_{selected_backend()}.backend")
    return backend.compile_exported(program, inputs, search_iterations=1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("axis", [2, -2])
@pytest.mark.parametrize("empty_position", [0, 1, 2])
def test_cat_rank_one_empty_with_higher_rank_tensors(dtype, axis, empty_position):
    class Model(torch.nn.Module):
        def forward(self, a, b, empty):
            if dtype.is_complex:
                a, b, empty = (torch.view_as_complex(v) for v in (a, b, empty))
            values = [a, b]
            values.insert(empty_position, empty)
            result = torch.cat(values, dim=axis)
            return torch.view_as_real(result) if dtype.is_complex else result

    a = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2).to(dtype)
    b = (torch.arange(8, dtype=torch.float32) + 20).reshape(1, 2, 2, 2).to(dtype)
    if dtype.is_complex:
        a = a + (a + 1) * 1j
        b = b - (b + 1) * 1j
    inputs = (a, b, torch.empty(0, dtype=dtype))
    if dtype.is_complex:
        # Keep the backend boundary real while testing complex ATen lowering.
        inputs = tuple(torch.view_as_real(v) for v in inputs)
    model = Model()
    compiled = _compile_cat(model, inputs)
    (actual,) = compiled(*inputs)
    torch.testing.assert_close(actual, model(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("empty_shape", [(0,), (2, 0)])
@pytest.mark.parametrize("empty_first", [True, False])
def test_cat_empty_operand_promotes_result(empty_shape, empty_first):
    class Model(torch.nn.Module):
        def forward(self, empty, value):
            values = [empty, value] if empty_first else [value, empty]
            return torch.cat(values, dim=-1)

    inputs = (
        torch.empty(empty_shape, dtype=torch.float32),
        torch.arange(6, dtype=torch.int32).reshape(2, 3),
    )
    model = Model()
    compiled = _compile_cat(model, inputs)
    (actual,) = compiled(*inputs)
    torch.testing.assert_close(actual, model(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("symbolic_first", [True, False])
@pytest.mark.parametrize(
    "runtime_zero",
    [
        False,
        pytest.param(
            True,
            marks=[
                pytest.mark.xfail(
                    strict=True,
                    raises=RuntimeError,
                    reason="Existing GraphTensor::concat_along padding reads an empty symbolic operand",
                ),
                pytest.mark.skipif(
                    selected_backend() != "reference",
                    reason="Exercise the known out-of-bounds read in the checked reference runtime",
                ),
            ],
        ),
    ],
)
def test_cat_symbolic_operand_survives_across_lengths(
    dtype, symbolic_first, runtime_zero
):
    class Model(torch.nn.Module):
        def forward(self, a, b):
            if dtype.is_complex:
                a, b = torch.view_as_complex(a), torch.view_as_complex(b)
            result = torch.cat([a, b] if symbolic_first else [b, a], dim=-1)
            return torch.view_as_real(result) if dtype.is_complex else result

    def inputs_for(n):
        a = torch.arange(n, dtype=torch.float32).to(dtype) + 10
        b = torch.arange(2, dtype=torch.float32).to(dtype) + 30
        if dtype.is_complex:
            a = a + (a + 1) * 1j
            b = b - (b + 1) * 1j
            return torch.view_as_real(a), torch.view_as_real(b)
        return a, b

    model = Model()
    compiled = _compile_cat(
        model,
        inputs_for(2),
        dynamic_shapes=(
            {0: torch.export.Dim("length", min=0 if runtime_zero else 1, max=5)},
            {},
        ),
    )
    for n in [1, 5, 0, 3] if runtime_zero else [1, 5, 3, 2]:
        inputs = inputs_for(n)
        try:
            (actual,) = compiled(*inputs)
        except RuntimeError as error:
            # Only the established empty-read failure is expected here.
            assert runtime_zero and n == 0
            assert "materialize index" in str(error) and "(extent 0)" in str(error)
            raise
        torch.testing.assert_close(actual, model(*inputs), rtol=0, atol=0)
