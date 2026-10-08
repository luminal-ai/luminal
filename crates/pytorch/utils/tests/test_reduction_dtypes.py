"""Cast-before-accumulate parity for the PT2 translator's dtype policy."""

import pytest
import torch
from backend_test_utils import compile_for_test, selected_backend


class Reduction(torch.nn.Module):
    def __init__(self, operation, dtype=None, full=False):
        super().__init__()
        self.operation = operation
        self.dtype = dtype
        self.full = full

    def forward(self, value):
        op = getattr(torch, self.operation)
        if self.full:
            return op(value, dtype=self.dtype)
        if self.operation.startswith("cum"):
            return op(value, dim=-1, dtype=self.dtype)
        return op(value, dim=-1, keepdim=True, dtype=self.dtype)


def assert_parity(module, value):
    expected = module(value)
    if (
        selected_backend() == "reference"
        and isinstance(module, Reduction)
        and module.operation == "sum"
        and expected.dtype == torch.int32
        and value.ndim > 0
    ):
        pytest.skip(
            "reference Int32 ReduceSum requires value bounds unavailable through "
            "the PT2 harness; Rust tests check translation/saturation"
        )
    compiled = compile_for_test(
        module, (value,), search_iterations=1, dynamic_shapes={}
    )
    (actual,) = compiled(value)
    # These small, exactly representable cases test conversions, not fold order.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["sum", "prod", "cumsum", "cumprod"])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int8, torch.int32, torch.int64, torch.float32, torch.bfloat16],
)
def test_default_reduction_dtype(operation, scalar, dtype):
    if selected_backend() == "reference":
        if dtype == torch.bfloat16:
            pytest.skip(
                "reference runtime has no BF16 storage; Rust tests check translation/saturation"
            )
        if not scalar and dtype != torch.float32:
            pytest.skip(
                "reference runtime has no Int64 reduction/scan kernels; Rust tests check translation/saturation"
            )
    value = torch.tensor(2 if scalar else [[1, 2, 3], [-1, 0, 2]], dtype=dtype)
    assert_parity(Reduction(operation), value)


@pytest.mark.parametrize("operation", ["sum", "prod", "cumsum", "cumprod"])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize(
    "input_dtype,output_dtype",
    [
        (torch.bool, torch.float32),
        (torch.int32, torch.float32),
        (torch.float32, torch.int32),
        (torch.int64, torch.int32),
    ],
)
def test_explicit_dtype_casts_before_accumulation(
    operation, scalar, input_dtype, output_dtype
):
    # Casting only the output would make sums/products of 1.75 disagree.
    value = torch.tensor(
        1.75 if scalar else [[1.75, 1.75, -1.75], [2.75, 0, -2.75]], dtype=input_dtype
    )
    assert_parity(Reduction(operation, output_dtype), value)


@pytest.mark.parametrize("operation", ["sum", "prod"])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.bool])
def test_full_and_dim_reductions_honor_explicit_dtype(operation, full, dtype):
    # A post-reduction Bool cast incorrectly cancels [-1, 1] for sum;
    # truncating after reduction also differs from truncating each element.
    value = torch.tensor([[-1.75, 1.75], [0.25, 0.5]])
    assert_parity(Reduction(operation, dtype, full), value)


def test_argmax_keeps_float_operand_for_int64_index():
    class Argmax(torch.nn.Module):
        def forward(self, value):
            return torch.argmax(value, dim=-1)

    assert_parity(Argmax(), torch.tensor([0.25, 0.75, 0.5]))


@pytest.mark.parametrize("operation", ["sum", "prod"])
@pytest.mark.parametrize(
    "values",
    [
        float("nan"),
        [float("nan")],
        [float("nan"), 0.0],
        [float("inf"), -float("inf")],
        [-0.0, 0.0],
        [],
    ],
)
def test_boolean_reduction_converts_nonfinite_and_empty_inputs(operation, values):
    assert_parity(Reduction(operation, torch.bool), torch.tensor(values))
