"""Regression coverage for PT2 binary elementwise lowerings."""

import os

import pytest
import torch

from luminal.pt2 import compile as luminal_compile


def _compile_and_run(module: torch.nn.Module, *inputs: torch.Tensor):
    compiled = luminal_compile(module, inputs, search_iterations=1)
    return compiled(*inputs)


class Atan2(torch.nn.Module):
    def forward(self, y, x):
        return torch.atan2(y, x)


def test_atan2_quadrants_and_signed_zero():
    y = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf")])
    x = torch.tensor([-1.0, -1.0, 0.0, -0.0, float("inf"), -float("inf")])
    expected = Atan2()(y, x)
    (actual,) = _compile_and_run(Atan2(), y, x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual)[zeros], torch.signbit(expected)[zeros])


class BinaryIeeeValues(torch.nn.Module):
    def forward(self, a, b):
        return torch.copysign(a, b), torch.fmax(a, b), torch.fmin(a, b)


@pytest.mark.xfail(
    os.getenv("LUMINAL_TEST_DEVICE", "cpu").lower() == "cuda"
    and torch.cuda.is_available(),
    reason=(
        "egglog's f64 value domain equates +0.0 and -0.0, so CUDA lowering "
        "cannot preserve every IEEE zero sign without a distinct representation"
    ),
)
def test_copysign_fmax_fmin_nan_and_zero_signs():
    a = torch.tensor([0.0, -0.0, float("nan"), 2.0, float("nan")])
    b = torch.tensor([-0.0, 0.0, 2.0, float("nan"), float("nan")])
    expected = BinaryIeeeValues()(a, b)
    actual = _compile_and_run(BinaryIeeeValues(), a, b)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
        classified = ~torch.isnan(reference)
        assert torch.equal(
            torch.signbit(result)[classified], torch.signbit(reference)[classified]
        )


class CopySignScalars(torch.nn.Module):
    def forward(self, value):
        return torch.copysign(value, -0.0), torch.copysign(value, 0.0)


def test_copysign_scalar_typed_constants_and_zero_signs():
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        value = torch.tensor([0.0, -0.0, 2.0, -2.0, float("nan")], dtype=dtype)
        expected = CopySignScalars()(value)
        actual = _compile_and_run(CopySignScalars(), value)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(
                result, reference, rtol=0, atol=0, equal_nan=True
            )
            classified = ~torch.isnan(reference)
            assert torch.equal(
                torch.signbit(result)[classified], torch.signbit(reference)[classified]
            )


class BoolNotEqualAndDiff(torch.nn.Module):
    def forward(self, left, right):
        return left.ne(right), torch.diff(left)


def test_not_equal_and_diff_preserve_boolean_output_dtype():
    left = torch.tensor([True, True, False, True, False])
    right = torch.tensor([False, True, False, False, True])
    expected = BoolNotEqualAndDiff()(left, right)
    actual = _compile_and_run(BoolNotEqualAndDiff(), left, right)

    for result, reference in zip(actual, expected):
        assert result.dtype == torch.bool
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
