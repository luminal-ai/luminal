"""Regression coverage for PT2 reduction and scan lowerings."""

import torch
from luminal.pt2 import compile as luminal_compile


def _compile_and_run(module: torch.nn.Module, *inputs: torch.Tensor):
    compiled = luminal_compile(module, inputs, search_iterations=1, dynamic_shapes={})
    return compiled(*inputs)


class CumulativeExtrema(torch.nn.Module):
    def forward(self, value):
        maxima = torch.cummax(value, 0)
        minima = torch.cummin(value, 0)
        return maxima.values, maxima.indices, minima.values, minima.indices


def test_cumulative_extrema_ties_nan_and_zero_signs():
    value = torch.tensor([0.0, -0.0, 0.0, 1.0, float("nan"), 2.0, float("nan")])
    expected = CumulativeExtrema()(value)
    actual = _compile_and_run(CumulativeExtrema(), value)

    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
    for result, reference in ((actual[0], expected[0]), (actual[2], expected[2])):
        ordered = ~torch.isnan(reference)
        assert torch.equal(
            torch.signbit(result)[ordered], torch.signbit(reference)[ordered]
        )


class CumulativeProduct(torch.nn.Module):
    def forward(self, value):
        return torch.cumprod(value, 1)


class CumulativeProductDimZero(torch.nn.Module):
    def forward(self, value):
        return torch.cumprod(value, 0)


def test_cumprod_integer_promotion_and_noncontiguous_input():
    value = torch.tensor([[-2, 3, 0, -4], [5, -1, 2, 3]], dtype=torch.int8).t()
    assert not value.is_contiguous()
    expected = CumulativeProduct()(value)
    (actual,) = _compile_and_run(CumulativeProduct(), value)

    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cumprod_complex_scalar_and_empty_inputs():
    for value in (
        torch.tensor([1 + 2j, -3 + 0.5j, 2j], dtype=torch.complex64),
        torch.tensor(2 - 3j, dtype=torch.complex128),
        torch.empty(0, dtype=torch.complex64),
    ):
        expected = torch.cumprod(value, 0)
        (actual,) = _compile_and_run(CumulativeProductDimZero(), value)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
