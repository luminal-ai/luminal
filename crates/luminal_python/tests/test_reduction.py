"""Regression coverage for PT2 reduction and scan lowerings."""

import warnings

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


class ComposedReductions(torch.nn.Module):
    def forward(self, value):
        variance, mean = torch.ops.aten.var_mean.correction(
            value, [0, 2], correction=0.5, keepdim=True
        )
        return (
            torch.ops.aten._log_softmax.default(value, -1, False),
            torch.ops.aten.linalg_vector_norm.default(value, 0, [1], False),
            torch.ops.aten.linalg_vector_norm.default(value, 1, [0, 2], True),
            torch.ops.aten.linalg_vector_norm.default(value, 2, [], False),
            torch.ops.aten.linalg_vector_norm.default(value, 3.5, [2], False),
            torch.ops.aten.linalg_vector_norm.default(value, float("inf"), [1], False),
            torch.ops.aten.linalg_vector_norm.default(value, float("-inf"), [1], False),
            torch.ops.aten.var.correction(value, [1], correction=1, keepdim=True),
            variance,
            mean,
        )


def test_composed_real_reductions_match_eager():
    value = torch.tensor(
        [
            [[-2.0, 0.0, 1.5, 4.0], [3.0, -1.0, 2.0, 0.5], [1.0, 2.0, 3.0, 4.0]],
            [[0.25, -0.5, 1.0, 2.0], [-3.0, 5.0, 0.0, 1.0], [2.5, -1.5, 0.5, 3.5]],
        ]
    )
    module = ComposedReductions()
    expected = module(value)
    actual = _compile_and_run(module, value)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        torch.testing.assert_close(
            result, reference, rtol=4e-4, atol=4e-5, equal_nan=True
        )


class ScalarLogSoftmax(torch.nn.Module):
    def forward(self, value):
        return torch.ops.aten._log_softmax.default(value, 0, False)


def test_scalar_log_softmax_preserves_ieee_behavior():
    module = ScalarLogSoftmax()
    for value in (torch.tensor(2.0), torch.tensor(float("inf"))):
        expected = module(value)
        (actual,) = _compile_and_run(module, value)
        torch.testing.assert_close(actual, expected, equal_nan=True)


class ComplexComposedReductions(torch.nn.Module):
    def forward(self, value):
        variance, mean = torch.ops.aten.var_mean.correction(
            value, [0], correction=0, keepdim=False
        )
        return (
            torch.ops.aten.linalg_vector_norm.default(value, 2, [1], True),
            torch.ops.aten.linalg_vector_norm.default(
                value, 2, [1], False, dtype=torch.complex128
            ),
            torch.ops.aten.linalg_vector_norm.default(value, 1.5, [], False),
            torch.ops.aten.var.correction(value, [1], correction=1, keepdim=False),
            variance,
            mean,
        )


def test_composed_complex_reductions_match_eager():
    real = torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.0, -1.5]])
    imag = torch.tensor([[0.25, 1.0, -3.0], [-2.0, 4.0, 0.5]])
    value = torch.complex(real, imag)
    module = ComplexComposedReductions()
    expected = module(value)
    actual = _compile_and_run(module, value)
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        torch.testing.assert_close(
            result, reference, rtol=5e-4, atol=5e-5, equal_nan=True
        )


class VarianceOverloads(torch.nn.Module):
    def forward(self, value):
        variance_default = torch.ops.aten.var.default(value, False)
        variance_dim = torch.ops.aten.var.dim(value, [1], True, True)
        var_mean_default = torch.ops.aten.var_mean.default(value, False)
        var_mean_dim = torch.ops.aten.var_mean.dim(value, [0], True, False)
        return (
            variance_default,
            variance_dim,
            var_mean_default[0],
            var_mean_default[1],
            var_mean_dim[0],
            var_mean_dim[1],
        )


def test_variance_default_and_dim_overloads_match_eager():
    value = torch.tensor([[-2.0, 0.5, 3.0], [4.0, -1.0, 2.0]], dtype=torch.float64)
    module = VarianceOverloads()
    expected = module(value)
    actual = _compile_and_run(module, value)
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        torch.testing.assert_close(
            result, reference, rtol=1e-7, atol=1e-8, equal_nan=True
        )


class DynamicComposedReductions(torch.nn.Module):
    def forward(self, value):
        variance, mean = torch.ops.aten.var_mean.correction(
            value, [0], correction=1, keepdim=True
        )
        return (
            torch.ops.aten.linalg_vector_norm.default(value, 2, [0], False),
            torch.ops.aten._log_softmax.default(value, 0, False),
            variance,
            mean,
        )


def test_composed_reductions_accept_symbolic_compile_time_extents():
    module = DynamicComposedReductions()
    example = torch.arange(6.0).reshape(2, 3)
    batch = torch.export.Dim("batch", min=2, max=8)
    compiled = luminal_compile(
        module,
        (example,),
        search_iterations=1,
        dynamic_shapes=({0: batch},),
    )

    for rows in (2, 5):
        value = torch.linspace(-3, 4, rows * 3).reshape(rows, 3)
        actual = compiled(value)
        expected = module(value)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=5e-4, atol=5e-5)


class ScatterAndIndexReductions(torch.nn.Module):
    def forward(self, value, scatter_index, scatter_source, index, index_source):
        return (
            torch.ops.aten.scatter_reduce.two(
                value, 1, scatter_index, scatter_source, "sum", include_self=True
            ),
            torch.ops.aten.scatter_reduce.two(
                value, 1, scatter_index, scatter_source, "prod", include_self=False
            ),
            torch.ops.aten.scatter_reduce.two(
                value, 1, scatter_index, scatter_source, "mean", include_self=False
            ),
            torch.ops.aten.scatter_reduce.two(
                value, 1, scatter_index, scatter_source, "amax", include_self=True
            ),
            torch.ops.aten.scatter_reduce.two(
                value, 1, scatter_index, scatter_source, "amin", include_self=False
            ),
            torch.ops.aten.index_reduce.default(
                value, 1, index, index_source, "prod", include_self=True
            ),
            torch.ops.aten.index_reduce.default(
                value, 1, index, index_source, "mean", include_self=False
            ),
            torch.ops.aten.index_reduce.default(
                value, 1, index, index_source, "amax", include_self=False
            ),
            torch.ops.aten.index_reduce.default(
                value, 1, index, index_source, "amin", include_self=True
            ),
        )


def test_scatter_and_index_reduce_duplicates_and_include_self():
    value = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    scatter_index = torch.tensor([[0, 0, 2], [1, 1, 1]])
    scatter_source = torch.tensor([[2.0, 3.0, 4.0], [0.5, 2.0, -1.0]])
    index = torch.tensor([1, 1, 2])
    index_source = torch.tensor([[2.0, 4.0, -3.0], [0.5, 2.0, 8.0]])
    module = ScatterAndIndexReductions()
    inputs = (value, scatter_index, scatter_source, index, index_source)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = module(*inputs)
        actual = _compile_and_run(module, *inputs)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(
            result, reference, rtol=2e-5, atol=2e-6, equal_nan=True
        )


class DynamicScatterAndIndexReductions(torch.nn.Module):
    def forward(self, value, index, source):
        return (
            torch.ops.aten.scatter_reduce.two(
                value, 0, index, source, "sum", include_self=False
            ),
            torch.ops.aten.index_reduce.default(
                value, 0, index, source, "prod", include_self=True
            ),
        )


def test_scatter_and_index_reduce_accept_bounded_symbolic_update_extents():
    module = DynamicScatterAndIndexReductions()
    example = (
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0, 2]),
        torch.tensor([4.0, 5.0]),
    )
    output_size = torch.export.Dim("output_size", min=2, max=8)
    update_size = torch.export.Dim("update_size", min=1, max=8)
    compiled = luminal_compile(
        module,
        example,
        search_iterations=1,
        dynamic_shapes=(
            {0: output_size},
            {0: update_size},
            {0: update_size},
        ),
    )

    runtime_inputs = (
        example,
        (
            torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0]),
            torch.tensor([1, 1, 4, 0]),
            torch.tensor([2.0, -3.0, 4.0, 0.5]),
        ),
    )
    for inputs in runtime_inputs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual = compiled(*inputs)
            expected = module(*inputs)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)


class IntegerScatterMean(torch.nn.Module):
    def forward(self, value, index, source):
        return torch.ops.aten.scatter_reduce.two(
            value, 0, index, source, "mean", include_self=False
        )


def test_scatter_reduce_integer_mean_floors_negative_results():
    value = torch.tensor([9, 9, 9], dtype=torch.int64)
    index = torch.tensor([1, 1, 2])
    source = torch.tensor([-2, -3, 4], dtype=torch.int64)
    module = IntegerScatterMean()
    (actual,) = _compile_and_run(module, value, index, source)
    expected = module(value, index, source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class ScatterNaNExtrema(torch.nn.Module):
    def forward(self, value, index, source):
        return (
            torch.ops.aten.scatter_reduce.two(
                value, 0, index, source, "amax", include_self=True
            ),
            torch.ops.aten.scatter_reduce.two(
                value, 0, index, source, "amin", include_self=False
            ),
        )


def test_scatter_reduce_extrema_propagate_nan():
    value = torch.tensor([10.0, 20.0])
    index = torch.tensor([0, 0])
    source = torch.tensor([float("nan"), 3.0])
    module = ScatterNaNExtrema()
    actual = _compile_and_run(module, value, index, source)
    expected = module(value, index, source)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, equal_nan=True)


class BooleanScatterReductions(torch.nn.Module):
    def forward(self, value, index, source):
        return tuple(
            torch.ops.aten.scatter_reduce.two(
                value, 0, index, source, reduction, include_self=True
            )
            for reduction in ("sum", "prod", "amax", "amin")
        )


def test_boolean_scatter_reductions_use_logical_semantics():
    value = torch.tensor([False, True, False])
    index = torch.tensor([0, 0, 1])
    source = torch.tensor([True, True, False])
    module = BooleanScatterReductions()
    actual = _compile_and_run(module, value, index, source)
    expected = module(value, index, source)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


class ScalarScatterAndIndexReductions(torch.nn.Module):
    def forward(self, value, scatter_index, index, source):
        return (
            torch.ops.aten.scatter_reduce.two(
                value, 0, scatter_index, source, "sum", include_self=False
            ),
            torch.ops.aten.index_reduce.default(
                value, 0, index, source, "mean", include_self=True
            ),
        )


def test_scalar_scatter_and_index_reductions():
    value = torch.tensor(6.0)
    scatter_index = torch.tensor(0)
    index = torch.tensor([0])
    source = torch.tensor(2.0)
    module = ScalarScatterAndIndexReductions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = _compile_and_run(module, value, scatter_index, index, source)
        expected = module(value, scatter_index, index, source)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
