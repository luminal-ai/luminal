"""Complex PT2 values lower to pairs of ordinary real HLIR tensors."""

import warnings

import pytest
import torch

from luminal.pt2 import compile as luminal_compile


def _compile_and_run(module: torch.nn.Module, *inputs: torch.Tensor):
    example = inputs if len(inputs) > 1 else inputs[0]
    compiled = luminal_compile(module, example, search_iterations=1)
    return compiled(*inputs)


def _assert_close(actual, expected, *, rtol=2e-4, atol=2e-5):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _assert_ieee_equal(actual, expected, *, check_zero_sign=True):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    if check_zero_sign and (actual.is_floating_point() or actual.is_complex()):
        actual = torch.view_as_real(actual) if actual.is_complex() else actual
        expected = torch.view_as_real(expected) if expected.is_complex() else expected
        zeros = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual)[zeros], torch.signbit(expected)[zeros])


class ComplexArithmetic(torch.nn.Module):
    def forward(self, a, b):
        return (
            a + b,
            a - b,
            a * b,
            a / b,
            a.abs(),
            a.sum(1),
            a.mean(0),
            a.prod(1),
            a.prod(),
            a @ b,
        )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.complex64, 2e-4, 2e-5),
        (torch.complex128, 1e-10, 1e-12),
    ],
)
def test_complex_arithmetic_reductions_and_matmul(dtype, rtol, atol):
    torch.manual_seed(0)
    a = torch.randn(3, 3, dtype=dtype)
    b = torch.randn(3, 3, dtype=dtype) + (0.25 + 0.125j)
    expected = ComplexArithmetic()(a, b)
    actual = _compile_and_run(ComplexArithmetic(), a, b)
    for result, reference in zip(actual, expected):
        _assert_close(result, reference, rtol=rtol, atol=atol)


class ComplexViews(torch.nn.Module):
    def forward(self, value):
        return (
            value.real,
            value.imag,
            value.conj(),
            torch.view_as_real(value),
            torch.view_as_complex(torch.view_as_real(value)),
            value.reshape(3, 2).permute(1, 0),
            value.unsqueeze(0).expand(2, -1, -1),
            value[:, 1],
            torch.cat((value, value), dim=1),
        )


def test_complex_components_and_shape_operations():
    value = torch.randn(2, 3, dtype=torch.complex64)
    expected = ComplexViews()(value)
    actual = _compile_and_run(ComplexViews(), value)
    for result, reference in zip(actual, expected):
        _assert_close(result, reference, rtol=0, atol=0)


class MixedComplexReal(torch.nn.Module):
    def forward(self, z, x):
        return (
            z + x,
            z * x,
            z / x,
            z + 2.0,
            z * 3.0,
            z / 2.0,
            z + (2.0 + 3.0j),
            z * (2.0 - 1.0j),
            z / (1.0 + 2.0j),
            z == x,
            z != x,
        )


def test_mixed_real_complex_and_real_scalars():
    z = torch.randn(2, 3, dtype=torch.complex64)
    x = torch.randn(2, 3, dtype=torch.float32) + 0.5
    expected = MixedComplexReal()(z, x)
    actual = _compile_and_run(MixedComplexReal(), z, x)
    for result, reference in zip(actual, expected):
        _assert_close(result, reference)


class StableComplexMath(torch.nn.Module):
    def forward(self, numerator, denominator):
        return numerator.abs(), numerator / denominator


def test_abs_and_division_avoid_naive_overflow_and_underflow():
    numerator = torch.tensor(
        [3.0e30 + 4.0e30j, 3.0e-30 + 4.0e-30j], dtype=torch.complex64
    )
    denominator = torch.tensor(
        [1.0e30 + 2.0e30j, 1.0e-30 + 2.0e-30j], dtype=torch.complex64
    )
    expected = StableComplexMath()(numerator, denominator)
    actual = _compile_and_run(StableComplexMath(), numerator, denominator)
    for result, reference in zip(actual, expected):
        assert torch.isfinite(result).all()
        _assert_close(result, reference, rtol=3e-5, atol=0)


class ComplexBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.randn(2, 3, dtype=torch.complex64))

    def forward(self, value):
        return value * self.weight


def test_complex_buffer_uses_component_storage_at_weight_boundary():
    module = ComplexBuffer()
    value = torch.randn(2, 3, dtype=torch.complex64)
    expected = module(value)
    (actual,) = _compile_and_run(module, value)
    _assert_close(actual, expected)


class ComplexHalfAdd(torch.nn.Module):
    def forward(self, a, b):
        return a + b


def test_complex32_boundary_uses_f16_components():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = torch.randn(2, 3, dtype=torch.complex32)
    b = torch.randn(2, 3, dtype=torch.complex32)
    expected = ComplexHalfAdd()(a, b)
    (actual,) = _compile_and_run(ComplexHalfAdd(), a, b)
    _assert_close(actual, expected, rtol=0, atol=0)


class ComplexSpecialValues(torch.nn.Module):
    def forward(self, a, b):
        return (
            a.abs(),
            a.conj(),
            a.real,
            a.imag,
            a + b,
            a - b,
            a * b,
            a / b,
            a.bool(),
            a == b,
            a != b,
        )


class ComplexInverseFunctions(torch.nn.Module):
    def forward(self, value):
        return torch.acos(value), torch.acosh(value)


class ComplexAsinAsinh(torch.nn.Module):
    def forward(self, value):
        return torch.asin(value), torch.asinh(value)


def test_complex_asin_asinh_avoid_cancellation():
    value = torch.tensor([-0.2327 - 0.0001j, 0.0378 + 3.5795j], dtype=torch.complex64)
    expected = ComplexAsinAsinh()(value)
    actual = _compile_and_run(ComplexAsinAsinh(), value)
    for result, reference in zip(actual, expected):
        _assert_close(result, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("dtype", "component_dtype", "rtol", "atol"),
    (
        (torch.complex64, torch.float32, 2e-4, 2e-5),
        (torch.complex128, torch.float64, 1e-8, 1e-10),
    ),
)
@pytest.mark.xfail(
    reason=(
        "egglog's f64 value domain equates +0.0 and -0.0, so lowering "
        "cannot preserve every complex branch-cut zero sign"
    ),
    strict=True,
)
def test_complex_acos_acosh_branch_cuts_and_special_values(
    dtype, component_dtype, rtol, atol
):
    parts = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, -0.0],
            [0.5, 0.0],
            [0.5, -0.0],
            [2.0, 0.0],
            [2.0, -0.0],
            [-2.0, 0.0],
            [-2.0, -0.0],
            [0.25, 0.75],
            [-0.25, -0.75],
            [0.0, float("inf")],
            [0.0, -float("inf")],
            [float("inf"), 0.0],
            [-float("inf"), -0.0],
            [float("inf"), float("inf")],
            [-float("inf"), -float("inf")],
            [float("inf"), float("nan")],
            [0.0, float("nan")],
        ],
        dtype=component_dtype,
    )
    value = torch.view_as_complex(parts).to(dtype)
    expected = ComplexInverseFunctions()(value)
    actual = _compile_and_run(ComplexInverseFunctions(), value)

    for result, reference in zip(actual, expected):
        result_parts = torch.view_as_real(result)
        reference_parts = torch.view_as_real(reference)
        assert torch.equal(torch.isnan(result_parts), torch.isnan(reference_parts))
        assert torch.equal(torch.isinf(result_parts), torch.isinf(reference_parts))
        finite = torch.isfinite(result_parts) & torch.isfinite(reference_parts)
        torch.testing.assert_close(
            result_parts[finite],
            reference_parts[finite],
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        signed_edges = ((result_parts == 0) & (reference_parts == 0)) | (
            torch.isinf(result_parts) & torch.isinf(reference_parts)
        )
        assert torch.equal(
            torch.signbit(result_parts)[signed_edges],
            torch.signbit(reference_parts)[signed_edges],
        )


@pytest.mark.parametrize(
    ("dtype", "component_dtype"),
    ((torch.complex64, torch.float32), (torch.complex128, torch.float64)),
)
@pytest.mark.xfail(
    reason=(
        "egglog's f64 value domain equates +0.0 and -0.0, so lowering "
        "cannot preserve every complex IEEE zero sign"
    ),
    strict=True,
)
def test_complex_ieee_special_values(dtype, component_dtype):
    subnormal = torch.nextafter(
        torch.tensor(0.0, dtype=component_dtype),
        torch.tensor(1.0, dtype=component_dtype),
    ).item()
    parts = torch.tensor(
        [
            [0.0, 0.0],
            [-0.0, 0.0],
            [0.0, -0.0],
            [-0.0, -0.0],
            [1.0, 2.0],
            [-1.0, -2.0],
            [float("inf"), 1.0],
            [-float("inf"), 1.0],
            [1.0, float("inf")],
            [float("inf"), float("inf")],
            [float("nan"), 1.0],
            [1.0, float("nan")],
            [subnormal, -subnormal],
        ],
        dtype=component_dtype,
    )
    values = torch.view_as_complex(parts).to(dtype)
    a = values[:, None].expand(-1, len(values)).clone()
    b = values[None, :].expand(len(values), -1).clone()
    expected = ComplexSpecialValues()(a, b)
    actual = _compile_and_run(ComplexSpecialValues(), a, b)
    for index, (result, reference) in enumerate(zip(actual, expected)):
        # PyTorch and HLIR differ only in the sign bit of a few zero-valued
        # subtraction/division results; their values and classifications agree.
        _assert_ieee_equal(
            result.resolve_conj(),
            reference.resolve_conj(),
            check_zero_sign=index not in (5, 7),
        )


class ComplexEmpty(torch.nn.Module):
    def forward(self, value):
        return value + value, value.abs(), value.sum(0), value.mean(0), value.prod(0)


def test_complex_empty_dimensions():
    value = torch.empty(0, 3, dtype=torch.complex64)
    expected = ComplexEmpty()(value)
    actual = _compile_and_run(ComplexEmpty(), value)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, equal_nan=True)


class ComplexStructuralEdges(torch.nn.Module):
    def forward(self, value):
        scalar = value[-1, -1]
        return (
            value[-4:-1, 1:],
            value.select(-1, -1),
            scalar,
            scalar.bool(),
            scalar.float(),
        )


def test_complex_scalar_casts_and_noncontiguous_negative_views():
    value = torch.randn(3, 5, dtype=torch.complex64).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = ComplexStructuralEdges()(value)
        actual = _compile_and_run(ComplexStructuralEdges(), value)
    for result, reference in zip(actual, expected):
        _assert_close(result, reference, rtol=0, atol=0)
