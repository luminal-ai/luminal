"""Regression coverage for PT2 binary elementwise lowerings."""

import os

import pytest
import torch
from backend_test_utils import compile_for_test as luminal_compile
from backend_test_utils import luminal_backend


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


def test_atan2_negative_infinity_and_zero_x_quadrants():
    """The quadrant correction reads the sign bit of `x`, and `atan(y/x)`
    reads the sign of the ratio; `-inf` and both signed zeros must agree."""
    inf = float("inf")
    y = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0, -0.0, inf, -inf, inf, -inf, 2.0]
    x = [0.0, 0.0, -0.0, -0.0, -inf, -inf, -inf, -inf, -inf, -inf, inf, inf, -3.0]
    for dtype in (torch.float32, torch.float64):
        y_t = torch.tensor(y, dtype=dtype)
        x_t = torch.tensor(x, dtype=dtype)
        expected = Atan2()(y_t, x_t)
        (actual,) = _compile_and_run(Atan2(), y_t, x_t)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


class CopySign(torch.nn.Module):
    def forward(self, magnitude, sign):
        return torch.copysign(magnitude, sign)


def test_copysign_infinite_sign_operand():
    magnitude = [5.0, 5.0, -5.0, -5.0, 2.0, -3.0]
    sign = [-float("inf"), float("inf"), -float("inf"), float("inf"), -1.0, 1.0]
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        a = torch.tensor(magnitude, dtype=dtype)
        b = torch.tensor(sign, dtype=dtype)
        expected = CopySign()(a, b)
        (actual,) = _compile_and_run(CopySign(), a, b)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    (torch.float16, torch.bfloat16, torch.float32, torch.float64),
    ids=str,
)
def test_copysign_takes_only_the_sign_bit_of_zero_tiny_and_infinite_signs(
    stage_one_only, dtype
):
    """copysign is |magnitude| with the sign BIT of `sign`. A zero magnitude
    becomes a signed zero, an infinite or subnormal `sign` is only a sign,
    and a `-0.0` magnitude loses its own sign. Multiplying by `sign(s)` gave
    0 for a zero `s` and NaN for an infinite one; `abs(-0.0)` is `-0.0`."""
    inf, nan = float("inf"), float("nan")
    magnitude = [0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 3.0, -3.0, 0.0, -0.0, 2.0, -2.0]
    sign = [inf, inf, -inf, -inf, 1e-45, -1e-45, -1e-30, 1e-30, nan, -0.0, 0.0, 0.0]
    magnitude += [inf, -inf, -0.0, 0.0]
    sign += [-1e-45, 1e-45, 1.0, -1.0]
    a = torch.tensor(magnitude, dtype=dtype)
    b = torch.tensor(sign, dtype=dtype)
    expected = CopySign()(a, b)
    actual = torch.compile(CopySign(), backend=luminal_backend)(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected)), (
        actual,
        expected,
    )


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


# ---- a Python or symbolic scalar on either side of add/sub/mul/div ----
#
# Export records `1 - x` as `rsub.Scalar(x, 1)` (undecomposed) or
# `sub.Tensor(1, x)` (decomposed), and `x.shape[0]` under dynamic shapes as
# a SymInt operand. All were refused once: the lowerings read operand 0 as
# a tensor and operand 1 as a literal.


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class Unary(Lambda):
    # One named input, so export's `dynamic_shapes` can address it.
    def forward(self, x):
        return self.fn(x)


def _assert_same(actual, expected, exact=True):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if exact:
        assert torch.equal(actual, expected), (actual, expected)
    else:
        torch.testing.assert_close(actual, expected)


# The integer tensor is clamped first: integer arithmetic on the reference
# backend needs a proven value range, which a raw integer input lacks.
SCALAR_FIRST_INPUTS = {
    torch.float32: lambda: torch.tensor([0.25, 2.0, -4.5, 1e-3]),
    torch.float64: lambda: torch.tensor([0.1, 2.0, -4.5, 1e-300], dtype=torch.float64),
    torch.float16: lambda: torch.tensor([0.25, 2.0, -4.5, 1e-3], dtype=torch.float16),
    torch.int64: lambda: torch.tensor([7, -3, 0, 100], dtype=torch.int64),
}


def _bounded(x):
    return x.clamp(-1000, 1000) if not x.is_floating_point() else x


@pytest.mark.parametrize("scalar", [1, 2.5], ids=["int", "float"])
@pytest.mark.parametrize("dtype", list(SCALAR_FIRST_INPUTS), ids=str)
def test_scalar_minus_tensor_real_path(stage_one_only, dtype, scalar):
    model = Unary(lambda x: scalar - _bounded(x))
    x = SCALAR_FIRST_INPUTS[dtype]()
    expected = model(x)
    actual = torch.compile(model, backend=luminal_backend)(x)
    _assert_same(actual, expected)


@pytest.mark.parametrize("scalar", [1, 2.5], ids=["int", "float"])
@pytest.mark.parametrize("dtype", list(SCALAR_FIRST_INPUTS), ids=str)
def test_scalar_minus_tensor_decomposed(dtype, scalar):
    # The decomposed spelling, `sub.Tensor(scalar, x)`: what stage 2 sees
    # when another op in the graph forces the whole graph to decompose.
    model = Unary(lambda x: scalar - _bounded(x))
    x = SCALAR_FIRST_INPUTS[dtype]()
    expected = model(x)
    (actual,) = luminal_compile(model, (x,), search_iterations=1)(x)
    _assert_same(actual, expected)


def test_float64_scalar_first_keeps_the_double(stage_one_only):
    # 0.1 read at F32 and widened would be off by ~1.5e-9.
    model = Unary(lambda x: 0.1 - x)
    x = torch.tensor([0.0, 1e-17, -0.2], dtype=torch.float64)
    expected = model(x)
    actual = torch.compile(model, backend=luminal_backend)(x)
    _assert_same(actual, expected)


@pytest.mark.parametrize(
    "fn",
    [
        lambda x, y: torch.rsub(x, 2.0, alpha=3.0),
        lambda x, y: torch.rsub(x, 2),
        lambda x, y: torch.rsub(x, y, alpha=2),
        lambda x, y: torch.rsub(x, y),
        lambda x, y: torch.sub(2, x, alpha=3),
        lambda x, y: x * (1.0 - y),
    ],
    ids=[
        "rsub-scalar-alpha",
        "rsub-scalar",
        "rsub-tensor-alpha",
        "rsub-tensor",
        "sub-scalar-first-alpha",
        "mask-idiom",
    ],
)
def test_reverse_subtraction_real_path(stage_one_only, fn):
    model = Lambda(fn)
    x = torch.tensor([[0.25, -1.5, 3.0], [8.0, 0.0, -0.75]])
    y = torch.tensor([0.0, 1.0, 0.5])
    expected = model(x, y)
    actual = torch.compile(model, backend=luminal_backend)(x, y)
    # A non-unit alpha records a multiply then a subtract; eager fuses them,
    # so the last ulp may differ.
    _assert_same(actual, expected, exact=False)


def test_reverse_subtraction_of_half_rounds_once(stage_one_only):
    model = Unary(lambda x: torch.rsub(x, 1.0))
    x = torch.tensor([0.25, 2.0, -4.5, 1e-3, 65504.0], dtype=torch.float16)
    expected = model(x)
    actual = torch.compile(model, backend=luminal_backend)(x)
    _assert_same(actual, expected)


# F32 division is recorded as a multiply by the reciprocal (the frontend's
# spelling for every non-F64 divisor), so it is compared with a tolerance.
SYMBOLIC_SCALAR_OPS = {
    "mul": (lambda x: x * x.shape[0], True),
    "truediv": (lambda x: x / x.shape[0], False),
    "add": (lambda x: x + x.shape[0], True),
    "sub": (lambda x: x - x.shape[0], True),
    "rsub": (lambda x: x.shape[0] - x, True),
    "mul-expression": (lambda x: x * (x.shape[0] + 1), True),
    # The `.Scalar` overloads take the SymInt as their `Scalar other`.
    "aten-mul-Scalar": (lambda x: torch.ops.aten.mul.Scalar(x, x.shape[0]), True),
    "aten-add-Scalar": (lambda x: torch.ops.aten.add.Scalar(x, x.shape[0]), True),
    "aten-div-Scalar": (lambda x: torch.ops.aten.div.Scalar(x, x.shape[0]), False),
}


@pytest.mark.parametrize("name", list(SYMBOLIC_SCALAR_OPS))
def test_symbolic_scalar_operand_real_path(stage_one_only, name):
    fn, exact = SYMBOLIC_SCALAR_OPS[name]
    model = Unary(fn)
    compiled = torch.compile(model, backend=luminal_backend, dynamic=True)
    # Every extent runs through the one dynamic graph: the scalar is the
    # dimension, never the hint it was traced at.
    for rows in (3, 5, 8):
        x = torch.linspace(-2.0, 2.0, rows * 2).reshape(rows, 2)
        _assert_same(compiled(x), model(x), exact)


@pytest.mark.parametrize("name", list(SYMBOLIC_SCALAR_OPS))
def test_symbolic_scalar_operand_decomposed(name):
    fn, exact = SYMBOLIC_SCALAR_OPS[name]
    model = Unary(fn)
    compiled = luminal_compile(
        model,
        (torch.ones(4, 2),),
        search_iterations=1,
        dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=64)},),
    )
    for rows in (3, 5, 8):
        x = torch.linspace(-2.0, 2.0, rows * 2).reshape(rows, 2)
        (actual,) = compiled(x)
        _assert_same(actual, model(x), exact)


@pytest.mark.parametrize(
    "fn",
    [lambda x: x * x.shape[0], lambda x: x / x.shape[0]],
    ids=["mul-stays-int64", "truediv-is-float"],
)
def test_symbolic_scalar_operand_follows_torch_promotion(fn):
    # int64 * SymInt is int64; int64 / SymInt is true division (float32).
    model = Unary(lambda x: fn(x.clamp(-1000, 1000)))
    compiled = luminal_compile(
        model,
        (torch.ones(4, 2, dtype=torch.int64),),
        search_iterations=1,
        dynamic_shapes=({0: torch.export.Dim("rows", min=2, max=64)},),
    )
    for rows in (3, 8):
        x = torch.arange(rows * 2, dtype=torch.int64).reshape(rows, 2) - 5
        (actual,) = compiled(x)
        _assert_same(actual, model(x), exact=x.dtype == actual.dtype)


# ---------------------------------------------------------------------------
# IEEE NaN semantics of the comparisons, maximum/minimum and clamp.
#
# These go through torch.compile with the real backend, so each ATen overload
# reaches its own stage-1 translator arm (eq.Tensor, ne.Scalar, maximum.default,
# clamp.default, clamp_min.Tensor, any.default, ...). `compile_for_test` always
# decomposes and would only reach the decomposed forms.
# ---------------------------------------------------------------------------

NAN = float("nan")
INF = float("inf")
# Every pairing of NaN with NaN, an infinity, a signed zero and a finite value,
# with NaN on the left, on the right and on both sides.
NAN_LHS = [NAN, 1.0, NAN, NAN, INF, -INF, 0.0, -0.0, NAN, 2.0, -INF]
NAN_RHS = [0.0, NAN, NAN, INF, NAN, NAN, -0.0, 0.0, -INF, 1.0, INF]


def _compile_real(module: torch.nn.Module):
    return torch.compile(
        module,
        backend=lambda gm, example_inputs: luminal_backend(
            gm, example_inputs, {"search_iterations": 1}
        ),
    )


def _assert_outputs_bitwise(actual, expected):
    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    assert len(actual) == len(expected)
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == reference.dtype, index
        torch.testing.assert_close(
            result, reference, rtol=0, atol=0, equal_nan=True, msg=f"output {index}"
        )


class TensorComparisons(torch.nn.Module):
    def forward(self, a, b):
        return (
            torch.eq(a, b),
            torch.ne(a, b),
            torch.le(a, b),
            torch.ge(a, b),
            torch.lt(a, b),
            torch.gt(a, b),
        )


class ScalarComparisons(torch.nn.Module):
    def forward(self, a):
        return (
            a == 1.0,
            a != 1.0,
            a <= 1.0,
            a >= 1.0,
            torch.eq(a, NAN),
            torch.ne(a, NAN),
            torch.le(a, NAN),
            torch.ge(a, NAN),
            a != a,
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_comparisons_are_false_for_nan_except_not_equal(dtype):
    a = torch.tensor(NAN_LHS, dtype=dtype)
    b = torch.tensor(NAN_RHS, dtype=dtype)
    _assert_outputs_bitwise(
        _compile_real(TensorComparisons())(a, b), TensorComparisons()(a, b)
    )
    _assert_outputs_bitwise(
        _compile_real(ScalarComparisons())(a), ScalarComparisons()(a)
    )


class MaximumMinimumBothOrders(torch.nn.Module):
    def forward(self, a, b):
        return (
            torch.maximum(a, b),
            torch.maximum(b, a),
            torch.minimum(a, b),
            torch.minimum(b, a),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_maximum_minimum_propagate_nan_from_either_operand(dtype):
    a = torch.tensor(NAN_LHS, dtype=dtype)
    b = torch.tensor(NAN_RHS, dtype=dtype)
    expected = MaximumMinimumBothOrders()(a, b)
    _assert_outputs_bitwise(_compile_real(MaximumMinimumBothOrders())(a, b), expected)


class ClampWithNan(torch.nn.Module):
    def forward(self, x, bound, nan_scalar):
        return (
            torch.clamp(x, -1.0, 1.0),
            torch.clamp(x, min=NAN),
            torch.clamp(x, max=NAN),
            torch.clamp(x, min=bound),
            torch.clamp(x, max=bound),
            torch.clamp_min(x, bound),
            torch.clamp_max(x, bound),
            torch.clamp_min(x, nan_scalar),
            torch.clamp_max(x, nan_scalar),
        )


def test_clamp_propagates_nan_bounds_and_nan_inputs():
    x = torch.tensor([NAN, -INF, -1.5, -0.0, 0.5, 2.0, INF])
    bound = torch.tensor([0.0, NAN, 0.0, NAN, 0.0, NAN, 0.0])
    nan_scalar = torch.tensor(NAN)
    expected = ClampWithNan()(x, bound, nan_scalar)
    _assert_outputs_bitwise(
        _compile_real(ClampWithNan())(x, bound, nan_scalar), expected
    )


class AnyOfRows(torch.nn.Module):
    def forward(self, x):
        return torch.any(x), torch.any(x, dim=1)


def test_any_counts_nan_as_true():
    x = torch.tensor([[NAN, 0.0], [0.0, -0.0], [0.0, 1.0]])
    _assert_outputs_bitwise(_compile_real(AnyOfRows())(x), AnyOfRows()(x))


class ThresholdNan(torch.nn.Module):
    def forward(self, x):
        return (
            torch.nn.functional.threshold(x, 0.5, -7.0),
            torch.nn.functional.threshold(x, NAN, -7.0),
        )


def test_threshold_keeps_nan_input_and_ignores_nan_threshold():
    x = torch.tensor([NAN, -INF, 0.0, 0.5, 1.0, INF])
    _assert_outputs_bitwise(_compile_real(ThresholdNan())(x), ThresholdNan()(x))


class FmaxFmin(torch.nn.Module):
    def forward(self, a, b):
        return torch.fmax(a, b), torch.fmin(a, b)


# Every ordered pair of these, in both operand orders, through broadcasting a
# column against a row: NaN against NaN, an infinity, a signed zero and a
# finite value, and each signed zero against both zeros.
FMAX_VALUES = [NAN, -INF, -2.0, -0.0, 0.0, 1.0, INF]


def _assert_same_with_zero_signs(actual, expected):
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == reference.dtype, index
        assert result.shape == reference.shape, index
        assert torch.equal(torch.isnan(result), torch.isnan(reference)), index
        ordered = ~torch.isnan(reference)
        assert torch.equal(result[ordered], reference[ordered]), index
        assert torch.equal(
            torch.signbit(result)[ordered], torch.signbit(reference)[ordered]
        ), index


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_fmax_fmin_skip_nan_and_order_signed_zeros(dtype):
    # torch.fmax/fmin return the other operand when exactly one is NaN, NaN
    # when both are, and order -0.0 below +0.0. Through torch.compile, so
    # fmax.default/fmin.default reach their stage-1 arm.
    values = torch.tensor(FMAX_VALUES, dtype=dtype)
    a, b = values.reshape(-1, 1), values.reshape(1, -1)
    expected = FmaxFmin()(a, b)
    _assert_same_with_zero_signs(_compile_real(FmaxFmin())(a, b), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fmax_fmin_integers_are_maximum_minimum(dtype):
    a = torch.tensor([[-7], [0], [5]], dtype=dtype)
    b = torch.tensor([[-8, 0, 5, 9]], dtype=dtype)
    expected = FmaxFmin()(a, b)
    _assert_same_with_zero_signs(_compile_real(FmaxFmin())(a, b), expected)


# ---------------------------------------------------------------------------
# Zero-element inputs, and the NaN test and truthiness the float lowerings
# share (`isnan`, the NaN passthrough of celu/mish/clamp_min/clamp_max, the
# operand truth of the logical ops).
#
# Real path pinned to stage 1, so each ATen overload reaches its own arm. Every
# op below used to fail on a zero-element input with "out of range integral
# type conversion attempted": a strided view of an empty domain was sized at a
# negative reach, and the NaN and truth helpers sliced such a view of the IEEE
# bits. The view now spans zero storage, and truthiness is the IEEE `x != 0`.
# ---------------------------------------------------------------------------

EMPTY_SHAPES = [(0,), (1, 0, 3), (0, 1, 3)]

EMPTY_UNARY_OPS = {
    "isnan": torch.isnan,
    "isinf": torch.isinf,
    "celu": lambda x: torch.nn.functional.celu(x, alpha=2.0),
    "mish": torch.nn.functional.mish,
    "logical_not": torch.logical_not,
    "where": lambda x: torch.where(x > 0, x, x * 2),
}

EMPTY_BINARY_OPS = {
    "clamp_min": torch.clamp_min,
    "clamp_max": torch.clamp_max,
    "atan2": torch.atan2,
    "hypot": torch.hypot,
    "logical_and": torch.logical_and,
    "logical_or": torch.logical_or,
    "logaddexp": torch.logaddexp,
}


@pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
@pytest.mark.parametrize("name", list(EMPTY_UNARY_OPS))
def test_zero_element_unary_float_ops(stage_one_only, name, shape):
    model = Unary(EMPTY_UNARY_OPS[name])
    x = torch.zeros(shape)
    _assert_outputs_bitwise(_compile_real(model)(x), model(x))


@pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
@pytest.mark.parametrize("name", list(EMPTY_BINARY_OPS))
def test_zero_element_binary_float_ops(stage_one_only, name, shape):
    model = Lambda(EMPTY_BINARY_OPS[name])
    a, b = torch.zeros(shape), torch.zeros(shape)
    _assert_outputs_bitwise(_compile_real(model)(a, b), model(a, b))


@pytest.mark.parametrize("name", ["clamp_min", "clamp_max"])
def test_zero_element_tensor_bound_broadcasts(stage_one_only, name):
    # OpInfo's sample: an empty (0, 1, 3) input against a (0, 10, 3) bound.
    model = Lambda(EMPTY_BINARY_OPS[name])
    x, bound = torch.zeros(0, 1, 3), torch.zeros(0, 10, 3)
    _assert_outputs_bitwise(_compile_real(model)(x, bound), model(x, bound))


def test_searchsorted_and_bucketize_of_an_empty_float_query(stage_one_only):
    model = Lambda(
        lambda sorted_, query: (
            torch.searchsorted(sorted_, query),
            torch.bucketize(query, sorted_),
        )
    )
    sorted_, query = torch.tensor([1.0, 2.0, 3.0]), torch.zeros(0)
    _assert_outputs_bitwise(_compile_real(model)(sorted_, query), model(sorted_, query))


class NanAndTruth(torch.nn.Module):
    def forward(self, x):
        one, zero = torch.ones_like(x), torch.zeros_like(x)
        return (
            torch.isnan(x),
            torch.logical_not(x),
            torch.logical_and(x, one),
            torch.logical_or(x, zero),
            torch.logical_xor(x, one),
        )


# The smallest positive subnormal of each dtype: truthy, and not NaN.
SMALLEST_SUBNORMAL = {
    torch.float32: 1e-45,
    torch.float64: 5e-324,
    torch.float16: 6e-8,
    torch.bfloat16: 1e-40,
}


@pytest.mark.parametrize("dtype", list(SMALLEST_SUBNORMAL), ids=str)
def test_isnan_and_truthiness_are_ieee(stage_one_only, dtype):
    # bool(nan) is True: logical_not(nan) is False and logical_and(nan, 1) is
    # True. Both signed zeros are the only falsy codes; subnormals are truthy.
    tiny = SMALLEST_SUBNORMAL[dtype]
    x = torch.tensor(
        [0.0, -0.0, NAN, -NAN, INF, -INF, 1.0, -2.5, tiny, -tiny], dtype=dtype
    )
    expected = NanAndTruth()(x)
    assert expected[0][2] and not expected[1][2] and expected[2][2]
    assert expected[2][8] and expected[2][9]
    _assert_outputs_bitwise(_compile_real(NanAndTruth())(x), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
    ids=str,
)
def test_isnan_of_fp8_storage_formats(stage_one_only, dtype):
    # The fp8 formats have NaN codes that do not follow the IEEE encoding rule;
    # they are classified on their exact F32 widening.
    model = Unary(torch.isnan)
    x = torch.tensor([1.0, NAN, 0.0, -0.0, -1.5]).to(dtype)
    _assert_outputs_bitwise(_compile_real(model)(x), model(x))
