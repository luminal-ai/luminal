"""Regression coverage for PT2 binary elementwise lowerings."""

import importlib
import os

import pytest
import torch
from backend_test_utils import compile_for_test as luminal_compile
from backend_test_utils import luminal_backend, selected_backend


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


@pytest.fixture
def stage_one_only(monkeypatch):
    """The real backend path, pinned to its first stage: the undecomposed
    graph must translate as it is. Re-decomposing the whole graph (the
    fallback for an unsupported op) fails the test instead of hiding a
    missing arm."""
    native = importlib.import_module(f"luminal_{selected_backend()}.backend")

    def refuse_decomposition():
        raise AssertionError("stage 1 refused the graph; it was re-decomposed")

    monkeypatch.setattr(native, "_decomp_table", refuse_decomposition)


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
