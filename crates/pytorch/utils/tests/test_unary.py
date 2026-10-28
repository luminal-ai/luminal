from typing import Callable

import pytest
import torch
import torch._dynamo
from backend_test_utils import luminal_backend
from test_models import (
    AbsAllNegativeModel,
    AbsInExpressionModel,
    AbsTestModel,
    ClipMaxOnlyTestModel,
    ClipMinOnlyTestModel,
    ClipTestModel,
    NegAllPositiveModel,
    NegInExpressionModel,
    NegTestModel,
    ReluAllNegativeModel,
    ReluInExpressionModel,
    ReluTestModel,
    SigmoidInExpressionModel,
    SigmoidTestModel,
    TanhInExpressionModel,
    TanhTestModel,
)

# ── Sigmoid ──────────────────────────────────────────────────────────────────


def test_sigmoid(device):
    model = SigmoidTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1  # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_sigmoid_in_expression(device):
    model = SigmoidInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Tanh ─────────────────────────────────────────────────────────────────────


def test_tanh(device):
    model = TanhTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_tanh_in_expression(device):
    model = TanhInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device)
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Relu ─────────────────────────────────────────────────────────────────────


def test_relu(device):
    model = ReluTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1  # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_relu_all_negative(device):
    model = ReluAllNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = -torch.rand((5, 5), device=device)  # all negative -> output all zeros
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_relu_in_expression(device):
    model = ReluInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Abs ──────────────────────────────────────────────────────────────────────


def test_abs(device):
    model = AbsTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1  # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_abs_all_negative(device):
    model = AbsAllNegativeModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = -torch.rand((5, 5), device=device)  # all negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_abs_in_expression(device):
    model = AbsInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Neg ──────────────────────────────────────────────────────────────────────


def test_neg(device):
    model = NegTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1  # mixed positive/negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_neg_all_positive(device):
    model = NegAllPositiveModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device)  # all positive -> output all negative
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_neg_in_expression(device):
    model = NegInExpressionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 2 - 1
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Clip ──────────────────────────────────────────────────────────────────────


def test_clip(device):
    """Clip tensor values to [-0.5, 0.5]."""
    model = ClipTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 4 - 2  # range [-2, 2]
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_clip_min_only(device):
    """Clip tensor values to [0.0, +inf]."""
    model = ClipMinOnlyTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 4 - 2
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


def test_clip_max_only(device):
    """Clip tensor values to [-inf, 0.5]."""
    model = ClipMaxOnlyTestModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.rand((5, 5), device=device) * 4 - 2
    assert torch.allclose(model_compiled(x), model(x), atol=1e-5)


# ── Logit ────────────────────────────────────────────────────────────────────

_NAN, _INF = float("nan"), float("inf")
# 0 and 1 hit the poles; -0.5, 1.5 and the infinities lie outside [0, 1].
_LOGIT_INPUT = (0.0, 1.0, 0.05, 0.2, 0.5, 0.7, 0.95, -0.5, 1.5, _NAN, _INF, -_INF)


def _compile_logit_real_path(monkeypatch, eps):
    """Compile `torch.logit(x, eps)` through the real two-stage backend path,
    recording the ATen targets of every program handed to the translator."""
    translated = []
    save = torch.export.save

    def recording_save(program, *args, **kwargs):
        translated.append(
            {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
        )
        return save(program, *args, **kwargs)

    monkeypatch.setattr(torch.export, "save", recording_save)

    def backend(gm, example_inputs):
        return luminal_backend(gm, example_inputs, options={"search_iterations": 1})

    def logit(x):
        return torch.logit(x, eps)

    compiled = torch.compile(logit, backend=backend, fullgraph=True, dynamic=False)
    return compiled, translated


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float64), ids=("float32", "float64")
)
@pytest.mark.parametrize("eps", (None, 0.0, 0.1, 0.5, 0.6, 0.9, -0.1))
def test_logit_eps_clamps_low_bound_first(device, monkeypatch, eps, dtype):
    """ATen clamps as `x < lo ? lo : (x > hi ? hi : x)` with lo = eps and
    hi = 1 - eps. For eps > 0.5 the bounds cross, so inputs below lo give
    logit(lo) and every other input gives logit(hi); NaN stays NaN, and a
    negative eps clamps nothing."""
    compiled, translated = _compile_logit_real_path(monkeypatch, eps)
    x = torch.tensor(_LOGIT_INPUT, dtype=dtype)
    actual = compiled(x)
    # Stage 1 must lower the undecomposed op: a stage-2 decomposition would
    # bypass the `logit.default` lowering under test.
    assert translated == [{"aten.logit.default"}]
    torch.testing.assert_close(actual, torch.logit(x, eps), equal_nan=True)


@pytest.mark.parametrize(
    ("values", "dtype"),
    (([0, 1, 2, -1], torch.int64), ([True, False], torch.bool)),
    ids=("int64", "bool"),
)
@pytest.mark.parametrize("eps", (None, 0.6))
def test_logit_promotes_integral_input_to_float(
    device, monkeypatch, values, dtype, eps
):
    compiled, translated = _compile_logit_real_path(monkeypatch, eps)
    values = torch.tensor(values, dtype=dtype)
    actual = compiled(values)
    assert translated == [{"aten.logit.default"}]
    torch.testing.assert_close(actual, torch.logit(values, eps), equal_nan=True)


# ── Signbit ──────────────────────────────────────────────────────────────────


class SignbitModel(torch.nn.Module):
    def forward(self, x):
        return torch.signbit(x)


def test_signbit_reads_negative_infinity_and_zero_signs(device):
    """`-inf` is negative even though `1/-inf` is `-0.0`, and only `-0.0`
    of the zeros is. Zero-element inputs must keep compiling."""
    inf, nan = float("inf"), float("nan")
    values = [-inf, -2.5, -1.0, -0.0, 0.0, 1.0, 2.5, inf, nan]
    for dtype in (torch.float32, torch.float64):
        full = torch.tensor(values, dtype=dtype, device=device)
        empty = torch.empty(0, dtype=dtype, device=device)
        for x in (full, empty):
            torch._dynamo.reset()
            model = SignbitModel().to(device)
            model_compiled: Callable = torch.compile(model, backend=luminal_backend)
            actual = model_compiled(x)
            assert actual.dtype == torch.bool
            assert torch.equal(actual.cpu(), model(x).cpu()), (dtype, x)


# ── Sign, angle, and the odd functions that carry a sign ─────────────────────


class _Elementwise(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# Both zeros, tiny normals, values that are f32 subnormals (1e-45 rounds to
# 2**-149), units, the infinities and NaN.
_SIGN_EDGES = (-_INF, -1.0, -1e-30, -1e-45, -0.0, 0.0, 1e-45, 1e-30, 1.0, _INF, _NAN)


def _assert_ieee_close(actual, expected, *, rtol=0.0, atol=0.0):
    """Close values (NaN matching NaN) with the same dtype, and the same
    sign bit wherever the reference is a zero."""
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual)[zeros], torch.signbit(expected)[zeros]), (
        actual,
        expected,
    )


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float64), ids=("float32", "float64")
)
@pytest.mark.parametrize("op", (torch.sign, torch.sgn), ids=("sign", "sgn"))
def test_sign_and_sgn_are_exact_at_ieee_edges(device, op, dtype):
    """torch.sign is `(x > 0) - (x < 0)`: exactly +-1 for every nonzero
    value, subnormals and infinities included, and +0 for both zeros and
    for NaN. (Real `sgn` has no stage-1 arm; stage 2 decomposes it to
    `sign`.)"""
    x = torch.tensor(_SIGN_EDGES, dtype=dtype, device=device)
    compiled = torch.compile(_Elementwise(op), backend=luminal_backend)
    _assert_ieee_close(compiled(x).cpu(), op(x).cpu())


@pytest.mark.parametrize(
    "op",
    (torch.erf, torch.erfc, torch.nn.functional.gelu),
    ids=("erf", "erfc", "gelu"),
)
def test_erf_family_is_odd_through_an_exact_sign(stage_one_only, device, op):
    """erf multiplies its magnitude by the core `sign()`, and gelu is built
    on erf. The old `x / (|x| + 1e-10)` was NaN at +-inf (inf / inf), so
    erf(+-inf), erfc(+-inf) and gelu(inf) were NaN."""
    x = torch.tensor(_SIGN_EDGES + (-2.0, 0.5, 3.0), device=device)
    compiled = torch.compile(_Elementwise(op), backend=luminal_backend)
    # Abramowitz & Stegun 7.1.26 is accurate to 1.5e-7 absolute.
    _assert_ieee_close(compiled(x).cpu(), op(x).cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "op", (torch.atan, torch.asinh, torch.asin), ids=("atan", "asinh", "asin")
)
def test_odd_inverse_functions_carry_the_input_sign_at_ieee_edges(
    stage_one_only, device, op
):
    """These lowerings compute a magnitude from |x| and copy the sign of x
    back on. Multiplying by an approximate sign gave NaN at +-inf and lost
    a tiny input (atan(1e-30) was ~1e-50)."""
    values = _SIGN_EDGES + (-2.0, 0.5, 3.0)
    if op is torch.asin:
        values = tuple(v for v in values if not abs(v) > 1.0) + (-0.5,)
    x = torch.tensor(values, device=device)
    compiled = torch.compile(_Elementwise(op), backend=luminal_backend)
    _assert_ieee_close(compiled(x).cpu(), op(x).cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "x",
    (
        torch.tensor((-2.0, -0.0, 0.0, 3.0, -_INF, _INF, _NAN, -1e-45, 1e-45)),
        torch.tensor(
            (-2.0, -0.0, 0.0, 3.0, -_INF, _INF, _NAN, -1e-300, 1e-300),
            dtype=torch.float64,
        ),
        torch.tensor((-2, 0, 3)),
        torch.empty(0),
    ),
    ids=("float32", "float64", "int64", "empty"),
)
def test_angle_of_a_real_input_is_pi_below_zero_and_plus_zero_otherwise(
    stage_one_only, device, x
):
    """Real torch.angle is pi for x < 0 (-inf and negative subnormals
    included), +0 for every other ordered value (both zeros: unlike
    atan2(0, -0.0) = pi), and NaN for NaN. It used to be 0 everywhere."""
    x = x.to(device)
    compiled = torch.compile(_Elementwise(torch.angle), backend=luminal_backend)
    _assert_ieee_close(compiled(x).cpu(), torch.angle(x).cpu())
