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
