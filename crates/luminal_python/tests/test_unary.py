from typing import Callable

import pytest
import torch
import torch._dynamo
from test_models import (
    SigmoidTestModel,
    SigmoidInExpressionModel,
    TanhTestModel,
    TanhInExpressionModel,
    ReluTestModel,
    ReluAllNegativeModel,
    ReluInExpressionModel,
    AbsTestModel,
    AbsAllNegativeModel,
    AbsInExpressionModel,
    NegTestModel,
    NegAllPositiveModel,
    NegInExpressionModel,
    ClipTestModel,
    ClipMinOnlyTestModel,
    ClipMaxOnlyTestModel,
)
from luminal import luminal_backend


class Exp2Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp2(x)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_exp2_integer_input_promotes_to_float32(device, dtype):
    if device.type != "cpu":
        pytest.skip("integer-input Exp2 is currently reference-backend only")
    model = Exp2Model().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x = torch.tensor(
        [-150, -149, -126, -2, -1, 0, 1, 30, 127, 128],
        dtype=dtype,
        device=device,
    )

    expected = model(x)
    actual = model_compiled(x)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)

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
