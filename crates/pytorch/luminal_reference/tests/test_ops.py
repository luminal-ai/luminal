"""Operator matrix: elementwise, unary, matmul, movement, reductions."""

import torch
import torch.nn as nn

import luminal_reference


def _check(model: nn.Module, *inputs: torch.Tensor, atol: float = 1e-5) -> None:
    torch.manual_seed(0)
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(*inputs)
    assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
    assert torch.allclose(out, eager, atol=atol), f"{out} != {eager}"


class Add(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class AddScalar(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 2.0


class MulScalar(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 3.0


class BroadcastAdd(nn.Module):
    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


class Sub(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b


class Div(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a / b


class Exp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)


class Log(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)


class Sqrt(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Relu(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class Gelu(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


class Matmul(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b


class Transpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1)


class Sum(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class SumDim(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)


class MeanDim(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class Amax(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.amax(x, dim=1)


class Cat(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=1)


class Floor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


class Ceil(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ceil(x)


class Trunc(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.trunc(x)


class RoundTensor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)


class ToInt(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32)


class ToLong(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.long()


def test_add() -> None:
    _check(Add(), torch.randn(3, 4), torch.randn(3, 4))


def test_add_scalar() -> None:
    _check(AddScalar(), torch.randn(3, 4))


def test_mul_scalar() -> None:
    _check(MulScalar(), torch.randn(3, 4))


def test_broadcast_add() -> None:
    _check(BroadcastAdd(), torch.randn(3, 4), torch.randn(4))


def test_sub() -> None:
    _check(Sub(), torch.randn(3, 4), torch.randn(3, 4))


def test_div() -> None:
    _check(Div(), torch.randn(3, 4), torch.rand(3, 4) + 0.5)


def test_exp() -> None:
    _check(Exp(), torch.randn(3, 4))


def test_log() -> None:
    _check(Log(), torch.rand(3, 4) + 0.5)


def test_sqrt() -> None:
    _check(Sqrt(), torch.rand(3, 4) + 0.5)


def test_sigmoid() -> None:
    _check(Sigmoid(), torch.randn(3, 4))


def test_tanh() -> None:
    _check(Tanh(), torch.randn(3, 4))


def test_relu() -> None:
    _check(Relu(), torch.randn(3, 4))


def test_gelu() -> None:
    _check(Gelu(), torch.randn(3, 4), atol=1e-5)


def test_matmul() -> None:
    _check(Matmul(), torch.randn(3, 4), torch.randn(4, 5))


def test_transpose() -> None:
    _check(Transpose(), torch.randn(3, 4))


def test_sum() -> None:
    _check(Sum(), torch.randn(3, 4))


def test_sum_dim() -> None:
    _check(SumDim(), torch.randn(3, 4))


def test_mean_dim() -> None:
    _check(MeanDim(), torch.randn(3, 4))


def test_amax() -> None:
    _check(Amax(), torch.randn(3, 4))


def test_cat() -> None:
    _check(Cat(), torch.randn(3, 2), torch.randn(3, 5))


def _check_input(model: nn.Module, x: torch.Tensor) -> None:
    eager = model(x)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(x.clone())
    assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
    assert torch.equal(out.to(torch.float32), eager.to(torch.float32)), (
        f"{out} != {eager}"
    )


def test_floor() -> None:
    _check(Floor(), torch.randn(3, 4) * 4.0)


def test_ceil() -> None:
    _check(Ceil(), torch.randn(3, 4) * 4.0)


def test_trunc() -> None:
    _check(Trunc(), torch.randn(3, 4) * 4.0)


def test_round_half_to_even() -> None:
    x = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 1.4, 1.6])
    _check_input(RoundTensor(), x)


def test_to_int_truncates() -> None:
    x = torch.tensor([1.9, 1.5, 0.5, -0.5, -1.5, -1.9])
    _check_input(ToInt(), x)


def test_to_long_truncates() -> None:
    x = torch.tensor([1.9, -1.9, 2.0, -2.0])
    _check_input(ToLong(), x)
