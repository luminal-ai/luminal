"""Translator batch 7/8: elementwise specials, squeeze/triangle, reductions,
and the addmm family."""

import luminal_reference
import torch
import torch.nn as nn
import torch.nn.functional as F


def _check(model: nn.Module, *inputs: torch.Tensor, atol: float = 1e-4) -> None:
    torch.manual_seed(0)
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference.Compiler())
    out = compiled(*inputs)
    inputs = tuple(t.clone() for t in inputs)
    if isinstance(eager, (tuple, list)):
        assert isinstance(out, (tuple, list)) and len(out) == len(eager)
        for got, want in zip(out, eager):
            assert got.shape == want.shape, f"{got.shape} != {want.shape}"
            assert torch.allclose(
                got.to(torch.float32), want.to(torch.float32), atol=atol
            ), f"{got} != {want}"
    else:
        assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
        assert torch.allclose(
            out.to(torch.float32), eager.to(torch.float32), atol=atol
        ), f"{out} != {eager}"


class Atan2(nn.Module):
    def forward(self, y, x):
        return torch.atan2(y, x)


class Copysign(nn.Module):
    def forward(self, a, b):
        return torch.copysign(a, b)


class Fmax(nn.Module):
    def forward(self, a, b):
        return torch.fmax(a, b)


class Hypot(nn.Module):
    def forward(self, a, b):
        return torch.hypot(a, b)


class Exp2(nn.Module):
    def forward(self, x):
        return torch.exp2(x)


class Log2(nn.Module):
    def forward(self, x):
        return torch.log2(x.abs() + 1.0)


class IsNan(nn.Module):
    def forward(self, x):
        return torch.isnan(x)


class LeakyReLU(nn.Module):
    def forward(self, x):
        return F.leaky_relu(x, 0.1)


class BitwiseAnd(nn.Module):
    def forward(self, a, b):
        return torch.bitwise_and(a, b)


class BitwiseOr(nn.Module):
    def forward(self, a, b):
        return torch.bitwise_or(a, b)


class SqueezeDefault(nn.Module):
    def forward(self, x):
        return x.squeeze()


class SqueezeDims(nn.Module):
    def forward(self, x):
        return x.squeeze((0, 2))


class Tril(nn.Module):
    def forward(self, x):
        return torch.tril(x)


class Triu(nn.Module):
    def forward(self, x):
        return torch.triu(x, diagonal=1)


class AnyDim(nn.Module):
    def forward(self, x):
        return torch.any(x, dim=1)


class VarMean(nn.Module):
    def forward(self, x):
        return torch.var_mean(x, dim=1)


class Addmm(nn.Module):
    def forward(self, b, m1, m2):
        return torch.addmm(b, m1, m2)


class Addbmm(nn.Module):
    def forward(self, b, x, y):
        return torch.addbmm(b, x, y)


class Addmv(nn.Module):
    def forward(self, b, m, v):
        return torch.addmv(b, m, v)


def test_atan2() -> None:
    _check(Atan2(), torch.randn(4, 4), torch.randn(4, 4))


def test_copysign() -> None:
    _check(Copysign(), torch.randn(4, 4), torch.randn(4, 4))


def test_fmax() -> None:
    _check(Fmax(), torch.randn(4, 4), torch.randn(4, 4))


def test_hypot() -> None:
    _check(Hypot(), torch.randn(4, 4), torch.randn(4, 4))


def test_exp2() -> None:
    _check(Exp2(), torch.randn(4, 4))


def test_log2() -> None:
    _check(Log2(), torch.randn(4, 4))


def test_isnan() -> None:
    _check(IsNan(), torch.randn(4, 4))


def test_leaky_relu() -> None:
    _check(LeakyReLU(), torch.randn(4, 4))


def test_bitwise_and() -> None:
    a = torch.randn(4, 4) > 0
    b = torch.randn(4, 4) > 0
    _check(BitwiseAnd(), a, b)


def test_bitwise_or() -> None:
    a = torch.randn(4, 4) > 0
    b = torch.randn(4, 4) > 0
    _check(BitwiseOr(), a, b)


def test_squeeze_default() -> None:
    _check(SqueezeDefault(), torch.randn(1, 3, 1, 4))


def test_squeeze_dims() -> None:
    _check(SqueezeDims(), torch.randn(1, 3, 1, 4))


def test_tril() -> None:
    _check(Tril(), torch.randn(4, 5))


def test_triu() -> None:
    _check(Triu(), torch.randn(4, 5))


def test_any_dim() -> None:
    _check(AnyDim(), torch.randn(3, 4))


def test_var_mean() -> None:
    _check(VarMean(), torch.randn(3, 4))


def test_addmm() -> None:
    _check(Addmm(), torch.randn(4), torch.randn(4, 5), torch.randn(5, 4))


def test_addbmm() -> None:
    _check(Addbmm(), torch.randn(4, 4), torch.randn(3, 4, 5), torch.randn(3, 5, 4))


def test_addmv() -> None:
    _check(Addmv(), torch.randn(4), torch.randn(4, 5), torch.randn(5))
