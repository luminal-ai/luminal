"""Translator batch 3: unary special functions and division modes."""

import torch
import torch.nn as nn

import luminal_reference


def _check(model: nn.Module, *inputs: torch.Tensor, atol: float = 1e-5) -> None:
    torch.manual_seed(0)
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(*inputs)
    assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
    assert torch.allclose(out.to(torch.float32), eager.to(torch.float32), atol=atol), (
        f"{out} != {eager}"
    )


def _check_exact(model: nn.Module, *inputs: torch.Tensor) -> None:
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(*inputs)
    assert torch.equal(out.to(torch.float32), eager.to(torch.float32)), f"{out} != {eager}"


class ExpM1(nn.Module):
    def forward(self, x):
        return torch.expm1(x)


class Log1P(nn.Module):
    def forward(self, x):
        return torch.log1p(x)


class Log10(nn.Module):
    def forward(self, x):
        return torch.log10(x)


class Rsqrt(nn.Module):
    def forward(self, x):
        return torch.rsqrt(x)


class Sinh(nn.Module):
    def forward(self, x):
        return torch.sinh(x)


class Cosh(nn.Module):
    def forward(self, x):
        return torch.cosh(x)


class Tan(nn.Module):
    def forward(self, x):
        return torch.tan(x)


class Asin(nn.Module):
    def forward(self, x):
        return torch.asin(x)


class Acos(nn.Module):
    def forward(self, x):
        return torch.acos(x)


class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class Asinh(nn.Module):
    def forward(self, x):
        return torch.asinh(x)


class Acosh(nn.Module):
    def forward(self, x):
        return torch.acosh(x)


class Atanh(nn.Module):
    def forward(self, x):
        return torch.atanh(x)


class Hardtanh(nn.Module):
    def forward(self, x):
        return torch.nn.functional.hardtanh(x, -0.5, 0.5)


class Elu(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x)


class Erf(nn.Module):
    def forward(self, x):
        return torch.erf(x)


class Erfc(nn.Module):
    def forward(self, x):
        return torch.erfc(x)


class Sign(nn.Module):
    def forward(self, x):
        return torch.sign(x)


class Signbit(nn.Module):
    def forward(self, x):
        return torch.signbit(x)


class Ldexp(nn.Module):
    def forward(self, x, e):
        return torch.ldexp(x, e)


class FloorDivide(nn.Module):
    def forward(self, a, b):
        return a // b


class DivFloor(nn.Module):
    def forward(self, a, b):
        return torch.div(a, b, rounding_mode="floor")


def test_expm1() -> None:
    _check(ExpM1(), torch.randn(3, 4) * 0.5)


def test_log1p() -> None:
    _check(Log1P(), torch.rand(3, 4) + 0.1)


def test_log10() -> None:
    _check(Log10(), torch.rand(3, 4) + 0.1)


def test_rsqrt() -> None:
    _check(Rsqrt(), torch.rand(3, 4) + 0.5)


def test_sinh() -> None:
    _check(Sinh(), torch.randn(3, 4), atol=1e-4)


def test_cosh() -> None:
    _check(Cosh(), torch.randn(3, 4), atol=1e-4)


def test_tan() -> None:
    _check(Tan(), torch.randn(3, 4) * 0.5, atol=1e-4)


def test_asin() -> None:
    _check(Asin(), torch.rand(3, 4) * 1.6 - 0.8, atol=1e-4)


def test_acos() -> None:
    _check(Acos(), torch.rand(3, 4) * 1.6 - 0.8, atol=1e-4)


def test_atan() -> None:
    _check(Atan(), torch.randn(3, 4) * 2.0, atol=1e-4)


def test_asinh() -> None:
    _check(Asinh(), torch.randn(3, 4), atol=1e-4)


def test_acosh() -> None:
    _check(Acosh(), torch.rand(3, 4) * 2.0 + 1.01, atol=1e-4)


def test_atanh() -> None:
    _check(Atanh(), torch.rand(3, 4) * 1.6 - 0.8, atol=1e-4)


def test_hardtanh() -> None:
    _check(Hardtanh(), torch.randn(3, 4))


def test_elu() -> None:
    _check(Elu(), torch.randn(3, 4), atol=1e-4)


def test_erf() -> None:
    _check(Erf(), torch.randn(3, 4), atol=1e-5)


def test_erfc() -> None:
    _check(Erfc(), torch.randn(3, 4), atol=1e-5)


def test_sign() -> None:
    _check_exact(Sign(), torch.randn(3, 4) * 3.0)


def test_signbit() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0, -0.0, 2.5, -2.5])
    _check_exact(Signbit(), x)


def test_ldexp() -> None:
    _check(
        Ldexp(),
        torch.rand(3, 4) + 0.5,
        torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 1]], dtype=torch.int32),
    )


def test_floor_divide_float() -> None:
    a = torch.tensor([-7.5, 7.5, -8.0, 8.0], dtype=torch.float32)
    b = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float32)
    _check_exact(FloorDivide(), a, b)


def test_div_floor_float() -> None:
    a = torch.tensor([-7.5, 7.5, -8.0, 8.0], dtype=torch.float32)
    b = torch.tensor([2.0, 2.0, 3.0, 3.0], dtype=torch.float32)
    _check_exact(DivFloor(), a, b)
