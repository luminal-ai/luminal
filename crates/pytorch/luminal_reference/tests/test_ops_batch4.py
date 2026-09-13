"""Translator batch 4/5: pooling, conv, norms, index/scatter, movement, special."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import luminal_reference


def _check(model: nn.Module, *inputs: torch.Tensor, atol: float = 1e-4) -> None:
    torch.manual_seed(0)
    eager = model(*inputs)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(*inputs)
    assert out.shape == eager.shape, f"{out.shape} != {eager.shape}"
    assert torch.allclose(out.to(torch.float32), eager.to(torch.float32), atol=atol), (
        f"{out} != {eager}"
    )


class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=3, stride=2)


class AdaptiveAvgPool(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (2, 2))


class MaxPool(nn.Module):
    def forward(self, x):
        return torch.max_pool2d(x, kernel_size=2, stride=2)


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv(x)


class GroupedConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=2)
    def forward(self, x):
        return self.conv(x)


class BatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
    def forward(self, x):
        return self.bn(x)


class GroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(2, 4)
    def forward(self, x):
        return self.gn(x)


class Flip(nn.Module):
    def forward(self, x):
        return torch.flip(x, [1])


class Diagonal(nn.Module):
    def forward(self, x):
        return torch.diagonal(x, dim1=1, dim2=2)


class Unfold(nn.Module):
    def forward(self, x):
        return F.unfold(x, kernel_size=2, stride=2)


class Narrow(nn.Module):
    def forward(self, x):
        return x.narrow(1, 1, 2)


class Unbind(nn.Module):
    def forward(self, x):
        return torch.unbind(x, dim=1)


class Split(nn.Module):
    def forward(self, x):
        return torch.split(x, 2, dim=1)


class ConstantPad(nn.Module):
    def forward(self, x):
        return F.pad(x, (1, 2, 0, 1), value=1.5)


class RepeatInterleave(nn.Module):
    def forward(self, x):
        return torch.repeat_interleave(x, 2, dim=1)


class IndexSelect(nn.Module):
    def forward(self, x, idx):
        return torch.index_select(x, 1, idx)


class GatherOp(nn.Module):
    def forward(self, x, idx):
        return torch.gather(x, 1, idx)


class ScatterOp(nn.Module):
    def forward(self, x, idx, src):
        return torch.scatter(x, 1, idx, src)


class ScatterAdd(nn.Module):
    def forward(self, x, idx, src):
        return torch.scatter_add(x, 1, idx, src)


class MaskedScatter(nn.Module):
    def forward(self, x, mask, src):
        return x.masked_scatter(mask, src)


class Lgamma(nn.Module):
    def forward(self, x):
        return torch.lgamma(x)


class Digamma(nn.Module):
    def forward(self, x):
        return torch.digamma(x)


class Erfcx(nn.Module):
    def forward(self, x):
        return torch.special.erfcx(x)


class I0(nn.Module):
    def forward(self, x):
        return torch.special.i0(x)


class Erfinv(nn.Module):
    def forward(self, x):
        return torch.erfinv(x)


class LogCumsumExp(nn.Module):
    def forward(self, x):
        return torch.logcumsumexp(x, dim=1)


class ChebyshevT(nn.Module):
    def forward(self, x):
        return torch.special.chebyshev_polynomial_t(x, 2.0)


def test_avg_pool() -> None:
    _check(AvgPool(), torch.randn(2, 3, 8, 8))


def test_adaptive_avg_pool() -> None:
    _check(AdaptiveAvgPool(), torch.randn(2, 3, 7, 7))


def test_max_pool_values() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    compiled = torch.compile(MaxPool(), backend=luminal_reference)
    out = compiled(x)
    eager = F.max_pool2d(x, 2, 2)
    assert out.shape == eager.shape and torch.allclose(out, eager, atol=1e-5)


def test_conv2d() -> None:
    _check(Conv(), torch.randn(2, 3, 8, 8))


def test_grouped_conv2d() -> None:
    _check(GroupedConv(), torch.randn(2, 4, 6, 6))


def test_batch_norm_eval() -> None:
    torch.manual_seed(0)
    model = BatchNorm().eval()
    x = torch.randn(2, 3, 4, 4)
    eager = model(x)
    compiled = torch.compile(model, backend=luminal_reference)
    out = compiled(x)
    assert out.shape == eager.shape
    assert torch.allclose(out, eager, atol=1e-4)


def test_group_norm() -> None:
    _check(GroupNorm(), torch.randn(2, 4, 4, 4))


def test_flip() -> None:
    _check(Flip(), torch.randn(2, 4))


def test_diagonal() -> None:
    _check(Diagonal(), torch.randn(2, 4, 4))


def test_unfold() -> None:
    _check(Unfold(), torch.randn(1, 2, 6, 6))


def test_narrow() -> None:
    _check(Narrow(), torch.randn(2, 5))


def test_constant_pad() -> None:
    _check(ConstantPad(), torch.randn(1, 2, 3, 3))


def test_repeat_interleave() -> None:
    _check(RepeatInterleave(), torch.randn(2, 3))


def test_index_select() -> None:
    _check(IndexSelect(), torch.randn(3, 5), torch.tensor([0, 2, 4]))


def test_gather() -> None:
    x = torch.randn(2, 4)
    idx = torch.tensor([[0, 2], [3, 1]])
    _check(GatherOp(), x, idx)


def test_scatter() -> None:
    x = torch.zeros(2, 4)
    idx = torch.tensor([[0, 2], [3, 1]])
    src = torch.randn(2, 2)
    _check(ScatterOp(), x, idx, src)


def test_scatter_add() -> None:
    x = torch.zeros(2, 4)
    idx = torch.tensor([[0, 2], [3, 1]])
    src = torch.randn(2, 2)
    _check(ScatterAdd(), x, idx, src)


def test_masked_scatter() -> None:
    x = torch.zeros(2, 4)
    mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
    src = torch.randn(4)
    _check(MaskedScatter(), x, mask, src)


def test_lgamma() -> None:
    _check(Lgamma(), torch.rand(2, 4) * 3.0 + 0.5)


def test_digamma() -> None:
    _check(Digamma(), torch.rand(2, 4) * 3.0 + 0.5, atol=1e-3)


def test_erfcx() -> None:
    _check(Erfcx(), torch.randn(2, 4), atol=1e-4)


def test_i0() -> None:
    _check(I0(), torch.randn(2, 4) * 2.0, atol=1e-4)


def test_erfinv() -> None:
    _check(Erfinv(), torch.rand(2, 4) * 1.6 - 0.8, atol=1e-3)


def test_logcumsumexp() -> None:
    _check(LogCumsumExp(), torch.randn(2, 4), atol=1e-4)


def test_chebyshev_t() -> None:
    _check(ChebyshevT(), torch.randn(2, 4), atol=1e-4)
