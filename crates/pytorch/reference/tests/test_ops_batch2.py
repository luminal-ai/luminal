"""Translator batch 2: movement, reductions, creation, where/clamp, softmax,
embedding, layer norm, power, comparisons."""

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


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(2, 6)


class ReshapeNeg1(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 4)


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)


class SliceCols(nn.Module):
    def forward(self, x):
        return x[:, 1:3]


class SelectRow(nn.Module):
    def forward(self, x):
        return x[1]


class Expand(nn.Module):
    def forward(self, x):
        return x.expand(3, 4)


class Repeat(nn.Module):
    def forward(self, x):
        return x.repeat(2, 1)


class Stack(nn.Module):
    def forward(self, a, b):
        return torch.stack([a, b], dim=0)


class SumKeepdim(nn.Module):
    def forward(self, x):
        return x.sum(dim=1, keepdim=True)


class ArgmaxDim(nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=1)


class VarDim(nn.Module):
    def forward(self, x):
        return torch.var(x, dim=1)


class CumsumDim(nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=1)


class Full(nn.Module):
    def forward(self, x):
        return torch.full((2, 3), 1.5)


class Arange(nn.Module):
    def forward(self, x):
        return torch.arange(6)


class Where(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x, torch.zeros_like(x))


class Clamp(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-0.5, max=0.5)


class SoftmaxDim(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=1)


class LogSoftmaxDim(nn.Module):
    def forward(self, x):
        return torch.log_softmax(x, dim=1)


class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(7, 4)

    def forward(self, idx):
        return self.emb(idx)


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)

    def forward(self, x):
        return self.ln(x)


class PowScalar(nn.Module):
    def forward(self, x):
        return x**2.0


class EqScalar(nn.Module):
    def forward(self, x):
        return x == 1.5


class LtTensor(nn.Module):
    def forward(self, a, b):
        return a < b


def test_reshape() -> None:
    _check(Reshape(), torch.randn(3, 4))


def test_reshape_neg1() -> None:
    _check(ReshapeNeg1(), torch.randn(3, 4))


def test_flatten() -> None:
    _check(Flatten(), torch.randn(3, 4))


def test_slice_cols() -> None:
    _check(SliceCols(), torch.randn(3, 5))


def test_select_row() -> None:
    _check(SelectRow(), torch.randn(3, 4))


def test_expand() -> None:
    _check(Expand(), torch.randn(1, 4))


def test_repeat() -> None:
    _check(Repeat(), torch.randn(3, 4))


def test_stack() -> None:
    _check(Stack(), torch.randn(3, 4), torch.randn(3, 4))


def test_sum_keepdim() -> None:
    _check(SumKeepdim(), torch.randn(3, 4))


def test_argmax_dim() -> None:
    _check(ArgmaxDim(), torch.randn(3, 4))


def test_var_dim() -> None:
    _check(VarDim(), torch.randn(3, 4))


def test_cumsum_dim() -> None:
    _check(CumsumDim(), torch.randn(3, 4))


def test_full() -> None:
    _check(Full(), torch.randn(1))


def test_arange() -> None:
    _check(Arange(), torch.randn(1))


def test_where() -> None:
    _check(Where(), torch.randn(3, 4))


def test_clamp() -> None:
    _check(Clamp(), torch.randn(3, 4))


def test_softmax_dim() -> None:
    _check(SoftmaxDim(), torch.randn(3, 4))


def test_log_softmax_dim() -> None:
    _check(LogSoftmaxDim(), torch.randn(3, 4), atol=1e-4)


def test_embedding() -> None:
    _check(Embedding(), torch.tensor([0, 3, 6, 1]))


def test_layer_norm() -> None:
    _check(LayerNorm(), torch.randn(3, 4), atol=1e-4)


def test_pow_scalar() -> None:
    _check(PowScalar(), torch.rand(3, 4) + 0.5)


def test_eq_scalar() -> None:
    _check(EqScalar(), torch.randn(3, 4))


def test_lt_tensor() -> None:
    _check(LtTensor(), torch.randn(3, 4), torch.randn(3, 4))


def test_max_dim_values_and_indices() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4)
    model = nn.Identity()
    compiled = torch.compile(model, backend=luminal_reference)
    # max.dim is exercised through the exported tuple-returning wrapper.
    class MaxDim(nn.Module):
        def forward(self, x):
            return torch.max(x, dim=1)

    out = torch.compile(MaxDim(), backend=luminal_reference)(x)
    eager = torch.max(x, dim=1)
    assert torch.allclose(out.values, eager.values)
    assert torch.equal(out.indices, eager.indices)
    _ = compiled
