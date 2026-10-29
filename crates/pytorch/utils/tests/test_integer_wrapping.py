"""PyTorch integer arithmetic wraps modulo 2^w.

Eager CPU computes int32/int64 `+`, `-`, `*`, `neg`, `abs` and `sum` in
two's complement: an overflow wraps, `neg(MIN) == MIN` and `abs(MIN) == MIN`
(de facto, pytorch/pytorch#108520). The translator lowers those ops to the
wrapping family, so overflowing programs compile and match eager exactly.
They run unattested: the exact Int ops would need value-range proofs.
"""

import pytest
import torch
from backend_test_utils import luminal_backend

WIDE = (
    pytest.param(torch.int32, id="int32"),
    pytest.param(torch.int64, id="int64"),
)


def _edges(dtype: torch.dtype) -> torch.Tensor:
    info = torch.iinfo(dtype)
    return torch.tensor(
        [info.min, info.min + 1, -7, -1, 0, 1, 3, info.max - 1, info.max], dtype=dtype
    )


class Binary(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.fn(a, b)


class Unary(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.fn(a)


def _run(model, *inputs) -> None:
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    actual, expected = compiled(*inputs), model(*inputs)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"


BINARY = (
    pytest.param(lambda a, b: a + b, id="add"),
    pytest.param(lambda a, b: a - b, id="sub"),
    pytest.param(lambda a, b: torch.rsub(a, b), id="rsub"),
    pytest.param(lambda a, b: torch.add(a, b, alpha=3), id="add_alpha"),
    pytest.param(lambda a, b: a * b, id="mul"),
    pytest.param(lambda a, b: a * 2 + b, id="mul_scalar_add"),
)


@pytest.mark.parametrize("dtype", WIDE)
@pytest.mark.parametrize("fn", BINARY)
def test_wide_integer_binary_ops_wrap_like_eager(stage_one_only, fn, dtype):
    a = _edges(dtype)
    _run(Binary(fn), a, a.flip(0))


UNARY = (
    pytest.param(torch.neg, id="neg"),
    pytest.param(torch.abs, id="abs"),
    pytest.param(torch.square, id="square"),
    pytest.param(lambda t: t**5, id="pow5"),
)


@pytest.mark.parametrize("dtype", WIDE)
@pytest.mark.parametrize("fn", UNARY)
def test_wide_integer_unary_ops_wrap_like_eager(stage_one_only, fn, dtype):
    _run(Unary(fn), _edges(dtype))


@pytest.mark.parametrize("dtype", WIDE)
def test_integer_sums_wrap_like_eager(stage_one_only, dtype):
    """int32 sums promote to int64; int64 sums wrap modulo 2^64."""
    info = torch.iinfo(dtype)
    x = torch.tensor([[info.max, info.max, 5], [info.min, -1, info.min]], dtype=dtype)
    _run(Unary(lambda t: t.sum()), x)
    _run(Unary(lambda t: t.sum(dim=1)), x)


SCANS = (
    pytest.param(lambda t: t.cumsum(0), id="cumsum"),
    pytest.param(lambda t: t.cumprod(0), id="cumprod"),
    pytest.param(lambda t: t.prod(), id="prod"),
    pytest.param(lambda t: torch.dot(t, t.flip(0)), id="dot"),
)


@pytest.mark.parametrize("dtype", WIDE)
@pytest.mark.parametrize("fn", SCANS)
def test_integer_scans_products_and_dots_wrap_like_eager(stage_one_only, fn, dtype):
    """Scans, products and dots accumulate modulo 2^w (int32 cumsum,
    cumprod and prod promote to int64, as eager does)."""
    _run(Unary(fn), _edges(dtype))


def test_narrow_integers_keep_wrapping(stage_one_only):
    """8-bit integer arithmetic already wrapped through the plain ops."""
    a = torch.tensor([127, -128, 100, -3], dtype=torch.int8)
    _run(Binary(lambda x, y: x + y), a, a)
    _run(Binary(lambda x, y: x * y), a, a)
    u = torch.tensor([255, 0, 128, 7], dtype=torch.uint8)
    _run(Binary(lambda x, y: x + y), u, u)


MATMULS = (
    pytest.param(lambda a, b: a @ b, id="mm"),
    pytest.param(lambda a, b: torch.bmm(a[None], b[None]), id="bmm"),
    pytest.param(lambda a, b: a[0] @ b, id="vector_matmul"),
    pytest.param(lambda a, b: torch.addmm(a, a, b, beta=2, alpha=3), id="addmm"),
    pytest.param(lambda a, b: torch.addmv(a[0], a, b[0], beta=2, alpha=3), id="addmv"),
    pytest.param(lambda a, b: torch.addbmm(a, a[None], b[None]), id="addbmm"),
    pytest.param(lambda a, b: torch.nn.functional.linear(a, b, a[0]), id="linear"),
)


@pytest.mark.parametrize("dtype", WIDE)
@pytest.mark.parametrize("fn", MATMULS)
def test_integer_matmuls_wrap_like_eager(stage_one_only, fn, dtype):
    """Integer matmuls accumulate in the operand dtype and wrap."""
    a = _edges(dtype).reshape(3, 3)
    _run(Binary(fn), a, a.flip(0))


def _scatter_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    info = torch.iinfo(dtype)
    data = torch.tensor([info.max, info.min, 2**24, -7], dtype=dtype)
    updates = torch.tensor([1, -1, 1, info.max, 3, -5], dtype=dtype)
    return data, updates


INDEX = torch.tensor([0, 1, 2, 0, 3, 3])

SCATTERS = (
    pytest.param(lambda t, u: t.scatter_add(0, INDEX, u), id="scatter_add"),
    pytest.param(lambda t, u: t.index_add(0, INDEX, u), id="index_add"),
    pytest.param(
        lambda t, u: t.index_put((INDEX,), u, accumulate=True), id="index_put_acc"
    ),
    pytest.param(lambda t, u: t.scatter_reduce(0, INDEX, u, "sum"), id="sum"),
    pytest.param(lambda t, u: t.scatter_reduce(0, INDEX, u, "prod"), id="prod"),
    pytest.param(lambda t, u: t.scatter_reduce(0, INDEX, u, "mean"), id="mean"),
    pytest.param(
        lambda t, u: t.scatter_reduce(0, INDEX, u, "mean", include_self=False),
        id="mean_exclude_self",
    ),
    pytest.param(lambda t, u: t.scatter_reduce(0, INDEX, u, "amax"), id="amax"),
    pytest.param(lambda t, u: t.scatter_reduce(0, INDEX, u, "amin"), id="amin"),
)


@pytest.mark.parametrize("dtype", WIDE)
@pytest.mark.parametrize("fn", SCATTERS)
def test_integer_scatter_accumulation_wraps_like_eager(stage_one_only, fn, dtype):
    """Scatter accumulation runs in the data dtype: exact past 2^24 and
    wrapping on overflow; an integer mean floor-divides."""
    _run(Binary(fn), *_scatter_inputs(dtype))
