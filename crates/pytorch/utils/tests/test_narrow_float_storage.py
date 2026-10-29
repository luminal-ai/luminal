"""Narrow floats are storage in the translated graph.

Torch computes half-precision math in F32 (opmath) and computes exact ops
(negation, magnitude, rounding to an integer) at their own dtype. The
translator records both through explicit casts: it widens F16/BF16 to F32,
computes there, and rounds back once. For an exact op that is bit-identical
to torch, so these tests compare bit for bit, through the real backend's
undecomposed stage (no decomposition may hide a missing widening).
"""

import pytest
import torch
from backend_test_utils import luminal_backend

NARROW = (
    pytest.param(torch.float16, id="float16"),
    pytest.param(torch.bfloat16, id="bfloat16"),
)


def _values(dtype: torch.dtype) -> torch.Tensor:
    """Signed zeros, ties, fractions on both sides of zero, large and
    subnormal magnitudes, both infinities and NaN."""
    finfo = torch.finfo(dtype)
    values = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        0.3,
        -0.7,
        3.75,
        -3.75,
        float(finfo.max),
        -float(finfo.max),
        float(finfo.tiny) / 4,
        -float(finfo.tiny) / 4,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    return torch.tensor(values, dtype=dtype)


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    view = width[expected.element_size()]
    # NaN payloads may differ; every NaN must stay NaN.
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(view), expected[~nan].view(view))


class Unary(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


EXACT_OPS = (
    pytest.param(torch.abs, id="abs"),
    pytest.param(torch.neg, id="neg"),
    pytest.param(torch.floor, id="floor"),
    pytest.param(torch.ceil, id="ceil"),
    pytest.param(torch.trunc, id="trunc"),
    pytest.param(torch.round, id="round"),
    pytest.param(lambda x: torch.where(x > 0, x, -x), id="where_neg"),
)


@pytest.mark.parametrize("dtype", NARROW)
@pytest.mark.parametrize("fn", EXACT_OPS)
def test_exact_ops_on_narrow_storage_match_bitwise(stage_one_only, fn, dtype):
    model = Unary(fn)
    x = _values(dtype)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    _assert_bitwise(compiled(x), model(x))


@pytest.mark.parametrize("dtype", NARROW)
@pytest.mark.parametrize(
    "target", (torch.int8, torch.int16, torch.uint8, torch.int32, torch.int64)
)
def test_narrow_float_to_integer_truncates_the_stored_value(
    stage_one_only, dtype, target
):
    model = Unary(lambda x: x.to(target))
    finite = _values(dtype)
    finite = finite[torch.isfinite(finite) & (finite.abs() < 100)]
    if target is torch.uint8:
        finite = finite.abs()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    actual = compiled(finite)
    expected = model(finite)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ),
)
def test_abs_clears_the_sign_of_negative_zero(stage_one_only, dtype):
    """A float's magnitude clears its sign bit at every width."""
    model = Unary(torch.abs)
    x = torch.tensor([-0.0, 0.0, -1.5, float("-inf")], dtype=dtype)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    _assert_bitwise(compiled(x), model(x))


def _ordered_values(dtype: torch.dtype) -> torch.Tensor:
    """Distinct finite values with both zeros' neighbours and a tie."""
    return torch.tensor(
        [[3.5, -0.0, 2.25, -7.0, 0.5, 2.25], [1.0, -2.5, 8.0, 0.25, -1.0, 4.0]],
        dtype=dtype,
    )


ORDERING_OPS = (
    pytest.param(lambda x: torch.sort(x, dim=1), id="sort"),
    pytest.param(
        lambda x: torch.sort(x, dim=1, stable=True, descending=True), id="sort_stable"
    ),
    pytest.param(lambda x: torch.argsort(x, dim=1), id="argsort"),
    pytest.param(lambda x: torch.topk(x, 3, dim=1), id="topk"),
    pytest.param(torch.min, id="min"),
    pytest.param(lambda x: torch.amin(x, dim=1), id="amin"),
    pytest.param(lambda x: torch.min(x, dim=1), id="min_dim"),
    pytest.param(lambda x: torch.median(x, dim=1), id="median_dim"),
    pytest.param(lambda x: torch.nanmedian(x), id="nanmedian"),
    pytest.param(torch.signbit, id="signbit"),
    pytest.param(
        lambda x: torch.nn.functional.max_pool1d(
            x.unsqueeze(0), 2, return_indices=True
        ),
        id="max_pool1d_indices",
    ),
)


@pytest.mark.parametrize("dtype", NARROW)
@pytest.mark.parametrize("fn", ORDERING_OPS)
def test_ordering_ops_on_narrow_storage_match_bitwise(stage_one_only, fn, dtype):
    """Sorting, selecting and comparing are exact: the widened graph returns
    torch's values and indices bit for bit."""
    model = Unary(fn)
    x = _ordered_values(dtype)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    actual, expected = compiled(x), model(x)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    for a, e in zip(actual, expected, strict=True):
        if e.is_floating_point():
            _assert_bitwise(a, e)
        else:
            assert torch.equal(a, e)


class Accumulate(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return self.fn(x, src)


_INDEX = torch.tensor([0, 2, 0, 1, 2, 0])

ACCUMULATING_OPS = (
    pytest.param(lambda x, s: x.scatter_add(0, _INDEX, s), id="scatter_add"),
    pytest.param(
        lambda x, s: x.index_put((_INDEX,), s, accumulate=True),
        id="index_put_accumulate",
    ),
    pytest.param(
        lambda x, s: x.scatter_reduce(0, _INDEX, s, "sum"), id="scatter_reduce_sum"
    ),
    pytest.param(
        lambda x, s: x.scatter_reduce(0, _INDEX, s, "prod"), id="scatter_reduce_prod"
    ),
    pytest.param(
        lambda x, s: x.scatter_reduce(0, _INDEX, s, "mean", include_self=False),
        id="scatter_reduce_mean",
    ),
    pytest.param(
        lambda x, s: x.index_reduce(0, _INDEX, s, "prod"), id="index_reduce_prod"
    ),
    pytest.param(lambda x, s: x.put(_INDEX, s, accumulate=True), id="put_accumulate"),
)


@pytest.mark.parametrize("dtype", NARROW)
@pytest.mark.parametrize("fn", ACCUMULATING_OPS)
def test_accumulations_into_narrow_storage_round_each_step(stage_one_only, fn, dtype):
    """Each accumulation step computes in F32 and rounds to the storage
    dtype, as torch's sequential CPU kernels do."""
    model = Accumulate(fn)
    x = torch.tensor([1.5, -2.0, 0.375], dtype=dtype)
    src = torch.tensor([0.1, 3.0, -1.25, 2.5, 0.5, 7.0], dtype=dtype)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    actual, expected = compiled(x, src), model(x, src)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", NARROW)
def test_histc_counts_into_narrow_storage(stage_one_only, dtype):
    model = Unary(lambda x: torch.histc(x, bins=4, min=0, max=3))
    x = torch.tensor([0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0], dtype=dtype)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    _assert_bitwise(compiled(x), model(x))
