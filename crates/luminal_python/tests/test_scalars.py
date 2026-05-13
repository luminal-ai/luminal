"""Tests for scalar (rank-0) tensor handling.

Most tests in this suite use ``torch.allclose`` which is silent about shape.
That hides discrepancies between PyTorch (where ``x.sum()`` is shape ``()``)
and luminal (where the same op may produce shape ``(1,)``). These tests
assert dtype, shape, AND values, so any rank-0 vs rank-1 drift fails loudly.

The tests cover:
 - Per-op rank-0 production (sum, max, mean, min, prod, indexing, constants)
 - Sequences of unsqueeze/squeeze/expand/reshape that round-trip through scalar
 - Scalars participating in arithmetic, comparisons, mod, where
 - Models that return scalars as their final output

Each test compiles a small ``nn.Module`` with the luminal backend and compares
to PyTorch eager.
"""

from typing import Callable

import pytest
import torch
import torch._dynamo

from luminal import luminal_backend


# ---------------------------------------------------------------------------
# Strict comparison helper: catches shape / dtype divergence in addition to
# value differences. This is the rigor that ``torch.allclose`` lacks.
# ---------------------------------------------------------------------------


def _strict_match(
    output: torch.Tensor,
    original: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    assert output.dtype == original.dtype, (
        f"dtype mismatch: luminal={output.dtype} vs eager={original.dtype}"
    )
    assert tuple(output.shape) == tuple(original.shape), (
        f"shape mismatch: luminal={tuple(output.shape)} vs "
        f"eager={tuple(original.shape)} (rank {output.dim()} vs {original.dim()})"
    )
    if output.numel() == 0:
        return
    if output.dtype.is_floating_point:
        assert torch.allclose(output, original, atol=atol, rtol=rtol), (
            f"value mismatch (max abs err: {(output - original).abs().max().item()})"
        )
    else:
        assert torch.equal(output, original), (
            f"value mismatch: luminal={output} vs eager={original}"
        )


def _run(
    model: torch.nn.Module, *inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eager_output, compiled_output) for matched comparison."""
    compiled: Callable = torch.compile(model, backend=luminal_backend)
    eager = model(*inputs)
    compiled_out = compiled(*inputs)
    return eager, compiled_out


# ---------------------------------------------------------------------------
# Section 1: Full reductions produce a rank-0 scalar.
# ---------------------------------------------------------------------------


class _SumAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _MaxAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max()


class _MinAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.min()


class _MeanAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class _ProdAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.prod()


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4), (2, 2, 3, 4)])
def test_sum_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_SumAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_max_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MaxAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_min_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MinAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_mean_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MeanAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_prod_all_produces_scalar(device: torch.device, shape: tuple) -> None:
    # Use values close to 1.0 so the product stays well-conditioned.
    x = torch.rand(shape, device=device) * 0.5 + 0.75
    eager, compiled = _run(_ProdAll(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 2: insert-dim / remove-dim sequences round-trip through scalar.
# ---------------------------------------------------------------------------


class _SumUnsqueezeSqueeze(torch.nn.Module):
    """sum -> () -> unsqueeze(0) -> (1,) -> squeeze(0) -> ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(0).squeeze(0)


class _SumDoubleUnsqueezeDoubleSqueeze(torch.nn.Module):
    """sum -> [u(0), u(0), s(0), s(0)] back to scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(0).unsqueeze(0).squeeze(0).squeeze(0)


class _UnsqueezeNegativeAxis(torch.nn.Module):
    """sum -> unsqueeze(-1) -> squeeze(-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().unsqueeze(-1).squeeze(-1)


class _NestedUnsqueezeSqueezeAll(torch.nn.Module):
    """sum -> u(0)*3 -> squeeze() (squeeze with no dim removes ALL size-1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = s.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return s.squeeze()


class _AlternatingUnsqueezeSqueeze(torch.nn.Module):
    """Insert and remove dims in alternation, ending at scalar."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = s.unsqueeze(0)  # (1,)
        s = s.squeeze(0)  # ()
        s = s.unsqueeze(0).unsqueeze(0)  # (1, 1)
        s = s.squeeze(-1).squeeze(-1)  # ()
        return s


class _ReduceKeepDimThenSqueeze(torch.nn.Module):
    """sum(keepdim=True) -> (1, 1) -> squeeze() -> ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=(0, 1), keepdim=True).squeeze()


class _ReshapeToFromScalar(torch.nn.Module):
    """sum -> reshape(()) -> reshape((1,)) -> reshape(()) — explicit scalar reshapes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        return s.reshape(()).reshape((1,)).reshape(())


class _UnsqueezeExpandSumBack(torch.nn.Module):
    """() -> (1,) -> (5,) (expand) -> sum back to ()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        return s.unsqueeze(0).expand(5).sum()


def test_unsqueeze_squeeze_roundtrip(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_SumUnsqueezeSqueeze(), x)
    _strict_match(compiled, eager)


def test_double_unsqueeze_double_squeeze(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_SumDoubleUnsqueezeDoubleSqueeze(), x)
    _strict_match(compiled, eager)


def test_unsqueeze_negative_axis(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_UnsqueezeNegativeAxis(), x)
    _strict_match(compiled, eager)


def test_nested_unsqueeze_squeeze_all(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_NestedUnsqueezeSqueezeAll(), x)
    _strict_match(compiled, eager)


def test_alternating_unsqueeze_squeeze(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_AlternatingUnsqueezeSqueeze(), x)
    _strict_match(compiled, eager)


def test_reduce_keepdim_then_squeeze_to_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReduceKeepDimThenSqueeze(), x)
    _strict_match(compiled, eager)


def test_reshape_to_and_from_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReshapeToFromScalar(), x)
    _strict_match(compiled, eager)


def test_unsqueeze_expand_sum_back(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_UnsqueezeExpandSumBack(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 3: Scalars broadcast against rank-N tensors in arithmetic chains.
# ---------------------------------------------------------------------------


class _NormalizeBySum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum()


class _CenterByMean(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean()


class _MinMaxScale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min() + 1e-6)


class _DoubleScalarBroadcast(torch.nn.Module):
    """Two independent scalar reductions multiplied, then broadcast onto x."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * (y.sum() * x.mean())


class _ScalarChainedArithmetic(torch.nn.Module):
    """Long chain of scalar ops: ((s+1) * 2 - 0.5) used to scale x."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.sum()
        s = (s + 1.0) * 2.0 - 0.5
        return x * s


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_normalize_by_sum(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) + 0.1
    eager, compiled = _run(_NormalizeBySum(), x)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_center_by_mean(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_CenterByMean(), x)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_minmax_scale(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MinMaxScale(), x)
    _strict_match(compiled, eager, atol=1e-4)


def test_double_scalar_broadcast(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    y = torch.rand((2, 5), device=device)
    eager, compiled = _run(_DoubleScalarBroadcast(), x, y)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_scalar_chained_arithmetic(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ScalarChainedArithmetic(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 4: 0-d tensor constants in the graph.
# ---------------------------------------------------------------------------


class _AddScalarTensorConst(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tensor(2.5).to(x.device)


class _MulScalarTensorConst(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tensor(0.5).to(x.device)


class _ClampWithScalarTensors(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(-0.25).to(x.device)
        hi = torch.tensor(0.75).to(x.device)
        return torch.clamp(x, lo, hi)


class _WhereWithScalarBranches(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0).to(x.device)
        one = torch.tensor(1.0).to(x.device)
        return torch.where(x > 0.5, one, zero)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_add_scalar_tensor_constant(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_AddScalarTensorConst(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mul_scalar_tensor_constant(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MulScalarTensorConst(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_clamp_with_scalar_tensors(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 2.0 - 1.0  # in [-1, 1]
    eager, compiled = _run(_ClampWithScalarTensors(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_where_with_scalar_branches(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_WhereWithScalarBranches(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 5: Comparisons that produce or consume scalars.
# ---------------------------------------------------------------------------


class _ScalarLtScalar(torch.nn.Module):
    """Compare two scalar reductions; result is a 0-d bool, cast to float."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x.sum() < y.sum()).float()


class _ThresholdByMean(torch.nn.Module):
    """tensor > scalar — scalar broadcasts in comparison."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > x.mean()).float()


class _MaskedByScalarThreshold(torch.nn.Module):
    """Use a scalar comparison as a mask back into the tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (x > x.mean()).to(x.dtype)


def test_scalar_lt_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    y = torch.rand((2, 5), device=device)
    eager, compiled = _run(_ScalarLtScalar(), x, y)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_threshold_by_mean(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ThresholdByMean(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_masked_by_scalar_threshold(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_MaskedByScalarThreshold(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 6: Mod with scalar RHS — exercises broadcasting through luminal Rem.
# ---------------------------------------------------------------------------


class _ModByScalarTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x % torch.tensor(3.0).to(x.device)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mod_by_scalar_tensor(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 10.0
    eager, compiled = _run(_ModByScalarTensor(), x)
    _strict_match(compiled, eager, atol=1e-4)


# ---------------------------------------------------------------------------
# Section 7: Indexing / select that produces a 0-d scalar.
# ---------------------------------------------------------------------------


class _Index1DToScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[0]


class _IndexAllDimsToScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1, 2, 3]


class _IndexThenAddScalarConst(torch.nn.Module):
    """Indexed scalar enters arithmetic with a scalar constant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[1, 2] + torch.tensor(7.0).to(x.device)


def test_index_1d_produces_scalar(device: torch.device) -> None:
    x = torch.rand((5,), device=device)
    eager, compiled = _run(_Index1DToScalar(), x)
    _strict_match(compiled, eager)


def test_index_all_dims_produces_scalar(device: torch.device) -> None:
    x = torch.rand((4, 5, 6), device=device)
    eager, compiled = _run(_IndexAllDimsToScalar(), x)
    _strict_match(compiled, eager)


def test_index_then_add_scalar_const(device: torch.device) -> None:
    x = torch.rand((4, 5), device=device)
    eager, compiled = _run(_IndexThenAddScalarConst(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 8: Models whose final output is a scalar.
# ---------------------------------------------------------------------------


class _ReturnScalarSum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _ReturnScalarFromIndex(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[0, 0]


class _ReturnDerivedScalar(torch.nn.Module):
    """Return scalar built from constant and reduction — no input shape leak."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(3.14).to(x.device) + 0.0 * x.sum()


def test_model_returns_scalar_sum(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnScalarSum(), x)
    _strict_match(compiled, eager)


def test_model_returns_scalar_from_index(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnScalarFromIndex(), x)
    _strict_match(compiled, eager)


def test_model_returns_derived_scalar(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReturnDerivedScalar(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 9: Mixed dtype scalars.
# ---------------------------------------------------------------------------


class _IntSumProducesIntScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _FloatScalarTimesIntTensor(torch.nn.Module):
    """Scalar float constant + int tensor — exercises promotion rules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * torch.tensor(0.5).to(x.device)


def test_int_sum_produces_int_scalar(device: torch.device) -> None:
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_IntSumProducesIntScalar(), x)
    _strict_match(compiled, eager)


def test_float_scalar_times_int_tensor(device: torch.device) -> None:
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_FloatScalarTimesIntTensor(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 10: Binary ops with INPUT 0-d (not reduction-derived).
#
# torch.compile may dispatch a 0-d graph input through aten.{op}.Tensor while
# routing a constant-folded 0-d through aten.{op}.Scalar. Both paths matter.
# ---------------------------------------------------------------------------


class _Add0dInputLhs(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return s + x


class _Add0dInputRhs(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return x + s


class _Sub0dInputLhs(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return s - x


class _Mul0dInputRhs(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return x * s


class _Div0dInputRhs(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return x / s


class _Mod0dInputRhs(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return x % s


class _Maximum0dInput(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(s, x)


class _Minimum0dInput(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.minimum(x, s)


class _PowNd0dExponent(torch.nn.Module):
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return torch.pow(x, e)


class _FloorDivide0dInput(torch.nn.Module):
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(x, d)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_add_input_0d_lhs(device: torch.device, shape: tuple) -> None:
    s = torch.tensor(2.5, device=device)
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_Add0dInputLhs(), s, x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4), (2, 3, 4)])
def test_add_input_0d_rhs(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    s = torch.tensor(2.5, device=device)
    eager, compiled = _run(_Add0dInputRhs(), x, s)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_sub_input_0d_lhs(device: torch.device, shape: tuple) -> None:
    s = torch.tensor(0.7, device=device)
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_Sub0dInputLhs(), s, x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mul_input_0d_rhs(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    s = torch.tensor(0.5, device=device)
    eager, compiled = _run(_Mul0dInputRhs(), x, s)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_div_input_0d_rhs(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    s = torch.tensor(2.0, device=device)
    eager, compiled = _run(_Div0dInputRhs(), x, s)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mod_input_0d_rhs(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 10.0
    s = torch.tensor(3.0, device=device)
    eager, compiled = _run(_Mod0dInputRhs(), x, s)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_maximum_input_0d(device: torch.device, shape: tuple) -> None:
    s = torch.tensor(0.5, device=device)
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_Maximum0dInput(), s, x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_minimum_input_0d(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    s = torch.tensor(0.5, device=device)
    eager, compiled = _run(_Minimum0dInput(), x, s)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_pow_nd_with_0d_exponent(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) + 0.1
    e = torch.tensor(2.0, device=device)
    eager, compiled = _run(_PowNd0dExponent(), x, e)
    _strict_match(compiled, eager, atol=1e-4)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_floor_divide_input_0d(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 10.0
    d = torch.tensor(3.0, device=device)
    eager, compiled = _run(_FloorDivide0dInput(), x, d)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 11: Pure 0-d ↔ 0-d arithmetic (no broadcasting required).
# ---------------------------------------------------------------------------


class _ScalarPlusScalar(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class _ScalarMulScalar(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class _ScalarPipelineMix(torch.nn.Module):
    """Pure scalar pipeline: (a + b) * (a - b)."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a + b) * (a - b)


def test_scalar_plus_scalar(device: torch.device) -> None:
    a = torch.tensor(1.5, device=device)
    b = torch.tensor(2.5, device=device)
    eager, compiled = _run(_ScalarPlusScalar(), a, b)
    _strict_match(compiled, eager)


def test_scalar_mul_scalar(device: torch.device) -> None:
    a = torch.tensor(1.5, device=device)
    b = torch.tensor(2.5, device=device)
    eager, compiled = _run(_ScalarMulScalar(), a, b)
    _strict_match(compiled, eager)


def test_scalar_only_pipeline(device: torch.device) -> None:
    a = torch.tensor(2.5, device=device)
    b = torch.tensor(1.5, device=device)
    eager, compiled = _run(_ScalarPipelineMix(), a, b)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 12: Comparisons producing 0-d bool from 0-d inputs.
# ---------------------------------------------------------------------------


class _GtInput0ds(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a > b).float()


class _GeInput0ds(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a >= b).float()


class _LeInput0ds(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a <= b).float()


class _EqInput0ds(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a == b).float()


class _NeInput0ds(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a != b).float()


class _ThresholdByInputScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (x > t).float()


class _MaskByScalarEq(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.where(s == 0, x, x * 2)


@pytest.mark.parametrize(
    "model_cls,a_val,b_val",
    [
        (_GtInput0ds, 0.7, 0.5),
        (_GeInput0ds, 0.5, 0.5),
        (_LeInput0ds, 0.5, 0.7),
        (_EqInput0ds, 0.5, 0.5),
        (_NeInput0ds, 0.5, 0.7),
    ],
)
def test_input_0d_comparisons(
    device: torch.device,
    model_cls: type,
    a_val: float,
    b_val: float,
) -> None:
    a = torch.tensor(a_val, device=device)
    b = torch.tensor(b_val, device=device)
    eager, compiled = _run(model_cls(), a, b)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_threshold_by_input_scalar(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    t = torch.tensor(0.5, device=device)
    eager, compiled = _run(_ThresholdByInputScalar(), x, t)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_mask_by_scalar_eq(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    s = torch.tensor(0.0, device=device)  # equals branch chosen
    eager, compiled = _run(_MaskByScalarEq(), x, s)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 13: Reduction extras.
# ---------------------------------------------------------------------------


class _ArgmaxAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax()


class _ArgminAll(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmin()


class _ArgmaxAllKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=None, keepdim=True)


class _ArgminAllKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmin(dim=None, keepdim=True)


class _ArgmaxKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=0, keepdim=True)


class _ArgminKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmin(dim=0, keepdim=True)


class _ArgmaxNegDimKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmax(dim=-1, keepdim=True)


class _ArgminNegDimKeepDim(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.argmin(dim=-1, keepdim=True)


class _SumEmptyDimTuple(torch.nn.Module):
    """sum(dim=()) is documented as a no-op identity in PyTorch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=())


class _SumOf0d(torch.nn.Module):
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return s.sum()


class _MeanOfOneElem(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


class _CumsumOf0d(torch.nn.Module):
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(s, 0)


@pytest.mark.parametrize("shape", [(), (5,), (3, 4), (2, 3, 4)])
def test_argmax_all(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ArgmaxAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(), (5,), (3, 4)])
def test_argmin_all(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ArgminAll(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize(
    "model_cls",
    [
        _ArgmaxAllKeepDim,
        _ArgminAllKeepDim,
        _ArgmaxKeepDim,
        _ArgminKeepDim,
        _ArgmaxNegDimKeepDim,
        _ArgminNegDimKeepDim,
    ],
)
def test_argextremum_0d_keepdim_returns_scalar(
    device: torch.device, model_cls: type[torch.nn.Module]
) -> None:
    x = torch.tensor(7.0, device=device)
    eager, compiled = _run(model_cls(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("model_cls", [_ArgmaxKeepDim, _ArgminKeepDim])
def test_argextremum_keepdim_1d(
    device: torch.device, model_cls: type[torch.nn.Module]
) -> None:
    x = torch.rand(5, device=device)
    eager, compiled = _run(model_cls(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_sum_empty_dim_tuple(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_SumEmptyDimTuple(), x)
    _strict_match(compiled, eager)


def test_sum_of_0d(device: torch.device) -> None:
    s = torch.tensor(7.5, device=device)
    eager, compiled = _run(_SumOf0d(), s)
    _strict_match(compiled, eager)


def test_mean_of_one_element(device: torch.device) -> None:
    x = torch.rand(1, device=device)
    eager, compiled = _run(_MeanOfOneElem(), x)
    _strict_match(compiled, eager)


def test_cumsum_of_0d(device: torch.device) -> None:
    s = torch.tensor(3.5, device=device)
    eager, compiled = _run(_CumsumOf0d(), s)
    _strict_match(compiled, eager)


def test_cumsum_of_1elem_1d(device: torch.device) -> None:
    x = torch.tensor([3.5], device=device)
    eager, compiled = _run(_CumsumOf0d(), x)
    assert compiled.shape == (1,)
    assert eager.shape == (1,)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 14: Shape-flattening ops on 0-d.
# PyTorch documents that flatten/ravel/reshape(-1)/view(-1) on a 0-d tensor
# return shape (1,) — the rank surprises many users. Worth pinning.
# ---------------------------------------------------------------------------


class _Flatten0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().flatten()


class _Ravel0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ravel(x.sum())


class _ReshapeMinusOne0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().reshape(-1)


class _ReshapeOneToScalar(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(())


class _ViewMinusOne0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().view(-1)


class _PermuteEmptyOn0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().permute([])


class _Contiguous0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().contiguous()


class _SqueezeAll1s(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze()


class _ExpandAs0dTo2d(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return s.expand_as(x)


def test_flatten_0d_returns_rank1(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_Flatten0d(), x)
    _strict_match(compiled, eager)


def test_ravel_0d_returns_rank1(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_Ravel0d(), x)
    _strict_match(compiled, eager)


def test_reshape_minus_one_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ReshapeMinusOne0d(), x)
    _strict_match(compiled, eager)


def test_reshape_to_empty_from_one_elem(device: torch.device) -> None:
    x = torch.rand(1, device=device)
    eager, compiled = _run(_ReshapeOneToScalar(), x)
    _strict_match(compiled, eager)


def test_view_minus_one_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ViewMinusOne0d(), x)
    _strict_match(compiled, eager)


def test_permute_empty_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_PermuteEmptyOn0d(), x)
    _strict_match(compiled, eager)


def test_contiguous_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_Contiguous0d(), x)
    _strict_match(compiled, eager)


def test_squeeze_no_arg_collapses_all_size_1_dims(device: torch.device) -> None:
    x = torch.rand((1, 1, 1, 1), device=device)
    eager, compiled = _run(_SqueezeAll1s(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_expand_as_0d_to_nd(device: torch.device, shape: tuple) -> None:
    s = torch.tensor(2.5, device=device)
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_ExpandAs0dTo2d(), s, x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 15: Indexing extras.
# ---------------------------------------------------------------------------


class _EllipsisOn0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.sum())[...]


class _IndexByScalarTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return x[i]


class _ReturnScalarIndexAndVector(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, i: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return i, x


class _SetItemReturnSameInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x[0] = x[0] + 1
        return x


def _xfail_cuda_io_name_collision(device: torch.device) -> None:
    if device.type == "cuda":
        pytest.xfail(
            "LUM-536: PT2 input/output boundary names can collide; runtime "
            "interface IDs are still name-keyed in this PR. "
            "https://linear.app/luminalai/issue/LUM-536/"
            "pt2-interface-ids-should-model-inputoutput-roles-and-passthrough"
        )


class _GatherWith0dIndex(torch.nn.Module):
    def forward(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 0, i)


class _NegativeIndexOn1d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[-1]


def test_ellipsis_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_EllipsisOn0d(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("index", [2, -1])
def test_index_by_scalar_tensor(device: torch.device, index: int) -> None:
    _xfail_cuda_io_name_collision(device)
    x = torch.rand(5, device=device)
    i = torch.tensor(index, device=device)
    eager, compiled = _run(_IndexByScalarTensor(), x, i)
    _strict_match(compiled, eager)


def test_passthrough_scalar_index_and_vector(device: torch.device) -> None:
    _xfail_cuda_io_name_collision(device)
    x = torch.rand(5, device=device)
    i = torch.tensor(2, device=device)
    eager, compiled = _run(_ReturnScalarIndexAndVector(), x, i)
    assert isinstance(compiled, tuple)
    _strict_match(compiled[0], eager[0])
    _strict_match(compiled[1], eager[1])


@pytest.mark.xfail(
    reason=(
        "LUM-538: PT2 can return a mutated input with the same boundary name; "
        "Luminal should support or explicitly reject this case. "
        "https://linear.app/luminalai/issue/LUM-538/"
        "pt2-mutated-input-can-be-returned-with-the-same-boundary-name-as-the"
    )
)
def test_mutated_input_returned_with_same_boundary_name(device: torch.device) -> None:
    x = torch.rand(5, device=device)
    eager, compiled = _run(_SetItemReturnSameInput(), x)
    _strict_match(compiled, eager)


def test_gather_with_0d_index(device: torch.device) -> None:
    x = torch.rand(5, device=device)
    i = torch.zeros((), dtype=torch.int64, device=device)
    eager, compiled = _run(_GatherWith0dIndex(), x, i)
    _strict_match(compiled, eager)


def test_negative_index_on_1d_to_scalar(device: torch.device) -> None:
    x = torch.rand(5, device=device)
    eager, compiled = _run(_NegativeIndexOn1d(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 16: Type promotion with 0-d.
# ---------------------------------------------------------------------------


class _Float0dPlusIntNd(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return s + x


class _Int0dPlusFloatNd(torch.nn.Module):
    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return s + x


class _CastRoundTrip0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().to(torch.int32).to(torch.float32)


class _DotShorthandFloat0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum().float()


class _DotShorthandInt0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.sum() * 10).int()


class _WhereMixedDtypeBranches(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x > 0.5
        return torch.where(
            c,
            torch.tensor(1).to(x.device),
            torch.tensor(0.0).to(x.device),
        )


def test_float_0d_plus_int_nd(device: torch.device) -> None:
    s = torch.tensor(2.5, device=device)
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_Float0dPlusIntNd(), s, x)
    _strict_match(compiled, eager)


def test_int_0d_plus_float_nd(device: torch.device) -> None:
    s = torch.tensor(3, device=device, dtype=torch.int64)
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_Int0dPlusFloatNd(), s, x)
    _strict_match(compiled, eager)


def test_int32_0d_plus_float_nd(device: torch.device) -> None:
    """Regression for LUM-498: int32 (not just int64) 0-d input must also work."""
    s = torch.tensor(3, device=device, dtype=torch.int32)
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_Int0dPlusFloatNd(), s, x)
    _strict_match(compiled, eager)


def test_cast_roundtrip_through_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_CastRoundTrip0d(), x)
    _strict_match(compiled, eager)


def test_dot_float_shorthand_on_0d(device: torch.device) -> None:
    x = torch.randint(0, 10, (3, 4), device=device, dtype=torch.int64)
    eager, compiled = _run(_DotShorthandFloat0d(), x)
    _strict_match(compiled, eager)


def test_dot_int_shorthand_on_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_DotShorthandInt0d(), x)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_where_mixed_dtype_branches(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_WhereMixedDtypeBranches(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 17: Unary math on 0-d (parametric).
# ---------------------------------------------------------------------------


class _UnaryOn0d(torch.nn.Module):
    """Parametric: applies a unary op to a reduction-derived 0-d."""

    def __init__(self, op: Callable) -> None:
        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x.sum())


@pytest.mark.parametrize(
    "op",
    [
        torch.abs,
        torch.neg,
        torch.exp,
        torch.sin,
        torch.cos,
        torch.tanh,
        torch.sigmoid,
        torch.sqrt,
        torch.sign,
        torch.floor,
        torch.ceil,
    ],
    ids=lambda f: f.__name__,
)
def test_unary_on_reduced_0d(device: torch.device, op: Callable) -> None:
    x = torch.rand((3, 4), device=device) + 0.1  # avoid log(0)/sqrt(neg)
    eager, compiled = _run(_UnaryOn0d(op), x)
    _strict_match(compiled, eager, atol=1e-5)


# ---------------------------------------------------------------------------
# Section 18: Logical / bitwise ops on 0-d bools.
# ---------------------------------------------------------------------------


class _AndBools(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a > 0) & (b > 0)).float()


class _OrBools(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a > 0) | (b > 0)).float()


class _XorBools(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_xor(a > 0, b > 0).float()


class _NotBoolFromCmp(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logical_not(a < b).float()


@pytest.mark.parametrize("model_cls", [_AndBools, _OrBools, _XorBools, _NotBoolFromCmp])
def test_bool_logic_on_0d(device: torch.device, model_cls: type) -> None:
    a = torch.tensor(0.5, device=device)
    b = torch.tensor(-0.5, device=device)
    eager, compiled = _run(model_cls(), a, b)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 19: Stack / cat near 0-d.
# ---------------------------------------------------------------------------


class _StackZeroDs(torch.nn.Module):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([a, b, c])


class _CatUnsqueezedZeroDs(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)


def test_stack_three_0d(device: torch.device) -> None:
    a = torch.tensor(1.0, device=device)
    b = torch.tensor(2.0, device=device)
    c = torch.tensor(3.0, device=device)
    eager, compiled = _run(_StackZeroDs(), a, b, c)
    _strict_match(compiled, eager)


def test_cat_two_unsqueezed_0d(device: torch.device) -> None:
    a = torch.tensor(1.0, device=device)
    b = torch.tensor(2.0, device=device)
    eager, compiled = _run(_CatUnsqueezedZeroDs(), a, b)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 20: full / full_like / scalar_tensor.
# ---------------------------------------------------------------------------


class _FullEmptyShape(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.full((), 3.0).to(x.device)


class _FullLike0d(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x.sum(), 7.0)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_full_empty_shape_constant(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device)
    eager, compiled = _run(_FullEmptyShape(), x)
    _strict_match(compiled, eager)


def test_full_like_of_0d(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_FullLike0d(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 21: Reduction edge cases — keepdim semantics, broadcast onto views.
# ---------------------------------------------------------------------------


class _SumKeepdimAllAxes(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.sum(dim=tuple(range(x.dim())), keepdim=True)


class _ScalarBroadcastOntoTransposed(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.t() - x.sum()


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_sum_keepdim_all_axes_then_div(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) + 0.1
    eager, compiled = _run(_SumKeepdimAllAxes(), x)
    _strict_match(compiled, eager, atol=1e-4)


def test_scalar_broadcast_onto_transposed(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    eager, compiled = _run(_ScalarBroadcastOntoTransposed(), x)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 22: where / clamp with mixed scalar + tensor + python.
# ---------------------------------------------------------------------------


class _ClampScalarTensorAndPyFloat(torch.nn.Module):
    def forward(self, x: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, lo, 0.5)


class _WhereScalarOther(torch.nn.Module):
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, s, x)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_clamp_scalar_tensor_lo_python_hi(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 2.0 - 1.0
    lo = torch.tensor(-0.25, device=device)
    eager, compiled = _run(_ClampScalarTensorAndPyFloat(), x, lo)
    _strict_match(compiled, eager)


@pytest.mark.parametrize("shape", [(5,), (3, 4)])
def test_where_with_input_scalar_branch(device: torch.device, shape: tuple) -> None:
    x = torch.rand(shape, device=device) * 2.0 - 1.0
    s = torch.tensor(99.0, device=device)
    eager, compiled = _run(_WhereScalarOther(), x, s)
    _strict_match(compiled, eager)


# ---------------------------------------------------------------------------
# Section 23: Models returning multiple outputs (one rank-0, one rank-N).
# ---------------------------------------------------------------------------


class _ReturnPairScalarAndTensor(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.sum(), x * 2


def test_return_pair_scalar_and_tensor(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device)
    model = _ReturnPairScalarAndTensor()
    compiled_fn = torch.compile(model, backend=luminal_backend)
    eager_outs = model(x)
    compiled_outs = compiled_fn(x)
    assert len(eager_outs) == len(compiled_outs)
    for c, e in zip(compiled_outs, eager_outs):
        _strict_match(c, e)


# ---------------------------------------------------------------------------
# Section 24: aten.clamp.Tensor shape coverage.
#
# PyTorch's clamp.Tensor accepts bounds with rank-0 (scalar), same-shape,
# or any NumPy-broadcastable shape. The translator must handle all three.
# ---------------------------------------------------------------------------


class _ClampTensorBothBounds(torch.nn.Module):
    """clamp(x, lo, hi) — both bounds are tensors of arbitrary broadcastable shape."""

    def forward(
        self, x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
    ) -> torch.Tensor:
        return torch.clamp(x, lo, hi)


class _ClampTensorMinOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=lo)


class _ClampTensorMaxOnly(torch.nn.Module):
    def forward(self, x: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, max=hi)


def test_clamp_tensor_same_shape_bounds(device: torch.device) -> None:
    """Per-element clamp: lo and hi same shape as x (e.g. learned bounds)."""
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    lo = torch.full_like(x, -0.5)
    hi = torch.full_like(x, 0.5)
    eager, compiled = _run(_ClampTensorBothBounds(), x, lo, hi)
    _strict_match(compiled, eager)


def test_clamp_tensor_per_row_bounds(device: torch.device) -> None:
    """Per-row clamp: x is (3, 4); lo/hi are (3, 1) — broadcasts across columns."""
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    lo = torch.tensor([[-0.5], [-0.25], [-0.1]], device=device)
    hi = torch.tensor([[0.5], [0.25], [0.1]], device=device)
    eager, compiled = _run(_ClampTensorBothBounds(), x, lo, hi)
    _strict_match(compiled, eager)


def test_clamp_tensor_per_col_bounds(device: torch.device) -> None:
    """Per-column clamp: x is (3, 4); lo/hi are (4,) — right-aligned broadcast."""
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    lo = torch.tensor([-0.5, -0.25, -0.1, 0.0], device=device)
    hi = torch.tensor([0.5, 0.25, 0.1, 0.2], device=device)
    eager, compiled = _run(_ClampTensorBothBounds(), x, lo, hi)
    _strict_match(compiled, eager)


def test_clamp_tensor_mixed_rank0_and_full_shape(device: torch.device) -> None:
    """One bound rank-0, the other matching x.shape."""
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    lo = torch.tensor(-0.25, device=device)  # rank-0
    hi = torch.full_like(x, 0.5)  # same shape as x
    eager, compiled = _run(_ClampTensorBothBounds(), x, lo, hi)
    _strict_match(compiled, eager)


def test_clamp_tensor_min_only_same_shape(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    lo = torch.full_like(x, -0.25)
    eager, compiled = _run(_ClampTensorMinOnly(), x, lo)
    _strict_match(compiled, eager)


def test_clamp_tensor_max_only_per_row(device: torch.device) -> None:
    x = torch.rand((3, 4), device=device) * 2.0 - 1.0
    hi = torch.tensor([[0.5], [0.25], [0.1]], device=device)
    eager, compiled = _run(_ClampTensorMaxOnly(), x, hi)
    _strict_match(compiled, eager)


def test_clamp_tensor_3d_with_2d_bounds(device: torch.device) -> None:
    """x is (2, 3, 4); bounds are (3, 4) — left-unsqueeze broadcast."""
    x = torch.rand((2, 3, 4), device=device) * 2.0 - 1.0
    lo = torch.full((3, 4), -0.5, device=device)
    hi = torch.full((3, 4), 0.5, device=device)
    eager, compiled = _run(_ClampTensorBothBounds(), x, lo, hi)
    _strict_match(compiled, eager)
