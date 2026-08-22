"""Regression coverage for composite linear algebra and movement lowerings."""

import warnings

import torch
from luminal.pt2 import compile as luminal_compile


def _compile(module: torch.nn.Module, *inputs: torch.Tensor, dynamic_shapes=None):
    return luminal_compile(
        module,
        inputs,
        search_iterations=1,
        dynamic_shapes={} if dynamic_shapes is None else dynamic_shapes,
    )


def _assert_outputs(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        if result.dtype.is_floating_point or result.dtype.is_complex:
            torch.testing.assert_close(
                result, reference, rtol=3e-4, atol=3e-5, equal_nan=True
            )
        else:
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def _complex_randn(*shape: int, dtype=torch.complex64) -> torch.Tensor:
    component = torch.float64 if dtype == torch.complex128 else torch.float32
    return torch.complex(
        torch.randn(*shape, dtype=component),
        torch.randn(*shape, dtype=component),
    )


class SmallAlgebraicAliases(torch.nn.Module):
    def forward(self, value, exponent, integers):
        return (
            torch.ops.aten.any.default(value),
            torch.ops.aten.any.dim(value, -1, True),
            torch.ops.aten.any.dims(value, [], False),
            torch.ops.aten.log1p.default(value),
            torch.ops.aten.expm1.default(value),
            torch.ops.aten.log10.default(value + 2.0),
            torch.ops.aten.sinh.default(value),
            torch.ops.aten.tan.default(value),
            torch.ops.aten.angle.default(value),
            torch.ops.aten.isinf.default(value),
            torch.ops.aten.isinf.default(integers),
            torch.ops.aten.ldexp.Tensor(value, exponent),
        )


class CopyAndStableSortAliases(torch.nn.Module):
    def forward(self, value):
        permuted = torch.ops.aten.permute_copy.default(value, [1, 0])
        unbound = torch.ops.aten.unbind_copy.int(permuted, 0)
        sorted_values, sorted_indices = torch.ops.aten.sort.stable(
            value, stable=True, dim=-1, descending=True
        )
        return (
            torch.ops.aten.view_copy.default(value, [2, 6]),
            permuted,
            torch.ops.aten.narrow_copy.default(permuted, -1, 1, 2),
            *unbound,
            sorted_values,
            sorted_indices,
        )


class ScatterAliases(torch.nn.Module):
    def forward(self, value, index, source):
        return (
            torch.ops.aten.scatter.reduce(value, 1, index, source, reduce="add"),
            torch.ops.aten.scatter.reduce(value, 1, index, source, reduce="multiply"),
            torch.ops.aten.scatter.value_reduce(value, 1, index, 1.5, reduce="add"),
            torch.ops.aten.scatter.value_reduce(
                value, 1, index, -0.5, reduce="multiply"
            ),
            torch.ops.aten.scatter_add.default(value, 1, index, source),
        )


class OptionalIndexPutAccumulate(torch.nn.Module):
    def forward(self, value, index, source):
        return torch.ops.aten.index_put.default(
            value, [None, index], source, accumulate=True
        )


def test_small_algebraic_aliases_use_existing_primitives() -> None:
    value = torch.tensor([[0.0, 0.25, -0.5], [1.0, 0.0, 0.75]])
    exponent = torch.tensor([[0, 1, -1], [2, -2, 3]], dtype=torch.int32)
    integers = torch.tensor([[0, 1, -2], [3, 0, 4]], dtype=torch.int64)
    module = SmallAlgebraicAliases()
    _assert_outputs(
        _compile(module, value, exponent, integers)(value, exponent, integers),
        module(value, exponent, integers),
    )


def test_copy_aliases_and_stable_sort_preserve_all_outputs() -> None:
    value = torch.tensor(
        [[3.0, 1.0, 3.0, 2.0], [0.0, 4.0, 4.0, -1.0], [2.0, 2.0, 1.0, 0.0]]
    )
    module = CopyAndStableSortAliases()
    _assert_outputs(_compile(module, value)(value), module(value))


def test_scatter_aliases_accumulate_duplicate_destinations_in_order() -> None:
    value = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    index = torch.tensor([[0, 0, 2], [1, 1, 1]])
    source = torch.tensor([[2.0, 3.0, 4.0], [0.5, 2.0, -1.0]])
    module = ScatterAliases()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected = module(value, index, source)
        actual = _compile(module, value, index, source)(value, index, source)
    _assert_outputs(actual, expected)


def test_optional_index_put_accumulates_real_and_complex_duplicates() -> None:
    index = torch.tensor([1, 1, 3])
    module = OptionalIndexPutAccumulate()
    for value, source in (
        (
            torch.arange(10, dtype=torch.float32).reshape(2, 5),
            torch.tensor([[2.0, 3.0, 4.0], [-1.0, 0.5, 7.0]]),
        ),
        (
            _complex_randn(2, 5),
            torch.tensor(
                [[2 + 1j, 3 - 4j, 4 + 0j], [-1 + 2j, 0.5j, 7 - 3j]],
                dtype=torch.complex64,
            ),
        ),
    ):
        _assert_outputs(
            _compile(module, value, index, source)(value, index, source),
            module(value, index, source),
        )


class LinearComposites(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, vector_input, matrix, vector, matrix_input, batch1, batch2):
        return (
            torch.ops.aten.addmv.default(
                vector_input,
                matrix,
                vector,
                beta=self.beta,
                alpha=self.alpha,
            ),
            torch.ops.aten.addbmm.default(
                matrix_input,
                batch1,
                batch2,
                beta=self.beta,
                alpha=self.alpha,
            ),
        )


class ZeroCoefficientLinearComposites(torch.nn.Module):
    def forward(self, vector_input, matrix, vector, matrix_input, batch1, batch2):
        return (
            torch.ops.aten.addmv.default(vector_input, matrix, vector, beta=0, alpha=0),
            torch.ops.aten.addbmm.default(
                matrix_input, batch1, batch2, beta=0, alpha=0
            ),
        )


class MovementComposites(torch.nn.Module):
    def forward(self, backing, diagonal, index, scalar_index):
        # Exercise every operation on a noncontiguous view. The output itself
        # is value-semantic, so the lowerings must honor the view's strides.
        value = backing.permute(1, 0)
        return (
            torch.ops.aten.diagonal_scatter.default(value, diagonal, 1, -2, -1),
            torch.ops.aten.index_select.default(value, -1, index),
            torch.ops.aten.index_select.default(value, 0, scalar_index),
            torch.ops.aten.unfold.default(value, -1, 2, 2),
        )


class DynamicMovement(torch.nn.Module):
    def forward(self, value, index):
        return (
            torch.ops.aten.index_select.default(value, 1, index),
            torch.ops.aten.unfold.default(value, 1, 2, 2),
        )


class ScalarMovement(torch.nn.Module):
    def forward(self, value, index):
        return (
            torch.ops.aten.index_select.default(value, -1, index),
            torch.ops.aten.unfold.default(value, 0, 0, 2),
            torch.ops.aten.unfold.default(value, -1, 1, 2),
        )


class CopyConversions(torch.nn.Module):
    def forward(
        self,
        real_destination,
        integer_source,
        complex_destination,
        real_source,
        bool_destination,
        complex_bool_source,
        discard_destination,
        complex_source,
    ):
        return (
            torch.ops.aten.copy.default(real_destination, integer_source),
            torch.ops.aten.copy.default(complex_destination, real_source),
            torch.ops.aten.copy.default(bool_destination, complex_bool_source),
            torch.ops.aten.copy.default(discard_destination, complex_source),
        )


def test_addmv_and_addbmm_preserve_real_dtypes_and_coefficients() -> None:
    torch.manual_seed(0)
    for dtype in (torch.float64, torch.bfloat16, torch.int16):
        inputs = (
            torch.arange(2, dtype=dtype),
            torch.arange(6, dtype=dtype).reshape(2, 3),
            torch.arange(3, dtype=dtype),
            torch.arange(8, dtype=dtype).reshape(2, 4),
            torch.arange(30, dtype=dtype).reshape(3, 2, 5),
            torch.arange(60, dtype=dtype).reshape(3, 5, 4),
        )
        module = LinearComposites(alpha=-2, beta=3)
        _assert_outputs(_compile(module, *inputs)(*inputs), module(*inputs))


def test_addmv_and_addbmm_support_complex_coefficients() -> None:
    torch.manual_seed(1)
    inputs = (
        _complex_randn(2, dtype=torch.complex128),
        _complex_randn(2, 3, dtype=torch.complex128),
        _complex_randn(3, dtype=torch.complex128),
        _complex_randn(2, 4, dtype=torch.complex128),
        _complex_randn(3, 2, 5, dtype=torch.complex128),
        _complex_randn(3, 5, 4, dtype=torch.complex128),
    )
    module = LinearComposites(alpha=1.5 - 0.75j, beta=-0.5 + 0.25j)
    _assert_outputs(_compile(module, *inputs)(*inputs), module(*inputs))


def test_zero_coefficients_structurally_ignore_nan_inputs() -> None:
    inputs = (
        torch.full((2,), float("nan")),
        torch.full((2, 3), float("nan")),
        torch.full((3,), float("nan")),
        torch.full((2, 4), float("nan")),
        torch.full((3, 2, 5), float("nan")),
        torch.full((3, 5, 4), float("nan")),
    )
    module = ZeroCoefficientLinearComposites()
    _assert_outputs(_compile(module, *inputs)(*inputs), module(*inputs))


def test_movement_lowerings_support_noncontiguous_real_and_complex_values() -> None:
    torch.manual_seed(2)
    for backing, diagonal in (
        (torch.randn(4, 3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)),
        (_complex_randn(4, 3), _complex_randn(3)),
    ):
        inputs = (backing, diagonal, torch.tensor([3, 1]), torch.tensor(1))
        module = MovementComposites()
        _assert_outputs(_compile(module, *inputs)(*inputs), module(*inputs))


def test_index_select_and_unfold_support_symbolic_lengths() -> None:
    module = DynamicMovement()
    example = (torch.randn(2, 6), torch.tensor([4, 1]))
    length = torch.export.Dim("length", min=2, max=10)
    count = torch.export.Dim("count", min=1, max=6)
    compiled = _compile(
        module,
        *example,
        dynamic_shapes=({1: length}, {0: count}),
    )
    for inputs in (
        (torch.randn(2, 4), torch.tensor([3])),
        (torch.randn(2, 8), torch.tensor([6, 2, 0])),
    ):
        _assert_outputs(compiled(*inputs), module(*inputs))


def test_index_select_and_unfold_match_scalar_semantics() -> None:
    module = ScalarMovement()
    for value in (torch.tensor(3.25), torch.tensor(2 - 4j)):
        inputs = (value, torch.tensor([0]))
        _assert_outputs(_compile(module, *inputs)(*inputs), module(*inputs))


def test_copy_supports_broadcast_and_real_complex_conversions() -> None:
    torch.manual_seed(3)
    inputs = (
        torch.empty(2, 3, dtype=torch.float64),
        torch.tensor([[7], [-2]], dtype=torch.int16),
        _complex_randn(2, 3),
        torch.tensor([[1.5], [-2.0]]),
        torch.empty(2, 3, dtype=torch.bool),
        torch.tensor([[0j, 2j, 3 + 0j]], dtype=torch.complex64),
        torch.empty(2, 3, dtype=torch.float64),
        _complex_randn(1, 3, dtype=torch.complex128),
    )
    module = CopyConversions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = module(*inputs)
        actual = _compile(module, *inputs)(*inputs)
    _assert_outputs(actual, expected)
