"""Regression tests for ATen composites built from the existing HLIR."""

import torch

from luminal.pt2 import compile as luminal_compile


def _compile(module: torch.nn.Module, *inputs: torch.Tensor, dynamic_shapes=None):
    return luminal_compile(
        module,
        inputs,
        search_iterations=1,
        dynamic_shapes={} if dynamic_shapes is None else dynamic_shapes,
    )


def _assert_close(actual, expected, *, rtol=5e-4, atol=5e-5) -> None:
    actual, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected, expected_spec = torch.utils._pytree.tree_flatten(expected)
    del actual_spec, expected_spec
    assert len(actual) == len(expected)
    for index, (result, reference) in enumerate(zip(actual, expected)):
        approximate = reference.dtype.is_floating_point or reference.is_complex()
        torch.testing.assert_close(
            result,
            reference,
            rtol=rtol if approximate else 0,
            atol=atol if approximate else 0,
            equal_nan=True,
            check_stride=False,
            msg=f"output {index}",
        )


def _check(module, *inputs, **tolerances) -> None:
    _assert_close(_compile(module, *inputs)(*inputs), module(*inputs), **tolerances)


class ElementwiseComposites(torch.nn.Module):
    def forward(self, value, other):
        return (
            torch.ops.aten.hardtanh.default(value, -0.5, 0.75),
            torch.ops.aten.elu.default(value, 1.2, 0.7, 0.8),
            torch.ops.aten.leaky_relu.default(value, 0.2),
            torch.ops.aten.round.default(value),
            torch.ops.aten.round.decimals(value, decimals=2),
            torch.ops.aten.signbit.default(value),
            torch.ops.aten.hypot.default(value, other),
            torch.ops.aten.erfc.default(value),
            torch.ops.aten.special_erfcx.default(value),
            torch.ops.aten.lgamma.default(value),
            torch.ops.aten.digamma.default(value),
            torch.ops.aten.polygamma.default(1, value),
            torch.ops.aten.i0.default(value),
            torch.ops.aten.special_i0e.default(value),
            torch.ops.aten.special_i1.default(value),
            torch.ops.aten.special_i1e.default(value),
            torch.ops.aten.special_modified_bessel_i0.default(value),
            torch.ops.aten.special_modified_bessel_i1.default(value),
            torch.ops.aten.special_spherical_bessel_j0.default(value),
            torch.ops.aten.erfinv.default(value / 4),
            torch.ops.aten.special_ndtri.default(value.sigmoid()),
            torch.ops.aten.logcumsumexp.default(value, -1),
            torch.ops.aten.pow.Scalar(2.0, value),
        )


def test_elementwise_composites_cover_ieee_edges() -> None:
    value = torch.tensor(
        [[-2.5, -0.0, 0.5, 1.25], [float("inf"), float("nan"), -0.25, 50.0]]
    )
    other = torch.tensor([[4.0], [2.0]])
    module = ElementwiseComposites()
    _check(module, value, other)


class PolygammaOrders(torch.nn.Module):
    def forward(self, value):
        return (
            torch.ops.aten.digamma.default(value),
            torch.ops.aten.polygamma.default(0, value),
            torch.ops.aten.polygamma.default(1, value),
            torch.ops.aten.polygamma.default(2, value),
            torch.ops.aten.polygamma.default(3, value),
            torch.ops.aten.polygamma.default(4, value),
        )


@torch.no_grad()
def test_polygamma_orders_reflection_poles_and_infinities() -> None:
    values = [
        -100.2,
        -20.5,
        -10.1,
        -5.0,
        -2.5,
        -1.0,
        -0.5,
        -0.1,
        -0.0,
        0.0,
        1e-6,
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        10.0,
        100.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    module = PolygammaOrders()
    for dtype in (torch.float32, torch.float64):
        value = torch.tensor(values, dtype=dtype)
        _check(module, value, rtol=2e-4, atol=2e-5)


class ModifiedBesselFamily(torch.nn.Module):
    def forward(self, value):
        return (
            torch.ops.aten.i0.default(value),
            torch.ops.aten.special_i0e.default(value),
            torch.ops.aten.special_i1.default(value),
            torch.ops.aten.special_i1e.default(value),
            torch.ops.aten.special_modified_bessel_i0.default(value),
            torch.ops.aten.special_modified_bessel_i1.default(value),
            torch.ops.aten.special_spherical_bessel_j0.default(value),
        )


@torch.no_grad()
def test_modified_bessel_family_across_approximation_intervals() -> None:
    values = [
        float("-inf"),
        -100.0,
        -20.0,
        -8.0,
        -1.0,
        -0.0,
        0.0,
        1e-8,
        1.0,
        8.0,
        20.0,
        100.0,
        float("inf"),
        float("nan"),
    ]
    module = ModifiedBesselFamily()
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-7)):
        value = torch.tensor(values, dtype=dtype)
        _check(module, value, rtol=tolerance, atol=tolerance)


class CylindricalBesselAndAiryFamily(torch.nn.Module):
    def forward(self, value):
        return (
            torch.ops.aten.special_bessel_j0.default(value),
            torch.ops.aten.special_bessel_j1.default(value),
            torch.ops.aten.special_bessel_y0.default(value),
            torch.ops.aten.special_bessel_y1.default(value),
            torch.ops.aten.special_modified_bessel_k0.default(value),
            torch.ops.aten.special_modified_bessel_k1.default(value),
            torch.ops.aten.special_scaled_modified_bessel_k0.default(value),
            torch.ops.aten.special_scaled_modified_bessel_k1.default(value),
            torch.ops.aten.special_airy_ai.default(value),
        )


@torch.no_grad()
def test_cylindrical_bessel_and_airy_piecewise_approximations() -> None:
    values = [
        float("-inf"),
        -100.0,
        -10.0,
        -5.001,
        -5.0,
        -2.1001,
        -2.09,
        -2.0,
        -1e-8,
        -0.0,
        0.0,
        1e-8,
        1.0,
        2.0,
        2.09,
        5.0,
        5.001,
        8.3203353,
        20.0,
        103.892,
        103.893,
        float("inf"),
        float("nan"),
    ]
    module = CylindricalBesselAndAiryFamily()
    for dtype, tolerance in ((torch.float32, 3e-5), (torch.float64, 3e-7)):
        value = torch.tensor(values, dtype=dtype)
        _check(module, value, rtol=tolerance, atol=tolerance / 10)


class InverseProbabilityFunctions(torch.nn.Module):
    def forward(self, erf_value, probability):
        return (
            torch.ops.aten.erfinv.default(erf_value),
            torch.ops.aten.special_ndtri.default(probability),
        )


@torch.no_grad()
def test_inverse_probability_functions_cover_tails_and_boundaries() -> None:
    module = InverseProbabilityFunctions()
    # Older supported PyTorch releases use a less accurate F64 erfinv tail
    # approximation than Luminal (about 7e-5 relative error at 1 - eps).
    # Keep the comparison tight enough to catch branch/coefficient mistakes
    # without requiring Luminal to reproduce that version-specific error.
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-4)):
        epsilon = torch.finfo(dtype).eps
        erf_value = torch.tensor(
            [
                float("-inf"),
                -2.0,
                -1.0,
                -1.0 + epsilon,
                -0.7,
                -0.0,
                0.0,
                0.7,
                1.0 - epsilon,
                1.0,
                2.0,
                float("inf"),
                float("nan"),
            ],
            dtype=dtype,
        )
        probability = torch.tensor(
            [
                float("-inf"),
                -1.0,
                -0.0,
                0.0,
                torch.finfo(dtype).tiny,
                epsilon,
                0.1,
                0.5,
                0.9,
                1.0 - epsilon,
                1.0,
                2.0,
                float("inf"),
                float("nan"),
            ],
            dtype=dtype,
        )
        _check(module, erf_value, probability, rtol=tolerance, atol=tolerance)


class ChebyshevPolynomial(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, value, degree):
        operations = (
            torch.ops.aten.special_chebyshev_polynomial_t.default,
            torch.ops.aten.special_chebyshev_polynomial_u.default,
            torch.ops.aten.special_chebyshev_polynomial_v.default,
            torch.ops.aten.special_chebyshev_polynomial_w.default,
            torch.ops.aten.special_shifted_chebyshev_polynomial_t.default,
            torch.ops.aten.special_shifted_chebyshev_polynomial_u.default,
            torch.ops.aten.special_shifted_chebyshev_polynomial_v.default,
            torch.ops.aten.special_shifted_chebyshev_polynomial_w.default,
        )
        return operations[self.index](value, degree)


@torch.no_grad()
def test_chebyshev_families_cover_runtime_degrees_and_domains() -> None:
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-7)):
        value = torch.tensor(
            [-2.0, -1.0, -0.75, 0.0, 0.3, 1.0, 2.0, -0.2, 0.8, 1.25],
            dtype=dtype,
        )
        degree = torch.tensor(
            [-2.0, 0.0, 1.0, 2.0, 3.9, 7.0, 9.0, 10.0, 20.0, 25.0],
            dtype=dtype,
        )
        for index in range(8):
            module = ChebyshevPolynomial(index)
            _check(module, value, degree, rtol=tolerance, atol=tolerance)


class HistogramComposites(torch.nn.Module):
    def forward(
        self,
        value,
        weight,
        bin_edges,
        coordinates,
        coordinate_weight,
        first_edges,
        second_edges,
    ):
        count_histogram = torch.ops.aten.histogram.bin_ct(
            value,
            4,
            range=[-2.0, 3.0],
            weight=weight,
            density=False,
        )
        edge_histogram = torch.ops.aten.histogram.bins_tensor(
            value,
            bin_edges,
            weight=weight,
            density=True,
        )
        generated_edges = torch.ops.aten._histogramdd_bin_edges.default(
            coordinates,
            [3, 4],
            range=[-2.0, 2.0, -1.0, 3.0],
            weight=coordinate_weight,
            density=False,
        )
        count_histogramdd = torch.ops.aten._histogramdd_from_bin_cts.default(
            coordinates,
            [3, 4],
            range=[-2.0, 2.0, -1.0, 3.0],
            weight=coordinate_weight,
            density=True,
        )
        edge_histogramdd = torch.ops.aten._histogramdd_from_bin_tensors.default(
            coordinates,
            [first_edges, second_edges],
            weight=coordinate_weight,
            density=False,
        )
        return (
            count_histogram,
            edge_histogram,
            generated_edges,
            count_histogramdd,
            edge_histogramdd,
        )


@torch.no_grad()
def test_histogram_composites_have_fixed_bin_shapes() -> None:
    module = HistogramComposites()
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-7)):
        inputs = (
            torch.tensor(
                [-3.0, -2.0, -1.0, -0.5, 0.0, 1.0, 3.0, 4.0, float("nan")],
                dtype=dtype,
            ),
            torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 4.0, 1.0, 5.0], dtype=dtype),
            torch.tensor([-2.0, -0.5, 1.0, 3.0], dtype=dtype),
            torch.tensor(
                [
                    [-2.0, -1.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [float("nan"), 0.0],
                ],
                dtype=dtype,
            ),
            torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0, 5.0, 7.0], dtype=dtype),
            torch.tensor([-2.0, -0.25, 0.5, 2.0], dtype=dtype),
            torch.tensor([-1.0, 0.0, 1.5, 3.0], dtype=dtype),
        )
        _check(module, *inputs, rtol=tolerance, atol=tolerance)


class FixedShapeIndexing(torch.nn.Module):
    def forward(self, value, mask, source, index, repeats, boundaries, queries):
        return (
            torch.ops.aten.slice_scatter.default(value, source[:2], 0, 1, 5, 2),
            torch.ops.aten.masked_scatter.default(value, mask, source),
            torch.ops.aten.put.default(value, index, source[:3], True),
            torch.ops.aten.repeat_interleave.Tensor(repeats, output_size=6),
            torch.ops.aten.nonzero_static.default(value, size=7, fill_value=-2),
            torch.ops.aten.bucketize.Tensor(queries, boundaries, right=True),
            torch.ops.aten.searchsorted.Tensor(boundaries, queries, right=False),
        )


def test_fixed_shape_indexing_and_search_composites() -> None:
    inputs = (
        torch.tensor([0.0, 1.0, 0.0, 2.0, 3.0, 0.0]),
        torch.tensor([False, True, False, True, True, False]),
        torch.tensor([9.0, 8.0, 7.0]),
        torch.tensor([1, 1, -1]),
        torch.tensor([2, 0, 1, 3]),
        torch.tensor([1.0, 3.0, 5.0]),
        torch.tensor([0.0, 1.0, 2.0, 5.0, float("nan")]),
    )
    module = FixedShapeIndexing()
    _check(module, *inputs)


class ComplexMovement(torch.nn.Module):
    def forward(self, value, source, mask, gather_index, fancy_index):
        return (
            torch.ops.aten.gather.default(value, 1, gather_index),
            torch.ops.aten.index.Tensor(value, [fancy_index]),
            torch.ops.aten.slice_scatter.default(value, source, 1, 1, 4, 2),
            torch.ops.aten.masked_scatter.default(value, mask, source.flatten()),
            torch.ops.aten.put.default(
                value, torch.tensor([1, 1, -1]), source.flatten()[:3], True
            ),
            torch.ops.aten.logcumsumexp.default(value + 80.0, 1),
            torch.ops.aten.dist.default(value, value * 2.0, 2.0),
        )


def test_complex_movement_is_componentwise() -> None:
    torch.manual_seed(0)
    value = torch.randn(3, 4) + 1j * torch.randn(3, 4)
    source = torch.randn(3, 2) + 1j * torch.randn(3, 2)
    mask = torch.tensor(
        [[False, True, False, True], [True, False, False, False], [False] * 4]
    )
    gather_index = torch.tensor([[3, 1], [0, 2], [1, 1]])
    fancy_index = torch.tensor([2, 0])
    module = ComplexMovement()
    inputs = (value, source, mask, gather_index, fancy_index)
    _check(module, *inputs)


class PoolingAndResize(torch.nn.Module):
    def forward(self, value):
        max_values, max_indices = torch.ops.aten.max_pool2d_with_indices.default(
            value, [2, 3], [1, 2], [1, 1], [1, 1], True
        )
        adaptive_values, adaptive_indices = torch.ops.aten.adaptive_max_pool2d.default(
            value, [3, 2]
        )
        max_backward = torch.ops.aten.max_pool2d_with_indices_backward.default(
            torch.ones_like(max_values),
            value,
            [2, 3],
            [1, 2],
            [1, 1],
            [1, 1],
            True,
            max_indices,
        )
        return (
            torch.ops.aten.avg_pool2d.default(
                value, [3, 2], [2, 1], [1, 1], True, False, None
            ),
            torch.ops.aten._adaptive_avg_pool2d.default(value, [3, 2]),
            max_values,
            max_indices,
            adaptive_values,
            adaptive_indices,
            max_backward,
            torch.ops.aten.upsample_bilinear2d.vec(value, [7, 5], False, None),
            torch.ops.aten._upsample_bilinear2d_aa.default(
                value, [2, 3], False, None, None
            ),
            torch.ops.aten._upsample_bilinear2d_aa.default(
                value, [7, 3], True, None, None
            ),
        )


def test_pooling_and_bilinear_resize_composites() -> None:
    torch.manual_seed(1)
    value = torch.randn(1, 2, 4, 5)
    module = PoolingAndResize()
    _check(module, value)


class AntialiasedBilinearResize(torch.nn.Module):
    def forward(self, value):
        return (
            torch.ops.aten._upsample_bilinear2d_aa.default(
                value, [2, 3], False, 0.5, 0.6
            ),
            torch.ops.aten._upsample_bilinear2d_aa.default(
                value, [7, 8], True, None, None
            ),
        )


def test_antialiased_bilinear_resize_float_and_uint8() -> None:
    torch.manual_seed(7)
    module = AntialiasedBilinearResize()
    values = (
        torch.randn(1, 3, 4, 5, dtype=torch.float32),
        torch.randn(1, 3, 4, 5, dtype=torch.float64),
        torch.randint(0, 256, (1, 3, 4, 5), dtype=torch.uint8),
    )
    for value in values:
        _check(module, value)


class FunctionalBatchNorm(torch.nn.Module):
    def forward(self, value, weight, bias, running_mean, running_var):
        return torch.ops.aten._native_batch_norm_legit_functional.default(
            value,
            weight,
            bias,
            running_mean,
            running_var,
            True,
            0.2,
            1e-5,
        )


def test_functional_batch_norm_returns_stats_and_updates() -> None:
    torch.manual_seed(2)
    inputs = (
        torch.randn(2, 3, 4, 5),
        torch.randn(3),
        torch.randn(3),
        torch.randn(3),
        torch.rand(3) + 0.5,
    )
    module = FunctionalBatchNorm()
    _check(module, *inputs)


class OrderingAndDistances(torch.nn.Module):
    def forward(self, value, other, points, points2):
        return (
            torch.ops.aten.max.dim(value, 1, False),
            torch.ops.aten.min.dim(value, 1, True),
            torch.ops.aten.median.dim(value, 1, False),
            torch.ops.aten.nanmedian.dim(value, 1, False),
            torch.ops.aten.dist.default(value, other, 2.5),
            torch.ops.aten._cdist_forward.default(points, points2, 2.0, None),
            torch.ops.aten._pdist_forward.default(points, 2.0),
        )


class IntegerGcd(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.ops.aten.gcd.default(lhs, rhs)


def test_integer_gcd_static_euclidean_rounds() -> None:
    module = IntegerGcd()
    values = {
        torch.uint8: ([0, 1, 6, 17, 200, 255], [0, 4, 8, 51, 125, 254]),
        torch.int8: ([-128, -10, -1, 0, 6, 127], [0, 4, 3, -8, 15, 126]),
        torch.int16: ([-32768, -144, -1, 0, 610, 32767], [0, 89, 3, -8, 987, 32766]),
        torch.int32: (
            [-(2**31), -10946, -1, 0, 46368, 2**31 - 1],
            [0, 6765, 3, -8, 75025, 2**31 - 2],
        ),
        torch.int64: (
            [-(2**63), -1836311903, -1, 0, 2971215073, 2**63 - 1],
            [0, 1134903170, 3, -8, 4807526976, 2**63 - 2],
        ),
    }
    for dtype, (lhs_values, rhs_values) in values.items():
        lhs = torch.tensor(lhs_values, dtype=dtype)
        rhs = torch.tensor(rhs_values, dtype=dtype)
        _check(module, lhs, rhs)


def test_ordering_and_distance_composites() -> None:
    value = torch.tensor([[1.0, 4.0, 2.0, 2.0], [float("nan"), 2.0, 3.0, 4.0]])
    other = torch.tensor([[0.0, 3.0, 1.0, 4.0], [1.0, 2.0, 5.0, 3.0]])
    points = torch.tensor([[0.0, 1.0], [2.0, 3.0], [-1.0, 4.0]])
    points2 = torch.tensor([[1.0, 0.0], [3.0, -2.0]])
    module = OrderingAndDistances()
    inputs = (value, other, points, points2)
    _check(module, *inputs)


class TrilinearAndEmbeddingBag(torch.nn.Module):
    def forward(self, left, right, weight, bias, indices, offsets, sample_weights):
        return (
            torch.nn.functional.bilinear(left, right, weight, bias),
            torch.nn.functional.embedding_bag(
                indices,
                weight.flatten(1),
                offsets,
                mode="sum",
                per_sample_weights=sample_weights,
                padding_idx=2,
            ),
            torch.nn.functional.embedding_bag(
                indices,
                weight.flatten(1),
                offsets,
                mode="mean",
                padding_idx=2,
            ),
            torch.nn.functional.embedding_bag(
                indices,
                weight.flatten(1),
                offsets,
                mode="max",
                padding_idx=2,
            ),
        )


def test_trilinear_and_embedding_bag_composites() -> None:
    torch.manual_seed(3)
    inputs = (
        torch.randn(2, 3),
        torch.randn(2, 4),
        torch.randn(5, 3, 4),
        torch.randn(5),
        torch.tensor([1, 2, 0, 3, 2]),
        torch.tensor([0, 2, 2, 5]),
        torch.tensor([0.5, 2.0, -1.0, 1.5, 3.0]),
    )
    module = TrilinearAndEmbeddingBag()
    _check(module, *inputs)


class Sampling2d(torch.nn.Module):
    def forward(self, value, grid, random_samples):
        return (
            torch.ops.aten.grid_sampler_2d.default(value, grid, 0, 2, False),
            torch.ops.aten.grid_sampler_2d.default(value, grid, 2, 1, True),
            torch.ops.aten.fractional_max_pool2d.default(
                value, [2, 3], [3, 2], random_samples
            ),
        )


def test_fixed_shape_2d_sampling_composites() -> None:
    torch.manual_seed(4)
    # NaN grid coordinates changed semantics across supported PyTorch
    # releases; finite out-of-bounds coordinates keep this regression focused
    # on Luminal's interpolation and padding rather than host-version behavior.
    inputs = (
        torch.randn(1, 2, 5, 6),
        torch.tensor(
            [
                [
                    [[-1.4, -1.2], [-0.5, 0.25], [1.3, 0.8]],
                    [[-0.75, 0.0], [0.2, -0.9], [1.0, 1.0]],
                ]
            ]
        ),
        torch.rand(1, 2, 2),
    )
    module = Sampling2d()
    _check(module, *inputs)


class Sampling3d(torch.nn.Module):
    def forward(self, value, grid, random_samples):
        return (
            torch.ops.aten.grid_sampler_3d.default(value, grid, 0, 2, False),
            torch.ops.aten.fractional_max_pool3d.default(
                value, [2, 2, 3], [2, 3, 2], random_samples
            ),
        )


def test_fixed_shape_3d_sampling_composites() -> None:
    torch.manual_seed(5)
    inputs = (
        torch.randn(1, 2, 4, 5, 6),
        torch.tensor(
            [
                [
                    [
                        [[-1.3, -1.1, -0.7], [0.2, 0.4, 0.6]],
                        [[1.2, 0.8, 1.4], [float("nan"), 0.0, 0.0]],
                    ]
                ]
            ]
        ),
        torch.rand(1, 2, 3),
    )
    module = Sampling3d()
    _check(module, *inputs)


class SegmentReductions(torch.nn.Module):
    def forward(self, value, lengths, offsets):
        return (
            torch.segment_reduce(value, "sum", lengths=lengths, axis=1, initial=2.0),
            torch.segment_reduce(value, "mean", offsets=offsets, axis=1, initial=1.0),
            torch.segment_reduce(value, "prod", lengths=lengths, axis=1, initial=-1.0),
            torch.segment_reduce(value, "max", offsets=offsets, axis=1, initial=0.5),
            torch.segment_reduce(value, "min", lengths=lengths, axis=1),
        )


def test_fixed_shape_segment_reduce_composites() -> None:
    value = torch.tensor(
        [
            [1.0, 3.0, -2.0, 5.0, 4.0],
            [-1.0, 2.0, 6.0, -3.0, 7.0],
        ]
    )
    lengths = torch.tensor([[0, 1, 2, 2], [2, 0, 3, 0]])
    offsets = torch.nn.functional.pad(lengths, [1, 0]).cumsum(1)
    module = SegmentReductions()
    inputs = (value, lengths, offsets)
    _check(module, *inputs)
