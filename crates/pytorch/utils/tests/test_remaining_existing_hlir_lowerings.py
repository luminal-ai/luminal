"""Regression tests for ATen composites built from the existing HLIR."""

import math

import pytest
import torch
from backend_test_utils import compile_for_test as luminal_compile
from backend_test_utils import luminal_backend


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


class Hypot(torch.nn.Module):
    def forward(self, value, other):
        return torch.hypot(value, other)


def _hypot_operands(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """(value, other) pairs covering C hypot's special values and range ends."""
    inf, nan = float("inf"), float("nan")
    info = torch.finfo(dtype)
    sub = info.smallest_normal * info.eps  # the smallest subnormal
    pairs = [
        # +-inf wins, even over NaN.
        (inf, inf),
        (-inf, -inf),
        (nan, inf),
        (inf, nan),
        (-inf, nan),
        (nan, -inf),
        (inf, 1.0),
        (-3.0, -inf),
        (0.0, inf),
        # Otherwise NaN wins over any finite value.
        (nan, 1.0),
        (1.0, nan),
        (nan, 0.0),
        (-0.0, nan),
        (nan, nan),
        # The origin is +0 for every zero sign.
        (0.0, 0.0),
        (-0.0, -0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, 3.0),
        (5.0, 0.0),
        (3.0, 4.0),
        (-5.0, 12.0),
        (7.0, -24.0),
        # Squaring would overflow; the scaled form must not.
        (1e38, 1e38),
        (0.6 * info.max, 0.6 * info.max),
        (info.max, 1.0),
        (info.max, info.max),
        (-info.max, info.max / 2),
        # Subnormal magnitudes, whose reciprocal overflows.
        (sub, sub),
        (sub, 0.0),
        (-0.0, sub),
        (3 * sub, 4 * sub),
        (info.smallest_normal / 2, info.smallest_normal / 2),
        (info.smallest_normal, sub),
        (info.max, sub),
    ]
    value, other = zip(*pairs)
    return torch.tensor(value, dtype=dtype), torch.tensor(other, dtype=dtype)


def _wide_hypot_operands(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded magnitudes spread over the dtype's whole exponent range."""
    info = torch.finfo(dtype)
    low = math.floor(math.log2(info.smallest_normal * info.eps))
    high = math.floor(math.log2(info.max))
    generator = torch.Generator().manual_seed(0)

    def draw() -> torch.Tensor:
        mantissa = torch.rand(64, generator=generator, dtype=torch.float64) * 2 - 1
        exponent = torch.randint(low, high, (64,), generator=generator)
        return (mantissa * exponent.to(torch.float64).exp2()).to(dtype)

    return draw(), draw()


def _assert_hypot_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    info = torch.finfo(expected.dtype)
    rtol = {torch.float64: 4 * info.eps}.get(expected.dtype, 2 * info.eps)
    # One subnormal ulp of slack: the result is rounded once into that range.
    atol = info.smallest_normal * info.eps
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True, check_stride=False
    )
    # assert_close equates the zero signs; hypot's zero is +0.
    zeros = expected == 0
    assert not torch.signbit(actual[zeros]).any(), actual[zeros]


@torch.no_grad()
@pytest.mark.parametrize(
    "operands", [_hypot_operands, _wide_hypot_operands], ids=["special", "wide"]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
def test_hypot_follows_c_special_values(dtype, operands, monkeypatch) -> None:
    """``hypot.default`` lowers at stage 1 of the real (torch.compile) path.

    It used to scale by the larger magnitude unconditionally, so two
    infinities gave ``inf / inf = NaN``, a subnormal pair gave ``inf`` from an
    overflowing reciprocal, and the origin kept ``-0.0``.
    """
    value, other = operands(dtype)
    module = Hypot()
    compiled, attempts = _backend_path(module, monkeypatch, dynamic=False)
    _assert_hypot_matches(compiled(value, other), module(value, other))
    _assert_stage_one(attempts, "aten.hypot.default")


@torch.no_grad()
@pytest.mark.parametrize(
    "value, other",
    [
        (
            torch.tensor([[float("inf")], [float("nan")], [-0.0], [3.0], [1e-45]]),
            torch.tensor([float("nan"), -0.0, 4.0, 1e-300], dtype=torch.float64),
        ),
        (
            torch.tensor([float("inf"), float("nan"), -0.0], dtype=torch.float16),
            torch.tensor([[float("nan")], [-0.0], [65504.0]]),
        ),
        (
            torch.tensor([float("inf"), float("nan"), 3.0], dtype=torch.bfloat16),
            torch.tensor(-0.0, dtype=torch.float16),
        ),
        (torch.empty(0, 3), torch.empty(3)),
    ],
    ids=["f32-f64", "f16-f32", "bf16-f16-scalar", "empty"],
)
def test_hypot_promotes_and_broadcasts(value, other, monkeypatch) -> None:
    module = Hypot()
    compiled, attempts = _backend_path(module, monkeypatch, dynamic=False)
    _assert_hypot_matches(compiled(value, other), module(value, other))
    _assert_stage_one(attempts, "aten.hypot.default")


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


def _histogram(values, *rest, **kwargs):
    return torch.histogram(values, *rest, **kwargs)


def _histogramdd(points, *rest, **kwargs):
    hist, edges = torch.histogramdd(points, *rest, **kwargs)
    return (hist, *edges)


# (callable, operand names). Every case normalises with ``density=True``:
# uniform (``bin_ct``) and tensor (non-uniform) edges, 1-D and N-D, with and
# without weights, and an all-zero histogram, where eager divides 0/0 into NaN.
_HISTOGRAM_DENSITY_CASES = {
    "1d_uniform": (lambda x: _histogram(x, 4, density=True), ("values",)),
    "1d_uniform_range_weight": (
        lambda x, w: _histogram(x, 4, range=(-1.0, 1.5), weight=w, density=True),
        ("values", "weight"),
    ),
    "1d_edges": (
        lambda x, e: _histogram(x, e, density=True),
        ("values", "edges"),
    ),
    "1d_edges_weight": (
        lambda x, e, w: _histogram(x, e, weight=w, density=True),
        ("values", "edges", "weight"),
    ),
    "1d_all_zero": (
        lambda x: _histogram(x, 3, range=(10.0, 11.0), density=True),
        ("values",),
    ),
    "nd_uniform": (
        lambda p: _histogramdd(p, [3, 4], density=True),
        ("points",),
    ),
    "nd_uniform_range_weight": (
        lambda p, w: _histogramdd(
            p, [3, 4], range=[-2.0, 2.0, -1.0, 3.0], weight=w, density=True
        ),
        ("points", "point_weight"),
    ),
    "nd_edges": (
        lambda p, a, b: _histogramdd(p, [a, b], density=True),
        ("points", "first_edges", "second_edges"),
    ),
    "nd_edges_weight": (
        lambda p, a, b, w: _histogramdd(p, [a, b], weight=w, density=True),
        ("points", "first_edges", "second_edges", "point_weight"),
    ),
    "nd_all_zero": (
        lambda p, w: _histogramdd(p, [3, 4], weight=w * 0, density=True),
        ("points", "point_weight"),
    ),
}


def _histogram_density_operands(dtype):
    # Built on CPU: the generator is a CPU generator and eager histogram is
    # CPU-only, whatever device the test selects for the compiled run.
    cpu = {"dtype": torch.float64, "device": "cpu"}
    generator = torch.Generator(device="cpu").manual_seed(0)
    # Fixed samples sit on interior edges, on the inclusive last edge and
    # outside the tensor edges. They are also each axis' extremes (values in
    # [-4, 3], points in [-3, 3]), so the uniform edges derived from min/max
    # are exact in binary and this density check does not depend on how
    # uniform edges round.
    values = torch.cat(
        (
            torch.randn(20, generator=generator, **cpu).clamp(-3.5, 2.5),
            torch.tensor([-0.5, 0.0, 3.0, -4.0], **cpu),
        )
    )
    points = torch.cat(
        (
            torch.randn(16, 2, generator=generator, **cpu).clamp(-2.5, 2.5),
            torch.tensor([[-3.0, -3.0], [3.0, 3.0], [-1.0, 1.5], [0.0, 0.5]], **cpu),
        )
    )
    operands = {
        "values": values,
        "weight": torch.rand(values.shape, generator=generator, **cpu),
        "edges": torch.tensor([-3.0, -0.5, 0.0, 0.7, 3.0], **cpu),
        "points": points,
        "point_weight": torch.rand(points.shape[0], generator=generator, **cpu),
        "first_edges": torch.tensor([-3.0, -1.0, 0.0, 3.0], **cpu),
        "second_edges": torch.tensor([-3.0, 0.0, 0.5, 1.0, 3.0], **cpu),
    }
    return {name: value.to(dtype) for name, value in operands.items()}


@torch.no_grad()
@pytest.mark.parametrize(
    "dtype, tolerance", ((torch.float32, 1e-5), (torch.float64, 1e-12))
)
@pytest.mark.parametrize("case", sorted(_HISTOGRAM_DENSITY_CASES))
def test_histogram_density_matches_eager(device, case, dtype, tolerance) -> None:
    """``density=True`` through ``torch.compile``, the backend's real path.

    The histogram overloads are lowered at stage 1 (the undecomposed graph);
    the density normalisation used to panic on a doubly expanded bin volume.
    """
    function, names = _HISTOGRAM_DENSITY_CASES[case]
    operands = _histogram_density_operands(dtype)
    inputs = tuple(operands[name] for name in names)
    expected = function(*inputs)

    compiles = 0

    def backend(gm, example_inputs, options=None):
        nonlocal compiles
        compiles += 1
        return luminal_backend(gm, example_inputs, options={"search_iterations": 1})

    compiled = torch.compile(function, backend=backend, fullgraph=True, dynamic=False)
    actual = compiled(*(value.to(device) for value in inputs))
    assert compiles == 1
    actual = tuple(value.cpu() for value in actual)
    _assert_close(actual, expected, rtol=tolerance, atol=tolerance)


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


def _backend_path(module: torch.nn.Module, monkeypatch, **compile_options):
    """`torch.compile` through the luminal backend, the real two-stage path.

    Returns the compiled module and the ATen targets of every program handed
    to the translator. Stage 1 is the undecomposed graph; a second attempt
    appears only if stage 1 raised "unsupported ATen op". The list doubles as
    a recompile guard: one entry per compilation.
    """
    attempts: list[set[str]] = []
    save = torch.export.save

    def recording_save(program, *args, **kwargs):
        attempts.append(
            {
                str(node.target)
                for node in program.graph.nodes
                if node.op == "call_function"
            }
        )
        return save(program, *args, **kwargs)

    monkeypatch.setattr(torch.export, "save", recording_save)
    compiled = torch.compile(
        module, backend=luminal_backend, fullgraph=True, **compile_options
    )
    return compiled, attempts


def _assert_stage_one(attempts, *targets: str) -> None:
    assert len(attempts) == 1, f"expected one stage-1 translation, got {attempts}"
    for target in targets:
        assert target in attempts[0], f"{target} was decomposed: {attempts[0]}"


class SearchVariants(torch.nn.Module):
    def forward(self, boundaries, queries, rows, row_queries, point, unsorted, sorter):
        empty = boundaries[:0]
        return (
            torch.searchsorted(boundaries, queries),
            torch.searchsorted(boundaries, queries, right=True),
            torch.searchsorted(boundaries, queries, side="right"),
            torch.searchsorted(boundaries, queries, out_int32=True),
            torch.searchsorted(boundaries, queries, out_int32=True, right=True),
            torch.searchsorted(rows, row_queries),
            torch.searchsorted(rows, row_queries, side="right"),
            torch.searchsorted(boundaries, row_queries),
            torch.searchsorted(boundaries, point),
            torch.searchsorted(boundaries, 3.0),
            torch.searchsorted(boundaries, float("nan"), right=True, out_int32=True),
            torch.searchsorted(unsorted, queries, sorter=sorter),
            torch.searchsorted(
                unsorted, queries, sorter=sorter, side="right", out_int32=True
            ),
            torch.searchsorted(empty, queries),
            torch.bucketize(queries, boundaries),
            torch.bucketize(queries, boundaries, right=True, out_int32=True),
            torch.bucketize(row_queries, boundaries),
        )


def test_search_composites_match_eager_on_the_backend_path(monkeypatch) -> None:
    # The insertion index is an exact integer count in [0, row_length]; NaN
    # queries sort past every boundary, and ±inf boundaries are ordinary.
    nan, inf = float("nan"), float("inf")
    unsorted = torch.tensor([5.0, 3.0, -inf, 1.0, 3.0, inf])
    inputs = (
        torch.tensor([-inf, 1.0, 3.0, 3.0, 5.0, inf]),
        torch.tensor([nan, -inf, 0.0, 1.0, 3.0, 4.0, 5.0, 8.0, inf]),
        torch.tensor([[1.0, 3.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0]]),
        torch.tensor([[0.0, 3.0, 7.0, nan], [-1.0, 2.0, 5.0, 9.0]]),
        torch.tensor(3.0),
        unsorted,
        torch.argsort(unsorted),
    )
    module = SearchVariants()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(
        attempts,
        "aten.searchsorted.Tensor",
        "aten.searchsorted.Scalar",
        "aten.bucketize.Tensor",
    )


class SearchDtypes(torch.nn.Module):
    def forward(self, boundaries, queries):
        return (
            torch.searchsorted(boundaries, queries),
            torch.searchsorted(boundaries, queries, right=True, out_int32=True),
            torch.bucketize(queries, boundaries),
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.float16, torch.bfloat16, torch.uint8, torch.int64],
)
def test_search_composites_match_eager_across_dtypes(dtype, monkeypatch) -> None:
    queries = [0.0, 1.0, 3.0, 4.0, 7.0, 8.0]
    if dtype.is_floating_point:
        queries += [float("nan"), float("inf"), -float("inf")]
    inputs = (
        torch.tensor([1.0, 3.0, 3.0, 5.0, 7.0]).to(dtype),
        torch.tensor(queries).to(dtype),
    )
    module = SearchDtypes()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.searchsorted.Tensor", "aten.bucketize.Tensor")


class SearchSorted(torch.nn.Module):
    def forward(self, boundaries, queries):
        return torch.searchsorted(boundaries, queries)


def test_searchsorted_matches_eager_with_symbolic_lengths(monkeypatch) -> None:
    # A symbolic row length is the reduced extent of the count, so it is
    # what the integer sum's width proof has to bound.
    module = SearchSorted()
    compiled, attempts = _backend_path(module, monkeypatch, dynamic=True)
    nan = float("nan")
    for boundaries, queries in (
        ([1.0, 3.0, 5.0, 7.0], [0.0, 2.0, 4.0, 6.0, 8.0, nan]),
        ([1.0, 3.0, 5.0, 7.0, 9.0, 11.0], [nan, 10.0, 3.0]),
    ):
        inputs = (torch.tensor(boundaries), torch.tensor(queries))
        _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.searchsorted.Tensor")


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
def test_searchsorted_count_is_exact_past_the_f32_integer_range(
    dtype, monkeypatch
) -> None:
    # 2^24 + 1 is the first count an F32 accumulator cannot hold; the float
    # row also needs the NaN redirect to produce that same row length.
    length = (1 << 24) + 1
    queries = [1.0, float("nan")] if dtype.is_floating_point else [1]
    inputs = (torch.zeros(length, dtype=dtype), torch.tensor(queries, dtype=dtype))
    module = SearchSorted()
    compiled, attempts = _backend_path(module, monkeypatch)
    expected = module(*inputs)
    assert expected.tolist() == [length] * len(queries)
    _assert_close(compiled(*inputs), expected)
    _assert_stage_one(attempts, "aten.searchsorted.Tensor")


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


class InferenceBatchNorm(torch.nn.Module):
    def __init__(self, affine: bool):
        super().__init__()
        self.affine = affine

    def forward(self, value, weight, bias, running_mean, running_var):
        return torch.ops.aten._native_batch_norm_legit_no_training.default(
            value,
            weight if self.affine else None,
            bias if self.affine else None,
            running_mean,
            running_var,
            0.2,
            1e-5,
        )


def test_inference_batch_norm_uses_running_stats_and_returns_empty_saved_stats() -> (
    None
):
    torch.manual_seed(11)
    inputs = (
        torch.randn(2, 3, 4, 5),
        torch.randn(3),
        torch.randn(3),
        torch.randn(3) + 3.0,
        torch.rand(3) + 0.5,
    )
    for affine in (False, True):
        module = InferenceBatchNorm(affine)
        expected = module(*inputs)
        assert expected[1].numel() == 0
        assert expected[2].numel() == 0
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


class ClampedGcd(torch.nn.Module):
    """gcd over operands clamped to ``[-bound, bound]``.

    The clamp attests the value range the reference backend needs before it
    plans integer arithmetic over caller data. ``bound`` is the dtype's max,
    so only the minimum (whose magnitude the dtype cannot hold) is excluded.
    """

    def __init__(self, bound: int) -> None:
        super().__init__()
        self.bound = bound

    def forward(self, lhs, rhs):
        return torch.ops.aten.gcd.default(
            lhs.clamp(-self.bound, self.bound), rhs.clamp(-self.bound, self.bound)
        )


def _fibonacci(k: int) -> int:
    """F(k), with F(1) = F(2) = 1."""
    previous, current = 0, 1
    for _ in range(k - 1):
        previous, current = current, previous + current
    return current


@pytest.mark.parametrize(
    ("dtype", "k"), [(torch.int32, 46), (torch.int64, 92)], ids=["int32", "int64"]
)
def test_integer_gcd_finishes_the_dtypes_worst_case_euclid_chain(dtype, k) -> None:
    # F(k) is the largest Fibonacci number the dtype holds, so (F(k-1), F(k))
    # needs k-1 divisions: the most any in-range pair needs (Lame's theorem).
    # A lowering that stops early returns a Fibonacci number instead of 1.
    bound = torch.iinfo(dtype).max
    small, large = _fibonacci(k - 1), _fibonacci(k)
    assert large <= bound < _fibonacci(k + 1)
    # F35, F34 needs 33 divisions, one past the old 32-round unroll.
    lhs = [small, large, -small, small, large, 0, 0, large, -12, 9227465]
    rhs = [large, small, large, -large, large, 0, -large, 0, 18, 5702887]
    _check(
        ClampedGcd(bound),
        torch.tensor(lhs, dtype=dtype),
        torch.tensor(rhs, dtype=dtype),
    )


def test_ordering_and_distance_composites() -> None:
    value = torch.tensor([[1.0, 4.0, 2.0, 2.0], [float("nan"), 2.0, 3.0, 4.0]])
    other = torch.tensor([[0.0, 3.0, 1.0, 4.0], [1.0, 2.0, 5.0, 3.0]])
    points = torch.tensor([[0.0, 1.0], [2.0, 3.0], [-1.0, 4.0]])
    points2 = torch.tensor([[1.0, 0.0], [3.0, -2.0]])
    module = OrderingAndDistances()
    inputs = (value, other, points, points2)
    _check(module, *inputs)


class PairwiseDistance(torch.nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, points):
        return torch.nn.functional.pdist(points, self.p)


def _check_pdist_real_path(monkeypatch, points, p) -> None:
    """Compile through `torch.compile`'s two-stage backend and require that
    stage 1 translated the undecomposed `_pdist_forward` (no stage-2 retry),
    so the pdist lowering itself is what runs."""
    saved = []
    save = torch.export.save

    def recording_save(program, *args, **kwargs):
        saved.append(
            {
                str(node.target)
                for node in program.graph.nodes
                if node.op == "call_function"
            }
        )
        return save(program, *args, **kwargs)

    monkeypatch.setattr(torch.export, "save", recording_save)
    module = PairwiseDistance(p)
    actual = torch.compile(module, backend=luminal_backend)(points)
    assert saved == [{"aten._pdist_forward.default"}]
    _assert_close(actual, module(points))


@pytest.mark.parametrize("p", [1.0, 2.0, float("inf"), 0.5])
def test_pdist_matches_eager_across_orders(monkeypatch, p) -> None:
    torch.manual_seed(5)
    _check_pdist_real_path(monkeypatch, torch.randn(7, 3), p)


# F32 cannot hold the flat pair index or 4n(n-1) exactly past 2^24; an
# F32 inversion of the pair index first goes wrong at n = 4609 rows.
@pytest.mark.slow
@pytest.mark.parametrize("rows", [4609, 6000])
def test_pdist_pair_order_is_exact_past_f32_integer_range(monkeypatch, rows) -> None:
    torch.manual_seed(6)
    _check_pdist_real_path(monkeypatch, torch.randn(rows, 1), 1.0)


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


class SegmentProduct(torch.nn.Module):
    def forward(self, value, lengths):
        return torch.segment_reduce(value, "prod", lengths=lengths)


def test_segment_reduce_product_sign_counts_negative_infinity() -> None:
    inf = float("inf")
    value = torch.tensor([-inf, 2.0, 3.0, inf, -2.0, -inf, 2.0, -1.0, -inf, -inf])
    lengths = torch.tensor([2, 1, 2, 3, 2])
    _check(SegmentProduct(), value, lengths)
