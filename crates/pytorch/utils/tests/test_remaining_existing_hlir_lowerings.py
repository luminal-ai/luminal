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


class _RealPath:
    """``torch.compile`` through the backend's real path, counting compiles.

    ``dynamic=False`` keeps one graph per input shape, so a test can push many
    value sets through a single compile.
    """

    def __init__(self, function):
        self.compiles = 0

        def backend(gm, example_inputs, options=None):
            self.compiles += 1
            return luminal_backend(gm, example_inputs, options={"search_iterations": 1})

        self._compiled = torch.compile(
            function, backend=backend, fullgraph=True, dynamic=False
        )

    def __call__(self, device, *inputs):
        actual = self._compiled(*(value.to(device) for value in inputs))
        return tuple(value.cpu() for value in actual)


def _assert_same(actual, expected, label) -> None:
    """Bit-for-bit equality, dtype included (NaN equal to NaN)."""
    assert len(actual) == len(expected), label
    for index, (result, reference) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(
            result,
            reference,
            rtol=0,
            atol=0,
            equal_nan=True,
            check_stride=False,
            msg=f"{label}: output {index}",
        )


def _uniform_edge_range(generator, dtype, *, tiny):
    """A random ``(lower, upper)`` in ``dtype``, over many binades.

    With ``tiny``, the range spans one to four ulps, so a few bins repeat each
    edge value.
    """

    def uniform(low, high):
        unit = torch.rand((), generator=generator, dtype=torch.float64).item()
        return low + (high - low) * unit

    scale = 10.0 ** torch.randint(-6, 7, (), generator=generator).item()
    lower = torch.tensor(uniform(-10.0, 10.0) * scale, dtype=dtype)
    if tiny:
        upper = lower
        for _ in range(torch.randint(1, 5, (), generator=generator).item()):
            upper = torch.nextafter(upper, torch.tensor(float("inf"), dtype=dtype))
    else:
        width = 10.0 ** torch.randint(-3, 2, (), generator=generator).item()
        upper = torch.tensor(
            lower.item() + uniform(0.0, 20.0) * width * scale, dtype=dtype
        )
    return lower.item(), upper.item()


def _uniform_edge_samples(generator, dtype, lower, upper, count, edges):
    """``count`` samples whose min is ``lower`` and max is ``upper``.

    The extremes are samples, so the uniform edges run exactly from one to the
    other and the maximum must land in the closed last bin. Every value of
    ``edges`` is a sample too. The rest are random in between, and the order
    is shuffled.
    """
    fixed = torch.cat((torch.tensor([lower, upper], dtype=dtype), edges.to(dtype)))
    unit = torch.rand(count - fixed.numel(), generator=generator, dtype=torch.float64)
    # Rounding a value inside [lower, upper] to the dtype keeps it inside.
    random = (lower + (upper - lower) * unit).to(dtype)
    samples = torch.cat((fixed, random))
    return samples[torch.randperm(count, generator=generator)]


def _linspace_interior(lower, upper, bins, dtype):
    # ``torch.linspace`` is the kernel eager histogram builds its edges with.
    return torch.linspace(lower, upper, bins + 1, dtype=dtype)[1:-1]


_UNIFORM_EDGE_CASES = 48
_UNIFORM_EDGE_SAMPLES = 24


@torch.no_grad()
@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("bins", (1, 4, 7))
def test_histogram_uniform_edges_match_eager_f32(device, bins, density) -> None:
    """``histogram.bin_ct`` edges and bins agree with eager bit-for-bit.

    Every case puts samples on all of eager's edges, including ranges a few
    ulps wide whose edges repeat. Repeated edges exercise the kernel's local
    search, whose bin for a sample on a repeated edge depends on the
    interpolated estimate. One graph serves every case. The F32 edges round
    once, through F64, exactly as eager's fused multiply-add does.
    """
    dtype = torch.float32
    generator = torch.Generator(device="cpu").manual_seed(1000 + bins)

    def function(values):
        return tuple(torch.histogram(values, bins, density=density))

    compiled = _RealPath(function)
    for case in range(_UNIFORM_EDGE_CASES):
        lower, upper = _uniform_edge_range(generator, dtype, tiny=case % 4 == 3)
        edges = _linspace_interior(lower, upper, bins, dtype)
        values = _uniform_edge_samples(
            generator, dtype, lower, upper, _UNIFORM_EDGE_SAMPLES, edges
        )
        expected = function(values)
        actual = compiled(device, values)
        label = f"case {case}: range ({lower!r}, {upper!r})"
        if density:
            # The normalising divide is the frontend's reciprocal multiply.
            _assert_same(actual[1:], expected[1:], label)
            _assert_close(actual[:1], expected[:1], rtol=1e-5, atol=0)
        else:
            _assert_same(actual, expected, label)
    assert compiled.compiles == 1


@torch.no_grad()
@pytest.mark.parametrize("density", (False, True))
@pytest.mark.parametrize("bins", (1, 4, 7))
def test_histogram_uniform_edges_match_eager_f64(device, bins, density) -> None:
    """``histogram.bin_ct`` in F64: exact ends, counts as eager.

    Samples sit on the first and last edges, and the maximum must land in the
    closed last bin. That is the case that used to drop it: the old last edge
    was ``lower + bins * width / bins``, which rounded below the maximum.
    Interior F64 edges are within one ulp of eager's, because the IR has no
    fused multiply-add, so no sample is placed on them here; see
    ``test_histogram_f64_samples_on_interior_edges_match_eager``.
    """
    dtype = torch.float64
    generator = torch.Generator(device="cpu").manual_seed(2000 + bins)

    def function(values):
        return tuple(torch.histogram(values, bins, density=density))

    compiled = _RealPath(function)
    for case in range(_UNIFORM_EDGE_CASES):
        lower, upper = _uniform_edge_range(generator, dtype, tiny=False)
        values = _uniform_edge_samples(
            generator, dtype, lower, upper, _UNIFORM_EDGE_SAMPLES, torch.empty(0)
        )
        expected = function(values)
        actual = compiled(device, values)
        label = f"case {case}: range ({lower!r}, {upper!r})"
        hist, edges = actual
        expected_hist, expected_edges = expected
        _assert_same(
            (edges[[0, -1]],), (expected_edges[[0, -1]],), f"{label}: end edges"
        )
        ulp = torch.finfo(dtype).eps * max(abs(lower), abs(upper))
        torch.testing.assert_close(
            edges, expected_edges, rtol=0, atol=ulp, msg=f"{label}: edges"
        )
        if density:
            _assert_close((hist,), (expected_hist,), rtol=1e-12, atol=0)
        else:
            _assert_same((hist,), (expected_hist,), f"{label}: counts")
    assert compiled.compiles == 1


@torch.no_grad()
@pytest.mark.xfail(
    strict=True,
    reason=(
        "ATen's linspace kernel rounds each `base + step * n` once (a fused "
        "multiply-add). The IR has no fused multiply-add and F64 has no wider "
        "type, so an F64 interior edge can sit one ulp from eager's and move "
        "a sample that sits on it."
    ),
)
def test_histogram_f64_samples_on_interior_edges_match_eager(device) -> None:
    dtype, bins = torch.float64, 7
    generator = torch.Generator(device="cpu").manual_seed(3000)

    def function(values):
        return tuple(torch.histogram(values, bins))

    compiled = _RealPath(function)
    for case in range(_UNIFORM_EDGE_CASES):
        lower, upper = _uniform_edge_range(generator, dtype, tiny=case % 4 == 3)
        edges = _linspace_interior(lower, upper, bins, dtype)
        values = _uniform_edge_samples(
            generator, dtype, lower, upper, _UNIFORM_EDGE_SAMPLES, edges
        )
        _assert_same(compiled(device, values), function(values), f"case {case}")


@torch.no_grad()
@pytest.mark.parametrize("density", (False, True))
def test_histogramdd_uniform_edges_match_eager(device, density) -> None:
    """``_histogramdd_bin_edges`` and ``_histogramdd_from_bin_cts`` in F32.

    Each axis has its own random range, both extremes are samples on that
    axis, and every eager edge of that axis is a sample coordinate.
    """
    dtype, bins = torch.float32, [3, 5]
    generator = torch.Generator(device="cpu").manual_seed(4000)

    def function(points):
        hist, edges = torch.histogramdd(points, bins, density=density)
        return (hist, *edges)

    compiled = _RealPath(function)
    for case in range(_UNIFORM_EDGE_CASES // 2):
        columns = []
        for axis_bins in bins:
            lower, upper = _uniform_edge_range(generator, dtype, tiny=case % 4 == 3)
            edges = _linspace_interior(lower, upper, axis_bins, dtype)
            columns.append(
                _uniform_edge_samples(
                    generator, dtype, lower, upper, _UNIFORM_EDGE_SAMPLES, edges
                )
            )
        points = torch.stack(columns, dim=1)
        expected = function(points)
        actual = compiled(device, points)
        label = f"case {case}"
        if density:
            _assert_same(actual[1:], expected[1:], label)
            _assert_close(actual[:1], expected[:1], rtol=1e-5, atol=0)
        else:
            _assert_same(actual, expected, label)
    assert compiled.compiles == 1


# (values, bins, histogram kwargs): fixed regressions for uniform edges.
_HISTOGRAM_UNIFORM_EDGE_REGRESSIONS = {
    # The old last edge, `lower + 4 * width / 4`, rounded to
    # 2.9999999999999996 and dropped the maximum: eager counts [1, 1, 0, 1].
    "max_on_last_edge_f64": (
        torch.tensor([-1.1195351830271254, 0.0, 3.0], dtype=torch.float64),
        4,
        {},
    ),
    # Ranges of one or two ulps repeat each edge. Eager's local search puts
    # the minimum in bin 1 (the last repeat inside its window), not bin 5.
    "repeated_edges_f32": (
        torch.tensor([1.0, 1.0 + 2**-23], dtype=torch.float32),
        10,
        {},
    ),
    "repeated_edges_f64": (
        torch.tensor([1.0, 1.0 + 2**-52], dtype=torch.float64),
        10,
        {},
    ),
    "repeated_edges_three_values": (
        torch.tensor([1.0, 1.0 + 2**-23, 1.0 + 2**-22], dtype=torch.float32),
        7,
        {},
    ),
    # An interior edge of an explicit F32 range: (2 - 1) / 3 must round once.
    "explicit_range_f32": (
        torch.tensor([1.0, 1.5, 5 / 3, 2.0], dtype=torch.float32),
        3,
        {"range": (1.0, 2.0)},
    ),
    # An explicit range with equal ends is widened by 0.5 before rounding.
    "explicit_degenerate_range": (
        torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32),
        2,
        {"range": (1.0, 1.0)},
    ),
    # No samples: eager's edges come from (0, 1), or from `range`.
    "empty": (torch.empty(0, dtype=torch.float32), 3, {}),
    "empty_with_range": (
        torch.empty(0, dtype=torch.float64),
        3,
        {"range": (1.0, 2.0)},
    ),
}


@torch.no_grad()
@pytest.mark.parametrize("case", sorted(_HISTOGRAM_UNIFORM_EDGE_REGRESSIONS))
def test_histogram_uniform_edge_regressions(device, case) -> None:
    values, bins, kwargs = _HISTOGRAM_UNIFORM_EDGE_REGRESSIONS[case]

    def function(values):
        return tuple(torch.histogram(values, bins, **kwargs))

    compiled = _RealPath(function)
    _assert_same(compiled(device, values), function(values), case)
    assert compiled.compiles == 1


@torch.no_grad()
def test_histogramdd_without_samples_matches_eager(device) -> None:
    points = torch.empty(0, 2, dtype=torch.float32)

    def function(points):
        hist, edges = torch.histogramdd(points, [2, 3])
        return (hist, *edges)

    compiled = _RealPath(function)
    _assert_same(compiled(device, points), function(points), "empty")
    assert compiled.compiles == 1


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


# Integer counts, indices, and offsets are exact integers that never pass
# through a float (F32 holds integers exactly only to 2^24). Each case runs
# the real two-stage backend path and checks that stage 1 handed the op to
# the translator, so the lowering under test is the one exercised.

_F24 = 1 << 24


class TrueCoordinates(torch.nn.Module):
    def forward(self, value, count, flags):
        return (
            torch.nonzero_static(value, size=7, fill_value=-2),
            torch.nonzero_static(value, size=2),
            torch.nonzero_static(count, size=5, fill_value=9),
            torch.nonzero_static(flags, size=4, fill_value=-1),
            torch.tril_indices(4, 5, 1),
            torch.tril_indices(3, 3, -1),
            torch.triu_indices(5, 3, -1),
            torch.triu_indices(3, 4, 2),
        )


def test_true_coordinate_composites_match_eager_on_the_backend_path(
    monkeypatch,
) -> None:
    # -0.0 is zero; rows past the true count hold the fill.
    inputs = (
        torch.tensor([[0.0, 1.5, 0.0], [-0.0, -4.0, 2.0]]),
        torch.tensor([[0, 3], [5, 0], [0, 7]]),
        torch.tensor([[[True, False], [False, True]], [[False, False], [True, True]]]),
    )
    module = TrueCoordinates()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(
        attempts,
        "aten.nonzero_static.default",
        "aten.tril_indices.default",
        "aten.triu_indices.default",
    )


class ScatterDestinations(torch.nn.Module):
    def forward(self, value, flat, rows, cols, updates, ints, index, src):
        return (
            torch.put(value, flat, updates, accumulate=True),
            torch.put(ints, flat, updates.to(torch.int64), accumulate=True),
            torch.index_put(value, (rows, cols), updates, accumulate=True),
            torch.scatter_add(value, 1, index, src),
            torch.scatter_reduce(value, 0, index, src, "mean", include_self=False),
            torch.scatter_reduce(value, 1, index, src, "amax"),
            torch.scatter_reduce(value, 0, index, src, "prod", include_self=False),
            torch.index_reduce(value, 0, index[:, 0], src[:, :1].expand(3, 4), "amin"),
        )


def test_scatter_destinations_match_eager_on_the_backend_path(monkeypatch) -> None:
    # Negative flat and per-axis indices wrap; duplicates accumulate in order.
    inputs = (
        torch.tensor(
            [[1.0, -2.0, 3.0, 0.5], [4.0, 5.0, -6.0, 7.0], [8.0, 9.0, 1.0, -1.0]]
        ),
        torch.tensor([-1, 11, 0, -12]),
        torch.tensor([2, -1, 0, 2]),
        torch.tensor([1, 3, -4, 1]),
        torch.tensor([0.5, -1.5, 2.0, 4.0]),
        torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        torch.tensor([[2, 0], [1, 1], [0, 2]]),
        torch.tensor([[1.0, -2.0], [3.0, 0.25], [-4.0, 5.0]]),
    )
    module = ScatterDestinations()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(
        attempts,
        "aten.put.default",
        "aten.index_put.default",
        "aten.scatter_add.default",
        "aten.scatter_reduce.two",
        "aten.index_reduce.default",
    )


class MaskedPrefix(torch.nn.Module):
    def forward(self, value, mask, source, ints, int_source, row_mask):
        none = torch.zeros_like(mask)
        return (
            value.masked_scatter(mask, source),
            ints.masked_scatter(mask, int_source),
            value.masked_scatter(row_mask, source),
            value.masked_scatter(none, source),
        )


def test_masked_scatter_prefix_counts_match_eager_on_the_backend_path(
    monkeypatch,
) -> None:
    # The source element for a true slot is that slot's rank among the trues;
    # the [1, 4] mask broadcasts across rows, and an all-false mask is a copy.
    # Every source has one element per slot: search-time planning executes
    # candidates on all-ones placeholder inputs, where every mask slot is
    # true and a shorter source reads out of bounds (LUM-1098).
    inputs = (
        torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
        ),
        torch.tensor(
            [
                [True, False, False, True],
                [False, False, False, False],
                [True, True, False, False],
            ]
        ),
        torch.arange(12, dtype=torch.float32) * -1.5,
        torch.arange(12).reshape(3, 4),
        torch.arange(12) * 10 - 7,
        torch.tensor([[False, True, True, False]]),
    )
    module = MaskedPrefix()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.masked_scatter.default")


class IntervalCounts(torch.nn.Module):
    def forward(self, repeats, value, lengths, offsets):
        return (
            torch.repeat_interleave(repeats, output_size=7),
            torch.segment_reduce(value, "sum", lengths=lengths, axis=1),
            torch.segment_reduce(value, "mean", lengths=lengths, axis=1),
            torch.segment_reduce(value, "mean", offsets=offsets, axis=1, initial=1.0),
            torch.segment_reduce(value, "prod", lengths=lengths, axis=1),
            torch.segment_reduce(value, "max", offsets=offsets, axis=1),
            torch.segment_reduce(value, "min", lengths=lengths, axis=1, initial=0.5),
        )


def test_interval_counts_match_eager_on_the_backend_path(monkeypatch) -> None:
    # Zero repeats and empty segments: an interval of length zero owns nothing.
    value = torch.tensor(
        [[1.0, -3.0, -2.0, 5.0, 4.0, -0.5], [-1.0, 2.0, 6.0, -3.0, -7.0, -2.0]]
    )
    lengths = torch.tensor([[0, 1, 3, 2], [2, 0, 3, 1]])
    inputs = (
        torch.tensor([2, 0, 3, 2]),
        value,
        lengths,
        torch.nn.functional.pad(lengths, [1, 0]).cumsum(1),
    )
    module = IntervalCounts()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(
        attempts, "aten.repeat_interleave.Tensor", "aten.segment_reduce.default"
    )


class AdaptiveWindows(torch.nn.Module):
    def forward(self, value):
        return (
            torch.nn.functional.adaptive_avg_pool2d(value, (3, 4)),
            torch.nn.functional.adaptive_max_pool2d(value, (4, 3), return_indices=True),
            torch.nn.functional.adaptive_avg_pool2d(value, (5, 7)),
        )


def test_adaptive_pool_windows_match_eager_on_the_backend_path(monkeypatch) -> None:
    # Non-dividing sizes give overlapping windows of unequal length.
    torch.manual_seed(5)
    inputs = (torch.randn(1, 2, 5, 7),)
    module = AdaptiveWindows()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(
        attempts,
        "aten._adaptive_avg_pool2d.default",
        "aten.adaptive_max_pool2d.default",
    )


class NanMedianSlots(torch.nn.Module):
    def forward(self, value):
        return torch.nanmedian(value, dim=1), torch.nanmedian(
            value, dim=1, keepdim=True
        )


def test_nanmedian_slot_matches_eager_on_the_backend_path(monkeypatch) -> None:
    # Odd and even non-NaN counts, a single value, and an all-NaN row: the
    # slot is (count - 1) // 2, or 0 with no values. The NaNs trail each row:
    # the lowering's NaN reorder is composed in the wrong frame when a NaN
    # precedes a value (reported separately), and this test is about the slot.
    nan = float("nan")
    inputs = (
        torch.tensor(
            [
                [3.0, 1.0, 2.0, nan, nan],
                [4.0, 1.0, 3.0, 2.0, 5.0],
                [7.0, nan, nan, nan, nan],
                [nan, nan, nan, nan, nan],
                [2.0, 2.0, 1.0, 1.0, nan],
            ]
        ),
    )
    module = NanMedianSlots()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.nanmedian.dim")


class EmbeddingBagCounts(torch.nn.Module):
    def forward(self, weight, indices, offsets):
        bag = torch.ops.aten._embedding_bag_forward_only.default
        return (
            bag(weight, indices, offsets, False, 0),
            bag(weight, indices, offsets, False, 0, padding_idx=2),
            bag(weight, indices, offsets, False, 1),
            bag(weight, indices, offsets, False, 1, padding_idx=2),
            bag(weight, indices, offsets, False, 2),
        )


def test_embedding_bag_counts_match_eager_on_the_backend_path(monkeypatch) -> None:
    # Bag sizes and offset-to-bag ids are exact counts; one bag is empty.
    torch.manual_seed(6)
    inputs = (
        torch.randn(6, 3),
        torch.tensor([1, 2, 0, 3, 2, 5, 4]),
        torch.tensor([0, 2, 2, 5]),
    )
    module = EmbeddingBagCounts()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten._embedding_bag_forward_only.default")


class FlatPut(torch.nn.Module):
    def forward(self, value, index, source):
        return torch.put(value, index, source, accumulate=True)


def test_put_flat_index_is_exact_past_the_f32_integer_range(monkeypatch) -> None:
    # -1 wraps to 2^24 + 1, the first index an F32 cannot hold. (accumulate:
    # search-time planning runs on all-ones placeholder indices, which
    # collide in an overwriting put; LUM-1098.)
    length = _F24 + 2
    inputs = (torch.zeros(length), torch.tensor([-1, -2]), torch.tensor([7.0, 5.0]))
    module = FlatPut()
    compiled, attempts = _backend_path(module, monkeypatch)
    expected = module(*inputs)
    assert expected[-2:].tolist() == [5.0, 7.0]
    _assert_close(compiled(*inputs), expected)
    _assert_stage_one(attempts, "aten.put.default")


class AccumulatedIndexPut(torch.nn.Module):
    def forward(self, value, rows, cols, updates):
        return torch.index_put(value, (rows, cols), updates, accumulate=True)


def test_accumulated_index_put_destination_is_exact_past_the_f32_integer_range(
    monkeypatch,
) -> None:
    # (4096, 4095) of a 4097 x 4097 tensor is flat element 2^24 + 8191, an
    # odd integer past 2^24 that an F32 flat index rounds to its neighbour.
    inputs = (
        torch.zeros(4097, 4097),
        torch.tensor([4096]),
        torch.tensor([4095]),
        torch.tensor([3.0]),
    )
    module = AccumulatedIndexPut()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.index_put.default")


class MaskedScatter(torch.nn.Module):
    def forward(self, value, mask, source):
        return value.masked_scatter(mask, source)


@pytest.mark.slow
def test_masked_scatter_prefix_is_exact_past_the_f32_integer_range(
    monkeypatch,
) -> None:
    # Distinct int64 sources expose any slot whose rank is off by one.
    length = _F24 + 3
    inputs = (
        torch.zeros(length, dtype=torch.int64),
        torch.ones(length, dtype=torch.bool),
        torch.arange(length),
    )
    module = MaskedScatter()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.masked_scatter.default")


class LongRepeat(torch.nn.Module):
    def forward(self, repeats):
        return torch.repeat_interleave(repeats, output_size=_F24 + 2)


@pytest.mark.slow
def test_repeat_interleave_intervals_are_exact_past_the_f32_integer_range(
    monkeypatch,
) -> None:
    inputs = (torch.tensor([1, _F24 + 1]),)
    module = LongRepeat()
    compiled, attempts = _backend_path(module, monkeypatch)
    _assert_close(compiled(*inputs), module(*inputs))
    _assert_stage_one(attempts, "aten.repeat_interleave.Tensor")


class LongSegmentMax(torch.nn.Module):
    def forward(self, value, lengths):
        return torch.segment_reduce(value, "max", lengths=lengths)


@pytest.mark.slow
def test_segment_reduce_bounds_are_exact_past_the_f32_integer_range(
    monkeypatch,
) -> None:
    # Segment 0 ends at 2^24 + 1, so its last element sits at position 2^24.
    value = torch.zeros(_F24 + 2)
    value[-2], value[-1] = 5.0, -3.0
    inputs = (value, torch.tensor([_F24 + 1, 1]))
    module = LongSegmentMax()
    compiled, attempts = _backend_path(module, monkeypatch)
    expected = module(*inputs)
    assert expected.tolist() == [5.0, -3.0]
    _assert_close(compiled(*inputs), expected)
    _assert_stage_one(attempts, "aten.segment_reduce.default")


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


class MaxPoolBackward(torch.nn.Module):
    def __init__(self, kernel, stride, padding, dilation, ceil_mode):
        super().__init__()
        self.window = (kernel, stride, padding, dilation, ceil_mode)

    def forward(self, grad, value, indices):
        return torch.ops.aten.max_pool2d_with_indices_backward.default(
            grad, value, *self.window, indices
        )


# (input shape, dtype, (kernel, stride, padding, dilation, ceil_mode)).
# Every case has overlapping windows, so some input positions receive several
# updates; the `ties` case rounds the input so equal maxima tie as well.
_MAX_POOL_BACKWARD_CASES = {
    "overlap": ((1, 2, 5, 5), torch.float32, ([3, 3], [1, 1], [0, 0], [1, 1], False)),
    "padding_stride": (
        (2, 2, 6, 6),
        torch.float32,
        ([3, 3], [2, 2], [1, 1], [1, 1], False),
    ),
    "dilation": ((1, 2, 6, 6), torch.float32, ([2, 2], [1, 1], [0, 0], [2, 2], False)),
    "ceil_mode": ((1, 2, 6, 6), torch.float32, ([3, 3], [2, 2], [0, 0], [1, 1], True)),
    "unbatched": ((2, 5, 5), torch.float32, ([3, 3], [1, 1], [0, 0], [1, 1], False)),
    "ties": ((1, 2, 5, 5), torch.float32, ([2, 2], [1, 1], [0, 0], [1, 1], False)),
    "float64": ((1, 2, 5, 5), torch.float64, ([3, 3], [1, 1], [1, 1], [1, 1], False)),
}


@pytest.mark.parametrize("case", sorted(_MAX_POOL_BACKWARD_CASES))
def test_max_pool2d_with_indices_backward_accumulates_like_eager(case) -> None:
    # Real torch.compile path, the op called directly with eager indices.
    shape, dtype, window = _MAX_POOL_BACKWARD_CASES[case]
    torch.manual_seed(0)
    value = torch.randn(shape, dtype=torch.float64)
    if case == "ties":
        value = value.round()
    value = value.to(dtype)
    _, indices = torch.ops.aten.max_pool2d_with_indices.default(value, *window)
    grad = torch.randn(indices.shape, dtype=dtype)
    module = MaxPoolBackward(*window)
    actual = torch.compile(module, backend=luminal_backend)(grad, value, indices)
    torch.testing.assert_close(actual, module(grad, value, indices))


@pytest.mark.slow
def test_max_pool2d_with_indices_backward_is_exact_past_f32_integers() -> None:
    # One window per 4097x4097 plane (16,785,409 > 2^24 elements). Each
    # plane's max sits at the odd in-plane position 2^24 + 1 (no f32 holds
    # it) and the second plane starts past 2^24 too, so a destination
    # routed through f32 arithmetic would round or refuse.
    side = 4097
    window = ([side, side], [side, side], [0, 0], [1, 1], False)
    value = torch.zeros(1, 2, side, side)
    row, column = divmod(2**24 + 1, side)
    value[0, :, row, column] = 1.0
    _, indices = torch.ops.aten.max_pool2d_with_indices.default(value, *window)
    assert indices.flatten().tolist() == [2**24 + 1, 2**24 + 1]
    grad = torch.tensor([3.0, 5.0]).reshape(1, 2, 1, 1)
    module = MaxPoolBackward(*window)
    actual = torch.compile(module, backend=luminal_backend)(grad, value, indices)
    assert torch.equal(actual, module(grad, value, indices))


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


# ---- zero tests read NaN as nonzero ----
#
# The order-0 "norm" counts the entries with `x != 0`. `NaN != 0` holds, so
# a NaN difference (`nan - x`, `inf - inf`) counts, while two signed zeros
# cancel to a zero that does not. Every other lowering that asks "is this
# zero?" (the translator's `is_zero`, which is torch's `== 0`) must also read
# NaN as nonzero, whether it counts nonzeros, tests truth, or substitutes a
# value at zero. Each op below is a stage-1 arm, so these run the real
# backend path.

_NAN, _INF = float("nan"), float("inf")
_NORM_ORDERS = (0, 1, 2, 3, 0.5, -1, _INF, -_INF)
_DIFFERENCE_CASES = {
    "nan-minus-finite": ([_NAN, 1.0, 0.0, 5.0], [0.0, 1.0, 0.0, 2.0]),
    "inf-minus-inf": ([_INF, 1.0, -_INF, 5.0], [_INF, 1.0, -_INF, 2.0]),
    "signed-zeros": ([-0.0, 0.0, -0.0, 3.0], [0.0, -0.0, -0.0, -1.0]),
}
_FLOAT_DTYPES = pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64], ids=str
)


class DistOrders(torch.nn.Module):
    def __init__(self, orders):
        super().__init__()
        self.orders = orders

    def forward(self, lhs, rhs):
        return tuple(torch.dist(lhs, rhs, p) for p in self.orders)


@_FLOAT_DTYPES
@pytest.mark.parametrize("case", sorted(_DIFFERENCE_CASES))
def test_dist_orders_count_nan_differences(case, dtype, monkeypatch) -> None:
    lhs, rhs = _DIFFERENCE_CASES[case]
    inputs = (torch.tensor(lhs, dtype=dtype), torch.tensor(rhs, dtype=dtype))
    # A `-0` difference is checked at order 0 only: at the negative orders
    # it meets a separate defect (the float `abs` keeps the sign of `-0`, so
    # `|-0|^-1` is `-inf`).
    orders = (0,) if case == "signed-zeros" else _NORM_ORDERS
    module = DistOrders(orders)
    compiled, attempts = _backend_path(module, monkeypatch)
    actual = compiled(*inputs)
    _assert_stage_one(attempts, "aten.dist.default")
    expected = module(*inputs)
    # The order-0 count is exact. The other orders go through pow at F32.
    _assert_close(actual[0], expected[0], rtol=0, atol=0)
    _assert_close(actual[1:], expected[1:], rtol=1e-5, atol=1e-6)


class VectorNormOrderZero(torch.nn.Module):
    def forward(self, value):
        return (
            torch.linalg.vector_norm(value, 0, dim=1),
            torch.linalg.vector_norm(value, 0),
        )


@_FLOAT_DTYPES
def test_vector_norm_order_zero_counts_nan(dtype, monkeypatch) -> None:
    value = torch.tensor(
        [[_NAN, 0.0, 1.0], [0.0, -0.0, _INF], [-0.0, _NAN, _NAN]], dtype=dtype
    )
    module = VectorNormOrderZero()
    compiled, attempts = _backend_path(module, monkeypatch)
    actual = compiled(value)
    _assert_stage_one(attempts, "aten.linalg_vector_norm.default")
    _assert_close(actual, module(value), rtol=0, atol=0)


class PairwiseOrderZero(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, 0), torch.pdist(torch.cat([x1, x2]), 0)


@_FLOAT_DTYPES
def test_cdist_pdist_order_zero_follow_the_cpu_kernel_on_nan(
    dtype, monkeypatch
) -> None:
    # Unlike dist and vector_norm, ATen's CPU cdist/pdist kernels map each
    # |difference| d to min(ceil(d), 1). A NaN difference therefore makes
    # that pair's count NaN, and an infinite one still counts 1. The CUDA
    # kernel counts `d != 0` instead. The lowering follows the CPU kernel,
    # so the reference is computed on CPU whatever the backend.
    x1 = torch.tensor(
        [[_NAN, 1.0, 0.0], [1.0, 2.0, 3.0], [_INF, 0.0, -0.0]], dtype=dtype
    )
    x2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 5.0, 3.0], [_INF, 0.0, 0.0]], dtype=dtype)
    module = PairwiseOrderZero()
    compiled, attempts = _backend_path(module, monkeypatch)
    actual = compiled(x1, x2)
    _assert_stage_one(
        attempts, "aten._cdist_forward.default", "aten._pdist_forward.default"
    )
    expected = module(x1.cpu(), x2.cpu())
    _assert_close(tuple(value.cpu() for value in actual), expected, rtol=0, atol=0)


class ZeroTestedFunctions(torch.nn.Module):
    def forward(self, value, x, degree, samples, edges):
        return (
            torch.nonzero_static(value, size=8, fill_value=-3),
            torch.special.spherical_bessel_j0(value),
            torch.special.ndtri(value),
            # Nonnegative and NaN inputs only: a negative real angle is a
            # separate defect (the lowering's copy_sign reads a zero sign as 0).
            torch.angle(value),
            torch.special.chebyshev_polynomial_t(x, degree),
            torch.histogram(samples, edges),
        )


@_FLOAT_DTYPES
def test_zero_tests_read_nan_as_nonzero(dtype, monkeypatch) -> None:
    # `tiny` is nonzero but below F32's range at F64: the test must not
    # narrow it to zero.
    tiny = 1e-300 if dtype == torch.float64 else 1e-30
    inputs = (
        torch.tensor([_NAN, 0.0, -0.0, tiny, 0.3, 1.0, 2.0, _INF], dtype=dtype),
        torch.tensor([_NAN, _NAN, _NAN, 0.3, 1.0, -1.0], dtype=dtype),
        torch.tensor([0.0, 1.0, 25.0, 25.0, 25.0, 25.0], dtype=dtype),
        # An infinite last edge holds an equal infinite sample.
        torch.tensor([0.5, _INF, 2.0, -1.0, _NAN, 1.0], dtype=dtype),
        torch.tensor([0.0, 1.0, _INF], dtype=dtype),
    )
    module = ZeroTestedFunctions()
    compiled, attempts = _backend_path(module, monkeypatch)
    actual = compiled(*inputs)
    _assert_stage_one(
        attempts,
        "aten.nonzero_static.default",
        "aten.special_spherical_bessel_j0.default",
        "aten.special_ndtri.default",
        "aten.angle.default",
        "aten.special_chebyshev_polynomial_t.default",
        "aten.histogram.bins_tensor",
    )
    _assert_close(actual, module(*inputs))


class ComplexZeroTests(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, real, imag, other_real, other_imag):
        return self.function(
            torch.complex(real, imag), torch.complex(other_real, other_imag)
        )


# name -> (function of (z, w), z parts, w parts, stage-1 target). The complex
# values are built in the graph from real inputs.
_COMPLEX_ZERO_TEST_CASES = {
    "bool": (
        lambda z, w: z.bool(),
        [[_NAN, 0.0], [0.0, _NAN], [0.0, 0.0], [1.0, 0.0]],
        None,
        "aten._to_copy.default",
    ),
    "any": (
        lambda z, w: z.reshape(2, 2).any(dim=1),
        [[_NAN, 0.0], [0.0, 0.0], [0.0, 0.0], [-0.0, 0.0]],
        None,
        "aten.any.dim",
    ),
    "logical_not": (
        lambda z, w: torch.logical_not(z),
        [[_NAN, 0.0], [0.0, _NAN], [0.0, 0.0], [1.0, 0.0]],
        None,
        "aten.logical_not.default",
    ),
    "sqrt": (
        lambda z, w: torch.view_as_real(torch.sqrt(z)),
        [[_NAN, 1.0], [0.0, _NAN], [0.0, 0.0], [4.0, 0.0]],
        None,
        "aten.sqrt.default",
    ),
    "div": (
        lambda z, w: torch.view_as_real(w / z),
        [[_NAN, 0.0], [0.0, _NAN], [2.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        "aten.div.Tensor",
    ),
    "pow": (
        lambda z, w: torch.view_as_real(w**z),
        [[_NAN, 0.0], [0.0, _NAN], [0.0, 0.0]],
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
        "aten.pow.Tensor_Tensor",
    ),
    "acos": (
        lambda z, w: torch.view_as_real(torch.acos(z)),
        [[_NAN, _NAN], [0.0, _NAN], [0.5, 0.2]],
        None,
        "aten.acos.default",
    ),
    # A negative real part is a separate defect (copy_sign of a zero sign).
    "angle": (
        lambda z, w: torch.angle(z),
        [[_NAN, 0.0], [0.0, _NAN], [0.0, 0.0], [1.0, 0.0]],
        None,
        "aten.angle.default",
    ),
}


@_FLOAT_DTYPES
@pytest.mark.parametrize("case", sorted(_COMPLEX_ZERO_TEST_CASES))
def test_complex_zero_tests_read_nan_as_nonzero(case, dtype, monkeypatch) -> None:
    function, z_parts, w_parts, target = _COMPLEX_ZERO_TEST_CASES[case]
    z = torch.tensor(z_parts, dtype=dtype)
    w = torch.tensor(z_parts if w_parts is None else w_parts, dtype=dtype)
    inputs = (z[:, 0], z[:, 1], w[:, 0], w[:, 1])
    inputs = tuple(value.contiguous() for value in inputs)
    module = ComplexZeroTests(function)
    compiled, attempts = _backend_path(module, monkeypatch)
    actual = compiled(*inputs)
    _assert_stage_one(attempts, target)
    _assert_close(actual, module(*inputs))


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


class FractionalMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, output_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size

    def forward(self, value, random_samples):
        return torch.ops.aten.fractional_max_pool2d.default(
            value, self.kernel_size, self.output_size, random_samples
        )


class FractionalMaxPool3d(FractionalMaxPool2d):
    def forward(self, value, random_samples):
        return torch.ops.aten.fractional_max_pool3d.default(
            value, self.kernel_size, self.output_size, random_samples
        )


def _check_fractional_max_pool_real_path(module, value, random_samples):
    """Compile through torch.compile, whose stage-1 translation keeps the
    undecomposed fractional_max_pool arm, and require eager's exact values
    and int64 indices."""
    actual = torch.compile(module, backend=luminal_backend)(value, random_samples)
    expected = module(value, random_samples)
    assert len(actual) == len(expected) == 2
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == reference.dtype, f"output {index}"
        assert result.shape == reference.shape, f"output {index}"
        assert torch.equal(result, reference), f"output {index}"


# (module, value shape, random_samples shape). The wide 2d case has
# inputSize - poolSize = 301, which bfloat16 cannot hold, so the window
# starts only match eager when every interval term rounds in the sample
# dtype the way ATen's generate_intervals<scalar_t> does.
_FRACTIONAL_MAX_POOL_CASES = {
    "2d": (FractionalMaxPool2d([2, 3], [3, 2]), (2, 3, 7, 8), (2, 3, 2)),
    "2d_unbatched": (FractionalMaxPool2d([3, 2], [4, 3]), (2, 9, 8), (1, 2, 2)),
    "2d_wide": (FractionalMaxPool2d([2, 2], [7, 7]), (1, 2, 303, 40), (1, 2, 2)),
    "3d": (FractionalMaxPool3d([2, 2, 3], [2, 3, 2]), (1, 2, 4, 5, 6), (1, 2, 3)),
    "3d_unbatched": (
        FractionalMaxPool3d([2, 2, 2], [3, 3, 3]),
        (2, 6, 6, 6),
        (1, 2, 3),
    ),
}


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.bfloat16, torch.float16],
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
@pytest.mark.parametrize("case", sorted(_FRACTIONAL_MAX_POOL_CASES))
def test_fractional_max_pool_matches_eager_for_every_float_dtype(case, dtype) -> None:
    # ATen requires random_samples in the input's dtype, so every float
    # input brings samples of its own width into the interval arithmetic.
    module, value_shape, samples_shape = _FRACTIONAL_MAX_POOL_CASES[case]
    torch.manual_seed(7)
    value = torch.randn(value_shape).to(dtype)
    random_samples = torch.rand(samples_shape).to(dtype)
    _check_fractional_max_pool_real_path(module, value, random_samples)


@pytest.mark.slow
@pytest.mark.parametrize("case", ["one_plane", "many_planes"])
def test_fractional_max_pool_indices_exact_past_f32_integers(case) -> None:
    # Inputs past 2**24 elements: a flat position there is not an F32
    # integer. One 4100 x 4100 plane puts the planted maximum at the odd
    # within-plane index 16,809,999; 8 x 130 planes put whole-input
    # positions past 2**24 while each plane stays small.
    torch.manual_seed(11)
    if case == "one_plane":
        value = torch.randn(1, 1, 4100, 4100)
        value[0, 0, 4099, 4099] = 100.0
        module = FractionalMaxPool2d([2, 2], [8, 8])
    else:
        value = torch.randn(8, 130, 128, 128)
        module = FractionalMaxPool2d([2, 2], [4, 4])
    random_samples = torch.rand(value.shape[0], value.shape[1], 2)
    _check_fractional_max_pool_real_path(module, value, random_samples)
    if case == "one_plane":
        assert module(value, random_samples)[1][0, 0, 7, 7].item() == 16_809_999


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
