import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from luminal import luminal_backend
from luminal.dtype_util import torch_dtype_code


class BoundaryNoopModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.bool:
            return x | torch.zeros((), dtype=torch.bool, device=x.device)
        return x + torch.zeros((), dtype=x.dtype, device=x.device)


class AbsModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class EmptyWeightModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.empty((0, 3), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight


class AddWithoutAlphaModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)


class AddWithAlphaModel(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y, alpha=self.alpha)


class AddScalarWithAlphaModel(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.add(x, 1.0, alpha=self.alpha)


class ItemModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x.item()


class ArangeModel(torch.nn.Module):
    def __init__(
        self,
        start,
        end,
        step,
        dtype: torch.dtype,
        *,
        keyword_end: bool = False,
    ) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype
        self.keyword_end = keyword_end

    def forward(self) -> torch.Tensor:
        if self.keyword_end:
            return torch.arange(
                self.start,
                end=self.end,
                step=self.step,
                dtype=self.dtype,
            )
        return torch.arange(
            self.start,
            self.end,
            self.step,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class DTypeCase:
    name: str
    dtype: torch.dtype
    values: Callable[[], torch.Tensor]
    xfail_reason: str | None = None


DTYPE_CASES = [
    DTypeCase(
        "bool",
        torch.bool,
        lambda: torch.tensor([True, False, True], dtype=torch.bool),
    ),
    DTypeCase(
        "uint8",
        torch.uint8,
        lambda: torch.tensor([0, 127, 255], dtype=torch.uint8),
    ),
    DTypeCase(
        "int8",
        torch.int8,
        lambda: torch.tensor([-128, -1, 127], dtype=torch.int8),
    ),
    DTypeCase(
        "int16",
        torch.int16,
        lambda: torch.tensor([-32768, -1, 32767], dtype=torch.int16),
    ),
    DTypeCase(
        "int32",
        torch.int32,
        lambda: torch.tensor(
            [-2147483648, -1, 2147483647],
            dtype=torch.int32,
        ),
    ),
    DTypeCase(
        "int64_i32_range",
        torch.int64,
        lambda: torch.tensor(
            [-2147483648, -1, 2147483647],
            dtype=torch.int64,
        ),
    ),
    DTypeCase(
        "float16",
        torch.float16,
        lambda: torch.tensor([1.0, 1.5, -2.0], dtype=torch.float16),
    ),
    DTypeCase(
        "bfloat16",
        torch.bfloat16,
        lambda: torch.tensor([1.0, 1.5, -2.0], dtype=torch.bfloat16),
    ),
    DTypeCase(
        "float32",
        torch.float32,
        lambda: torch.tensor([1.0, 1.5, -2.0], dtype=torch.float32),
    ),
    DTypeCase(
        "float64_f32_exact",
        torch.float64,
        lambda: torch.tensor([1.0, 1.5, float(2**40)], dtype=torch.float64),
    ),
    DTypeCase(
        "int64_outside_i32_range",
        torch.int64,
        lambda: torch.tensor([-(2**40), -1, 2**40], dtype=torch.int64),
    ),
    DTypeCase(
        "float64_precision_sensitive",
        torch.float64,
        lambda: torch.tensor(
            [1.0, 1.0000000000000002, float(2**40) + 0.25],
            dtype=torch.float64,
        ),
    ),
]


def _cuda_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"

    try:
        from luminal.luminal import _cuda_lite_factory_capsule

        _cuda_lite_factory_capsule()
    except (ImportError, AttributeError, RuntimeError) as exc:
        return f"luminal_python was not built with CUDA support: {exc}"

    return None


@pytest.fixture(params=["cpu", "cuda"], ids=["cpu", "cuda"])
def boundary_device(request) -> torch.device:
    device_name = request.param
    if device_name == "cuda":
        skip_reason = _cuda_skip_reason()
        if skip_reason is not None:
            pytest.skip(skip_reason)
    return torch.device(device_name)


# Dtypes that round-trip the BoundaryNoopModel without an explicit cast at the
# call site. Each retains both its declared dtype and exact values across input
# upload, reference execution, and output readback.
_FIRST_CLASS_NOOP_DTYPES = {
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64_i32_range",
    "int64_outside_i32_range",
    "float16",
    "bfloat16",
    "float32",
    "float64_f32_exact",
    "float64_precision_sensitive",
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            marks=pytest.mark.xfail(reason=case.xfail_reason, strict=True)
            if case.xfail_reason is not None
            else (),
            id=case.name,
        )
        for case in DTYPE_CASES
        if case.name in _FIRST_CLASS_NOOP_DTYPES
    ],
)
def test_boundary_noop_preserves_dtype_and_values(
    boundary_device: torch.device,
    case: DTypeCase,
) -> None:
    model = BoundaryNoopModel().to(boundary_device)
    compiled = torch.compile(model, backend=luminal_backend)

    x = case.values().to(boundary_device)
    expected = model(x)
    actual = compiled(x)

    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.name)
        for case in DTYPE_CASES
        if case.name in {"uint8", "int8", "int16"}
    ],
)
def test_narrow_integer_abs_preserves_wrapping_semantics(case: DTypeCase) -> None:
    model = AbsModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    x = case.values()

    expected = model(x)
    actual = compiled(x)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bool, id="bool"),
        pytest.param(torch.uint8, id="uint8"),
        pytest.param(torch.int8, id="int8"),
        pytest.param(torch.int16, id="int16"),
        pytest.param(torch.int32, id="int32"),
        pytest.param(torch.int64, id="int64"),
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.bfloat16, id="bfloat16"),
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ],
)
def test_empty_cpu_input_preserves_shape_and_dtype(dtype: torch.dtype) -> None:
    """A zero-byte CPU tensor may have a null data pointer and must still
    cross the compiled boundary with its shape and dtype intact."""
    model = BoundaryNoopModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    x = torch.empty((1, 0, 3), dtype=dtype)
    expected = model(x)
    actual = compiled(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_empty_cpu_weight_preserves_shape_and_dtype() -> None:
    """Zero-byte registered weights use the same null-pointer contract as
    runtime inputs."""
    model = EmptyWeightModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    x = torch.empty((0, 3), dtype=torch.float32)
    expected = model(x)
    actual = compiled(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_float64_add_without_alpha_matches_eager() -> None:
    """An omitted alpha stays a direct add rather than adding a multiply."""
    model = AddWithoutAlphaModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    x = torch.tensor([1.0, 1.0000000000000002], dtype=torch.float64)
    y = torch.tensor([2.0, -0.5], dtype=torch.float64)

    actual = compiled(x, y)
    expected = model(x, y)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("alpha", [2.0, -3.125, 1.0000000000000002])
def test_float64_add_with_explicit_alpha_matches_eager(alpha: float) -> None:
    """Explicit alpha is represented and multiplied as F64 without narrowing."""
    model = AddWithAlphaModel(alpha)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    x = torch.zeros(3, dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, -4.0], dtype=torch.float64)

    actual = compiled(x, y)
    expected = model(x, y)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_float64_add_scalar_with_explicit_alpha_matches_eager() -> None:
    """The scalar-operand lowering also keeps an explicit F64 alpha exact."""
    alpha = 1.0000000000000002
    model = AddScalarWithAlphaModel(alpha)
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    x = torch.zeros(3, dtype=torch.float64)

    actual = compiled(x)
    expected = model(x)

    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("dtype", "value", "python_type"),
    [
        pytest.param(torch.float16, 1.5, float, id="float16"),
        pytest.param(torch.bfloat16, -2.25, float, id="bfloat16"),
        pytest.param(torch.float32, 1.25, float, id="float32"),
        pytest.param(
            torch.float64,
            1.0000000000000002,
            float,
            id="float64-precision-sensitive",
        ),
        pytest.param(torch.int32, 2**30 + 3, int, id="int32"),
        pytest.param(torch.int64, 2**40 + 3, int, id="int64"),
        pytest.param(torch.bool, True, bool, id="bool"),
    ],
)
def test_item_returns_exact_python_scalar(
    dtype: torch.dtype,
    value,
    python_type: type,
) -> None:
    """HLIR keeps item values as typed rank-zero tensors and the backend
    reconstructs the Python scalar only at its output boundary."""
    model = ItemModel()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    x = torch.tensor(value, dtype=dtype)

    expected = model(x)
    actual = compiled(x)

    assert type(actual) is python_type
    assert type(actual) is type(expected)
    assert actual == expected


@pytest.mark.parametrize(
    ("start", "end", "step", "dtype", "keyword_end"),
    [
        pytest.param(-1, 2, 2, torch.float32, False, id="ceil-length"),
        pytest.param(0.0, -8.000001, -4.0, torch.float64, False, id="negative-step"),
        pytest.param(-0.9, 2.1, 2.0, torch.int32, False, id="fractional-to-int"),
        pytest.param(False, True, True, torch.bfloat16, False, id="bool-scalars"),
        pytest.param(0, 3.1, 1, torch.float16, True, id="keyword-end"),
        pytest.param(1, 5, 2, torch.int64, False, id="int64-output"),
        pytest.param(0, 6, 2, torch.uint8, False, id="uint8-output"),
        pytest.param(-3, 4, 3, torch.int8, False, id="int8-output"),
        pytest.param(-300, 301, 300, torch.int16, False, id="int16-output"),
        pytest.param(1.1, 1.1, -1.0, torch.float32, False, id="empty"),
    ],
)
def test_arange_uses_exported_shape_and_declared_dtype(
    start,
    end,
    step,
    dtype: torch.dtype,
    keyword_end: bool,
) -> None:
    """PT2 metadata owns arange's ceiling/endpoint shape semantics, while
    the lowering preserves every scalar slot and materializes the declared
    output dtype instead of leaking Iota's internal I32 dtype."""
    model = ArangeModel(
        start,
        end,
        step,
        dtype,
        keyword_end=keyword_end,
    )
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    expected = model()
    actual = compiled()

    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_nonempty_cpu_input_rejects_null_pointer() -> None:
    """Null is valid only for an empty host buffer; non-empty buffers fail
    with a Python exception instead of reaching unsafe pointer conversion."""
    captured = []

    def capture_backend(gm, example_inputs, options=None):
        compiled_model = luminal_backend(gm, example_inputs, options)
        captured.append(compiled_model)
        return compiled_model

    compiled = torch.compile(
        BoundaryNoopModel(), backend=capture_backend, fullgraph=True
    )
    compiled(torch.ones(1, dtype=torch.float32))

    compiled_model = captured[0]
    with pytest.raises(
        ValueError,
        match="input pointer is null for a non-empty buffer of 4 bytes",
    ):
        compiled_model._graph.set_input_from_ptr(
            compiled_model._input_names[0],
            0,
            4,
            torch_dtype_code(torch.float32),
        )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.name)
        for case in DTYPE_CASES
        if case.name
        in {
            "bool",
            "uint8",
            "int8",
            "int16",
            "int32",
            "float16",
            "bfloat16",
            "float32",
            # int64 / float64 are first-class in the IR — passing a tensor
            # of either dtype matches the graph's input dtype directly, no
            # conversion needed.
            "int64_i32_range",
            "int64_outside_i32_range",
            "float64_f32_exact",
            "float64_precision_sensitive",
        }
    ],
)
def test_matching_dtype_does_not_raise(
    boundary_device: torch.device,
    case: DTypeCase,
) -> None:
    """Round-trip contract: a user input whose dtype matches the graph's
    declared input dtype runs without raising, with no warnings emitted at
    the boundary."""
    model = BoundaryNoopModel().to(boundary_device)
    compiled = torch.compile(model, backend=luminal_backend)
    x = case.values().to(boundary_device)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        compiled(x)

    boundary_warnings = [
        record
        for record in records
        if "boundary" in str(record.message).lower()
        or "convert" in str(record.message).lower()
    ]
    assert boundary_warnings == [], (
        f"unexpected boundary-related warning(s): {boundary_warnings}"
    )
