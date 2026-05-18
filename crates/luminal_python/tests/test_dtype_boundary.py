from dataclasses import dataclass
import warnings
from typing import Callable

import pytest
import torch

from luminal import luminal_backend
from luminal.compiled_model import DTypeBoundaryError


class BoundaryNoopModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.bool:
            return x | torch.zeros((), dtype=torch.bool, device=x.device)
        return x + torch.zeros((), dtype=x.dtype, device=x.device)


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


# Dtypes that round-trip the BoundaryNoopModel without an explicit
# `x.to(model.input_dtypes[0])` cast at the call site. Anything not in this
# set is a narrow integer (uint8 / int8 / int16) that luminal collapses to
# `DType::Int` internally — the hard-reject contract makes the boundary
# refuse the mismatched dtype, and the test for those lives in
# `test_input_dtype_mismatch_rejects` instead.
_FIRST_CLASS_NOOP_DTYPES = {
    "bool",
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
        # Narrower integer widths still collapse to luminal's `Int` (i32) at
        # the boundary; user inputs of those dtypes don't match the graph's
        # declared input dtype and so trigger a hard reject. int64 / float64
        # are first-class in the IR — they match the graph's declared input
        # dtype directly and don't reject.
        if case.name in {"uint8", "int8", "int16"}
    ],
)
def test_input_dtype_mismatch_rejects(
    boundary_device: torch.device,
    case: DTypeCase,
) -> None:
    """Hard-reject contract: a user input whose dtype doesn't match the
    graph's declared input dtype raises `DTypeBoundaryError` before the
    graph runs. Previously the boundary silently cast on every call, which
    hid real precision bugs (e.g. f64 → f32 truncation on values outside
    the f32 range) and burnt cycles on a per-call allocation+copy.
    """
    model = BoundaryNoopModel().to(boundary_device)
    compiled = torch.compile(model, backend=luminal_backend)
    x = case.values().to(boundary_device)

    with pytest.raises(DTypeBoundaryError, match="Convert at the call site"):
        compiled(x)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.name)
        for case in DTYPE_CASES
        if case.name
        in {
            "bool",
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
