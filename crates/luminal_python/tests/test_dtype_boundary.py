from dataclasses import dataclass
import warnings
from typing import Callable

import pytest
import torch

from luminal import luminal_backend
from luminal.compiled_model import DTypeBoundaryWarning


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
        xfail_reason=(
            "Luminal currently collapses integer inputs through i32 at the "
            "compiled boundary, so out-of-range int64 values lose information."
        ),
    ),
    DTypeCase(
        "float64_precision_sensitive",
        torch.float64,
        lambda: torch.tensor(
            [1.0, 1.0000000000000002, float(2**40) + 0.25],
            dtype=torch.float64,
        ),
        xfail_reason=(
            "Luminal currently routes float64 no-op computation through f32 "
            "storage/outputs before restoring the PyTorch-visible dtype."
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
        if case.name
        in {
            "uint8",
            "int8",
            "int16",
            "int64_i32_range",
            "int64_outside_i32_range",
            "float64_f32_exact",
            "float64_precision_sensitive",
        }
    ],
)
def test_boundary_warns_when_input_dtype_requires_conversion(
    boundary_device: torch.device,
    case: DTypeCase,
) -> None:
    model = BoundaryNoopModel().to(boundary_device)
    compiled = torch.compile(model, backend=luminal_backend)
    x = case.values().to(boundary_device)

    with pytest.warns(DTypeBoundaryWarning, match="allocate/copy input data"):
        compiled(x)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, id=case.name)
        for case in DTYPE_CASES
        if case.name in {"bool", "int32", "float16", "bfloat16", "float32"}
    ],
)
def test_boundary_does_not_warn_when_input_dtype_matches_graph(
    boundary_device: torch.device,
    case: DTypeCase,
) -> None:
    model = BoundaryNoopModel().to(boundary_device)
    compiled = torch.compile(model, backend=luminal_backend)
    x = case.values().to(boundary_device)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        compiled(x)

    dtype_boundary_warnings = [
        record
        for record in records
        if issubclass(record.category, DTypeBoundaryWarning)
    ]
    assert dtype_boundary_warnings == []
