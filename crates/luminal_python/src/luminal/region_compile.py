"""Compile a normalized vLLM-style region from exported tensor metadata."""

from __future__ import annotations

from .compiled_model import CompiledModel
from .pt2 import _save_and_compile
from .region_export import RegionExport


def compile_region(
    region: RegionExport,
    *,
    device_type: str = "cuda",
    search_iterations: int = 1,
) -> CompiledModel:
    """Compile a region without borrowing storage from its example inputs.

    The PT2 program contains the shape and dtype metadata needed to allocate
    Luminal's search inputs. Real tensor addresses are bound by CompiledModel
    only when the returned callable is invoked.
    """

    if device_type != "cuda":
        raise ValueError(f"unsupported region device type: {device_type!r}")

    try:
        from .luminal import _cuda_lite_factory_capsule
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "region compilation requires luminal_python built with CUDA support"
        ) from error

    return _save_and_compile(
        region.program,
        _cuda_lite_factory_capsule(),
        search_iterations,
        user_indices=region.input_indices,
        input_device_ptrs=None,
    )
