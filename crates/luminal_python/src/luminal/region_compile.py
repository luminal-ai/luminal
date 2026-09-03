"""Compile a normalized vLLM-style region from exported tensor metadata."""

from __future__ import annotations

from .artifact_cache import region_artifact_key
from .compiled_model import CompiledModel
from .pt2 import _save_and_compile
from .region_export import RegionExport


def compile_region(
    region: RegionExport,
    *,
    device_type: str = "cuda",
    search_iterations: int = 1,
    static_outputs: bool = False,
    external_cuda_graph: bool = False,
) -> CompiledModel:
    """Compile a region without borrowing storage from its example inputs.

    The PT2 program contains the shape and dtype metadata needed to allocate
    Luminal's search inputs. Real tensor addresses are bound by CompiledModel
    only when the returned callable is invoked.
    """

    if device_type != "cuda":
        raise ValueError(f"unsupported region device type: {device_type!r}")
    if region.device_index is None:
        raise ValueError("CUDA region compilation requires CUDA tensor inputs")
    if region.device_index != 0:
        raise ValueError(
            "Luminal currently supports only logical CUDA device 0, "
            f"got {region.device_index}"
        )

    try:
        from .luminal import _cuda_lite_factory_capsule
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "region compilation requires luminal_python built with CUDA support"
        ) from error

    artifact_key = region_artifact_key(
        region.program,
        device_type=device_type,
        device_index=region.device_index,
        search_iterations=search_iterations,
        external_cuda_graph=external_cuda_graph,
    )

    return _save_and_compile(
        region.program,
        _cuda_lite_factory_capsule(),
        search_iterations,
        user_indices=region.input_indices,
        output_spec=region.output_spec,
        input_device_ptrs=None,
        device_index=region.device_index,
        use_current_stream=True,
        static_outputs=static_outputs,
        external_cuda_graph=external_cuda_graph,
        artifact_key=artifact_key,
    )
