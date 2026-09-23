"""Compile compiler-owned FX regions with the native PyTorch runtimes."""

from .artifact_cache import CompiledArtifact, get_or_load
from .frontend import compile_exported


def compile_region(
    region,
    *,
    device_type="cpu",
    search_iterations=1,
    search_log=False,
    static_outputs=False,
    external_cuda_graph=False,
):
    if region.device_index not in (None, 0):
        raise ValueError("only logical CUDA device 0 is supported")
    return compile_exported(
        region.program,
        device_type=device_type,
        search_iterations=search_iterations,
        search_log=search_log,
        user_indices=region.input_indices,
        output_spec=region.output_spec,
        static_outputs=static_outputs,
        external_cuda_graph=external_cuda_graph,
    )


def load_region_artifact(
    data,
    *,
    input_indices=None,
    output_spec=None,
    device_index=None,
    device_type=None,
    search_log=False,
    static_outputs=False,
    external_cuda_graph=False,
):
    if device_index not in (None, 0):
        raise ValueError("only logical CUDA device 0 is supported")
    artifact = get_or_load(
        data,
        lambda: CompiledArtifact.deserialize(
            data, device_type=device_type, search_log=search_log
        ),
        device_type=device_type,
        device_index=device_index,
    )
    return artifact.bind(
        user_indices=input_indices,
        output_spec=output_spec,
        static_outputs=static_outputs,
        external_cuda_graph=external_cuda_graph,
    )
