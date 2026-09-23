"""Internal exported-program compilation for region artifacts."""

import torch


def compile_exported(
    program, *, device_type=None, search_iterations=1, search_log=False, **binding
):
    from .artifact_cache import CompiledArtifact, get_or_compile, region_artifact_key

    placeholders = [
        n for n in program.graph_module.graph.nodes if n.op == "placeholder"
    ]
    user_inputs = [
        n.meta["val"]
        for n, s in zip(placeholders, program.graph_signature.input_specs)
        if s.kind.name == "USER_INPUT"
    ]
    device_type = device_type or next(
        (t.device.type for t in user_inputs if torch.is_tensor(t)), "cpu"
    )
    for value in user_inputs:
        if isinstance(value, torch.Tensor) and value.device.type != device_type:
            raise ValueError(
                f"{device_type} backend cannot compile {value.device.type} inputs"
            )
    key = region_artifact_key(
        program, device_type=device_type, search_iterations=search_iterations
    )

    def build():
        if device_type == "cpu":
            from luminal_reference.backend import compile_exported as native_compile
        elif device_type == "cuda":
            from luminal_cuda_lite.backend import compile_exported as native_compile
        else:
            raise ValueError(f"unsupported region device type: {device_type!r}")
        native = native_compile(
            program, user_inputs, search_iterations, search_log=search_log
        )
        return CompiledArtifact(program, native, device_type, search_iterations, key)

    return get_or_compile(key, build).bind(program=program, **binding)
