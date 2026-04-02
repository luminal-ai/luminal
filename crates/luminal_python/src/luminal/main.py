import os
import tempfile

import torch
import torch._dynamo

import luminal

from .compiled_model import CompiledModel


# ---------------------------------------------------------------------------
# Shared helpers (used by both ONNX and PT2 paths)
# ---------------------------------------------------------------------------


def _detect_backend(example_inputs):
    """Detect backend from input device. Returns 'cuda' or 'native'."""
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    return "cuda" if device.type == "cuda" else "native"


def _collect_weight_pointers(weights, backend):
    """Partition weight tensors into CUDA device pointers and CPU host pointers.

    Args:
        weights: dict of name -> torch.Tensor
        backend: "cuda", "gpu", "cpu", or "native"

    Returns:
        (keep_alive, device_ptrs, cpu_ptrs) where:
        - keep_alive: list[Tensor] to prevent GC of shared weight memory
        - device_ptrs: {name: (device_ptr, n_bytes)}
        - cpu_ptrs: {name: (host_ptr, n_elements)}
    """
    keep_alive = []
    device_ptrs = {}
    cpu_ptrs = {}
    for name, tensor in weights.items():
        t = tensor.detach().contiguous()
        if t.dtype != torch.float32:
            t = t.float()
        if backend in ("cuda", "gpu") and t.is_cuda:
            keep_alive.append(t)
            device_ptrs[name] = (t.data_ptr(), t.numel() * 4)
        else:
            t = t.cpu() if t.is_cuda else t
            keep_alive.append(t)
            cpu_ptrs[name] = (t.data_ptr(), t.numel())
    return keep_alive, device_ptrs, cpu_ptrs


def _load_cpu_weights(compiled_graph, cpu_weights):
    """Load CPU weight data into a compiled graph after Rust compilation."""
    for name, (ptr, n_elements) in cpu_weights.items():
        compiled_graph.set_weight_from_ptr(name, ptr, n_elements)


# ---------------------------------------------------------------------------
# torch.compile backend entry point
# ---------------------------------------------------------------------------


def luminal_backend(gm, example_inputs, options=None):
    """Luminal torch.compile backend.

    Usage:
        torch.compile(model, backend=luminal_backend)
        torch.compile(model, backend=luminal_backend, options={"export_mode": "pt2"})

    Options:
        export_mode: "onnx" (default) or "pt2"
        opset: ONNX opset version (default 20)
    """
    options = options or {}

    # Env var override
    env_mode = os.getenv("LUMINAL_EXPORT_MODE", "").lower()
    export_mode = (
        env_mode if env_mode in ("pt2", "onnx") else options.get("export_mode", "onnx")
    )
    opset = options.get("opset", 20)

    backend = _detect_backend(example_inputs)

    if export_mode == "pt2":
        return _compile_pt2(gm, example_inputs, backend)
    return _compile_onnx(gm, example_inputs, backend, opset=opset)


# ---------------------------------------------------------------------------
# ONNX compilation path
# ---------------------------------------------------------------------------


def _compile_onnx(gm, example_inputs, backend, opset=20):
    """ONNX compilation path."""
    # Identify weight vs user inputs from FX graph placeholders.
    # torch.compile lifts model parameters into graph inputs — we detect them by name prefix.
    weight_tensors = {}  # onnx_name -> tensor
    user_indices = []
    ph_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            onnx_name = f"input_{ph_idx}"
            if node.name.startswith(("l_self_", "l_model_", "l__self_")):
                weight_tensors[onnx_name] = example_inputs[ph_idx]
            else:
                user_indices.append(ph_idx)
            ph_idx += 1

    # Collect weight pointers for Rust (avoids duplicate GPU buffer allocation)
    weight_refs, weight_device_ptrs, cpu_weights = _collect_weight_pointers(
        weight_tensors, backend
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    _ = gm.eval()
    try:
        _ = torch.onnx.export(
            gm,
            tuple(example_inputs),
            tmp_path,
            opset_version=opset,
            input_names=[f"input_{i}" for i in range(len(example_inputs))],
        )

        result = luminal.process_onnx(
            tmp_path, backend, weight_device_ptrs=weight_device_ptrs
        )
    finally:
        os.unlink(tmp_path)

    # Load CPU weights after compilation
    _load_cpu_weights(result, cpu_weights)

    # Only expose user input names to CompiledModel (weights are pre-loaded).
    # user_indices tells __call__ which args from torch.compile are real user inputs.
    user_input_names = [f"input_{i}" for i in user_indices]
    return CompiledModel(
        result,
        weight_refs=weight_refs,
        input_names=user_input_names,
        user_indices=user_indices,
    )


# ---------------------------------------------------------------------------
# PT2 compilation path (delegates to pt2 module)
# ---------------------------------------------------------------------------


def _compile_pt2(gm, example_inputs, backend):
    """PT2/torch.export path — delegates to pt2.pt2_backend."""
    from .pt2 import pt2_backend

    return pt2_backend(gm, example_inputs, backend=backend)
