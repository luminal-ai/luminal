import os
import tempfile

import onnx
import torch
import torch._dynamo

import luminal

from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel


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
    export_mode = env_mode if env_mode in ("pt2", "onnx") else options.get("export_mode", "onnx")
    opset = options.get("opset", 20)

    _register_cache_serialization()
    device = example_inputs[0].device if example_inputs else torch.device("cpu")
    backend = "cuda" if device.type == "cuda" else "native"

    if export_mode == "pt2":
        return _compile_pt2(gm, example_inputs, backend)
    return _compile_onnx(gm, example_inputs, backend, opset=opset)


def _compile_onnx(gm, example_inputs, backend, opset=20):
    """ONNX compilation path."""
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

        result = luminal.process_onnx(tmp_path, backend)
    finally:
        os.unlink(tmp_path)
    compiled = CompiledModel(result)
    return compiled


def _compile_pt2(gm, example_inputs, backend):
    """PT2/torch.export path — delegates to pt2.pt2_backend."""
    from .pt2 import pt2_backend
    return pt2_backend(gm, example_inputs, backend=backend)
