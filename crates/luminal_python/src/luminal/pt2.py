"""PT2 compilation pipeline for Luminal.

Provides:
  - compile(model, example_input, ...) — standalone PT2 path
  - pt2_backend(gm, example_inputs)    — torch.compile compatible backend
"""

import inspect
import os
import shutil
import tempfile

import torch
from safetensors.torch import save_file

from .cache_utils import _register_cache_serialization
from .compiled_model import CompiledModel
from .luminal import compile_pt2 as _compile_pt2_rust


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _export_kwargs():
    """Build common kwargs for torch.export.export()."""
    kwargs = dict(strict=False)
    if (
        "prefer_deferred_runtime_asserts_over_guards"
        in inspect.signature(torch.export.export).parameters
    ):
        kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
    return kwargs


def _save_and_compile(ep, backend, search_iterations):
    """Save ExportedProgram + weights to temp files, compile via Rust, return CompiledModel."""
    tmpdir = tempfile.mkdtemp(prefix="luminal_")
    try:
        pt2_path = os.path.join(tmpdir, "model.pt2")
        weights_path = os.path.join(tmpdir, "weights.safetensors")

        torch.export.save(ep, pt2_path)

        state_dict = {k: v.float().clone() for k, v in ep.state_dict.items()}
        if state_dict:
            save_file(state_dict, weights_path)
        else:
            weights_path = ""

        compiled = _compile_pt2_rust(pt2_path, weights_path, backend, search_iterations)
        return CompiledModel(compiled)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _reinternalize_lifted_params(gm, example_inputs):
    """Re-internalize lifted params as buffers so torch.export sees them as model state.

    torch.compile lifts model parameters out of the module and passes them as
    extra elements in example_inputs.  The Rust PT2 compiler expects weights in
    the .pt2 state dict, not as runtime inputs.  This function reverses the
    lifting by registering them as buffers and replacing the placeholder nodes
    with get_attr nodes.

    Returns (gm, user_inputs) where user_inputs contains only the real inputs.
    """
    buffer_indices = []
    user_indices = []
    buffer_nodes = []
    placeholder_idx = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            name = node.name
            if name.startswith("l_self_") or name.startswith("l_model_"):
                buffer_indices.append(placeholder_idx)
                buffer_nodes.append(node)
            else:
                user_indices.append(placeholder_idx)
            placeholder_idx += 1

    if buffer_nodes:
        for i, node in enumerate(buffer_nodes):
            attr_name = f"_luminal_param_{i}"
            gm.register_buffer(attr_name, example_inputs[buffer_indices[i]].detach().clone())
            with gm.graph.inserting_before(node):
                new_node = gm.graph.create_node("get_attr", attr_name)
                new_node.meta = node.meta.copy()
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()

    user_inputs = [example_inputs[i] for i in user_indices] if user_indices else list(example_inputs)
    return gm, user_inputs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile(
    model,
    example_input,
    search_iterations=25,
    backend=None,
    export_kwargs=None,
    dynamic_dim=None,
):
    """Compile a PyTorch model to run on Luminal via PT2 pipeline.

    Args:
        model: A PyTorch nn.Module.
        example_input: Example input tensor(s) for tracing.
        search_iterations: Number of optimization search iterations.
        backend: "cpu" or "cuda". Auto-detected if None.
        export_kwargs: Extra kwargs passed to torch.export.export.
        dynamic_dim: Which input dimension to make dynamic.

    Returns:
        A CompiledModel callable.
    """
    _register_cache_serialization()

    if dynamic_dim is None:
        dynamic_dim = "auto"

    if backend is None:
        backend = os.environ.get("LUMINAL_BACKEND", None)
        if backend is None:
            backend = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = export_kwargs or {}
    extra = _export_kwargs()
    ep = None

    # Try dynamic dimension export
    candidate_dims = []
    if isinstance(dynamic_dim, int):
        candidate_dims = [dynamic_dim]
    elif dynamic_dim == "auto" and example_input.dim() >= 2:
        if not example_input.is_floating_point():
            candidate_dims = [example_input.dim() - 1]

    if candidate_dims:
        from torch.export import Dim

        for dim_idx in candidate_dims:
            try:
                seq = Dim("seq", min=2)
                arg_shapes = {dim_idx: seq}
                kwarg_shapes = {k: None for k in kwargs}
                dynamic_shapes = (
                    (arg_shapes,) + tuple(kwarg_shapes.values())
                    if kwarg_shapes
                    else (arg_shapes,)
                )
                ep = torch.export.export(
                    model,
                    (example_input,),
                    kwargs=kwargs,
                    dynamic_shapes=dynamic_shapes,
                    **extra,
                )
                break
            except Exception:
                continue

    if ep is None:
        ep = torch.export.export(
            model, (example_input,), kwargs=kwargs, dynamic_shapes=None, **extra,
        )

    return _save_and_compile(ep, backend, search_iterations)


def pt2_backend(gm, example_inputs, backend=None):
    """torch.compile backend using PT2 pipeline.

    Usage: torch.compile(model, backend=luminal.pt2.pt2_backend)
    """
    _register_cache_serialization()
    if backend is None:
        device = example_inputs[0].device if example_inputs else torch.device("cpu")
        backend = "cuda" if device.type == "cuda" else "cpu"
    gm = gm.eval()
    gm, user_inputs = _reinternalize_lifted_params(gm, example_inputs)
    ep = torch.export.export(gm, tuple(user_inputs), **_export_kwargs())
    return _save_and_compile(ep, backend, 10)
