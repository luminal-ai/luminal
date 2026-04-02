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

from .compiled_model import CompiledModel
from .luminal import process_pt2
from .main import _collect_weight_pointers, _detect_backend, _load_cpu_weights

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


def _save_and_compile(ep_or_path, backend, search_iterations, original_weights=None):
    """Compile a PT2 model via Rust, return CompiledModel.

    Args:
        ep_or_path: Either an ExportedProgram (will be saved to a temp file) or
            a path to an already-saved .pt2 file.
        original_weights: Optional dict mapping state_dict key -> original PyTorch tensor.
            When provided, device pointers are taken from these tensors instead of
            ep.state_dict (which torch.export may have cloned), enabling true zero-copy
            sharing with the original model's GPU memory.
    """
    owns_tmpdir = not isinstance(ep_or_path, str)
    tmpdir = tempfile.mkdtemp(prefix="luminal_") if owns_tmpdir else None
    try:
        if owns_tmpdir:
            pt2_path = os.path.join(tmpdir, "model.pt2")
            torch.export.save(ep_or_path, pt2_path)
            weight_source = (
                original_weights if original_weights else ep_or_path.state_dict
            )
        else:
            pt2_path = ep_or_path
            weight_source = original_weights or {}

        # Collect weight pointers for Rust (avoids duplicate GPU buffer allocation)
        keep_alive, weight_device_ptrs, cpu_weights = _collect_weight_pointers(
            weight_source, backend
        )

        # Compile with device pointers — search uses actual weight memory (zero-copy)
        compiled = process_pt2(
            pt2_path, "", backend, search_iterations, weight_device_ptrs
        )

        # Load CPU weights after compilation
        _load_cpu_weights(compiled, cpu_weights)

        return CompiledModel(compiled, weight_refs=keep_alive)
    finally:
        if owns_tmpdir and tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _reinternalize_lifted_params(gm, example_inputs):
    """Re-internalize lifted params as buffers so torch.export sees them as model state.

    torch.compile lifts model parameters out of the module and passes them as
    extra elements in example_inputs.  The Rust PT2 compiler may expect weights in
    the .pt2 state dict, not as runtime inputs.  This function reverses the
    lifting by registering them as buffers and replacing the placeholder nodes
    with get_attr nodes.

    Returns (gm, user_inputs, original_weights) where:
      - user_inputs contains only the real inputs
      - original_weights maps buffer name -> original tensor (for zero-copy device pointers)
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

    original_weights = {}
    if buffer_nodes:
        for i, node in enumerate(buffer_nodes):
            attr_name = f"_luminal_param_{i}"
            # Keep a reference to the original tensor for zero-copy device pointers.
            # torch.export.export may clone the registered buffer, so we bypass
            # the EP's state_dict and use the originals directly.
            original_weights[attr_name] = example_inputs[buffer_indices[i]]
            gm.register_buffer(attr_name, example_inputs[buffer_indices[i]].detach())
            with gm.graph.inserting_before(node):
                new_node = gm.graph.create_node("get_attr", attr_name)
                new_node.meta = node.meta.copy()
                node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
        gm.graph.lint()
        gm.recompile()

    user_inputs = (
        [example_inputs[i] for i in user_indices]
        if user_indices
        else list(example_inputs)
    )
    return gm, user_inputs, original_weights


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
        backend: "native" or "cuda". Auto-detected if None.
        export_kwargs: Extra kwargs passed to torch.export.export.
        dynamic_dim: Which input dimension to make dynamic.

    Returns:
        A CompiledModel callable.
    """
    if dynamic_dim is None:
        dynamic_dim = "auto"

    if backend is None:
        backend = os.environ.get("LUMINAL_BACKEND", None)
        if backend is None:
            backend = "cuda" if torch.cuda.is_available() else "native"

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
            model,
            (example_input,),
            kwargs=kwargs,
            dynamic_shapes=None,
            **extra,
        )

    return _save_and_compile(ep, backend, search_iterations)


def pt2_backend(gm, example_inputs, backend=None):
    """torch.compile backend using PT2 pipeline.

    Usage: torch.compile(model, backend=luminal.pt2.pt2_backend)
    """
    import gc

    if backend is None:
        backend = _detect_backend(example_inputs)

    gm = gm.eval()
    gm, user_inputs, original_weights = _reinternalize_lifted_params(gm, example_inputs)

    ep = torch.export.export(gm, tuple(user_inputs), **_export_kwargs())

    # When using shared memory (original_weights), strip large weight buffers from
    # the EP before saving.  The Rust side uses device pointers for these weights,
    # not the .pt2 file data, so serializing them is pure IO waste (~32 GB for 8B
    # models).  Replacing with tiny CPU scalars shrinks the .pt2 to < 1 MB.
    if original_weights:
        for key in list(ep._state_dict.keys()):
            if key in original_weights:
                orig = ep._state_dict[key]
                ep._state_dict[key] = torch.zeros(1, dtype=orig.dtype, device="cpu")
                del orig

    # Save the exported program to disk, then free it and the traced graph module
    # BEFORE Rust compilation. torch.export clones the state_dict internally, so
    # holding ep alive during compilation would double the weight memory on GPU.
    tmpdir = tempfile.mkdtemp(prefix="luminal_")
    pt2_path = os.path.join(tmpdir, "model.pt2")
    torch.export.save(ep, pt2_path)

    del ep, gm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        result = _save_and_compile(
            pt2_path, backend, 10, original_weights=original_weights
        )
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
