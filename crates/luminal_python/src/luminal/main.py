import torch
import torch._dynamo

from .dtype_util import torch_dtype_code as _torch_dtype_code


# ---------------------------------------------------------------------------
# Shared helpers (used by PT2 path and compiled_model)
# ---------------------------------------------------------------------------


def _cuda_device_index(tensors, expected=None):
    """Return one CUDA device index and enforce the current cuda:0 contract."""
    indices = {
        tensor.device.index
        for tensor in tensors
        if torch.is_tensor(tensor) and tensor.device.type == "cuda"
    }
    if None in indices:
        raise ValueError("CUDA tensor metadata must include a logical device index")
    if len(indices) > 1:
        raise ValueError(
            f"CUDA tensors span multiple logical devices: {sorted(indices)}"
        )

    observed = next(iter(indices), None)
    if expected is not None and observed is not None and observed != expected:
        raise ValueError(
            f"CUDA tensor is on logical device {observed}, but the compiled "
            f"runtime uses logical device {expected}"
        )
    selected = expected if expected is not None else observed
    if selected not in (None, 0):
        raise ValueError(
            f"Luminal currently supports only logical CUDA device 0, got {selected}"
        )
    return selected


def _detect_factory_capsule(example_inputs):
    """Pick the best built-in factory capsule based on input device."""
    # Dynamo can prefix `example_inputs` with SymInt entries when shapes are
    # dynamic — those have no `.device`. Pick the first real tensor instead.
    first_tensor = next((t for t in (example_inputs or []) if torch.is_tensor(t)), None)
    device = first_tensor.device if first_tensor is not None else torch.device("cpu")
    if device.type == "cuda":
        try:
            from .luminal import _cuda_lite_factory_capsule

            return _cuda_lite_factory_capsule()
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "CUDA input was provided, but luminal_python was not built with "
                "the cuda feature. Rebuild with `maturin develop --features cuda` "
                "or run through `run_tests_cuda.sh`/the Modal CUDA test runner."
            ) from exc
    from .luminal import _reference_factory_capsule

    return _reference_factory_capsule()


def _collect_weight_pointers(weights, device_index=None):
    """Partition weight tensors into CUDA device pointers and CPU host pointers.

    Preserves native dtype — no forced conversion to float32.

    Args:
        weights: dict of name -> torch.Tensor

    Returns:
        (keep_alive, device_ptrs, cpu_ptrs) where:
        - keep_alive: list[Tensor] to prevent GC of shared weight memory
        - device_ptrs: {name: (device_ptr, n_bytes)}
        - cpu_ptrs: {name: (host_ptr, n_bytes, dtype_code)}
    """
    _cuda_device_index(weights.values(), expected=device_index)
    keep_alive = []
    device_ptrs = {}
    cpu_ptrs = {}
    for name, tensor in weights.items():
        t = tensor.detach().contiguous()
        n_bytes = t.numel() * t.element_size()
        if t.is_cuda:
            keep_alive.append(t)
            device_ptrs[name] = (t.data_ptr(), n_bytes)
        else:
            t = t.cpu() if t.is_cuda else t
            keep_alive.append(t)
            cpu_ptrs[name] = (t.data_ptr(), n_bytes, _torch_dtype_code(t.dtype))
    return keep_alive, device_ptrs, cpu_ptrs


def _load_cpu_weights(compiled_graph, cpu_weights):
    """Load CPU weight data into a compiled graph after Rust compilation."""
    for name, (ptr, n_bytes, dtype_code) in cpu_weights.items():
        compiled_graph.set_weight_from_ptr(name, ptr, n_bytes, dtype_code)


# ---------------------------------------------------------------------------
# Backend registration
# ---------------------------------------------------------------------------


def register_backend(factory_capsule):
    """Wrap a backend factory PyCapsule into a torch.compile-compatible callable.

    Args:
        factory_capsule: PyCapsule wrapping a BackendFactory fn pointer.

    Returns:
        A callable(gm, example_inputs, options=None) suitable for torch.compile.
    """

    def backend(gm, example_inputs, options=None):
        return _compile_pt2(
            gm,
            example_inputs,
            factory_capsule,
            search_iterations=(options or {}).get("search_iterations"),
        )

    return backend


# ---------------------------------------------------------------------------
# torch.compile backend entry point (auto-detecting)
# ---------------------------------------------------------------------------


def luminal_backend(gm, example_inputs, options=None):
    """Auto-detecting torch.compile backend.

    Picks cuda_lite if inputs are on CUDA (and cuda feature is compiled in),
    reference otherwise. `options={"search_iterations": N}` (torch.compile's
    backend-options dict) sets the schedule-search budget; default lives in
    pt2._BACKEND_DEFAULT_SEARCH_ITERATIONS.

    For external backends, use register_backend with the backend's factory capsule.
    """
    capsule = _detect_factory_capsule(example_inputs)
    return _compile_pt2(
        gm,
        example_inputs,
        capsule,
        search_iterations=(options or {}).get("search_iterations"),
    )


# ---------------------------------------------------------------------------
# PT2 compilation path (delegates to pt2 module)
# ---------------------------------------------------------------------------


def _compile_pt2(gm, example_inputs, factory_capsule, search_iterations=None):
    """PT2/torch.export path — delegates to pt2.pt2_backend."""
    from .pt2 import pt2_backend

    return pt2_backend(
        gm,
        example_inputs,
        factory=factory_capsule,
        search_iterations=search_iterations,
    )
