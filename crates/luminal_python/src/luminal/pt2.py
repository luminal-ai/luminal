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
from .main import _collect_weight_pointers, _detect_factory_capsule, _load_cpu_weights

# ---------------------------------------------------------------------------
# DynamicCache <> pytree registration
#
# Without this, torch.export.export raises when handed an HF model that
# returns CausalLMOutputWithPast(past_key_values=DynamicCache(...)), which
# is every model with use_cache=True. The registration mirrors the one in
# transformers.integrations.executorch.register_dynamic_cache_export_support
# — same dict-based flatten (key_cache / value_cache lists), same replay via
# cache.update(k, v, idx), and the matching torch.fx._pytree spec for FX
# graphs. Done at module import so both entry points (pt2_backend via
# torch.compile and the direct compile() call) get it for free.
# ---------------------------------------------------------------------------


def _get_cache_dict(cache):
    """Flatten a DynamicCache to a dict of parallel key/value lists."""
    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [
            layer.values for layer in cache.layers if layer.values is not None
        ],
    }


def _flatten_dynamic_cache(cache):
    return torch.utils._pytree._dict_flatten(_get_cache_dict(cache))


def _flatten_with_keys_dynamic_cache(cache):
    return torch.utils._pytree._dict_flatten_with_keys(_get_cache_dict(cache))


def _unflatten_dynamic_cache(values, context):
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        k = key_list[idx] if idx < len(key_list) else None
        v = value_list[idx] if idx < len(value_list) else None
        cache.update(k, v, idx)
    return cache


def _register_cache_serialization():
    """Register DynamicCache with both torch.utils._pytree and torch.fx._pytree.

    Idempotent: a second call is a no-op. Silently skipped if transformers is
    not installed.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        return

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        _flatten_dynamic_cache,
        _unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=_flatten_with_keys_dynamic_cache,
    )
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache,
        lambda cache, spec: torch.fx._pytree._dict_flatten_spec(
            _get_cache_dict(cache), spec
        ),
    )


_register_cache_serialization()


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


def _decomp_table():
    """Decomposition table for `ep.run_decompositions()` that preserves SDPA.

    The default table decomposes `aten.scaled_dot_product_attention.default`
    into ~20 ops (matmul/softmax + an `eq.Scalar`/`logical_not`/`any.dim`/
    `where`/`full_like` "all-masked" sentinel chain). We translate SDPA as a
    single fused op via `translate_sdpa`, so we strip the SDPA decompositions
    here to let them survive into the FX graph the translator walks.
    """
    try:
        from torch.export import default_decompositions
    except ImportError:
        return None
    table = default_decompositions()
    sdpa_ops = [
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    ]
    for op in sdpa_ops:
        table.pop(op, None)
    return table


def _save_and_compile(
    ep_or_path, factory, search_iterations, original_weights=None, user_indices=None
):
    """Compile a PT2 model via Rust, return CompiledModel.

    Args:
        ep_or_path: Either an ExportedProgram (will be saved to a temp file) or
            a path to an already-saved .pt2 file.
        factory: PyCapsule wrapping the BackendFactory to use.
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
            weight_source
        )

        # Compile with device pointers — search uses actual weight memory (zero-copy)
        compiled = process_pt2(
            pt2_path, "", search_iterations, factory, weight_device_ptrs
        )

        # Load CPU weights after compilation
        _load_cpu_weights(compiled, cpu_weights)

        return CompiledModel(
            compiled, weight_refs=keep_alive, user_indices=user_indices
        )
    finally:
        if owns_tmpdir and tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _safe_int_bound(value):
    """Coerce a sympy/symbolic-shape range bound to a finite int, or None.

    Range bounds returned by ShapeEnv can be sympy `Infinity` / `-Infinity`
    (as well as the internal `int_oo` sentinel), which both raise on `int(...)`.
    Treat anything non-finite — and anything that simply doesn't coerce — as
    "no bound."
    """
    if value is None:
        return None
    # Stringify is robust against the various sentinel types: sympy.Infinity,
    # torch.utils._sympy.numbers.IntInfinity, etc. all stringify to "oo"/"-oo".
    s = str(value)
    if "oo" in s or "inf" in s.lower():
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None


def _strip_symint_placeholders(gm, example_inputs):
    """Rewrite SymInt graph inputs into tensor.size(d) calls, then drop them.

    When Dynamo decides a dim is dynamic it emits the symbol as a separate
    placeholder (e.g. `s77`) alongside the user's tensor (whose FakeTensor shape
    references the same symbol). torch.export.export rejects mixed
    SymInt/Tensor positional args, and the Rust pipeline doesn't model SymInt
    inputs anyway — so we replace each SymInt placeholder's uses with
    `aten.sym_size.int(tensor, dim)` for the first tensor placeholder whose
    example_value's shape[dim] matches the symbol, then erase the placeholder.

    Returns `(post_strip_inputs, kept_indices, ok)` where:
      - `post_strip_inputs` is `example_inputs` filtered to tensor-only entries
      - `kept_indices` is the indices into `example_inputs` we kept (used by
        the caller to compose with any prior input filter, e.g. lifted-weight
        re-internalization, when handing `user_indices` to CompiledModel)
      - `ok` is False when at least one SymInt placeholder couldn't be
        rewritten (compound expression with users, or no matching tensor dim);
        the caller should fall back to no-dynamic export in that case.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    # Collect (placeholder_node, example_input_idx) for every SymInt placeholder.
    symint_entries = []
    tensor_entries = []
    for idx, node in enumerate(placeholders):
        ev = node.meta.get("example_value")
        if isinstance(ev, torch.SymInt) or (
            ev is None
            and idx < len(example_inputs)
            and isinstance(example_inputs[idx], torch.SymInt)
        ):
            symint_entries.append((node, idx))
        else:
            tensor_entries.append((node, idx))

    if not symint_entries:
        return example_inputs, list(range(len(example_inputs))), True

    # Build a symbol -> (tensor_node, dim) lookup from the tensor placeholders'
    # example FakeTensor shapes. Any tensor whose shape[d] is the SymInt
    # is a valid source — pick the first.
    sym_to_source = {}
    for t_node, _ in tensor_entries:
        ev = t_node.meta.get("example_value")
        if not torch.is_tensor(ev):
            continue
        for d, s in enumerate(ev.shape):
            if isinstance(s, torch.SymInt):
                key = str(s.node.expr)
                sym_to_source.setdefault(key, (t_node, d))

    # Rewrite each SymInt placeholder's uses to sym_size calls, then erase it.
    all_clean = True
    for s_node, _ in symint_entries:
        ev = s_node.meta.get("example_value")
        if ev is None:
            all_clean = False
            continue
        # The placeholder's example_value is the SymInt itself; its expr is the
        # symbol name (or a compound expression we can't lift this way).
        expr_str = str(ev.node.expr)
        source = sym_to_source.get(expr_str)
        if source is None:
            # Compound expression or no tensor carries this symbol — bail.
            if len(s_node.users) > 0:
                all_clean = False
                continue
            gm.graph.erase_node(s_node)
            continue

        if len(s_node.users) > 0:
            t_node, dim = source
            with gm.graph.inserting_after(t_node):
                size_node = gm.graph.call_function(
                    torch.ops.aten.sym_size.int, (t_node, dim)
                )
                size_node.meta["val"] = ev
                size_node.meta["example_value"] = ev
            s_node.replace_all_uses_with(size_node)
        gm.graph.erase_node(s_node)

    if not all_clean:
        # Recompile defensively even on partial success — some erases may have
        # happened. Caller will decide whether to proceed.
        gm.graph.lint()
        gm.recompile()
        return example_inputs, list(range(len(example_inputs))), False

    gm.graph.lint()
    gm.recompile()
    # Filter the runtime example_inputs to drop the stripped SymInt entries.
    kept_indices = [idx for _, idx in tensor_entries]
    keep_set = set(kept_indices)
    new_inputs = [v for i, v in enumerate(example_inputs) if i in keep_set]
    return new_inputs, kept_indices, True


def _build_dynamic_shapes_from_gm(gm):
    """Construct a torch.export.export `dynamic_shapes` spec from FX metadata.

    Walks each tensor placeholder's `meta['example_value']` FakeTensor and
    marks every SymInt dim as `Dim.AUTO`. Sharing/equality relationships
    between symbolic dims are already encoded in the FakeTensor shapes —
    torch.export's symbolic-shape engine recovers them during the trace, so
    we don't need to allocate named `Dim` objects ourselves.

    The returned spec is wrapped under `{"args": (...)}` because Dynamo's
    `GraphModule.forward(*args, **kwargs)` signature treats positional inputs
    as the `args` tuple.

    Returns None if there are no symbolic dims to mark.
    """
    from torch.export import Dim

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    per_input_spec = []
    saw_dynamic = False
    for node in placeholders:
        ev = node.meta.get("example_value")
        if not torch.is_tensor(ev):
            per_input_spec.append(None)
            continue
        spec = {}
        for d, s in enumerate(ev.shape):
            if isinstance(s, torch.SymInt):
                spec[d] = Dim.AUTO
                saw_dynamic = True
        per_input_spec.append(spec if spec else None)

    if not saw_dynamic:
        return None
    return {"args": tuple(per_input_spec)}


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
    return gm, user_inputs, original_weights, user_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compile(
    model,
    example_input,
    search_iterations=25,
    factory=None,
    export_kwargs=None,
    dynamic_dim=None,
    dynamic_shapes=None,
):
    """Compile a PyTorch model to run on Luminal via PT2 pipeline.

    Args:
        model: A PyTorch nn.Module.
        example_input: Example input tensor — or a list/tuple of tensors for
            multi-input models.
        search_iterations: Number of optimization search iterations.
        factory: PyCapsule wrapping a BackendFactory. Auto-detected if None.
        export_kwargs: Extra kwargs passed to torch.export.export.
        dynamic_dim: Convenience controls for `dynamic_shapes` when only one
            symbolic dim is needed.
                * `None` (default): leave shapes static.
                * `int`: mark that dim of the (first) input as `Dim.AUTO`.
                * `Iterable[int]`: mark each listed dim of the first input.
                * `"auto"`: mark every non-trivial dim (size > 1) of the
                  first input as `Dim.AUTO` — works for floating-point and
                  integer inputs alike.
        dynamic_shapes: Direct passthrough to `torch.export.export`'s
            `dynamic_shapes` argument. When provided, takes precedence over
            `dynamic_dim`. Use this for full control: per-input specs,
            `Dim("name", min=, max=)` ranges, shared dims across inputs, etc.

    Returns:
        A CompiledModel callable.
    """
    if factory is None:
        factory = _detect_factory_capsule(
            example_input
            if isinstance(example_input, (list, tuple))
            else [example_input]
        )

    if isinstance(example_input, (list, tuple)):
        example_args = tuple(example_input)
    else:
        example_args = (example_input,)

    kwargs = export_kwargs or {}
    extra = _export_kwargs()

    # Build dynamic_shapes from the convenience knob if the caller didn't
    # hand us a full spec. `dynamic_dim=None` falls back to the legacy
    # `"auto"` behavior (mark the last axis of an integer input as dynamic)
    # so callers that relied on the previous default keep working.
    if dynamic_shapes is None:
        if dynamic_dim is None:
            dynamic_dim = _legacy_auto_dim(example_args)
        if dynamic_dim is not None:
            dynamic_shapes = _build_dynamic_shapes_from_dim_arg(
                dynamic_dim, example_args
            )

    # `torch.export.export` is finicky: when `dynamic_shapes` is set it
    # validates the spec against the example shapes and raises on any
    # disagreement (e.g. the user marked a dim as dynamic but their model
    # specialises it to a constant). Fall back to a static export so the
    # caller still gets a usable CompiledModel rather than a hard error.
    ep = None
    if dynamic_shapes is not None:
        try:
            ep = torch.export.export(
                model,
                example_args,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                **extra,
            )
            ep = ep.run_decompositions(_decomp_table())
        except Exception:
            ep = None

    if ep is None:
        ep = torch.export.export(
            model,
            example_args,
            kwargs=kwargs,
            dynamic_shapes=None,
            **extra,
        )
        ep = ep.run_decompositions(_decomp_table())

    return _save_and_compile(ep, factory, search_iterations)


def _legacy_auto_dim(example_args):
    """Match the historical `dynamic_dim="auto"` heuristic.

    Returns the last axis of the first input when that input is a 2-D-or-
    larger integer tensor (the typical token-id sequence pattern), and
    `None` otherwise. Float inputs and 1-D tensors fall through to the
    static export path the legacy code did.
    """
    if not example_args:
        return None
    first = example_args[0]
    if not torch.is_tensor(first):
        return None
    if first.is_floating_point():
        return None
    if first.dim() < 2:
        return None
    return first.dim() - 1


def _build_dynamic_shapes_from_dim_arg(dynamic_dim, example_args):
    """Translate the `dynamic_dim` shorthand into a full `dynamic_shapes` spec.

    Always targets the first positional input — multi-input dynamic specs
    require the caller to use `dynamic_shapes=` directly so they can name
    which input each dim belongs to.
    """
    from torch.export import Dim

    if not example_args:
        return None
    first = example_args[0]
    if not torch.is_tensor(first):
        return None

    if isinstance(dynamic_dim, int):
        dims = [dynamic_dim]
    elif isinstance(dynamic_dim, str) and dynamic_dim == "auto":
        # Mark every dim with size > 1 as dynamic. Dim.AUTO leaves
        # torch.export to pick a Dim per axis and infer relationships from
        # the example FakeTensor.
        dims = [d for d, s in enumerate(first.shape) if int(s) > 1]
    elif hasattr(dynamic_dim, "__iter__"):
        dims = [int(d) for d in dynamic_dim]
    else:
        return None

    if not dims:
        return None

    spec = {d: Dim.AUTO for d in dims}
    rest = (None,) * (len(example_args) - 1)
    return (spec,) + rest


def _eager_pt2_compile(
    gm, user_inputs, original_weights, user_indices, dynamic_shapes, factory
):
    """Run torch.export → save → Rust compile end-to-end. Returns CompiledModel.

    Factored out so both the eager (static-shapes) and lazy (dynamic-shapes)
    backend paths share a single implementation.
    """
    import gc

    try:
        ep = torch.export.export(
            gm,
            tuple(user_inputs),
            dynamic_shapes=dynamic_shapes,
            **_export_kwargs(),
        )
    except Exception:
        # If torch.export rejects the dynamic spec (e.g. user code introduced
        # a constraint we didn't model), retry without it. Better to lose the
        # dynamic-dim optimization than to hand the user a hard failure.
        if dynamic_shapes is None:
            raise
        ep = torch.export.export(gm, tuple(user_inputs), **_export_kwargs())
    ep = ep.run_decompositions(_decomp_table())

    # When using shared memory (original_weights), strip large weight buffers
    # from the EP before saving. The Rust side uses device pointers for these
    # weights, not the .pt2 file data, so serializing them is pure IO waste
    # (~32 GB for 8B models). Replace with tiny CPU scalars to shrink to <1 MB.
    if original_weights:
        for key in list(ep._state_dict.keys()):
            if key in original_weights:
                orig = ep._state_dict[key]
                ep._state_dict[key] = torch.zeros(1, dtype=orig.dtype, device="cpu")
                del orig

    # Save EP to disk, then free it and the traced graph module before Rust
    # compilation. torch.export clones the state_dict internally; holding ep
    # alive during compile would double weight memory on GPU.
    tmpdir = tempfile.mkdtemp(prefix="luminal_")
    pt2_path = os.path.join(tmpdir, "model.pt2")
    torch.export.save(ep, pt2_path)

    del ep, gm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        return _save_and_compile(
            pt2_path,
            factory,
            10,
            original_weights=original_weights,
            user_indices=user_indices,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class _LazyDynamicCompiledModel:
    """Defers torch.export + Rust compile to the first invocation.

    Calling `torch.export.export(..., dynamic_shapes=...)` from inside a
    Dynamo backend frame triggers an internal "Guard failed on the same
    frame it was created" assertion in PyTorch — `torch.export`'s symbolic
    tracer mutates the ShapeEnv that Dynamo is also relying on for the
    surrounding compile, leaving the just-installed guards in an
    inconsistent state. Punting all of that work to the first runtime call
    sidesteps the issue: by then Dynamo's guard installation is finished,
    so the shape-env mutations no longer matter.

    This wrapper is API-compatible with `CompiledModel` for the bits the
    caller cares about (`__call__`, `has_dynamic_dims`, `dim_params`,
    `set_dim`). Subsequent calls forward straight to the inner CompiledModel.
    """

    def __init__(
        self,
        gm,
        user_inputs,
        original_weights,
        user_indices,
        dynamic_shapes,
        factory,
    ):
        self._gm = gm
        self._user_inputs = user_inputs
        self._original_weights = original_weights
        self._user_indices = user_indices
        self._dynamic_shapes = dynamic_shapes
        self._factory = factory
        self._compiled = None

    def _ensure_compiled(self):
        if self._compiled is None:
            self._compiled = _eager_pt2_compile(
                self._gm,
                self._user_inputs,
                self._original_weights,
                self._user_indices,
                self._dynamic_shapes,
                self._factory,
            )
            # Drop references to inputs we no longer need — the Rust side
            # holds onto weights via device pointers / CPU buffers.
            self._gm = None
            self._user_inputs = None
            self._original_weights = None
        return self._compiled

    def __call__(self, *inputs, **kwargs):
        return self._ensure_compiled()(*inputs, **kwargs)

    @property
    def has_dynamic_dims(self):
        return self._ensure_compiled().has_dynamic_dims

    @property
    def dim_params(self):
        return self._ensure_compiled().dim_params

    def set_dim(self, name, value):
        return self._ensure_compiled().set_dim(name, value)


def pt2_backend(gm, example_inputs, factory=None):
    """torch.compile backend using PT2 pipeline.

    Usage: torch.compile(model, backend=luminal.register_backend(capsule))
    """
    import copy as _copy

    if factory is None:
        factory = _detect_factory_capsule(example_inputs)

    # Work on a private copy of the GraphModule. Dynamo holds onto the
    # original to install guards and to retrace on shape changes; mutating it
    # here (erasing SymInt placeholders, re-internalizing lifted weights)
    # corrupts that bookkeeping and surfaces as cryptic "guard failed on the
    # same frame" assertions on the next call. The deepcopy is cheap relative
    # to the rest of the export pipeline.
    gm = _copy.deepcopy(gm).eval()
    gm, user_inputs, original_weights, post_lift_indices = _reinternalize_lifted_params(
        gm, example_inputs
    )

    # Lift any SymInt placeholders Dynamo emitted alongside the tensor inputs
    # into `aten.sym_size.int` calls so the re-export sees a tensor-only
    # signature, then derive the `dynamic_shapes` spec from the surviving
    # tensor placeholders' FakeTensor shapes. If the strip can't fully clean
    # the graph (e.g. a compound-expr SymInt with users), we drop dynamic
    # info and fall back to per-shape recompilation — same as today.
    user_inputs, post_strip_subindices, strip_ok = _strip_symint_placeholders(
        gm, user_inputs
    )
    dynamic_shapes = _build_dynamic_shapes_from_gm(gm) if strip_ok else None

    # Compose both filter steps into a single user_indices list relative to
    # the *original* example_inputs Dynamo will pass at runtime — so
    # CompiledModel.__call__ can drop both lifted weights and SymInt args.
    user_indices = [post_lift_indices[i] for i in post_strip_subindices]

    if dynamic_shapes is not None:
        # See `_LazyDynamicCompiledModel` for why dynamic-shape compiles must
        # be deferred — torch.export with dynamic_shapes mutates ShapeEnv state
        # Dynamo is still relying on, and running it inside the backend frame
        # corrupts the freshly-installed guards.
        return _LazyDynamicCompiledModel(
            gm, user_inputs, original_weights, user_indices, dynamic_shapes, factory
        )

    return _eager_pt2_compile(
        gm, user_inputs, original_weights, user_indices, None, factory
    )
