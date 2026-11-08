"""PT2 export-preparation helpers ported from the legacy luminal_python pipeline.

The reference backend self-exports a Dynamo GraphModule with
``torch.export.export``.  The legacy backend ran a small set of preprocessing
passes around that export so the resulting ExportedProgram only contains ops
the translator lowers (and survives ``torch.export.save``).  This module is the
new home of those passes:

  - DynamicCache <> pytree registration, so ``use_cache=True`` HF models export
  - scalar graph outputs boxed as rank-zero tensors
  - the SDPA-preserving decomposition table
  - the ``sym_sum`` serde workaround
  - the input-guard / dead data-dependent-op cleanups
"""

# ---------------------------------------------------------------------------
# DynamicCache <> pytree registration
#
# Without this, torch.export.export raises when handed an HF model that
# returns CausalLMOutputWithPast(past_key_values=DynamicCache(...)), which
# is every model with use_cache=True. The registration mirrors the one in
# transformers.integrations.executorch.register_dynamic_cache_export_support
# — same dict-based flatten (key_cache / value_cache lists), same replay via
# cache.update(k, v, idx), and the matching torch.fx._pytree spec for FX
# graphs. Done lazily by luminal_reference() so both entry points get it.
# ---------------------------------------------------------------------------


def private_graph_copy(gm):
    """A private copy of `gm` that SHARES its parameters and buffers.

    Dynamo keeps the original GraphModule and re-checks its guards against it
    after the backend returns, so the export passes need a module of their own
    to edit and recompile. They only ever edit the GRAPH, so only the graph has
    to be private. `copy.deepcopy(gm)` deep-copies the module's own dict, which
    clones every parameter and buffer on the device; a writeback then lands in
    the clone and the caller's module never sees it.

    `copy.copy` goes through `GraphModule.__copy__`, which builds a new module,
    carries `meta` over, and takes graph-referenced attributes BY REFERENCE.
    Assigning `.graph` recompiles onto that new module. Deep-copying a Graph
    copies node structure but shares `node.meta`, so the traced fake values
    stay the same objects.
    """
    import copy

    private = copy.copy(gm)
    private.graph = copy.deepcopy(gm.graph)
    return private


def _get_cache_dict(cache):
    """Flatten a DynamicCache to a dict of parallel key/value lists."""
    return {
        "key_cache": [layer.keys for layer in cache.layers if layer.keys is not None],
        "value_cache": [
            layer.values for layer in cache.layers if layer.values is not None
        ],
    }


def flatten_dynamic_cache(cache):
    """Pytree flatten function for DynamicCache."""
    import torch

    return torch.utils._pytree._dict_flatten(_get_cache_dict(cache))


def unflatten_dynamic_cache(values, context):
    """Pytree unflatten function for DynamicCache."""
    import torch
    from transformers.cache_utils import DynamicCache

    dictionary = torch.utils._pytree._dict_unflatten(values, context)
    cache = DynamicCache()
    key_list = dictionary.get("key_cache", [])
    value_list = dictionary.get("value_cache", [])
    for idx in range(max(len(key_list), len(value_list))):
        key = key_list[idx] if idx < len(key_list) else None
        value = value_list[idx] if idx < len(value_list) else None
        cache.update(key, value, idx)
    return cache


def flatten_with_keys_dynamic_cache(cache):
    """Pytree flatten-with-keys function for DynamicCache."""
    import torch

    return torch.utils._pytree._dict_flatten_with_keys(_get_cache_dict(cache))


def _register_cache_serialization():
    """Register DynamicCache with both torch.utils._pytree and torch.fx._pytree.

    Idempotent: a second call is a no-op. Silently skipped if transformers is
    not installed.
    """
    import torch

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        return

    if DynamicCache in torch.utils._pytree.SUPPORTED_NODES:
        return

    torch.utils._pytree.register_pytree_node(
        DynamicCache,
        flatten_dynamic_cache,
        unflatten_dynamic_cache,
        serialized_type_name=f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        flatten_with_keys_fn=flatten_with_keys_dynamic_cache,
    )
    torch.fx._pytree.register_pytree_flatten_spec(
        DynamicCache,
        lambda cache, spec: torch.fx._pytree._dict_flatten_spec(
            _get_cache_dict(cache), spec
        ),
    )


# ---------------------------------------------------------------------------
# Output-boundary canonicalization
# ---------------------------------------------------------------------------


def _box_scalar_graph_outputs(gm, *, should_box=None):
    """Represent backend scalar outputs as typed zero-dimensional tensors.

    Dynamo's backend graph may return SymInt/SymFloat/SymBool values even
    though Luminal's HLIR is tensor-only. Box those output leaves before the
    second torch.export capture and remember their flattened output positions;
    CompiledModel calls ``.item()`` at those positions after execution to
    restore the backend contract.

    This is an output-boundary canonicalization, not a symbolic-scalar IR:
    inside Luminal the values remain ordinary typed rank-zero tensors.
    """
    import torch

    output = next(node for node in gm.graph.nodes if node.op == "output")
    flat_outputs, output_spec = torch.utils._pytree.tree_flatten(output.args[0])
    scalar_output_positions = []
    compiled_position = 0

    def scalar_dtype(value):
        if isinstance(value, (torch.SymBool, bool)):
            return torch.bool
        if isinstance(value, (torch.SymInt, int)):
            return torch.int64
        if isinstance(value, (torch.SymFloat, float)):
            return torch.float64
        return None

    boxed_outputs = []
    for value in flat_outputs:
        example = (
            value.meta.get("example_value", value.meta.get("val"))
            if isinstance(value, torch.fx.Node)
            else value
        )
        if isinstance(example, torch.Tensor):
            boxed_outputs.append(value)
            compiled_position += 1
            continue

        dtype = scalar_dtype(example)
        if dtype is None:
            # Non-value outputs such as None remain outside the compiled tensor
            # output list, matching the existing PT2 parser behavior.
            boxed_outputs.append(value)
            continue
        if should_box is not None and not should_box(example):
            boxed_outputs.append(value)
            continue

        with gm.graph.inserting_before(output):
            boxed = gm.graph.call_function(
                torch.scalar_tensor, (value,), {"dtype": dtype}
            )
        boxed_outputs.append(boxed)
        scalar_output_positions.append(compiled_position)
        compiled_position += 1

    if scalar_output_positions:
        output.args = (torch.utils._pytree.tree_unflatten(boxed_outputs, output_spec),)
        gm.graph.lint()
        gm.recompile()

    return tuple(scalar_output_positions)


# ---------------------------------------------------------------------------
# Tensor-backed SDPA scale canonicalization
# ---------------------------------------------------------------------------


_tensor_scale_sdpa_op = None


def _get_tensor_scale_sdpa_op():
    """An opaque export spelling for SDPA whose scale is a rank-zero tensor.

    ATen's public SDPA schema accepts a Python float for ``scale``. Dynamo can
    nevertheless capture a model parameter as ``scale_tensor.item()``; nested
    non-strict export then tries to guard that data-dependent float. Keeping
    the tensor in the custom-op schema lets PT2 carry the authored runtime
    value to the translator without specializing it.
    """
    global _tensor_scale_sdpa_op
    if _tensor_scale_sdpa_op is not None:
        return _tensor_scale_sdpa_op

    import torch

    @torch.library.custom_op("luminal::sdpa_tensor_scale", mutates_args=())
    def sdpa_tensor_scale(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        dropout_p: float,
        is_causal: bool,
        scale: torch.Tensor,
        enable_gqa: bool,
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale.item(),
            enable_gqa=enable_gqa,
        )

    @sdpa_tensor_scale.register_fake
    def _fake(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    ):
        del key, attn_mask, dropout_p, is_causal, scale, enable_gqa
        return query.new_empty((*query.shape[:-1], value.shape[-1]))

    # FX nodes target dispatcher overloads. ``torch.library.custom_op`` returns
    # a ``CustomOpDef`` wrapper, which is callable but has no ``__name__`` and
    # therefore cannot be inserted with ``Graph.call_function``.
    _tensor_scale_sdpa_op = torch.ops.luminal.sdpa_tensor_scale.default
    return _tensor_scale_sdpa_op


def _canonicalize_tensor_scale_sdpa(gm):
    """Replace only ``SDPA(..., scale=t.item())`` with its tensor-scale op."""
    import torch

    sdpa_targets = {
        torch.nn.functional.scaled_dot_product_attention,
        torch._C._nn.scaled_dot_product_attention,
        torch.ops.aten.scaled_dot_product_attention.default,
    }

    def argument(node, position, name, default):
        if name in node.kwargs:
            return node.kwargs[name]
        return node.args[position] if len(node.args) > position else default

    changed = False
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in sdpa_targets:
            continue
        scale_item = argument(node, 6, "scale", None)
        if not isinstance(scale_item, torch.fx.Node):
            continue
        is_item = (scale_item.op == "call_method" and scale_item.target == "item") or (
            scale_item.op == "call_function"
            and getattr(scale_item.target, "_overloadpacket", None)
            is torch.ops.aten.item
        )
        if not is_item or not scale_item.args:
            continue
        scale_tensor = scale_item.args[0]
        if not isinstance(scale_tensor, torch.fx.Node):
            continue

        with graph.inserting_before(node):
            replacement = graph.call_function(
                _get_tensor_scale_sdpa_op(),
                (
                    node.args[0],
                    node.args[1],
                    node.args[2],
                    argument(node, 3, "attn_mask", None),
                    argument(node, 4, "dropout_p", 0.0),
                    argument(node, 5, "is_causal", False),
                    scale_tensor,
                    argument(node, 7, "enable_gqa", False),
                ),
            )
            replacement.meta = node.meta.copy()
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()
    return changed


# ---------------------------------------------------------------------------
# Decomposition table
# ---------------------------------------------------------------------------


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
    import torch

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


# ---------------------------------------------------------------------------
# Symbolic-sum serde workaround
# ---------------------------------------------------------------------------


def _lower_sym_sum(ep) -> None:
    """Rewrite `torch.sym_sum` nodes into chains of `operator.add` so the
    ExportedProgram survives `torch.export.save`.

    WORKAROUND for a torch.export inconsistency: the export VERIFIER allows
    sym_sum in graphs (pytorch#159111, landed 2025-07), but the PT2 serde's
    `_SYM_OPS` table never got the matching entry, so `torch.export.save`
    raises `AssertionError: op sym_sum is not in _SYM_OPS` on any graph the
    verifier just blessed. sym_sum appears whenever shape arithmetic sums
    three or more SymInts (sympy's Add is n-ary) — e.g. HF llama under
    attn_implementation="sdpa" with a growing KV cache.

    The upstream one-line fix exists inside the (larger, still-open)
    duck-sizing PR: https://github.com/pytorch/pytorch/pull/186373
    This pass checks torch's own table first, so it becomes a no-op — and
    can be DELETED — once we run a torch that includes that fix.
    """
    import operator

    import torch

    if not hasattr(torch, "sym_sum"):
        return  # torch too old to ever emit it
    try:
        from torch._export.serde.serialize import _SYM_OPS

        if torch.sym_sum in _SYM_OPS:
            return  # torch can serialize it natively (pytorch#186373 landed)
    except ImportError:
        pass  # private module moved — fall through and lower defensively

    def _val(x):
        # SymInt/int value of a term: fx Nodes carry it in meta["val"];
        # bare ints are their own value.
        return x.meta["val"] if hasattr(x, "meta") else x

    gm = ep.graph_module
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.sym_sum:
            continue
        (terms,) = node.args
        terms = list(terms)
        with gm.graph.inserting_before(node):
            acc = terms[0]
            acc_val = _val(acc)
            for term in terms[1:]:
                acc = gm.graph.call_function(operator.add, (acc, term))
                # Every ExportedProgram node must carry meta["val"] (the
                # verifier's _check_val rejects the graph otherwise). The
                # partial sums are real SymInt arithmetic over the terms'
                # vals; the FINAL node inherits the original node's meta
                # wholesale so downstream consumers see an exact swap.
                acc_val = acc_val + _val(term)
                acc.meta["val"] = acc_val
        acc.meta = {**node.meta, **acc.meta}
        node.replace_all_uses_with(acc)
        gm.graph.erase_node(node)
        changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()


# ---------------------------------------------------------------------------
# Input-guard / dead data-dependent-op cleanups
# ---------------------------------------------------------------------------


def _drop_input_guards(ep):
    """Discard ``ep._guards_code`` so unlift does not emit a ``_guards_fn``.

    LUM-499: When a 0-d int tensor flows into a tensor index (``x[i]`` with
    ``i = torch.tensor(2)``), torch.export records two equivalent input
    guards: ``L['i'].item() == 2`` (referencing the original local source)
    and ``L['args'][1].item() == 2`` (referencing the rewrapped flat args).
    Two failures stack on top of each other:

    1. ``ep.module()`` (invoked inside ``run_decompositions``) rewrites
       ``L['args'][1]`` → ``args[1]`` but cannot resolve ``L['i']``, leaving
       a literal ``L`` reference in the generated ``_guards_fn`` and raising
       ``NameError: name 'L' is not defined`` during retracing.
    2. Even after dropping the unresolvable guard, the surviving
       ``args[1].item()`` is data-dependent: AOT autograd's fake-tensor pass
       raises ``DataDependentOutputException(_local_scalar_dense)``, forcing
       a graph break.

    These guards exist solely to validate inputs at runtime in eager-mode
    consumers of the ExportedProgram; the luminal compiler does its own
    input shape/dtype checks against the compiled graph signature, so we
    are not losing any safety by clearing them.
    """

    if hasattr(ep, "_guards_code"):
        ep._guards_code = []


def _run_decompositions_without_guards(ep, decomposition_table):
    """Run export decompositions without carrying eager-only input guards.

    ``run_decompositions`` retraces the unlifted program and can populate a
    fresh ``_guards_code`` list on the returned ExportedProgram even when the
    input program's guards were already cleared.  A later decomposition pass
    would then execute that generated ``_guards_fn`` while retracing.  Besides
    the data-dependent guards described in :func:`_drop_input_guards`, PyTorch
    can emit guard expressions that refer to generated
    ``__meta_utils_unknown_tensor*`` globals which are not installed in the
    unlifted module's namespace.

    Luminal validates the compiled boundary itself, so those eager consumer
    guards are neither part of the authored tensor program nor needed by the
    backend.  Clear them on both sides of every decomposition pass.  Doing so
    centrally also makes repeated passes obey the same invariant.
    """

    _drop_input_guards(ep)
    decomposed = ep.run_decompositions(decomposition_table)
    _drop_input_guards(decomposed)
    return decomposed


def _drop_dead_data_dependent_ops(gm):
    """Remove ``aten.item.default`` (and other data-dependent ops) with no users.

    When dynamo specializes a 0-d int input by tracing through ``.item()``,
    the resulting graph may contain a dead ``aten.item.default`` node whose
    output is never consumed. luminal's translator does not lower
    ``aten._local_scalar_dense`` / ``aten.item.default``, so leaving the dead
    node in the graph causes a graph break at compile time. Eliminating it
    keeps the (correctly specialized) downstream graph in a single subgraph.
    """
    import torch

    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and getattr(node.target, "_overloadpacket", None) is torch.ops.aten.item
            and len(node.users) == 0
        ):
            graph.erase_node(node)
            changed = True

    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()


__all__ = [
    "_box_scalar_graph_outputs",
    "_canonicalize_tensor_scale_sdpa",
    "_decomp_table",
    "_drop_dead_data_dependent_ops",
    "_drop_input_guards",
    "_run_decompositions_without_guards",
    "_get_cache_dict",
    "_lower_sym_sum",
    "_register_cache_serialization",
    "flatten_dynamic_cache",
    "flatten_with_keys_dynamic_cache",
    "unflatten_dynamic_cache",
]
