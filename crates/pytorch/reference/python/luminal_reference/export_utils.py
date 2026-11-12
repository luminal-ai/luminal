"""Graph-preparation helpers for the PyTorch backends.

AOTAutograd hands a backend functional ATen graphs. These helpers prepare
one for the Rust translator:

  - a private copy of a graph module that shares its parameters
  - the decomposition tables: which composites are decomposed while
    AOTAutograd traces, and which the translator lowers itself
  - the ``sym_sum`` serializer workaround
  - the input-guard and dead data-dependent-op cleanups
"""


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


# Composite ATen ops the translator lowers as single Luminal ops. This mirrors
# the dispatch table in crates/pytorch/utils/src/translate.rs (follow-up: export
# that list from Rust instead of restating it here). The core ATen
# decomposition table would split these into primitives that search then has
# to reassemble, so they are preserved for the translator.
_NATIVELY_LOWERED = frozenset(
    {
        "__and__.Tensor",
        "__or__.Tensor",
        "__xor__.Tensor",
        "_fused_rms_norm.default",
        "_scaled_dot_product_flash_attention_for_cpu.default",
        "_unsafe_view.default",
        "_upsample_bilinear2d_aa.vec",
        "_upsample_nearest_exact1d.default",
        "_upsample_nearest_exact1d.vec",
        "_upsample_nearest_exact2d.default",
        "_upsample_nearest_exact2d.vec",
        "_upsample_nearest_exact3d.default",
        "_upsample_nearest_exact3d.vec",
        "adaptive_avg_pool2d.default",
        "adaptive_avg_pool3d.default",
        "arange.default",
        "arange.start",
        "argsort.default",
        "binary_cross_entropy_with_logits.default",
        "cat.default",
        "celu.default",
        "clamp_max.Tensor",
        "clamp_min.Tensor",
        "conv2d.default",
        "detach.default",
        "dot.default",
        "floor_divide.default",
        "glu.default",
        "group_norm.default",
        "im2col.default",
        "isfinite.default",
        "item.default",
        "layer_norm.default",
        "leaky_relu.default",
        "linalg_vector_norm.default",
        "linear.default",
        "log_softmax.int",
        "logaddexp.default",
        "logaddexp2.default",
        "logit.default",
        "masked_fill.Scalar",
        "matmul.default",
        "max_pool2d.default",
        "mish.default",
        "mvlgamma.default",
        "nansum.default",
        "ones.default",
        "ones_like.default",
        "pad.default",
        "repeat_interleave.self_Tensor",
        "repeat_interleave.self_int",
        "rms_norm.default",
        "rsub.Scalar",
        "rsub.Tensor",
        "scaled_dot_product_attention.default",
        "select_scatter.default",
        "silu.default",
        "sinc.default",
        "smooth_l1_loss.default",
        "soft_margin_loss.default",
        "softmax.int",
        "softplus.default",
        "special_i0.default",
        "special_xlog1py.default",
        "split.Tensor",
        "square.default",
        "squeeze.default",
        "squeeze.dim",
        "stack.default",
        "std.correction",
        "std.default",
        "std.dim",
        "sum.default",
        "sym_size.int",
        "t.default",
        "transpose.int",
        "tril.default",
        "triu.default",
        "unbind.int",
        "upsample_bicubic2d.vec",
        "upsample_linear1d.default",
        "upsample_linear1d.vec",
        "upsample_nearest1d.default",
        "upsample_nearest1d.vec",
        "upsample_nearest2d.default",
        "upsample_nearest3d.default",
        "upsample_nearest3d.vec",
        "upsample_trilinear3d.vec",
        "var.default",
        "var.dim",
        "var_mean.default",
        "var_mean.dim",
        "vdot.default",
        "where.ScalarOther",
        "xlogy.Tensor",
        "zeros_like.default",
    }
)


def _aot_decomposition_table():
    """The one decomposition table AOTAutograd applies while tracing.

    Core ATen decompositions, minus SDPA (see `_decomp_table`), minus the
    composites the translator lowers directly. Every other op the translator
    cannot lower is decomposed into primitives it can, and an op outside both
    sets fails translation by name. Passed to ``aot_autograd`` as a thunk, so
    the table is built once per compiler.
    """
    table = _decomp_table()
    if table is None:
        return {}
    for op in list(table):
        name = f"{op._schema.name.split('::')[-1]}.{op._overloadname}"
        if name in _NATIVELY_LOWERED:
            table.pop(op, None)
    return table.materialize() if hasattr(table, "materialize") else dict(table)


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
    "_aot_decomposition_table",
    "_decomp_table",
    "_drop_dead_data_dependent_ops",
    "_drop_input_guards",
    "_lower_sym_sum",
    "private_graph_copy",
]
