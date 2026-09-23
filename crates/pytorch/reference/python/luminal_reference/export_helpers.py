"""Shape normalization shared by direct and region exports."""

import inspect

import torch


def _export_kwargs():
    """Build common kwargs for torch.export.export()."""
    kwargs = {"strict": False}
    if (
        "prefer_deferred_runtime_asserts_over_guards"
        in inspect.signature(torch.export.export).parameters
    ):
        kwargs["prefer_deferred_runtime_asserts_over_guards"] = True
    return kwargs


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
      - `kept_indices` is the indices into `example_inputs` we kept (handed
        to CompiledModel as `user_indices` so __call__ drops stripped SymInts)
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


def _build_dynamic_shapes_from_gm(gm, dynamic_range=None):
    """Construct a torch.export.export `dynamic_shapes` spec from FX metadata.

    Walks each tensor placeholder's `meta['example_value']` FakeTensor and
    marks every SymInt dimension as dynamic. Bounded exports reuse the same
    `Dim` object wherever the same original FakeTensor symbol appears.

    The returned spec is wrapped under `{"args": (...)}` because Dynamo's
    `GraphModule.forward(*args, **kwargs)` signature treats positional inputs
    as the `args` tuple.

    Returns None if there are no symbolic dims to mark.
    """
    from torch.export import Dim

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    if dynamic_range is not None:
        minimum, maximum = dynamic_range
        if minimum < 0 or maximum < minimum:
            raise ValueError(f"invalid dynamic range [{minimum}, {maximum}]")
        if minimum == maximum:
            return None

        # torch.export specializes dimensions of size 0 and 1 instead of
        # representing them with a backed symbolic Dim. Ranged artifacts must
        # therefore start at 2 or higher; exact 0/1 artifacts take the static
        # path above.
        effective_minimum = max(2, minimum)
        if maximum < effective_minimum:
            raise ValueError(
                f"dynamic range [{minimum}, {maximum}] has no symbolic values"
            )

    per_input_spec = []
    saw_dynamic = False
    bounded_dims = {}
    for node in placeholders:
        ev = node.meta.get("example_value")
        if not torch.is_tensor(ev):
            per_input_spec.append(None)
            continue
        spec = {}
        for d, s in enumerate(ev.shape):
            if isinstance(s, torch.SymInt):
                if dynamic_range is None:
                    spec[d] = Dim.AUTO
                else:
                    # The fresh export inputs deliberately carry independent
                    # symbols. Reusing this Dim is the one place that restores
                    # equality between occurrences of the original symbol.
                    key = str(s.node.expr)
                    if key not in bounded_dims:
                        bounded_dims[key] = Dim(
                            f"luminal_dim_{len(bounded_dims)}",
                            min=effective_minimum,
                            max=dynamic_range[1],
                        )
                    spec[d] = bounded_dims[key]
                saw_dynamic = True
        per_input_spec.append(spec if spec else None)

    if not saw_dynamic:
        return None
    return {"args": tuple(per_input_spec)}


def _cuda_device_index(tensors, expected=None):
    indices = {
        t.device.index
        for t in tensors
        if isinstance(t, torch.Tensor) and t.device.type == "cuda"
    }
    if None in indices or len(indices) > 1:
        raise ValueError(
            "CUDA inputs span multiple logical devices or lack an explicit index"
        )
    observed = next(iter(indices), None)
    if expected is not None and observed is not None and observed != expected:
        raise ValueError(f"expected CUDA device {expected}, got {observed}")
    selected = expected if expected is not None else observed
    if selected not in (None, 0):
        raise ValueError(
            f"Luminal currently supports only logical CUDA device 0, got {selected}"
        )
    return selected
