"""Strict ``torch.export`` normalization for compiler-owned FX regions."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx

from .pt2 import (
    _build_dynamic_shapes_from_gm,
    _decomp_table,
    _drop_dead_data_dependent_ops,
    _drop_input_guards,
    _export_kwargs,
    _strip_symint_placeholders,
)
from .region_abi import DynamicRange


@dataclass(frozen=True, slots=True)
class RegionExport:
    program: torch.export.ExportedProgram
    input_indices: tuple[int, ...]
    dynamic_ranges: tuple[DynamicRange, ...]


def export_region(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    dynamic_range: tuple[int, int] | None = None,
) -> RegionExport:
    """Normalize an FX region without storage access or static fallback.

    Dynamo may expose a symbolic tensor dimension as a separate ``SymInt``
    argument.  Replace that argument with a tensor-size operation before
    re-exporting so the exported program has a tensor-only runtime signature.
    The caller's graph is never modified.
    """

    graph = copy.deepcopy(graph).eval()
    _strip_data_attr(graph)
    inputs, input_indices, strip_ok = _strip_symint_placeholders(
        graph, list(example_inputs)
    )
    if not strip_ok:
        raise RuntimeError(
            "cannot export region: a SymInt input could not be derived from "
            "a tensor dimension"
        )

    dynamic_shapes = _build_dynamic_shapes_from_gm(graph, dynamic_range)
    export_inputs = _fresh_export_inputs(inputs)
    has_varargs = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in inspect.signature(graph.forward).parameters.values()
    )
    if dynamic_shapes is not None and not has_varargs:
        dynamic_shapes = dynamic_shapes["args"]
    try:
        # This backend runs inside Dynamo's tracing context. The normalized
        # region is a separate export, so it must build its own shape guards
        # instead of adding them to vLLM's outer ShapeEnv.
        with torch._guards.tracing(None):
            exported = torch.export.export(
                graph,
                tuple(export_inputs),
                dynamic_shapes=dynamic_shapes,
                **_export_kwargs(),
            )
        _drop_input_guards(exported)
        _drop_dead_data_dependent_ops(exported.graph_module)
        exported = exported.run_decompositions(_decomp_table())
        _drop_input_guards(exported)
        ranges = _set_range_constraints(exported, dynamic_range)
        return RegionExport(exported, tuple(input_indices), ranges)
    except Exception as error:
        raise RuntimeError(
            "torch.export failed for compiler-owned FX region: "
            f"{type(error).__name__}: {error}"
        ) from error


def _fresh_export_inputs(inputs: list[Any]) -> list[Any]:
    """Detach a second export from Dynamo's original symbolic ShapeEnv.

    Dynamo FakeTensors still refer to symbols and guards from the outer vLLM
    trace. Recreate only their metadata so our dynamic_shapes specification,
    rather than those stale outer-trace guards, defines the exported symbols.
    """

    from torch._dynamo.source import LocalSource
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    if not any(isinstance(value, FakeTensor) for value in inputs):
        return inputs

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    symbol_index = 0

    def fresh_dim(dim: Any) -> int | torch.SymInt:
        nonlocal symbol_index
        if not isinstance(dim, torch.SymInt):
            return int(dim)
        hint = dim.node.hint
        if hint is None:
            raise RuntimeError(
                f"symbolic dimension {dim.node.expr} has no concrete hint"
            )
        name = f"luminal_export_dim_{symbol_index}"
        symbol_index += 1
        return shape_env.create_symintnode(
            shape_env.create_symbol(
                int(hint), LocalSource(name), dynamic_dim=DimDynamic.DYNAMIC
            ),
            hint=int(hint),
        )

    with fake_mode:
        return [
            torch.empty_strided(
                tuple(fresh_dim(dim) for dim in value.shape),
                tuple(int(dim) for dim in value.stride()),
                dtype=value.dtype,
                device=value.device,
                requires_grad=value.requires_grad,
            )
            if torch.is_tensor(value)
            else value
            for value in inputs
        ]


def _strip_data_attr(graph: fx.GraphModule) -> None:
    """Treat ``Tensor.data`` as identity inside an inference-only region.

    Dynamo represents ``tensor.data`` with the private ``_get_data_attr`` op.
    If that op reaches ``torch.export``, export may lift the aliased tensor as a
    constant instead of preserving the original region input. Luminal regions
    run without autograd, so the detached alias has the same observable value
    and storage semantics as its source tensor.
    """

    target = getattr(torch._C._autograd, "_get_data_attr", None)
    if target is None:
        return

    changed = False
    for node in list(graph.graph.nodes):
        if node.op != "call_function" or node.target is not target:
            continue
        if len(node.args) != 1 or node.kwargs:
            raise RuntimeError("unexpected _get_data_attr invocation")
        node.replace_all_uses_with(node.args[0])
        graph.graph.erase_node(node)
        changed = True

    if changed:
        graph.graph.lint()
        graph.recompile()


def _set_range_constraints(
    exported: torch.export.ExportedProgram,
    dynamic_range: tuple[int, int] | None,
) -> tuple[DynamicRange, ...]:
    from torch.utils._sympy.value_ranges import ValueRanges

    used_symbols = set()
    for node in exported.graph_module.graph.nodes:
        for value in torch.utils._pytree.tree_leaves(node.meta.get("val")):
            values = (
                (*value.shape, *value.stride())
                if isinstance(value, torch.Tensor)
                else (value,)
            )
            for item in values:
                if isinstance(item, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                    used_symbols.update(item.node.expr.free_symbols)
    for symbol in list(exported.range_constraints):
        if symbol not in used_symbols:
            del exported.range_constraints[symbol]

    if dynamic_range is None or dynamic_range[0] == dynamic_range[1]:
        return ()
    if len(exported.range_constraints) != 1:
        raise RuntimeError(
            "ranged region export requires exactly one dynamic dimension, "
            f"found {len(exported.range_constraints)}"
        )

    minimum, maximum = dynamic_range
    effective_minimum = max(2, minimum)
    if maximum < effective_minimum:
        raise RuntimeError(
            f"dynamic range [{minimum}, {maximum}] has no values supported "
            "by torch.export; sizes 0 and 1 require exact artifacts"
        )

    symbol, constraint = next(iter(exported.range_constraints.items()))
    constraint &= ValueRanges(effective_minimum, maximum)
    exported.range_constraints[symbol] = constraint
    try:
        lower = int(constraint.lower)
        upper = int(constraint.upper)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"dynamic dimension {symbol} does not have finite bounds"
        ) from error
    return (DynamicRange(str(symbol), lower, upper),)
