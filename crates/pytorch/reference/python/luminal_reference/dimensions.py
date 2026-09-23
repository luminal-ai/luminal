"""ShapeVar identity, PyTorch bounds, and reference-runtime bucket policies."""

import sys
from dataclasses import dataclass
from itertools import pairwise

import sympy
import torch
from torch.fx.experimental.dynamic_spec import ShapeVar

from .export_helpers import _safe_int_bound


@dataclass(frozen=True, kw_only=True)
class DimBucket:
    """Inclusive dimension interval and the size used to profile its plan."""

    min: int
    max: int
    representative: int | None = None

    def __post_init__(self):
        if (
            type(self.min) is not int
            or type(self.max) is not int
            or not 0 <= self.min <= self.max <= sys.maxsize - 1
        ):
            raise ValueError(
                "bucket bounds must be nonnegative integers with min <= max"
            )
        if self.representative is None:
            object.__setattr__(self, "representative", (self.min + self.max) // 2)
        if (
            type(self.representative) is not int
            or not self.min <= self.representative <= self.max
        ):
            raise ValueError(
                "bucket representative must be an integer within [min, max]"
            )

    def as_tuple(self):
        return self.min, self.max, self.representative


def normalize_buckets(config):
    result = {}
    for dim, buckets in (config or {}).items():
        if not isinstance(dim, ShapeVar):
            raise TypeError("dim_buckets keys must be PyTorch ShapeVar objects")
        buckets = tuple(buckets)
        if not buckets or any(not isinstance(b, DimBucket) for b in buckets):
            raise TypeError(
                "each ShapeVar requires a nonempty sequence of DimBucket objects"
            )
        for bucket in buckets:
            if bucket.min < dim.min or (dim.max is not None and bucket.max > dim.max):
                raise ValueError(f"bucket {bucket} lies outside {dim!r} bounds")
        if any(a.max >= b.min for a, b in pairwise(buckets)):
            raise ValueError("buckets must be sorted and non-overlapping")
        result[dim] = buckets
    return result


def bounds(env, symbol):
    vr = env.var_to_range[symbol]
    lo, hi = _safe_int_bound(vr.lower), _safe_int_bound(vr.upper)
    return max(0, lo or 0), min(
        sys.maxsize - 1, hi
    ) if hi is not None else sys.maxsize - 1


def resolve_policies(env, config):
    mapping = getattr(env, "_spec_symbol_to_compile_symbol", {})
    policies = {}
    for dim, buckets in config.items():
        expr = mapping.get(dim.sympy_sym)
        if expr is None:
            raise ValueError(
                f"bucket ShapeVar {dim!r} is not present in this graph's ShapesSpec"
            )
        expr = env.replace(expr)
        if not isinstance(expr, sympy.Symbol):
            raise TypeError(
                f"bucket ShapeVar {dim!r} no longer denotes a base dimension: {expr}"
            )
        if expr in policies and policies[expr] != buckets:
            raise ValueError(
                "PyTorch unified ShapeVars with conflicting bucket policies"
            )
        policies[expr] = buckets
    return policies


def profile_value(value):
    """Choose concrete profiling storage without specializing symbolic metadata."""
    if not isinstance(value, torch.SymInt):
        return value
    if value.node.hint is not None:
        return value.node.hint
    env = value.node.shape_env
    substitutions = {}
    for symbol in value.node.expr.free_symbols:
        # Only input-backed unbacked symbols from ShapesSpec are supported.
        if symbol not in env.unbacked_inputs and symbol not in env.backed_var_to_val:
            raise TypeError(f"data-dependent dimension {symbol} has no profiling input")
        lo, hi = bounds(env, symbol)
        substitutions[symbol] = env.var_to_hint_override.get(
            symbol, env.backed_var_to_val.get(symbol, min(hi, max(lo, 2)))
        )
    return int(value.node.expr.subs(substitutions))


def export_specs(shapes):
    """Re-export using shared Dim objects, preserving input symbol relationships."""
    ranges = {}
    for shape in shapes:
        for size in shape:
            if not isinstance(size, torch.SymInt):
                continue
            expr = size.node.expr
            for symbol in expr.free_symbols:
                ranges.setdefault(symbol, bounds(size.node.shape_env, symbol))
            if len(expr.free_symbols) == 1:
                symbol = next(iter(expr.free_symbols))
                scale = expr.diff(symbol)
                offset = sympy.expand(expr - scale * symbol)
                if scale.is_Integer and scale > 0 and offset.is_Integer:
                    lo, hi = ranges[symbol]
                    # A derived tensor extent must itself fit PyTorch's signed
                    # shape domain, even when the root has no finite upper bound.
                    ranges[symbol] = (
                        max(lo, int(sympy.ceiling(-offset / scale))),
                        min(hi, (sys.maxsize - 1 - int(offset)) // int(scale)),
                    )
    dims = {
        symbol: torch.export.Dim(f"luminal_dim_{i}", min=lo, max=hi)
        for i, (symbol, (lo, hi)) in enumerate(ranges.items())
    }

    def convert(size):
        expr = size.node.expr
        if isinstance(expr, sympy.Symbol):
            return dims[expr]
        if len(expr.free_symbols) == 1:
            symbol = next(iter(expr.free_symbols))
            scale = expr.diff(symbol)
            offset = sympy.expand(expr - scale * symbol)
            if scale.is_Integer and scale > 0 and offset.is_Integer:
                return int(scale) * dims[symbol] + int(offset)
        return torch.export.Dim.AUTO

    return [
        {
            i: convert(d)
            for i, d in enumerate(shape)
            if isinstance(d, torch.SymInt) and d.node.expr.free_symbols
        }
        for shape in shapes
    ]


def remap_buckets(ep, original_shapes, policies):
    """Map original AOT symbols through PT2's symbol renaming using input axes."""
    if not policies:
        return {}
    inputs = [
        n.meta["val"]
        for n, spec in zip(
            (n for n in ep.graph_module.graph.nodes if n.op == "placeholder"),
            ep.graph_signature.input_specs,
        )
        if spec.kind.name == "USER_INPUT"
    ]
    # Distinct Dummy symbols prevent accidental old/new name collisions.
    roots = {s: sympy.Dummy(str(s), integer=True) for s in policies}
    equations = []
    for original, exported in zip(original_shapes, inputs):
        for old, new in zip(original, getattr(exported, "shape", ())):
            if isinstance(old, torch.SymInt) and isinstance(new, torch.SymInt):
                equations.append(new.node.expr - old.node.expr.xreplace(roots))
    new_symbols = [s for s in ep.range_constraints if isinstance(s, sympy.Symbol)]
    solutions = sympy.solve(equations, new_symbols, dict=True)
    result = {}
    if not solutions:
        raise ValueError("cannot map configured ShapeVar buckets into local export")
    for new, expr in solutions[0].items():
        for old, root in roots.items():
            if expr == root:
                result[str(new)] = [b.as_tuple() for b in policies[old]]
            elif root in expr.free_symbols:
                raise ValueError(
                    f"bucket dimension {old} was transformed to {expr}; unsupported bucket remapping"
                )
    return result
