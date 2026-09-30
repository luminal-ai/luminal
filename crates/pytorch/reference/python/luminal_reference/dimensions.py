"""PyTorch dimension bounds and profiling values."""

import sys

import sympy
import torch

from .export_helpers import _safe_int_bound


def bounds(env, symbol):
    vr = env.var_to_range[symbol]
    lo, hi = _safe_int_bound(vr.lower), _safe_int_bound(vr.upper)
    return max(0, lo or 0), min(
        sys.maxsize - 1, hi
    ) if hi is not None else sys.maxsize - 1


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
