"""PyTorch dimension bounds and profiling values."""

import sys

import torch


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
