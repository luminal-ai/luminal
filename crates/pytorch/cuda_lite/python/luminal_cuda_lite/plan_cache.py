"""Process-local reuse of searched CUDA-lite plans.

The cache owns immutable Rust ``PlanTemplate`` objects only.  A caller always
builds a fresh translated graph and binds its own tensors; a hit merely skips
the genetic search and installs the selected plan on that fresh runtime.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.fx as fx


_SCHEMA_VERSION = 1
_SYMBOL = re.compile(r"(?<![A-Za-z0-9_])(?:s|u)\d+(?![A-Za-z0-9_])")


class _Symbols:
    def __init__(self) -> None:
        self._names: dict[str, str] = {}

    def text(self, value: Any) -> str:
        text = str(value)

        def replace(match: re.Match[str]) -> str:
            name = match.group(0)
            return self._names.setdefault(name, f"d{len(self._names)}")

        return _SYMBOL.sub(replace, text)


def _target(target: Any) -> str:
    module = getattr(target, "__module__", None)
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    return f"{module}.{name}" if module and name else str(target)


def _encode(value: Any, nodes: Mapping[fx.Node, int], symbols: _Symbols) -> Any:
    if isinstance(value, fx.Node):
        return ["node", nodes[value]]
    if isinstance(value, tuple):
        return ["tuple", *(_encode(item, nodes, symbols) for item in value)]
    if isinstance(value, list):
        return ["list", *(_encode(item, nodes, symbols) for item in value)]
    if isinstance(value, dict):
        return [
            "dict",
            *(
                [symbols.text(key), _encode(item, nodes, symbols)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ),
        ]
    if isinstance(value, slice):
        return [
            "slice",
            _encode(value.start, nodes, symbols),
            _encode(value.stop, nodes, symbols),
            _encode(value.step, nodes, symbols),
        ]
    if isinstance(value, (str, torch.dtype, torch.device)):
        return [type(value).__name__, symbols.text(value)]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return [type(value).__name__, symbols.text(value)]


def _tensor_meta(value: Any, symbols: _Symbols) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": [symbols.text(axis) for axis in value.shape],
            "stride": [symbols.text(axis) for axis in value.stride()],
            "storage_offset": symbols.text(value.storage_offset()),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_meta(item, symbols) for item in value]
    if value is None:
        return None
    # Scalar symbolic values affect generated code even though they are not
    # tensor boundary rows.
    return [type(value).__name__, symbols.text(value)]


def _boundary_name_map(ep: Any) -> dict[str, str]:
    names: dict[str, str] = {}
    for prefix, specs in (
        ("i", ep.graph_signature.input_specs),
        ("o", ep.graph_signature.output_specs),
    ):
        for index, spec in enumerate(specs):
            name = getattr(spec.arg, "name", None)
            if name is not None:
                names.setdefault(name, f"{prefix}{index}")
    return names


def structural_fingerprint(
    ep: Any,
    input_layouts: Sequence[tuple[str, str, Sequence[str]]],
    output_layouts: Sequence[tuple[str, str, Sequence[str]]],
    output_aliases: Sequence[tuple[str, str, int]],
    *,
    search_iterations: int | None,
    dynamic_range: tuple[int, int] | None,
    search_configuration: str | None = None,
) -> str:
    """Hash normalized PT2 structure and every plan-affecting contract.

    Node and boundary names are positional, and generated PT2 symbols are
    alpha-renamed by first occurrence. Parameter contents and addresses are
    deliberately absent: they are external boundary storage on every bound
    instance.
    """

    symbols = _Symbols()
    graph_nodes = list(ep.graph_module.graph.nodes)
    node_ids = {node: index for index, node in enumerate(graph_nodes)}
    nodes = [
        [
            node.op,
            _target(node.target) if node.op not in ("placeholder", "output") else node.op,
            _encode(node.args, node_ids, symbols),
            _encode(node.kwargs, node_ids, symbols),
            _tensor_meta(node.meta.get("val"), symbols),
        ]
        for node in graph_nodes
    ]
    boundary_names = _boundary_name_map(ep)

    def layouts(rows: Sequence[tuple[str, str, Sequence[str]]]) -> list[Any]:
        return [
            [
                boundary_names.get(name, name),
                tag,
                [symbols.text(stride) for stride in strides],
            ]
            for name, tag, strides in rows
        ]

    signature = {
        "inputs": [spec.kind.name for spec in ep.graph_signature.input_specs],
        "outputs": [
            [spec.kind.name, boundary_names.get(str(spec.target), str(spec.target))]
            for spec in ep.graph_signature.output_specs
        ],
    }
    constraints = sorted(symbols.text(row) for row in ep.range_constraints.items())
    capability = None
    device_name = None
    device_memory = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        device_name = torch.cuda.get_device_name(device)
        device_memory = torch.cuda.get_device_properties(device).total_memory
    manifest = {
        "schema": _SCHEMA_VERSION,
        "nodes": nodes,
        "signature": signature,
        "constraints": constraints,
        "input_layouts": layouts(input_layouts),
        "output_layouts": layouts(output_layouts),
        "aliases": [
            [
                boundary_names.get(name, name),
                boundary_names.get(owner, owner),
                offset,
            ]
            for name, owner, offset in output_aliases
        ],
        "dynamic_range": dynamic_range,
        "search": search_configuration or {"generations": search_iterations},
        "cuda": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "capability": capability,
            "device": device_name,
            "total_memory": device_memory,
        },
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    entries: int


class _Pending:
    def __init__(self, condition: threading.Condition) -> None:
        self.condition = condition


_lock = threading.RLock()
_entries: OrderedDict[str, Any] = OrderedDict()
_hits = 0
_misses = 0


def get_or_create(key: str, create: Callable[[], Any]) -> tuple[Any, bool]:
    """Return ``(template, hit)`` with one creator per structural key."""

    global _hits, _misses
    with _lock:
        while True:
            entry = _entries.get(key)
            if isinstance(entry, _Pending):
                entry.condition.wait()
                continue
            if entry is not None:
                _hits += 1
                _entries.move_to_end(key)
                return entry, True
            pending = _Pending(threading.Condition(_lock))
            _entries[key] = pending
            _misses += 1
            break
    try:
        template = create()
    except BaseException:
        with _lock:
            if _entries.get(key) is pending:
                del _entries[key]
            pending.condition.notify_all()
        raise
    with _lock:
        _entries[key] = template
        _entries.move_to_end(key)
        limit = max(1, int(os.getenv("LUMINAL_PLAN_CACHE_SIZE", "256")))
        while len(_entries) > limit:
            oldest, value = next(iter(_entries.items()))
            if isinstance(value, _Pending):
                _entries.move_to_end(oldest)
                continue
            _entries.popitem(last=False)
        pending.condition.notify_all()
    return template, False


def cache_stats() -> CacheStats:
    with _lock:
        entries = sum(not isinstance(value, _Pending) for value in _entries.values())
        return CacheStats(_hits, _misses, entries)


def clear_plan_cache() -> None:
    global _hits, _misses
    with _lock:
        if any(isinstance(value, _Pending) for value in _entries.values()):
            raise RuntimeError("cannot clear the Luminal plan cache while a search is running")
        _entries.clear()
        _hits = 0
        _misses = 0
