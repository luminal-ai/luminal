"""Process-local reuse for structurally identical compiled regions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import torch
import torch.fx as fx

from .compiled_model import CompiledModel

_SYMBOL = re.compile(r"\b[su]\d+\b")
_artifacts: dict[str, CompiledArtifact] = {}
_reuse_hits = 0
_searches = 0


@dataclass(frozen=True)
class ArtifactCacheStats:
    unique_artifacts: int
    reuse_hits: int
    searches: int


class CompiledArtifact:
    """Compiled runtime shared by separate positional tensor bindings."""

    def __init__(self, graph, weight_refs=()):
        self.graph = graph
        self.weight_refs = tuple(weight_refs)
        self._active_binding = None

    def bind(self, **kwargs):
        return CompiledModel(
            self.graph,
            weight_refs=self.weight_refs,
            artifact=self,
            **kwargs,
        )

    def activate(self, binding) -> bool:
        changed = self._active_binding is not binding
        self._active_binding = binding
        return changed


def region_artifact_key(program, **options) -> str | None:
    """Return None when the program contains weights that cannot be rebound."""

    if program.state_dict or program.constants:
        return None

    nodes = list(program.graph_module.graph.nodes)
    indices = {node: index for index, node in enumerate(nodes)}
    symbols = {}

    def normalize_symbol(value):
        def replace(match):
            name = match.group(0)
            return symbols.setdefault(name, f"s{len(symbols)}")

        return _SYMBOL.sub(replace, str(value))

    def normalize(value):
        if isinstance(value, fx.Node):
            return ("node", indices[value])
        if isinstance(value, torch.Tensor):
            return (
                "tensor",
                tuple(normalize_symbol(dim) for dim in value.shape),
                tuple(normalize_symbol(dim) for dim in value.stride()),
                str(value.dtype),
                value.device.type,
                str(value.layout),
                normalize_symbol(value.storage_offset()),
            )
        if isinstance(value, (tuple, list)):
            return tuple(normalize(item) for item in value)
        if isinstance(value, dict):
            items = ((str(key), normalize(item)) for key, item in value.items())
            return tuple(sorted(items))
        if isinstance(value, (str, torch.SymInt, torch.SymFloat, torch.SymBool)):
            return normalize_symbol(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)

    graph = []
    for node in nodes:
        graph.append(
            (
                node.op,
                None if node.op == "placeholder" else str(node.target),
                normalize(node.args),
                normalize(node.kwargs),
                normalize(node.meta.get("val")),
            )
        )

    ranges = sorted(
        (normalize_symbol(symbol), normalize_symbol(bounds))
        for symbol, bounds in program.range_constraints.items()
    )
    payload = repr((graph, ranges, sorted(options.items()))).encode()
    return hashlib.sha256(payload).hexdigest()


def get_or_compile(key, compile_artifact):
    global _reuse_hits, _searches
    artifact = _artifacts.get(key)
    if artifact is not None:
        _reuse_hits += 1
        return artifact
    artifact = compile_artifact()
    _artifacts[key] = artifact
    _searches += 1
    return artifact


def artifact_cache_stats() -> ArtifactCacheStats:
    return ArtifactCacheStats(len(_artifacts), _reuse_hits, _searches)


def clear_artifact_cache() -> None:
    global _reuse_hits, _searches
    _artifacts.clear()
    _reuse_hits = 0
    _searches = 0
