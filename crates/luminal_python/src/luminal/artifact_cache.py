"""Process-local reuse for structurally identical compiled regions."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

import torch
import torch.fx as fx

from .compiled_model import CompiledModel

_SYMBOL = re.compile(r"\b[su]\d+\b")
_artifacts: dict[str, CompiledArtifact] = {}
_reuse_hits = 0
_searches = 0
_loads = 0
_load_reuse_hits = 0
_search_seconds = 0.0
_load_seconds = 0.0
_CACHE_KEY_FIELD = "luminal_artifact_key"


@dataclass(frozen=True)
class ArtifactCacheStats:
    unique_artifacts: int
    reuse_hits: int
    searches: int
    loads: int = 0
    load_reuse_hits: int = 0
    search_seconds: float = 0.0
    load_seconds: float = 0.0


class CompiledArtifact:
    """Compiled runtime shared by separate positional tensor bindings."""

    def __init__(self, graph, weight_refs=(), cache_key=None):
        self.graph = graph
        self.weight_refs = tuple(weight_refs)
        self.cache_key = cache_key
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

    def serialize(self) -> bytes:
        if self.weight_refs:
            raise RuntimeError(
                "compiled artifacts with bound weights are not serializable"
            )
        return bytes(self.graph.serialize_artifact(self.cache_key))

    @classmethod
    def deserialize(
        cls,
        data,
        factory,
        *,
        device_index=None,
        external_cuda_graph=False,
    ):
        from .luminal import load_compiled_artifact

        graph = load_compiled_artifact(
            data,
            factory,
            device_index=device_index,
            external_cuda_graph=external_cuda_graph,
        )
        return cls(graph)


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
    global _reuse_hits, _searches, _search_seconds
    artifact = _artifacts.get(key)
    if artifact is not None:
        _reuse_hits += 1
        return artifact
    started = time.perf_counter()
    artifact = compile_artifact()
    _search_seconds += time.perf_counter() - started
    if isinstance(artifact, CompiledArtifact):
        artifact.cache_key = key
    _artifacts[key] = artifact
    _searches += 1
    return artifact


def get_or_load(data, load_artifact, **options):
    """Load identical serialized artifacts only once per process."""

    global _loads, _load_reuse_hits, _load_seconds

    data = bytes(data)
    payload = json.loads(data)
    identity = payload.get(_CACHE_KEY_FIELD)
    if identity is None:
        identity = hashlib.sha256(data).hexdigest()

    compatibility = (
        payload.get("schema_version"),
        payload.get("backend"),
        payload.get("device_index"),
        payload.get("external_cuda_graph"),
    )
    digest = hashlib.sha256(repr((identity, compatibility)).encode())
    digest.update(repr(sorted(options.items())).encode())
    key = f"loaded:{digest.hexdigest()}"

    artifact = _artifacts.get(key)
    if artifact is not None:
        _load_reuse_hits += 1
        return artifact

    started = time.perf_counter()
    artifact = load_artifact()
    _load_seconds += time.perf_counter() - started
    _loads += 1
    _artifacts[key] = artifact
    return artifact


def artifact_cache_stats() -> ArtifactCacheStats:
    return ArtifactCacheStats(
        len(_artifacts),
        _reuse_hits,
        _searches,
        _loads,
        _load_reuse_hits,
        _search_seconds,
        _load_seconds,
    )


def clear_artifact_cache() -> None:
    global _reuse_hits, _searches, _loads, _load_reuse_hits
    global _search_seconds, _load_seconds
    _artifacts.clear()
    _reuse_hits = 0
    _searches = 0
    _loads = 0
    _load_reuse_hits = 0
    _search_seconds = 0.0
    _load_seconds = 0.0
