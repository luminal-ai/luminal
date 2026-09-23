"""Process-local reuse for structurally identical compiled regions."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass

import torch
from torch import fx

from .region_model import RegionModel

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
    if key is not None:
        _artifacts[key] = artifact
    _searches += 1
    return artifact


def get_or_load(data, load_artifact, **options):
    """Load identical serialized artifacts only once per process."""

    global _loads, _load_reuse_hits, _load_seconds

    data = bytes(data)
    payload = json.loads(data)
    identity = hashlib.sha256(data).hexdigest()

    compatibility = (
        payload.get("schema_version"),
        payload.get("backend"),
        payload.get("torch_version"),
        payload.get("device_type"),
        payload.get("search_iterations"),
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


class CompiledArtifact:
    """Native plan plus a portable, versioned PT2 representation.

    Serialized artifacts contain the exported program, not device pointers or
    a legacy HLIR plan. Loading compiles once for the receiving runtime; the
    process cache then reuses that plan for subsequent bindings.
    """

    SCHEMA_VERSION = 3

    def __init__(
        self, program, native, device_type, search_iterations=1, cache_key=None
    ):
        self.program = program
        self.native = native
        self.device_type = device_type
        self.search_iterations = search_iterations
        self.cache_key = cache_key

    def bind(self, **kwargs):
        return RegionModel(self, **kwargs)

    def serialize(self):
        if self.program.state_dict or self.program.constants:
            raise RuntimeError(
                "compiled artifacts with bound weights are not serializable"
            )
        buffer = io.BytesIO()
        torch.export.save(self.program, buffer)
        return json.dumps(
            {
                "schema_version": self.SCHEMA_VERSION,
                "torch_version": str(torch.__version__),
                "device_type": self.device_type,
                "search_iterations": self.search_iterations,
                "luminal_artifact_key": self.cache_key,
                "program": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        ).encode()

    @classmethod
    def deserialize(cls, data, *, device_type=None, search_log=False, **options):
        from .frontend import compile_exported

        payload = json.loads(data)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported artifact schema {payload.get('schema_version')}"
            )
        if payload["torch_version"] != str(torch.__version__):
            raise RuntimeError(
                "artifact PyTorch version differs from the current environment"
            )
        target = device_type or payload["device_type"]
        if target != payload["device_type"]:
            raise RuntimeError(
                "artifact device type differs from the requested runtime"
            )
        program = torch.export.load(io.BytesIO(base64.b64decode(payload["program"])))
        return compile_exported(
            program,
            device_type=target,
            search_iterations=payload["search_iterations"],
            search_log=search_log,
        ).artifact
