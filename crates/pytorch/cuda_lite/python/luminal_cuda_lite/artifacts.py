"""Versioned persistent storage for CUDA-lite searched plan templates."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ._luminal import PlanTemplate


_FILE_SCHEMA = 1


def artifact_path(cache_dir: str, prefix: str, cache_key: str) -> Path:
    safe_prefix = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in prefix
    )
    return Path(cache_dir) / "luminal_cuda_lite" / f"{safe_prefix}{cache_key}.json"


def load_artifact(path: Path, fingerprint: str) -> Any:
    document = json.loads(path.read_text())
    if document.get("schema") != _FILE_SCHEMA:
        raise RuntimeError(
            f"Luminal artifact {path} has schema {document.get('schema')!r}, "
            f"expected {_FILE_SCHEMA}"
        )
    if document.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Luminal artifact {path} has the wrong structural fingerprint")
    payload = base64.b64decode(document["plan"], validate=True)
    return PlanTemplate.deserialize_artifact(
        payload,
        fingerprint,
        document["input_buffers"],
        document["output_buffers"],
        document["dim_symbols"],
    )


def save_artifact(path: Path, fingerprint: str, template: Any) -> None:
    payload, input_buffers, output_buffers, dim_symbols = template.serialize_artifact(
        fingerprint
    )
    document = {
        "schema": _FILE_SCHEMA,
        "fingerprint": fingerprint,
        "input_buffers": input_buffers,
        "output_buffers": output_buffers,
        "dim_symbols": dim_symbols,
        "plan": base64.b64encode(bytes(payload)).decode("ascii"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
