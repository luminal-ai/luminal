#!/bin/bash
# Build CUDA and its reference dependency; run GPU and shared parity tests.
set -euo pipefail
cd "$(dirname "$0")"
export LUMINAL_TEST_BACKEND=cuda_lite
export LUMINAL_TEST_DEVICE=cuda
uv sync --group dev
if [[ -z "${LUMINAL_CUDA_HEADERS_DIR:-}" ]]; then
    cuda_headers="$(uv run --no-sync python -c 'import site; from pathlib import Path; print(next((str(p) for root in site.getsitepackages() for p in [Path(root) / "nvidia" / "cuda_runtime" / "include"] if (p / "cuda_fp16.h").is_file()), ""))')"
    if [[ -n "$cuda_headers" ]]; then
        export LUMINAL_CUDA_HEADERS_DIR="$cuda_headers"
    fi
fi
# Install the sibling project itself. Using maturin with only its manifest
# path can pair a fresh native module with stale reference Python sources.
uv pip install --python .venv/bin/python --reinstall --no-deps -e ../reference
uv run --no-sync --group dev maturin develop
uv run --no-sync --group dev pytest -c pyproject.toml tests ../utils/tests ../reference/tests/models "$@"
