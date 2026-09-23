#!/bin/bash
# Build CUDA and its reference dependency; run GPU and shared parity tests.
set -euo pipefail
cd "$(dirname "$0")"
export LUMINAL_TEST_BACKEND=cuda_lite
export LUMINAL_TEST_DEVICE=cuda
uv run --group dev maturin develop --manifest-path ../reference/Cargo.toml
uv run --group dev maturin develop
uv run --group dev pytest -c pyproject.toml tests ../utils/tests ../reference/tests/models "$@"
