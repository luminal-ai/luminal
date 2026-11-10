#!/bin/bash
# Build reference, then run CPU runtime and shared translator parity tests.
set -euo pipefail
cd "$(dirname "$0")"
export LUMINAL_TEST_BACKEND=reference
export LUMINAL_TEST_DEVICE=cpu
uv run --group dev maturin develop
for suite in tests ../utils/tests; do
    uv run --group dev pytest -c pyproject.toml "$suite" "$@"
done
