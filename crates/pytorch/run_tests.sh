#!/bin/bash
# Run the PyTorch-backend test suites against a freshly built extension.
#
# `maturin develop` rebuilds the Rust and installs it into the project's uv
# environment on every run, so a test never runs against a stale extension —
# and, unlike `maturin build`, it never rewrites the shared object's library
# names, which is what made a hand-built wheel fail to load.
#
#   ./run_tests.sh            # both suites
#   ./run_tests.sh reference  # CPU backend only
#   ./run_tests.sh cuda_lite  # CUDA backend only (needs a GPU)
#   ./run_tests.sh all -m "not slow"  # complete pre-integration gate
#
# Extra pytest arguments are passed through after the suite name.
set -e

cd "$(dirname "$0")"
suite="${1:-all}"
[ $# -gt 0 ] && shift

run_reference() {
    echo "=== reference backend: build ==="
    (cd reference && uv run --group dev maturin develop)
    echo "=== reference backend: pytest ==="
    (cd reference && uv run --no-sync --group dev pytest "$@")
}

run_cuda_lite() {
    # Both extensions go into the cuda_lite environment: its Python imports
    # the reference package's export helpers, which load the reference's own
    # compiled module. Synchronize the environment once, then keep uv from
    # replacing the freshly built editable reference package with its cached
    # path-dependency wheel before pytest starts.
    echo "=== cuda-lite backend: build (reference, then cuda_lite) ==="
    (cd cuda_lite && uv run --group dev maturin develop --manifest-path ../reference/Cargo.toml)
    if [[ -z "${LUMINAL_CUDA_HEADERS_DIR:-}" ]]; then
        cuda_headers="$({
            cd cuda_lite
            uv run --no-sync python -c 'import site; from pathlib import Path; print(next((str(p) for root in site.getsitepackages() for p in [Path(root) / "nvidia" / "cuda_runtime" / "include"] if (p / "cuda_fp16.h").is_file()), ""))'
        })"
        if [[ -n "$cuda_headers" ]]; then
            export LUMINAL_CUDA_HEADERS_DIR="$cuda_headers"
        fi
    fi
    (cd cuda_lite && uv run --no-sync --group dev maturin develop)
    echo "=== cuda-lite backend: pytest ==="
    (cd cuda_lite && uv run --no-sync --group dev pytest "$@")
}

case "$suite" in
    reference) run_reference "$@" ;;
    cuda_lite) run_cuda_lite "$@" ;;
    all)       run_reference "$@"; run_cuda_lite "$@" ;;
    *) echo "unknown suite '$suite' (expected: reference, cuda_lite, all)" >&2; exit 2 ;;
esac

echo "=== done ==="
