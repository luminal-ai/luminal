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
    (cd reference && uv run --group dev pytest "$@")
}

run_cuda_lite() {
    # Both extensions go into the cuda_lite environment: its Python imports
    # the reference package's export helpers, which load the reference's own
    # compiled module.
    echo "=== cuda-lite backend: build (reference, then cuda_lite) ==="
    (cd cuda_lite && uv run --group dev maturin develop --manifest-path ../reference/Cargo.toml)
    (cd cuda_lite && uv run --group dev maturin develop)
    echo "=== cuda-lite backend: pytest ==="
    (cd cuda_lite && uv run --group dev pytest "$@")
}

case "$suite" in
    reference) run_reference "$@" ;;
    cuda_lite) run_cuda_lite "$@" ;;
    all)       run_reference "$@"; run_cuda_lite "$@" ;;
    *) echo "unknown suite '$suite' (expected: reference, cuda_lite, all)" >&2; exit 2 ;;
esac

echo "=== done ==="
