#!/bin/bash
set -e

echo "=== Luminal Python Test Runner (CUDA Backend) ==="
echo ""

export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"
export MATURIN_PEP517_ARGS="${MATURIN_PEP517_ARGS:---features cuda --profile release}"

PYTEST_MARK='not slow'
if [[ "${1:-}" == "--include-slow" ]]; then
    PYTEST_MARK=''
elif [[ "${1:-}" == "--slow-only" ]]; then
    PYTEST_MARK='slow'
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: ./run_tests_cuda.sh [--include-slow|--slow-only]"
    exit 2
fi

# Force clean rebuild of Rust extension
echo "Step 1: Cleaning previous builds..."
rm -rf rust/target/wheels rust/target/debug rust/target/release

# Rebuild in development mode (faster compilation)
echo "Step 2: Building Rust extension..."
uv run --group dev maturin develop --manifest-path rust/Cargo.toml --features cuda -r

# Run pytest with CUDA backend
echo "Step 3: Running pytest with CUDA backend..."
if [[ -n "$PYTEST_MARK" ]]; then
    RUST_BACKTRACE=1 LUMINAL_TEST_DEVICE=cuda uv run --group dev pytest tests/ -m "$PYTEST_MARK" -v -s
else
    RUST_BACKTRACE=1 LUMINAL_TEST_DEVICE=cuda uv run --group dev pytest tests/ -v -s
fi

echo ""
echo "=== Tests Complete ==="
