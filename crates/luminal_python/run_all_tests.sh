#!/bin/bash
set -e

export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12080}"
export MATURIN_PEP517_ARGS="${MATURIN_PEP517_ARGS:---features cuda --profile release}"

echo "=========================================="
echo "  Luminal Python: Full Test Suite"
echo "=========================================="

NATIVE_TESTS="tests/test_hlir_ops.py tests/test_unary.py"
CUDA_TESTS="tests/"

# ── Phase 1: Native Backend ─────────────────────────────────

echo ""
echo "=== Phase 1: Building native backend ==="
rm -rf rust/target/wheels rust/target/debug rust/target/release
uv run --group dev maturin develop --manifest-path rust/Cargo.toml

echo ""
echo "--- 1a: Native backend tests ---"
uv run --group dev pytest $NATIVE_TESTS -v

# ── Phase 2: CUDA Backend ───────────────────────────────────

echo ""
echo "=== Phase 2: Building CUDA backend ==="
rm -rf rust/target/wheels rust/target/debug rust/target/release
uv run --group dev maturin develop --manifest-path rust/Cargo.toml --features cuda -r

echo ""
echo "--- 2a: CUDA ---"
RUST_BACKTRACE=1 LUMINAL_TEST_DEVICE=cuda uv run --group dev pytest $CUDA_TESTS -m "not slow" -v

echo ""
echo "Slow CUDA tests are opt-in. To include them, run:"
echo "  RUST_BACKTRACE=1 LUMINAL_TEST_DEVICE=cuda uv run pytest tests/ -v -s"
echo "Or, for only slow tests:"
echo "  RUST_BACKTRACE=1 LUMINAL_TEST_DEVICE=cuda uv run pytest tests/ -m slow -v -s"

echo ""
echo "=========================================="
echo "  All tests passed!"
echo "=========================================="
