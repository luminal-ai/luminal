#!/bin/bash
set -e

echo "=========================================="
echo "  Luminal Python: Full Test Suite"
echo "=========================================="

NATIVE_TESTS="tests/test_hlir_ops.py tests/test_unary.py"
CUDA_TESTS="tests/test_hlir_ops.py tests/test_unary.py tests/test_llama3.py"

# ── Phase 1: Native Backend ─────────────────────────────────

echo ""
echo "=== Phase 1: Building native backend ==="
rm -rf rust/target/wheels rust/target/debug rust/target/release
uv run maturin develop --manifest-path rust/Cargo.toml

echo ""
echo "--- 1a: Native + ONNX ---"
uv run pytest $NATIVE_TESTS -v

echo ""
echo "--- 1b: Native + PT2 ---"
LUMINAL_EXPORT_MODE=pt2 uv run pytest $NATIVE_TESTS -v

# ── Phase 2: CUDA Backend ───────────────────────────────────

echo ""
echo "=== Phase 2: Building CUDA backend ==="
rm -rf rust/target/wheels rust/target/debug rust/target/release
uv run maturin develop --manifest-path rust/Cargo.toml --features cuda -r

echo ""
echo "--- 2a: CUDA + ONNX ---"
RUST_BACKTRACE=1 LUMINAL_BACKEND=cuda uv run pytest $CUDA_TESTS -m "not slow" -v

echo ""
echo "--- 2b: CUDA + PT2 ---"
RUST_BACKTRACE=1 LUMINAL_BACKEND=cuda LUMINAL_EXPORT_MODE=pt2 uv run pytest $CUDA_TESTS -m "not slow" -v

echo ""
echo "=========================================="
echo "  All tests passed!"
echo "=========================================="
