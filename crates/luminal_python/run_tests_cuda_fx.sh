#!/bin/bash
set -e

echo "=== Luminal Python Test Runner (CUDA + PT2 Export Mode) ==="
echo ""

# Force clean rebuild of Rust extension
echo "Step 1: Cleaning previous builds..."
rm -rf rust/target/wheels rust/target/debug rust/target/release

# Rebuild in development mode (faster compilation)
echo "Step 2: Building Rust extension..."
uv run maturin develop --manifest-path rust/Cargo.toml --features cuda -r

# Run pytest with CUDA backend and PT2 export mode
echo "Step 3: Running pytest with CUDA backend + PT2 export mode..."
RUST_BACKTRACE=1 LUMINAL_BACKEND=cuda LUMINAL_EXPORT_MODE=pt2 uv run pytest tests/test_hlir_ops.py tests/test_unary.py tests/test_llama3.py -m "not slow" -v
echo ""
echo "=== Tests Complete ==="
