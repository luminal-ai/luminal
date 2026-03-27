#!/bin/bash
set -e

echo "=== Luminal Python Test Runner (PT2 Export Mode) ==="
echo ""

# Force clean rebuild of Rust extension
echo "Step 1: Cleaning previous builds..."
rm -rf rust/target/wheels rust/target/debug rust/target/release

# Rebuild in development mode (faster compilation)
echo "Step 2: Building Rust extension..."
uv run maturin develop --manifest-path rust/Cargo.toml

# Run pytest with PT2 export mode
echo "Step 3: Running pytest with PT2 export mode..."
LUMINAL_EXPORT_MODE=pt2 uv run pytest tests/test_hlir_ops.py tests/test_unary.py -v

echo ""
echo "=== Tests Complete ==="
