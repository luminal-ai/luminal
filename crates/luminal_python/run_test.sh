#!/bin/bash
set -e

echo "=== Luminal Python Test Runner ==="
echo ""

# Force clean rebuild of Rust extension
echo "Step 1: Cleaning previous builds..."
rm -rf rust/target/wheels rust/target/debug rust/target/release

# Rebuild in development mode (faster compilation)
echo "Step 2: Building Rust extension..."
uv run maturin develop --manifest-path rust/Cargo.toml

# Run pytest
echo "Step 3: Running pytest..."
# it is best not to add the full model tests, they end up running billion parameter models
# on the CPU and it takes far to long
uv run pytest tests/test_hlir_ops.py tests/test_unary.py -v

echo ""
echo "=== Tests Complete ==="
