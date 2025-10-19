#!/usr/bin/env bash
set -euo pipefail

echo "== Luminal: Colab validation helper =="
echo "Ensure you run this in a Colab bash cell. It installs Rust, checks GPU, and builds the CUDA crate." 

if ! command -v rustc >/dev/null 2>&1; then
  echo "Installing rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source $HOME/.cargo/env
fi

echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "GPU status:"; nvidia-smi || true
echo "nvcc:"; nvcc --version || true

echo "Cloning repo and checking out branch..."
git clone https://github.com/salavathhari/luminal.git /content/luminal || true
cd /content/luminal
git fetch origin
git checkout fix/cudarc-cuda122 || git checkout main

echo "Updating cudarc and building CUDA crate..."
cargo update -p cudarc || true
cargo build -p luminal_cuda --release --features cuda

echo "Build finished. Run demos using: cargo run -p matmul --release --features cuda"
