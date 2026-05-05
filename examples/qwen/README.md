# Qwen3 4B

Run Qwen3-4B through Luminal's CUDA backend:

```bash
cargo run --release -p qwen --features cuda
```

Run Qwen3-4B through Luminal's Metal backend on Apple targets:

```bash
cargo run --release -p qwen --features metal
```

The first run downloads `Qwen/Qwen3-4B`, converts the safetensors weights to a combined FP32 file, compiles the selected backend graph, and then generates text.
