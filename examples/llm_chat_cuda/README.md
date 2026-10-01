# CUDA chat

This application runs on Linux or Windows with an NVIDIA GPU and CUDA toolkit. It owns its runtime bindings,
device allocations, and explicit uploads/readbacks in `src/backend.rs` and
`src/memory.rs`. Executables borrow that memory per call.

```sh
cargo run --release -p llm_chat_cuda -- \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B
cargo test -p llm_chat_cuda
```

See the [shared application guide](../llm_chat/README.md) for models,
checkpoint formats, options, and session behavior.
