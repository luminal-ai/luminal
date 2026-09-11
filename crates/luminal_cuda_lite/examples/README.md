# CUDA Lite model applications

The `model_zoo` crate contains the logical model definitions. This directory
owns the CUDA-specific search, binding, execution, and readback.

Run any application with a CUDA device and the `device` feature:

```sh
cargo run --release -p luminal_cuda_lite --example llama3 --features device
```

The full logical implementations have these runners: `llama3`, `qwen3`,
`gemma3`, `qwen3_moe`, `gemma4_moe`, `whisper`, `yolo_v11`, and `flux2`.
Each runner instantiates the released model dimensions and complete layer
count. These are attended, hardware-sized applications rather than smoke
tests. They bind deterministic synthetic parameters so the examples exercise
the full logical graph without owning checkpoint download or preprocessing.
The largest configurations require correspondingly large host and device
memory; the applications do not silently truncate layers to fit smaller GPUs.
The mini models remain runtime-neutral execution-smoke fixtures exercised by
the runtime test suites; they are not CUDA Lite applications.
