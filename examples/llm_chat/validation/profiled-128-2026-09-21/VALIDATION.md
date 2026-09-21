# Validation of profiling defaults and benchmark reruns

- Release build of the chat binary and benchmark passed.
- CLI parser regression test passes for omitted `--profile`, bare `--profile`, `--profile=true`, and `--profile=false`.
- Full CUDA + chat device suite: **188 passed, 0 failed, 3 ignored**. The mini-Llama profiling test inherits production defaults, verifies that CUDA graph launches occurred during search, and compares the selected plan’s outputs with ReferenceRuntime.
- Host suites for `luminal_reference`, `luminal_cuda_lite`, `test_runtime`, `luminal_nn`, and `llm_chat_cuda`: **443 passed, 0 failed, 6 ignored**.
- Core initially reported **272 passed, 1 failed, 6 ignored**: `frontend::movement::tests::test_pad_2d`. Its minimized case has shape 1×1 and zero padding on all sides; the reference recorder rejects that identity graph with `bindings name no output`. Core and reference implementation files are unchanged in this task. The adjacent 1D padding test already excludes zero padding with a documented identity-graph limitation.
- Rerunning the core package with that recorded failure excluded completed the remaining checks: **278 passed, 0 failed, 6 ignored** (including integration tests and doctests). This does not turn the initial full-sweep result green.
- Strict Clippy passes for all CUDA and chat targets, both with and without the device feature. Three pre-existing test-only lint findings were corrected before PR submission: unnecessary parentheses, fixed-size byte chunk iteration, and atomic array initialization. The two affected GPU test suites were rerun: **10 passed, 0 failed**. Initial lint failure logs are retained alongside the passing PR checks.
- `cargo fmt --all -- --check` and `git diff --check` passed.
- All four full-model benchmarks completed successfully. Each report verifies exactly 128 input and output tokens, matching prompt IDs against the earlier run, and identical generated IDs across all four repetitions and against the earlier heuristic-selected output.

GPU tests ran before or after the full-model benchmark sequence, never concurrently with model measurements. Host linting finished during early Qwen3-0.6B compilation. The broad test suites began only after Gemma finished.

## Commands

```sh
cargo test --release -p luminal -p luminal_reference -p luminal_cuda_lite -p test_runtime -p luminal_nn -p llm_chat_cuda
cargo test --release -p luminal_reference -p luminal_cuda_lite -p test_runtime -p luminal_nn -p llm_chat_cuda --no-fail-fast
cargo test --release -p luminal --no-fail-fast -- --skip frontend::movement::tests::test_pad_2d
cargo test --release -p luminal_cuda_lite -p llm_chat_cuda --features llm_chat_cuda/device -- --test-threads=1
cargo clippy --release -p luminal_cuda_lite -p llm_chat_cuda --features llm_chat_cuda/device --all-targets -- -D warnings
cargo clippy --release -p luminal_cuda_lite -p llm_chat_cuda --all-targets -- -D warnings
cargo test --release -p luminal_cuda_lite --features device --test dtype_support --test execution_traits -- --test-threads=1
cargo fmt --all -- --check
```

Compressed command output is saved under `checks/`.
