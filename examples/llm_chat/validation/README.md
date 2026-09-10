# Real-checkpoint validation

These tools exercise the actual chat tokenizer, model adapter, checkpoint
loader, GPU backend, sampler, and `Session`. They require local, immutable
Hugging Face checkpoints and a GPU that fits the selected model in F32.

First generate reference data with PyTorch, Transformers, Accelerate, and
Safetensors installed:

```sh
python examples/llm_chat/validation/generate_reference.py \
  --model qwen3 --checkpoint /path/to/Qwen3-0.6B --output /tmp/qwen-reference
```

The generator uses Transformers' model implementation, F32 weights, eager
attention, disabled TF32, and greedy generation. It tests arithmetic, remembering
a name, translation, follow-ups, and an explicit replay after resetting history.
It records exact chat-template token IDs, generated tokens, decoded responses,
and the full vocabulary logits at every position. A full causal replay must also
agree with Transformers' cached generation before a fixture is accepted.
Dependency versions and an optional checkpoint `REVISION` file are recorded in
`suite.json`.

Then compare the shared chat runner at multiple prefill chunk sizes:

```sh
for chunk in 1 4 8; do
  cargo run --release -p llm_chat --features cuda_lite --example validate -- \
    --model qwen3 --checkpoint /path/to/Qwen3-0.6B \
    --suite /tmp/qwen-reference/suite.json --prefill-chunk "$chunk" \
    --report "/tmp/qwen-reference/cuda-chunk${chunk}.json"
done
```

Use the same commands with `--model llama3`, `gemma3`, or `qwen3-moe` and the
corresponding checkpoint. On a Mac, replace the Cargo feature with `metal`.
The reference process exits before the runner starts, so they need not hold
their models in GPU memory simultaneously.

The validator compares every execution's last-token logits, including prefill
boundaries, decode, and consumed EOS tokens. It also checks the complete
generated token sequence and decoded text, builds follow-up history from the
actual responses, and tests explicit cache resets. The report includes reset
counts, checked steps, maximum absolute error, and the largest fraction of the
allowed tolerance used. Compiler search uses the chat app's defaults of two
generations and four candidates, with seed zero; `--search-generations` and
`--search-population` allow explicit overrides recorded in each report.
Defaults are `atol=0.005`, `rtol=0.0005`; all entries must
satisfy `abs(actual-reference) <= atol + rtol*abs(reference)`. Greedy token IDs
must match exactly, regardless of the logit tolerance.

For a sharded checkpoint, `--inspect` checks its configuration and tensor names
without loading its weights:

```sh
cargo run -p llm_chat --features cuda_lite --example validate -- \
  --model gemma3 --checkpoint /path/to/gemma-3-4b-it --inspect
```

Matching the reference establishes implementation parity for these cases. It
does not establish that every generated answer is factually correct or follows
instructions: the reports retain model mistakes as well as correct answers.
Large logit fixtures and checkpoint weights are kept outside the repository;
small result reports can be retained alongside a validation summary.

See [recorded results](RESULTS.md) for the tested checkpoints and coverage.
