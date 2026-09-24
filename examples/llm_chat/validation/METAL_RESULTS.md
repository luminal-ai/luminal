# Metal chat validation — 2026-09-10

Tested locally on an Apple M1 Pro (16 GPU cores, 32 GiB unified memory),
using repository revision `305cbb7c69b93032e22f01f13f799bcab4b46547` plus
`tests/metal_model_validation.rs`. No model or runtime implementation changed.
The device reports a maximum Metal buffer length of 20,100,448,256 bytes.

## Family-level numerical validation

All four small instances of the actual `LlmGraph` model adapters pass against
`ReferenceRuntime`: Llama3, Qwen3, Gemma3, and Qwen3-MoE. Deterministic weights,
two layers, capacity six, and chunk size two exercise prefill, single-token
decode, a subsequent two-token chunk, another decode, and reset/replay.
Gemma includes both local and global attention and crosses its window of three;
MoE routes to two of four experts. All 20 last-token logit rows meet absolute
error < 0.0001. Search uses two generations, four candidates, seed zero.
These checks establish small-graph runtime parity, not full-checkpoint parity.

```sh
cargo test -p llm_chat --features metal
cargo test -p llm_chat --features metal --test metal_model_validation -- --test-threads=1
cargo test -p luminal_metal --test resident_feedback --test dynamic_execution
cargo clippy -p llm_chat --features metal --all-targets -- -D warnings
```

The original chat suite passed 10 tests; the added family suite passed four;
the resident-feedback and dynamic-execution suites passed five. Strict clippy
passed, as did `cargo fmt --all -- --check`. The family suite took 160.99 seconds in the debug profile.

## Qwen3-4B real chat

Checkpoint: `Qwen/Qwen3-4B`, revision
`9e1b55c76f4b5bf0d14d37da8010110060f512e0`. Native Metal, release build,
F32 weight loading, greedy decoding, seed zero, default search and chunk size.
A single-turn arithmetic invocation (capacity 128, limit 16 new tokens)
completed with `4.` and exit status zero.

A second, interactive invocation used capacity 256 and 24 new tokens per turn.
All seven turns completed; the process exited zero. Explicit resets separate
the arithmetic, memory, translation, and final replay conversations.

| Prompt | Observed answer |
| --- | --- |
| What is 2 + 2? Answer with the number only. | 4 |
| Now multiply that result by 3. Answer with the number only. | 12 |
| My dog's name is Pixel. Remember it. Reply briefly. | Pixel is a great name! 🐾 |
| What is my dog's name? Reply with the name only. | Pixel |
| Translate 'good morning' into French. Reply briefly. | Bonjour. |
| Now translate 'thank you' into French. Reply briefly. | Merci. |
| Reset, then repeat the first arithmetic prompt | 4 |

This is end-to-end chat execution coverage; full-size Qwen logits were not
compared against Transformers.

## Gemma3-4B real checkpoint

All seven cases passed against the existing Transformers fixtures for revision
`bf46152c47f5dd20b896357cb51abc4c03b8ee8c` of `unsloth/gemma-3-4b-it`.
All 48 full-vocabulary logit rows passed `atol=0.005, rtol=0.0005`.
The maximum absolute error was 0.0005931854, and the maximum fraction of the
allowed tolerance used was 0.1013654. Generated token IDs (including EOS),
chat-template token IDs, and decoded text matched the reference exactly.
The validator exited zero. See [machine-readable results](metal-results.json).

This run uses chunk size eight, context capacity 62, greedy decoding, search
seed zero, two generations, and four candidates. The reference was generated
in F32 with eager attention and disabled TF32 using PyTorch 2.13.0+cu129,
Transformers 5.16.1, and Safetensors 0.8.0. Its cached generation was checked
against full causal replay by the existing reference generator.

| Case | Answer | Checked steps |
| --- | --- | ---: |
| Arithmetic | `4\n` | 6 |
| Arithmetic follow-up | `12\n` | 10 |
| Remember Pixel | Okay, Pixel! 😊 | 9 |
| Name recall | Pixel | 5 |
| Translate good morning | Bonjour ! | 6 |
| Translate thank you | Merci ! | 6 |
| Reset and repeat arithmetic | `4\n` | 6 |

The memory and translation follow-ups reused cached history. The arithmetic
follow-up refilled history because the template removes the previous answer's
newline, as in the CUDA validation. Inspection finds all 444 required text
parameters; F32 text weights total 15,521,052,672 bytes.

Reproduce using the reference-generation instructions in [README.md](README.md):

```sh
cargo run --release -p llm_chat --features metal --example validate -- \
  --model gemma3 --checkpoint /path/to/gemma3 \
  --suite /path/to/gemma-reference/suite.json --prefill-chunk 8 \
  --report /path/to/metal-results.json
```

## Full-size hardware blockers

Checkpoint inspection succeeds with no missing parameter names for both:

| Checkpoint | Parameters mapped | F32 weight bytes |
| --- | ---: | ---: |
| NousResearch/Meta-Llama-3-8B-Instruct | 291 | 32,121,044,992 |
| Qwen/Qwen3-30B-A3B | 18,867 | 122,128,490,496 |

Each exceeds this Mac's single-buffer limit before KV cache or intermediate
storage. The current backend uses one arena. These full checkpoints were not
loaded or executed; a larger-memory Metal device is required. Cached Llama3.2
models are not substitutes for this adapter's unscaled-RoPE Llama3 contract.

Coverage is bounded to the stated models, shapes, prompts, and compiler seed.
