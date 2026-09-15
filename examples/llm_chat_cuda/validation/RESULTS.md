# CUDA chat checkpoint validation — 2026-09-10

All 63 turn/chunk combinations passed for the three checkpoints below: seven
turns per checkpoint at prefill chunk sizes 1, 4, and 8. Chat-template token IDs,
generated token IDs (including EOS), and decoded text matched Transformers
exactly. All 1,116 checked full-vocabulary logit rows passed the numerical
comparison; the largest absolute difference was 0.00077057.

The runs use `llm_chat`'s actual checkpoint loader, tokenizer, graph adapter,
GPU backend, sampler, and session. Compiler search uses the app defaults:
two generations, four candidates, seed zero, without device profiling.
This validates the selected graphs at that seed, not every possible extraction.

## Checkpoints and reference

- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca): `c1899de289a04d12100db370d81485cdf75e47ca`.
- [NousResearch/Meta-Llama-3-8B-Instruct](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/tree/53346005fb0ef11d3b6a83b12c895cca40156b6c): `53346005fb0ef11d3b6a83b12c895cca40156b6c`.
- [unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it/tree/bf46152c47f5dd20b896357cb51abc4c03b8ee8c): `bf46152c47f5dd20b896357cb51abc4c03b8ee8c`.

Both implementations use F32 weights on an NVIDIA H200, Linux, driver
580.126.09. The reference uses PyTorch 2.13.0+cu129, Transformers 5.16.1,
Safetensors 0.8.0, and Accelerate 1.10.1, with eager attention, TF32 disabled,
greedy generation, thinking disabled, and at most 24 new tokens per turn.
Every response reached EOS. Transformers' cached generation also matched
its independent full causal replay before the fixtures were accepted.

Rust used rustc 1.98.1 with `profile.dev.opt-level=2` and
`profile.dev.package.llm_chat.opt-level=3`. The tested code is based on
`311e4bb3`, including the Gemma config correction and validation tools in
this commit. Earlier exploratory runs with one search candidate also passed;
the matrix here contains only the default-search runs.

## Numerical results

Each step compares the complete vocabulary row returned by the application,
covering chunked prefill boundaries, decode, and consumed EOS tokens. Every
element must satisfy `abs(actual-reference) <= 0.005 + 0.0005*abs(reference)`.
The tolerance ratio is the largest error divided by that element's allowed
tolerance; values at or below one pass. Exact token equality is also required.

| Checkpoint | Prefill chunk | Turns passed | Logit rows | Max absolute error | Max tolerance ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 1 | 7/7 | 310 | 0.00014305 | 0.0206 |
| Qwen3-0.6B | 4 | 7/7 | 115 | 0.00014687 | 0.0267 |
| Qwen3-0.6B | 8 | 7/7 | 83 | 0.00012398 | 0.0206 |
| Llama3-8B-Instruct | 1 | 7/7 | 179 | 0.00060761 | 0.1071 |
| Llama3-8B-Instruct | 4 | 7/7 | 63 | 0.00031090 | 0.0354 |
| Llama3-8B-Instruct | 8 | 7/7 | 42 | 0.00015640 | 0.0177 |
| Gemma3-4B-IT (text) | 1 | 7/7 | 205 | 0.00047684 | 0.0835 |
| Gemma3-4B-IT (text) | 4 | 7/7 | 71 | 0.00077057 | 0.0806 |
| Gemma3-4B-IT (text) | 8 | 7/7 | 48 | 0.00042152 | 0.0819 |

The [machine-readable results](results.json) retain every case's output,
step/reset counts, error metrics, checkpoint revision, and reference settings.
See [reproduction instructions](README.md) and the fixed prompts in
[the fixture generator](generate_reference.py).

## Prompts, follow-ups, and answer quality

Each checkpoint answered an arithmetic prompt and follow-up, remembered a
dog's name across turns, translated two phrases, and repeated the initial
arithmetic prompt after an explicit reset. These are the decoded answers
(Gemma's arithmetic outputs additionally end in a newline):

| Prompt / follow-up | Qwen3-0.6B | Llama3-8B-Instruct | Gemma3-4B-IT |
| --- | --- | --- | --- |
| 2 + 2, number only | 2 + 2 = 4 | 4 | 4 |
| Multiply that result by 3 | 4 × 3 = 12 | 12 | 12 |
| Remember my dog's name, Pixel | Pixel is a great name for your dog! 🐾 | Pixel! | Okay, Pixel! 😊 |
| What is my dog's name? | Pixel | Pixel | Pixel |
| Translate “good morning” | Bonjour! | "Bonne journée" | Bonjour ! |
| Now translate “thank you” | Bonjour! | "Merci" | Merci ! |
| Reset, then 2 + 2 again | 2 + 2 = 4 | 4 | 4 |

Qwen's translation follow-up is incorrect, and it does not obey the initial
number-only instruction. Llama's “Bonne journée” means “have a good day,”
so it is an imperfect translation here. These are also the exact Transformers
answers; neither prompts nor model definitions were changed to hide them.
Runtime parity does not imply that every model answer is correct.

Llama reuses its KV cache on all three follow-ups. Gemma reuses it on the
name and translation follow-ups; its arithmetic newline is removed by the
checkpoint's history template, so the session correctly refills that history.
Qwen's template changes the rendered token prefix on each follow-up, which
also correctly triggers a refill. All models reuse state during decode and
return the same initial answer after an explicit reset.

## Fix and limits

The Gemma checkpoint omits `tie_word_embeddings`, whose model-config default
is true. The app previously rejected it. The adapter now applies that default
and still rejects explicitly untied embeddings; a unit regression covers
omission and both root/nested overrides. No compiler/runtime numerical
mismatches were found in these runs.

The app's nine unit tests and CUDA prefill/decode/reset reference test pass.
Strict chat clippy passes with CUDA Lite, Metal, and no backend features;
Rust formatting and Python syntax checks pass.

Metal execution and Qwen3-30B-A3B are deferred at the user's request on this
device. Coverage here is greedy, text-only, short-context chat (capacities
74 for Qwen, 60 for Llama, and 62 for Gemma). It does not cover long contexts,
Gemma's sliding-window boundary, image inputs, stochastic sampling parity,
other checkpoint sizes, or different compiler-search seeds.
