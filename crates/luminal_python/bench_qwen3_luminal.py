"""Benchmark Qwen3-4B (non-MoE) through the luminal torch.compile backend.

Mirrors the Rust `qwen` example: one prefill (TTFT) + a greedy autoregressive
decode loop with a KV cache (TPOT over 100 tokens). Uses the dynamic-decode-graph
reuse path validated by tests/test_kv_cache_growing.py — automatic_dynamic_shapes
+ DynamicCache settle to a single reused decode graph after 3 backend compiles.
"""
import time
import torch
import torch._dynamo
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM, DynamicCache
from luminal import luminal_backend

MODEL = "Qwen/Qwen3-4B"
PROMPT = "Explain what a neural network is in a paragraph."
DECODE_TOKENS = 100
WARMUP_DECODES = 8


def chat_prompt(p: str) -> str:
    # Matches the Rust example's template exactly (qwen3_chat_prompt in lib.rs).
    return (f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.recompile_limit = 32
    # NB: deliberately NOT setting torch.set_float32_matmul_precision("highest").
    # That's a test-correctness knob (conftest uses it to keep compiled output
    # bit-close to eager); it disables TF32 on fp32 matmuls. The model runs in
    # bf16, so it's irrelevant to most of the math and would only bias the
    # benchmark by slowing residual fp32 ops. Use PyTorch defaults instead.

    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.use_cache = True
    cfg._attn_implementation = "eager"
    model = (Qwen3ForCausalLM
             .from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16)
             .eval().cuda())

    compiles = []

    def backend(gm, ex, options=None):
        compiles.append(1)
        return luminal_backend(gm, ex, options)

    compiled = torch.compile(model, backend=backend, fullgraph=True)

    input_ids = tok(chat_prompt(PROMPT), return_tensors="pt").input_ids.cuda()
    prompt_len = input_ids.shape[1]

    def run(steps):
        cache = DynamicCache(config=model.config)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = compiled(input_ids=input_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        nxt = out.logits[0, -1].argmax().view(1, 1)
        per = []
        for _ in range(steps):
            torch.cuda.synchronize()
            s = time.perf_counter()
            with torch.no_grad():
                out = compiled(input_ids=nxt,
                               past_key_values=out.past_key_values, use_cache=True)
            torch.cuda.synchronize()
            per.append(time.perf_counter() - s)
            nxt = out.logits[0, -1].argmax().view(1, 1)
        return ttft, per

    run(WARMUP_DECODES)                       # warmup: trigger all compiles
    n_after_warmup = sum(compiles)
    ttft, per = run(DECODE_TOKENS)            # measured
    assert sum(compiles) == n_after_warmup, \
        f"recompiled during measured run ({sum(compiles)} vs {n_after_warmup})"

    tpot = sum(per) / len(per)
    print(f"[luminal_python] Qwen3-4B bf16  prompt_len={prompt_len}  "
          f"backend_compiles={sum(compiles)}")
    print(f"  TTFT: {ttft * 1e3:.2f} ms")
    print(f"  TPOT: {tpot * 1e3:.2f} ms  ({1.0 / tpot:.1f} tok/s over {DECODE_TOKENS})")


if __name__ == "__main__":
    main()
