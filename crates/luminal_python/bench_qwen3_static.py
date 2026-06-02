"""TTFT/TPOT for Qwen3-4B via StaticCache (preallocated, scatter cache) through luminal.

Uses TorchExportableModuleWithStaticCache (cache as in-place buffers, returns
logits). Now correct after the search-corruption fix. Mirrors bench_qwen3_luminal.py
(DynamicCache) for an apples-to-apples comparison: same prompt, bf16, 100 decode
tokens. Note: StaticCache attention covers the full max_cache_len each step
(masked), so max_cache is kept tight (prompt + 100) to bound the extra work.
"""
import os
import time
import statistics
import torch
import torch._dynamo

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
DTYPE_NAME = os.environ.get("LUMINAL_BENCH_DTYPE", "bf16")
DTYPE = _DTYPES[DTYPE_NAME]
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from luminal import luminal_backend

MODEL = "Qwen/Qwen3-4B"
PROMPT = "Explain what a neural network is in a paragraph."
DECODE = 100
WARMUP = 8


def chat_prompt(p):
    return (f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    assert torch.cuda.is_available()
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.recompile_limit = 32

    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.use_cache = True
    cfg._attn_implementation = "eager"
    model = (Qwen3ForCausalLM
             .from_pretrained(MODEL, config=cfg, torch_dtype=DTYPE)
             .eval().cuda())
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"

    input_ids = tok(chat_prompt(PROMPT), return_tensors="pt").input_ids.cuda()
    L = input_ids.shape[1]
    max_cache = L + DECODE + 8

    wrapper = TorchExportableModuleWithStaticCache(
        model, batch_size=1, max_cache_len=max_cache, device=torch.device("cuda"))

    compiles = []

    def backend(gm, ex, options=None):
        compiles.append(1)
        return luminal_backend(gm, ex, options)

    compiled = torch.compile(wrapper, backend=backend, fullgraph=True)

    def zero_cache():
        for layer in wrapper.static_cache.layers:
            layer.keys.zero_()
            layer.values.zero_()

    def gen(steps):
        zero_cache()
        cp = torch.arange(L, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = compiled(input_ids=input_ids, cache_position=cp)
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0
        nxt = logits[0, -1].argmax().view(1, 1)
        per = []
        for i in range(steps):
            cp1 = torch.tensor([L + i], device="cuda")
            torch.cuda.synchronize()
            s = time.perf_counter()
            with torch.no_grad():
                logits = compiled(input_ids=nxt, cache_position=cp1)
            torch.cuda.synchronize()
            per.append(time.perf_counter() - s)
            nxt = logits[0, -1].argmax().view(1, 1)
        return ttft, per

    gen(WARMUP)                       # warmup: compile + clean search corruption
    n_after = sum(compiles)
    ttft, per = gen(DECODE)           # measured
    tpot = statistics.mean(per)
    print(f"[StaticCache {DTYPE_NAME}] L={L} max_cache={max_cache} backend_compiles={sum(compiles)} "
          f"(recompiles_in_measured={sum(compiles) - n_after})")
    print(f"  TTFT: {ttft * 1e3:.2f} ms")
    print(f"  TPOT: {tpot * 1e3:.2f} ms  ({1.0 / tpot:.1f} tok/s over {DECODE})")


if __name__ == "__main__":
    main()
