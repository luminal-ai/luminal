"""Apples-to-apples TTFT/TPOT: Rust qwen vs luminal_python, ONE measurement protocol.

Mirrors luminal-benchmarks: every path exposes generate_tokens(prompt) -> iterator
of token strings, and the driver times token *arrivals* externally (TTFT = first
token; TPOT = mean inter-token). Both paths: same prompt, same GEN tokens, greedy,
and fp32 (the Rust example is fp32, so the luminal path is run fp32 too).

- rust:    the qwen_stdio --stdio subprocess (emits TOK\\t<text> per token, EOQ at end)
- luminal: TorchExportableModuleWithStaticCache greedy decode (in-process iterator)
"""
import os
import sys
import time
import statistics
import subprocess
from pathlib import Path

import torch
import torch._dynamo
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM, DynamicCache
from transformers.integrations.executorch import TorchExportableModuleWithStaticCache
from luminal import luminal_backend

PROMPT = "Explain what a neural network is in a paragraph."
GEN = 100
EOS, STOP = 151645, 151643
RUST_BIN = "/lambda/nfs/tucker-fs/compare/luminal/target/release/qwen_stdio"
HF_HOME_RUST = "/lambda/nfs/tucker-fs/compare/hf_cache_rust"


def chat_prompt(p):
    return (f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def measure(gen_fn, label, warmups=1):
    for _ in range(warmups):
        for _ in gen_fn(PROMPT):
            pass
    t0 = time.perf_counter()
    stamps = []
    toks = []
    for t in gen_fn(PROMPT):
        stamps.append(time.perf_counter())
        toks.append(t)
    ttft = (stamps[0] - t0) * 1e3
    inter = [stamps[j] - stamps[j - 1] for j in range(1, len(stamps))]
    tpot = statistics.mean(inter) * 1e3 if inter else float("nan")
    text = "".join(toks)
    print(f"[{label}] TTFT={ttft:.2f}ms  TPOT={tpot:.2f}ms  "
          f"({1000.0/tpot:.1f} tok/s)  tokens={len(stamps)}", flush=True)
    print(f"[{label}] TEXT[:160]={text[:160]!r}", flush=True)


# ---------------- Rust path ----------------
def start_rust():
    env = {**os.environ, "HF_HOME": HF_HOME_RUST, "CUDARC_CUDA_VERSION": "12080",
           "GEN_TOKENS": str(GEN), "SEARCH_GRAPHS": "10"}
    proc = subprocess.Popen([RUST_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=sys.stderr, text=True, bufsize=1, env=env)
    # wait for READY (compile done)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("rust exited before READY")
        if line.rstrip() == "READY":
            return proc
    raise TimeoutError("no READY")


def make_rust_gen(proc):
    def gen(prompt):
        proc.stdin.write(prompt.replace("\n", " ") + "\n")
        proc.stdin.flush()
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("rust exited mid-stream")
            line = line.rstrip("\n")
            if line.startswith("TOK\t"):
                yield line[4:]
            elif line == "EOQ":
                return
    return gen


# ---------------- luminal path ----------------
def make_luminal_gen():
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.recompile_limit = 32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
    cfg.use_cache = True
    cfg._attn_implementation = "eager"
    model = (Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4B", config=cfg,
             torch_dtype=torch.float32).eval().cuda())
    model.generation_config.use_cache = True
    model.generation_config.cache_implementation = "static"
    # Size the static cache to the generation length. StaticCache attends over
    # the full max_cache_len each step (masked), so a 4096 buffer would do ~33x
    # the attention of Rust (which attends only the actual length). prompt+GEN
    # makes the attention window comparable to Rust's.
    _L0 = len(AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")(
        chat_prompt(PROMPT)).input_ids)
    max_cache = _L0 + GEN + 8
    wrapper = TorchExportableModuleWithStaticCache(
        model, batch_size=1, max_cache_len=max_cache, device=torch.device("cuda"))
    compiled = torch.compile(wrapper, backend=luminal_backend, fullgraph=True)

    def zero_cache():
        for layer in wrapper.static_cache.layers:
            layer.keys.zero_()
            layer.values.zero_()

    def gen(prompt):
        zero_cache()
        ids = tok(chat_prompt(prompt), return_tensors="pt").input_ids.cuda()
        L = ids.shape[1]
        with torch.no_grad():
            logits = compiled(input_ids=ids, cache_position=torch.arange(L, device="cuda"))
        torch.cuda.synchronize()
        nxt = int(logits[0, -1].argmax())
        for i in range(GEN):
            if nxt in (EOS, STOP):
                return
            yield tok.decode([nxt])
            with torch.no_grad():
                logits = compiled(input_ids=torch.tensor([[nxt]], device="cuda"),
                                  cache_position=torch.tensor([L + i], device="cuda"))
            torch.cuda.synchronize()
            nxt = int(logits[0, -1].argmax())
    return gen


# ---------------- luminal DynamicCache path (the standard luminal_python path) ----------------
def make_dynamic_gen():
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.recompile_limit = 32
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3-4B")
    cfg.use_cache = True
    cfg._attn_implementation = "eager"
    model = (Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4B", config=cfg,
             torch_dtype=torch.float32).eval().cuda())
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)

    def gen(prompt):
        cache = DynamicCache(config=model.config)
        ids = tok(chat_prompt(prompt), return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = compiled(input_ids=ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        nxt = int(out.logits[0, -1].argmax())
        for _ in range(GEN):
            if nxt in (EOS, STOP):
                return
            yield tok.decode([nxt])
            with torch.no_grad():
                out = compiled(input_ids=torch.tensor([[nxt]], device="cuda"),
                               past_key_values=out.past_key_values, use_cache=True)
            torch.cuda.synchronize()
            nxt = int(out.logits[0, -1].argmax())
    return gen


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which == "dynamic":
        measure(make_dynamic_gen(), "luminal DynamicCache fp32")
        return
    if which in ("both", "rust"):
        proc = start_rust()
        measure(make_rust_gen(proc), "Rust fp32")
        proc.stdin.close()
    if which in ("both", "luminal"):
        measure(make_luminal_gen(), "luminal StaticCache fp32")


if __name__ == "__main__":
    main()
