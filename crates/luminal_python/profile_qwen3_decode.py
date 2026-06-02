"""Profile a single steady-state decode step of Qwen3-4B through luminal_python.

Goal: attribute the ~27 ms/token decode cost. Four measurements:
  (1) wall-clock per step (sync each side) — the headline TPOT number
  (2) CUDA-event *device span* of just the compiled() call (GPU timeline width)
  (3) torch.profiler CPU/GPU split + top kernels + #kernel launches/step
  (4) per-_graph-method host (FFI) timing via a transparent proxy

Run after warmup so all 3 backend graphs are compiled and the dynamic decode
graph is reused with no recompiles.
"""
import os
import time
import statistics
import torch
import torch._dynamo
from collections import defaultdict
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM, DynamicCache
from luminal import luminal_backend
from luminal.pt2 import _LazyDynamicCompiledModel

MODEL = "Qwen/Qwen3-4B"
PROMPT = "Explain what a neural network is in a paragraph."
DECODE = 100
WARMUP = 8

# Precision is selectable so we can isolate the bf16 cast-tax from fusion:
#   LUMINAL_BENCH_DTYPE=fp32 -> all-fp32 (matches the Rust example's precision)
#   LUMINAL_BENCH_DTYPE=bf16 -> default HF bf16
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}
DTYPE_NAME = os.environ.get("LUMINAL_BENCH_DTYPE", "bf16")
DTYPE = _DTYPES[DTYPE_NAME]


def chat_prompt(p: str) -> str:
    return (f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _dev_self_us(e):
    # PyTorch renamed cuda_* -> device_* around 2.x; support both.
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0.0


def _dev_total_us(e):
    for attr in ("device_time_total", "cuda_time_total"):
        if hasattr(e, attr):
            return getattr(e, attr)
    return 0.0


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
    print(f"dtype = {DTYPE_NAME} ({DTYPE})")

    compiled_models = []

    def backend(gm, ex, options=None):
        r = luminal_backend(gm, ex, options)
        compiled_models.append(r)
        return r

    compiled = torch.compile(model, backend=backend, fullgraph=True)
    input_ids = tok(chat_prompt(PROMPT), return_tensors="pt").input_ids.cuda()

    def prefill():
        cache = DynamicCache(config=model.config)
        out = compiled(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return out, out.logits[0, -1].argmax().view(1, 1)

    def step(out, nxt):
        out = compiled(input_ids=nxt, past_key_values=out.past_key_values,
                       use_cache=True)
        return out, out.logits[0, -1].argmax().view(1, 1)

    # ---- warmup: compile all 3 graphs, settle the dynamic decode graph ----
    out, nxt = prefill()
    for _ in range(WARMUP):
        out, nxt = step(out, nxt)
    torch.cuda.synchronize()
    print(f"warmup done: backend graphs compiled = {len(compiled_models)}")

    # ================= (1) wall-clock per step =================
    out, nxt = prefill(); torch.cuda.synchronize()
    per = []
    for _ in range(DECODE):
        torch.cuda.synchronize(); s = time.perf_counter()
        out, nxt = step(out, nxt)
        torch.cuda.synchronize(); per.append(time.perf_counter() - s)
    wall = 1e3 * statistics.mean(per)
    print(f"\n(1) wall-clock/step:  mean={wall:.3f} ms  "
          f"min={1e3*min(per):.3f}  p50={1e3*statistics.median(per):.3f}")

    # ============ (2) CUDA-event device span of the call ============
    out, nxt = prefill(); torch.cuda.synchronize()
    spans = []
    for _ in range(DECODE):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        out = compiled(input_ids=nxt, past_key_values=out.past_key_values,
                       use_cache=True)
        e1.record()
        torch.cuda.synchronize()
        spans.append(e0.elapsed_time(e1))      # ms, GPU timeline span
        nxt = out.logits[0, -1].argmax().view(1, 1)
    span = statistics.mean(spans)
    print(f"(2) device span/step: mean={span:.3f} ms  (GPU-timeline width of the "
          f"compiled() call, incl. intra-step idle gaps)")

    # ================= (3) torch.profiler =================
    out, nxt = prefill(); torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(DECODE):
            out, nxt = step(out, nxt)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    total_self_dev = sum(_dev_self_us(e) for e in ka) / 1e3   # ms over window
    total_self_cpu = sum(e.self_cpu_time_total for e in ka) / 1e3
    # kernel launches = sum of counts over rows that ran on device
    n_launches = sum(e.count for e in ka if _dev_self_us(e) > 0)
    print(f"\n(3) torch.profiler over {DECODE} steps:")
    print(f"    GPU busy (sum self device time)/step = {total_self_dev/DECODE:.3f} ms")
    print(f"    CPU (sum self cpu time)/step         = {total_self_cpu/DECODE:.3f} ms")
    print(f"    device kernel launches/step          = {n_launches/DECODE:.1f}")
    try:
        print(ka.table(sort_by="self_device_time_total", row_limit=20))
    except Exception:
        print(ka.table(sort_by="self_cuda_time_total", row_limit=20))
    print("---- top by CPU ----")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=15))
    trace = "/tmp/qwen3_decode_trace.json"
    prof.export_chrome_trace(trace)
    print(f"    chrome trace -> {trace}")

    # ============ (4) per-_graph-method host (FFI) timing ============
    timers = defaultdict(lambda: [0.0, 0])  # name -> [total_s, calls]

    class Proxy:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, name):
            attr = getattr(object.__getattribute__(self, "_inner"), name)
            if callable(attr):
                def wrapped(*a, **k):
                    s = time.perf_counter()
                    r = attr(*a, **k)
                    t = timers[name]
                    t[0] += time.perf_counter() - s
                    t[1] += 1
                    return r
                return wrapped
            return attr

    # Resolve every compiled graph to its inner CompiledModel and wrap _graph.
    wrapped = 0
    for m in compiled_models:
        inner = m
        if isinstance(m, _LazyDynamicCompiledModel):
            inner = m._ensure_compiled()
        if hasattr(inner, "_graph"):
            inner._graph = Proxy(inner._graph)
            wrapped += 1

    out, nxt = prefill(); torch.cuda.synchronize()
    timers.clear()
    for _ in range(DECODE):
        out, nxt = step(out, nxt)
    torch.cuda.synchronize()
    print(f"\n(4) host-side _graph FFI per step ({wrapped} graphs wrapped):")
    print(f"    {'method':<34}{'ms/step':>10}{'calls/step':>12}")
    rows = sorted(timers.items(), key=lambda kv: kv[1][0], reverse=True)
    host_total = 0.0
    for name, (tot, calls) in rows:
        host_total += tot
        print(f"    {name:<34}{1e3*tot/DECODE:>10.3f}{calls/DECODE:>12.1f}")
    print(f"    {'TOTAL host FFI':<34}{1e3*host_total/DECODE:>10.3f}")


if __name__ == "__main__":
    main()
