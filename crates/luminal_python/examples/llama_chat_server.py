"""Browser-accessible chat over Llama-3-8B-Instruct compiled by luminal.

Loads NousResearch/Meta-Llama-3-8B-Instruct with `use_cache=True` and eager
attention, compiles it through `torch.compile(backend=luminal_backend)`, and
serves a FastAPI chat UI on 0.0.0.0:<port>. Each browser turn POSTs the full
conversation; the server prefills, then greedy-decodes streaming tokens back
as SSE.

Run:
    cd crates/luminal_python
    CUDARC_CUDA_VERSION=12080 \
      uv run --group dev python examples/llama_chat_server.py --port 8000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any

import torch
import torch._dynamo
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, DynamicCache, LlamaForCausalLM

from luminal import luminal_backend

REPO_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # overridden by --dtype on the CLI
# Llama-3 chat-template stop tokens (mirrors examples/llama/src/main.rs).
EOS_TOKENS = {128009, 128001}
REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_TOKENS = 1500
SESSION_TTL_SEC = 30 * 60
MAX_SESSION_CACHES = 4

log = logging.getLogger("llama_chat")


@dataclass
class SessionState:
    token_ids: list[int]
    past_key_values: Any
    updated_at: float


_session_states: dict[str, SessionState] = {}


# --- HTML chat UI (inlined) ------------------------------------------------

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Llama-3 8B (luminal)</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem;background:#fafafa;color:#222}
 h1{font-size:1.1rem;color:#666;font-weight:500}
 #log{display:flex;flex-direction:column;gap:.6rem;margin-bottom:1rem}
 .m{padding:.7rem .9rem;border-radius:10px;white-space:pre-wrap;line-height:1.4}
 .user{background:#dbeafe;align-self:flex-end;max-width:80%}
 .assistant{background:#fff;border:1px solid #e5e7eb;align-self:flex-start;max-width:90%}
 .sys{font-size:.8rem;color:#888;align-self:center}
 form{display:flex;gap:.5rem}
 textarea{flex:1;padding:.6rem;border:1px solid #d1d5db;border-radius:8px;font:inherit;resize:vertical;min-height:2.5rem}
 button{padding:.6rem 1.1rem;border:0;border-radius:8px;background:#2563eb;color:#fff;font:inherit;cursor:pointer}
 button[disabled]{background:#9ca3af;cursor:default}
</style></head>
<body>
<h1>Llama-3-8B-Instruct &middot; compiled by luminal</h1>
<div id="log"></div>
<form id="f">
  <textarea id="inp" placeholder="Say something..." autofocus></textarea>
  <button id="send" type="submit">Send</button>
</form>
<script>
const messages = [];
const logEl = document.getElementById("log");
const f = document.getElementById("f");
const inp = document.getElementById("inp");
const sendBtn = document.getElementById("send");
const SESSION_KEY = "luminal_session_id";
let sessionId = sessionStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = (self.crypto && self.crypto.randomUUID)
    ? self.crypto.randomUUID()
    : "sess-" + Date.now().toString(36) + Math.random().toString(36).slice(2);
  sessionStorage.setItem(SESSION_KEY, sessionId);
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = "m " + role;
  el.textContent = text;
  logEl.appendChild(el);
  window.scrollTo(0, document.body.scrollHeight);
  return el;
}

f.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = inp.value.trim();
  if (!text) return;
  inp.value = "";
  sendBtn.disabled = true;
  messages.push({role: "user", content: text});
  bubble("user", text);
  const asst = bubble("assistant", "");
  asst.textContent = "…";
  let acc = "";
  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({session_id: sessionId, messages})
    });
    if (!r.ok || !r.body) throw new Error("HTTP " + r.status);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      while (true) {
        const i = buf.indexOf("\\n\\n");
        if (i < 0) break;
        const frame = buf.slice(0, i);
        buf = buf.slice(i + 2);
        for (const line of frame.split("\\n")) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") continue;
          try {
            const ev = JSON.parse(payload);
            if (ev.token) {
              if (acc === "") asst.textContent = "";
              acc += ev.token;
              asst.textContent = acc;
              window.scrollTo(0, document.body.scrollHeight);
            } else if (ev.status) {
              const s = document.createElement("div");
              s.className = "m sys";
              s.textContent = ev.status;
              logEl.insertBefore(s, asst);
            } else if (ev.error) {
              asst.textContent = "[error] " + ev.error;
            }
          } catch (e) { /* ignore parse errors */ }
        }
      }
    }
    if (acc) messages.push({role: "assistant", content: acc});
  } catch (e) {
    asst.textContent = "[error] " + e.message;
  } finally {
    sendBtn.disabled = false;
    inp.focus();
  }
});
</script>
</body></html>
"""


# --- Model + compile -------------------------------------------------------


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    session_id: str | None = None
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS


def load_model(
    repo_id: str,
    dtype: torch.dtype,
    use_cache: bool,
    local_files_only: bool,
):
    log.info(
        "Loading tokenizer + config from %s (device=%s, local_files_only=%s)",
        repo_id,
        DEVICE,
        local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, local_files_only=local_files_only
    )
    config = AutoConfig.from_pretrained(repo_id, local_files_only=local_files_only)
    config.use_cache = use_cache
    config._attn_implementation = "eager"
    log.info("Loading weights (%s, use_cache=%s) to %s ...", dtype, use_cache, DEVICE)
    model = (
        LlamaForCausalLM.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        .eval()
        .to(DEVICE)
    )
    return tokenizer, model


def make_compiled_kvcache(model: torch.nn.Module):
    # luminal's KV-cache pattern (tests/test_kv_cache_growing.py): bump dynamo
    # cache size and disable automatic dynamic shapes so each new cache seq_len
    # cuts its own graph (torch.export can't SymInt the cache-seq axis).
    torch._dynamo.config.cache_size_limit = 4096
    torch._dynamo.config.automatic_dynamic_shapes = False
    return torch.compile(model, backend=luminal_backend)


def make_compiled_dynamic(model: torch.nn.Module):
    torch._dynamo.reset()
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.recompile_limit = 16
    torch._dynamo.config.automatic_dynamic_shapes = True
    log.info(
        "Compiling HF DynamicCache decode with automatic dynamic shapes "
        "(cache_size_limit=%d, recompile_limit=%d)",
        torch._dynamo.config.cache_size_limit,
        torch._dynamo.config.recompile_limit,
    )
    return torch.compile(model, backend=luminal_backend, fullgraph=True)


def warmup_kvcache(compiled, steps: int) -> None:
    if steps <= 0:
        log.info("Warmup skipped (steps=0)")
        return
    log.info(
        "Warmup: prefill + %d decode shapes. First few will be SLOW (egglog + codegen).",
        steps,
    )
    t0 = time.time()
    with torch.no_grad():
        ids = torch.tensor([[1]], device=DEVICE)
        out = _compiled_call(compiled, ids)
        log.info("  prefill[seq=1] compiled in %.1fs", time.time() - t0)
        for i in range(steps):
            t = time.time()
            nxt = torch.tensor([[2]], device=DEVICE)
            out = _compiled_call(compiled, nxt, past_key_values=out.past_key_values)
            log.info("  decode[cache=%d] compiled in %.1fs", i + 1, time.time() - t)
    log.info("Warmup done in %.1fs total", time.time() - t0)


def warmup_dynamic_cache(compiled, model_config, steps: int) -> None:
    if steps <= 0:
        log.info("Warmup skipped (steps=0)")
        return
    log.info(
        "Warmup: prefill seq_len=1 plus %d decode steps to build the reusable cache graph.",
        steps,
    )
    t0 = time.time()
    with torch.no_grad():
        ids = torch.tensor([[1]], device=DEVICE)
        cache = DynamicCache(config=model_config)
        out = None
        t = time.time()
        out = _compiled_call(
            compiled, input_ids=ids, past_key_values=cache, use_cache=True
        )
        log.info("  prefill[seq=1] %.1fs", time.time() - t)
        assert out is not None
        for i in range(steps):
            nxt = out.logits[:, -1:].argmax(dim=-1)
            t = time.time()
            out = _compiled_call(
                compiled,
                input_ids=nxt,
                past_key_values=out.past_key_values,
                use_cache=True,
            )
            cache_len = out.past_key_values.layers[0].keys.shape[2]
            log.info("  decode[cache=%d] %.1fs", cache_len, time.time() - t)
    log.info("Warmup done in %.1fs total", time.time() - t0)


# --- Generation -------------------------------------------------------------

_gen_lock = threading.Lock()


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _compiled_call(compiled, *args, **kwargs):
    reload_fn = getattr(compiled, "reload_original_weights", None)
    if reload_fn is not None:
        reload_fn()
    return compiled(*args, **kwargs)


def _prune_session_states(
    now: float | None = None, preserve: set[str] | None = None
) -> None:
    now = time.time() if now is None else now
    preserve = preserve or set()

    expired = [
        session_id
        for session_id, state in _session_states.items()
        if session_id not in preserve and now - state.updated_at > SESSION_TTL_SEC
    ]
    for session_id in expired:
        del _session_states[session_id]

    while len(_session_states) > MAX_SESSION_CACHES:
        victim = min(
            (
                (session_id, state.updated_at)
                for session_id, state in _session_states.items()
                if session_id not in preserve
            ),
            default=None,
            key=lambda item: item[1],
        )
        if victim is None:
            break
        del _session_states[victim[0]]


def _get_session_state(session_id: str | None) -> SessionState | None:
    if not session_id:
        return None
    now = time.time()
    _prune_session_states(now)
    state = _session_states.get(session_id)
    if state is None:
        return None
    state.updated_at = now
    return state


def _store_session_state(
    session_id: str | None, token_ids: list[int], past_key_values: Any
) -> None:
    if not session_id:
        return
    now = time.time()
    _session_states[session_id] = SessionState(
        token_ids=token_ids,
        past_key_values=past_key_values,
        updated_at=now,
    )
    _prune_session_states(now, preserve={session_id})


def stream_chat_kvcache(tokenizer, compiled, messages, max_new_tokens):
    acquired = _gen_lock.acquire(blocking=False)
    if not acquired:
        yield _sse({"error": "busy"})
        yield b"data: [DONE]\n\n"
        return
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(DEVICE)
        prompt_len = int(ids.shape[1])
        yield _sse(
            {
                "status": f"prefilling {prompt_len} tokens (kv-cache mode: each new seq_len recompiles ~30s on 8B)..."
            }
        )

        t0 = time.time()
        with torch.no_grad():
            out = _compiled_call(compiled, ids)
        log.info("prefill[seq=%d] %.2fs", prompt_len, time.time() - t0)

        seen: set[int] = set()
        for step in range(max_new_tokens):
            logits = out.logits[0, -1].float()
            if seen:
                idx = torch.tensor(sorted(seen), device=logits.device)
                vals = logits.index_select(0, idx)
                vals = torch.where(
                    vals > 0, vals / REPETITION_PENALTY, vals * REPETITION_PENALTY
                )
                logits.index_copy_(0, idx, vals)
            nxt = int(torch.argmax(logits).item())
            if nxt in EOS_TOKENS:
                break
            seen.add(nxt)
            piece = tokenizer.decode([nxt], skip_special_tokens=True)
            if piece:
                yield _sse({"token": piece})
            t = time.time()
            with torch.no_grad():
                out = _compiled_call(
                    compiled,
                    torch.tensor([[nxt]], device=DEVICE),
                    past_key_values=out.past_key_values,
                )
            dt = time.time() - t
            if dt > 1.0:
                log.info("decode step %d slow: %.1fs (new shape compile?)", step, dt)
        yield b"data: [DONE]\n\n"
    except Exception as e:
        log.exception("generation failed")
        yield _sse({"error": str(e)})
        yield b"data: [DONE]\n\n"
    finally:
        _gen_lock.release()


async def stream_chat_dynamic(
    tokenizer,
    compiled,
    model_config,
    messages,
    max_new_tokens,
    session_id: str | None,
):
    """Pure compiled-cache decode path.

    Prefill runs once on the full prompt. Each decode step then feeds one
    token plus `DynamicCache`, letting Dynamo promote the cache sequence axis
    to symbolic and reuse one compiled decode graph as the cache grows.
    """
    acquired = _gen_lock.acquire(blocking=False)
    if not acquired:
        yield _sse({"error": "busy"})
        yield b"data: [DONE]\n\n"
        return
    try:
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        ids = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(DEVICE)
        prompt_ids = ids[0].tolist()
        prompt_len = len(prompt_ids)
        generated_ids: list[int] = []
        all_step_ms: list[float] = []
        state = _get_session_state(session_id)
        out = None
        if (
            state is not None
            and len(prompt_ids) >= len(state.token_ids)
            and prompt_ids[: len(state.token_ids)] == state.token_ids
            and len(prompt_ids) > len(state.token_ids)
        ):
            suffix = prompt_ids[len(state.token_ids) :]
            yield _sse(
                {
                    "status": (
                        f"reusing {len(state.token_ids)} cached tokens; "
                        f"replaying {len(suffix)} new tokens through cache ..."
                    )
                }
            )
            with torch.no_grad():
                t0 = time.time()
                cache = state.past_key_values
                for token_id in suffix:
                    out = _compiled_call(
                        compiled,
                        input_ids=torch.tensor([[token_id]], device=DEVICE),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = out.past_key_values
            log.info(
                "dynamic cached replay[reuse=%d suffix=%d total=%d] %.2fs",
                len(state.token_ids),
                len(suffix),
                prompt_len,
                time.time() - t0,
            )
        else:
            if state is not None:
                log.info(
                    "session cache miss: cached=%d new=%d",
                    len(state.token_ids),
                    prompt_len,
                )
            yield _sse({"status": f"prefilling {prompt_len} prompt tokens ..."})
            with torch.no_grad():
                cache = DynamicCache(config=model_config)
                t0 = time.time()
                out = _compiled_call(
                    compiled,
                    input_ids=ids,
                    past_key_values=cache,
                    use_cache=True,
                )
            log.info("dynamic prefill[seq=%d] %.2fs", prompt_len, time.time() - t0)
        assert out is not None
        for step in range(max_new_tokens):
            logits = out.logits[0, -1].float()
            seen = set(generated_ids)
            if seen:
                idx = torch.tensor(sorted(seen), device=logits.device)
                vals = logits.index_select(0, idx)
                vals = torch.where(
                    vals > 0, vals / REPETITION_PENALTY, vals * REPETITION_PENALTY
                )
                logits.index_copy_(0, idx, vals)
            nxt = int(torch.argmax(logits).item())
            if nxt in EOS_TOKENS:
                break
            generated_ids.append(nxt)
            piece = tokenizer.decode([nxt], skip_special_tokens=True)
            if piece:
                yield _sse({"token": piece})
            t = time.time()
            with torch.no_grad():
                out = _compiled_call(
                    compiled,
                    input_ids=torch.tensor([[nxt]], device=DEVICE),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
            all_step_ms.append((time.time() - t) * 1000)
        if all_step_ms:
            log.info(
                "decode: %d steps, mean=%.1fms, last=%.1fms",
                len(all_step_ms),
                sum(all_step_ms) / len(all_step_ms),
                all_step_ms[-1],
            )
        _store_session_state(
            session_id, prompt_ids + generated_ids, out.past_key_values
        )
        yield b"data: [DONE]\n\n"
    except Exception as e:
        log.exception("generation failed")
        yield _sse({"error": str(e)})
        yield b"data: [DONE]\n\n"
    finally:
        _gen_lock.release()


# --- App --------------------------------------------------------------------


def build_app(
    tokenizer,
    compiled,
    mode: str,
    repo_id: str,
    model_config=None,
    max_context_tokens: int = 0,
) -> FastAPI:
    app = FastAPI()
    stream_fn = stream_chat_dynamic if mode == "dynamic" else stream_chat_kvcache

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX_HTML

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": repo_id,
            "dtype": str(DTYPE),
            "device": DEVICE,
            "mode": mode,
            "max_context_tokens": max_context_tokens,
            "cached_sessions": len(_session_states),
        }

    @app.post("/chat")
    async def chat(req: ChatRequest):
        if mode == "dynamic":
            body = stream_fn(
                tokenizer,
                compiled,
                model_config,
                req.messages,
                req.max_new_tokens,
                req.session_id,
            )
        else:
            body = stream_fn(tokenizer, compiled, req.messages, req.max_new_tokens)
        return StreamingResponse(
            body,
            media_type="text/event-stream",
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. `auto` prefers CUDA when available, "
        "otherwise falls back to CPU.",
    )
    ap.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic", "kv-cache"],
        help="dynamic: HF DynamicCache through torch.compile with "
        "automatic dynamic shapes; after prefill/static "
        "decode/dynamic decode compile, cached decode reuses "
        "one compiled graph. kv-cache: force per-cache-length "
        "specialization by disabling automatic dynamic shapes.",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=4,
        help="dynamic: after a short prefill, run this "
        "many decode steps at startup to build the reusable "
        "cache graph. kv-cache: decode cache lengths to "
        "pre-compile at startup.",
    )
    ap.add_argument(
        "--example-seq",
        type=int,
        default=64,
        help="Legacy no-op kept for CLI compatibility.",
    )
    ap.add_argument(
        "--search-iterations",
        type=int,
        default=25,
        help="Legacy no-op kept for CLI compatibility.",
    )
    ap.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="Legacy no-op in compiled-cache dynamic mode; kept for CLI compatibility.",
    )
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer/config only from the local Hugging Face cache.",
    )
    ap.add_argument(
        "--dtype",
        default="fp32",
        choices=["bf16", "fp32", "fp16"],
        help="Weight/activation dtype. fp32 matches eager argmax "
        "exactly on full 8B; bf16 drifts and breaks coherence "
        "(empirically picks gibberish tokens).",
    )
    args = ap.parse_args()
    global DTYPE, DEVICE
    DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[
        args.dtype
    ]
    if args.device == "auto":
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "`--device cuda` was requested, but torch.cuda.is_available() is False."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log.info(
        "Starting server with repo=%s device=%s dtype=%s mode=%s",
        args.repo_id,
        DEVICE,
        DTYPE,
        args.mode,
    )

    use_cache = True
    tokenizer, model = load_model(
        args.repo_id,
        DTYPE,
        use_cache,
        args.local_files_only,
    )

    if args.mode == "kv-cache":
        compiled = make_compiled_kvcache(model)
        warmup_kvcache(compiled, args.warmup_steps)
    else:
        if args.example_seq != ap.get_default("example_seq"):
            log.info("--example-seq is ignored in compiled-cache dynamic mode")
        if args.search_iterations != ap.get_default("search_iterations"):
            log.info("--search-iterations is ignored in compiled-cache dynamic mode")
        if args.max_context_tokens:
            log.info("--max-context-tokens is ignored in compiled-cache dynamic mode")
        compiled = make_compiled_dynamic(model)
        warmup_dynamic_cache(compiled, model.config, args.warmup_steps)

    app = build_app(
        tokenizer,
        compiled,
        args.mode,
        args.repo_id,
        model_config=model.config,
        max_context_tokens=args.max_context_tokens,
    )

    import uvicorn

    log.info(
        "mode=%s device=%s   Open http://<this-host>:%d/   (bind=%s)",
        args.mode,
        DEVICE,
        args.port,
        args.host,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
