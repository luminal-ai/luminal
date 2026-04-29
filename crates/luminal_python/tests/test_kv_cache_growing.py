"""KV Cache growing decode loop test.

Compiles a tiny 1-layer Llama model with use_cache=True, then runs a
multi-step autoregressive decode loop:

  1. Prefill:  model(input_ids) -> logits + initial KV cache
  2. Decode x N: model(next_token, past_key_values=cache) -> logits + grown KV cache

At each step, prints the KV cache tensor shapes so you can see the
sequence dimension grow: (1, n_kv_heads, 4, head_dim) -> (1, n_kv_heads, 5, ...) -> ...

Verifies luminal output matches PyTorch reference at every step.
"""

import pytest
import torch
import torch._dynamo

from luminal import luminal_backend

NUM_DECODE_STEPS = 5


def test_kv_cache_growing():
    """Multi-step prefill + decode loop showing KV cache growth."""
    from transformers import LlamaConfig, LlamaForCausalLM

    # We need 1 compilation for prefill + 1 per unique decode cache size
    torch._dynamo.config.cache_size_limit = NUM_DECODE_STEPS + 2
    # Disable automatic dynamic shapes — dynamo would otherwise try to use SymInt
    # for the varying cache seq_len dimension, which torch.export doesn't support.
    # Instead, we want a fresh recompilation for each new cache size.
    torch._dynamo.config.automatic_dynamic_shapes = False

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
        use_cache=True,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval()
    compiled = torch.compile(model, backend=luminal_backend)

    input_ids = torch.tensor([[1, 2, 3, 4]])

    # ---- Prefill ----
    with torch.no_grad():
        ref_out = model(input_ids)
        lum_out = compiled(input_ids)

    assert ref_out.past_key_values is not None, "Reference should return KV cache"
    assert lum_out.past_key_values is not None, "Luminal should return KV cache"

    assert torch.allclose(lum_out.logits, ref_out.logits, atol=1e-5), (
        f"Prefill mismatch: max_diff="
        f"{torch.max(torch.abs(lum_out.logits - ref_out.logits)).item():.2e}"
    )

    _print_cache_shapes("Prefill", ref_out.past_key_values, lum_out.past_key_values)

    ref_cache = ref_out.past_key_values
    lum_cache = lum_out.past_key_values

    # ---- Decode loop ----
    for step in range(NUM_DECODE_STEPS):
        # Greedy next token from reference logits
        next_token = ref_out.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            ref_out = model(next_token, past_key_values=ref_cache)
            lum_out = compiled(next_token, past_key_values=lum_cache)

        assert torch.allclose(lum_out.logits, ref_out.logits, atol=1e-5), (
            f"Decode step {step} mismatch: max_diff="
            f"{torch.max(torch.abs(lum_out.logits - ref_out.logits)).item():.2e}"
        )

        ref_cache = ref_out.past_key_values
        lum_cache = lum_out.past_key_values

        _print_cache_shapes(f"Decode step {step}", ref_cache, lum_cache)

    # Final sanity check: cache seq_len should equal prompt + decode steps
    expected_seq = input_ids.shape[1] + NUM_DECODE_STEPS
    final_k = ref_cache.layers[0].keys
    assert final_k.shape[2] == expected_seq, (
        f"Expected cache seq_len={expected_seq}, got {final_k.shape[2]}"
    )
    print(
        f"\nAll {NUM_DECODE_STEPS} decode steps passed. "
        f"Cache grew from seq_len={input_ids.shape[1]} to {expected_seq}."
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="R1 full-width 1-layer is too memory-heavy for CPU native backend",
)
def test_kv_cache_growing_r1_mla(device: torch.device):
    """Growing-cache decode loop on DeepSeek-R1 (MLA + decoupled RoPE), 1 layer.

    Exercises MLA: q_lora / kv_lora low-rank projections, decoupled RoPE split
    (qk_nope_head_dim + qk_rope_head_dim), and DynamicCache crossing the compile
    boundary through the MLA update path (`cache_utils.py:102-121`).

    Runs in fp32 — in bf16, MLA's empty-tensor-cat inside DynamicLayer.update
    has a precision drift on the compiled path (logits ~3.7 on 1 layer) that
    does not affect standard GQA (Llama in bf16 is bit-identical). Investigate
    separately.
    """
    from transformers import AutoConfig, DeepseekV3ForCausalLM

    torch._dynamo.config.cache_size_limit = NUM_DECODE_STEPS + 2
    torch._dynamo.config.automatic_dynamic_shapes = False

    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1")
    config.num_hidden_layers = 1
    # first_k_dense_replace=3 (default) makes the 1 layer dense, so we avoid
    # the 256-expert MoE path and the associated memory pressure.
    config._attn_implementation = "eager"
    config.torch_dtype = torch.float32
    model = DeepseekV3ForCausalLM(config).eval().to(dtype=torch.float32, device=device)
    compiled = torch.compile(model, backend=luminal_backend)

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)

    with torch.no_grad():
        ref_out = model(input_ids)
        lum_out = compiled(input_ids)

    # fp32 MLA matches to ~1e-5 — see diagnose_dtype.py. Keep the tolerance
    # tight here so regressions in the MLA cat/split path show up immediately.
    assert torch.allclose(lum_out.logits, ref_out.logits, atol=1e-4), (
        f"Prefill: max_diff={torch.max(torch.abs(lum_out.logits - ref_out.logits)).item():.2e}"
    )

    ref_cache = ref_out.past_key_values
    lum_cache = lum_out.past_key_values

    # Run a single decode step — enough to confirm the cache flows through as an
    # explicit input on the second compile (the key signal from
    # _test_kv_cache_comparison.py's "decode has more inputs than prefill"
    # assertion). Full 5-step growth is covered by the Llama test above.
    next_token = ref_out.logits[0, -1, :].argmax().view(1, 1).to(device)
    with torch.no_grad():
        ref_dec = model(next_token, past_key_values=ref_cache)
        lum_dec = compiled(next_token, past_key_values=lum_cache)

    assert torch.allclose(lum_dec.logits, ref_dec.logits, atol=1e-4), (
        f"Decode: max_diff={torch.max(torch.abs(lum_dec.logits - ref_dec.logits)).item():.2e}"
    )


def _print_cache_shapes(label, ref_cache, lum_cache):
    """Print KV cache shapes for both reference and luminal."""
    print(f"\n--- {label} ---")
    for layer_idx, ref_layer in enumerate(ref_cache.layers):
        ref_k, ref_v = ref_layer.keys, ref_layer.values
        lum_layer = lum_cache.layers[layer_idx]
        lum_k, lum_v = lum_layer.keys, lum_layer.values
        print(
            f"  Layer {layer_idx}:  "
            f"K ref={list(ref_k.shape)}  lum={list(lum_k.shape)}  |  "
            f"V ref={list(ref_v.shape)}  lum={list(lum_v.shape)}"
        )
        # Verify cache tensors match
        assert torch.allclose(lum_k, ref_k, atol=1e-5), (
            f"{label} layer {layer_idx} K mismatch: "
            f"max_diff={torch.max(torch.abs(lum_k - ref_k)).item():.2e}"
        )
        assert torch.allclose(lum_v, ref_v, atol=1e-5), (
            f"{label} layer {layer_idx} V mismatch: "
            f"max_diff={torch.max(torch.abs(lum_v - ref_v)).item():.2e}"
        )
