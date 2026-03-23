"""KV Cache decode loop test.

Compiles a tiny 1-layer Llama model with use_cache=True, then:
  1. Prefill: model(input_ids) -> logits + K/V cache
  2. Decode:  model(next_token, past_key_values=cache) -> logits + updated K/V

Verifies correctness of both steps and writes DOT graphs for comparison.
"""

import os

import torch

from luminal import luminal_backend


def _capturing_backend(captured):
    """Wrap luminal_backend to capture CompiledModels for DOT extraction."""

    def backend(gm, example_inputs):
        compiled = luminal_backend(gm, example_inputs)
        captured.append(compiled)
        return compiled

    return backend


def test_kv_cache_decode_loop():
    """Full prefill -> decode loop through luminal with KV cache."""
    from transformers import LlamaConfig, LlamaForCausalLM

    # Allow both prefill and decode compilations (conftest sets limit=1)
    torch._dynamo.config.cache_size_limit = 2

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
        use_cache=True,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    captured = []
    compiled = torch.compile(model, backend=_capturing_backend(captured))

    # --- Prefill step ---
    with torch.no_grad():
        ref_prefill = model(input_ids)
        out_prefill = compiled(input_ids)

    assert torch.allclose(out_prefill.logits, ref_prefill.logits, atol=1e-5)
    assert out_prefill.past_key_values is not None, "Prefill should return KV cache"

    # --- Decode step ---
    next_token = ref_prefill.logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ref_decode = model(next_token, past_key_values=ref_prefill.past_key_values)
        out_decode = compiled(next_token, past_key_values=out_prefill.past_key_values)

    assert torch.allclose(out_decode.logits, ref_decode.logits, atol=1e-5)

    # --- DOT graph comparison ---
    # captured[0] = prefill graph, captured[1] = decode graph (recompiled by dynamo)
    assert len(captured) >= 2, (
        f"Expected 2 compilations (prefill+decode), got {len(captured)}"
    )

    out_dir = "/tmp/luminal_kv_cache_comparison"
    os.makedirs(out_dir, exist_ok=True)

    prefill_dot = captured[0]._graph.to_dot()
    decode_dot = captured[1]._graph.to_dot()

    with open(os.path.join(out_dir, "prefill.dot"), "w") as f:
        f.write(prefill_dot)
    with open(os.path.join(out_dir, "decode.dot"), "w") as f:
        f.write(decode_dot)

    print(f"\n=== DOT files written to {out_dir} ===")
    print(f"Prefill: {len(prefill_dot)} chars, inputs: {captured[0]._input_names}")
    print(f"Decode:  {len(decode_dot)} chars, inputs: {captured[1]._input_names}")

    # Decode graph should have more inputs (past K/V cache tensors)
    assert len(captured[1]._input_names) > len(captured[0]._input_names), (
        f"Decode should have more inputs than prefill: "
        f"{len(captured[1]._input_names)} vs {len(captured[0]._input_names)}"
    )
