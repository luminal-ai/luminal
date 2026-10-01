"""KV cache decode loop test.

Compiles a tiny 1-layer Llama model with use_cache=True, then:
  1. Prefill: model(input_ids) -> logits + K/V cache
  2. Decode:  model(next_token, past_key_values=cache) -> logits + updated K/V

Verifies correctness of both steps.
"""

import torch
from backend_test_utils import luminal_backend


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
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    compiled = torch.compile(model, backend=luminal_backend)

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
