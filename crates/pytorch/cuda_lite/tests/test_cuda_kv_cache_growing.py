import pytest
import torch
import torch._dynamo
from luminal_cuda_lite import Compiler
from reference.tests.models.test_kv_cache_growing import NUM_DECODE_STEPS

luminal_backend = Compiler()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="dynamic-cache torch.compile reuse requires CUDA coverage",
)
@pytest.mark.slow
def test_dynamic_kv_cache_torch_compile_matches_reference_and_reuses_decode_graph():
    """End-to-end server-style path: torch.compile + DynamicCache on CUDA."""
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

    backend_invocations = []

    def counting_backend(gm, example_inputs, options=None):
        backend_invocations.append((gm, example_inputs))
        return Compiler(**(options or {}))(gm, example_inputs)

    prev_auto = torch._dynamo.config.automatic_dynamic_shapes
    prev_cache_limit = torch._dynamo.config.cache_size_limit
    prev_recompile_limit = torch._dynamo.config.recompile_limit
    torch._dynamo.config.automatic_dynamic_shapes = True
    torch._dynamo.config.cache_size_limit = 16
    torch._dynamo.config.recompile_limit = 16

    try:
        model = (
            LlamaForCausalLM(
                LlamaConfig(
                    hidden_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    num_hidden_layers=1,
                    intermediate_size=128,
                    vocab_size=256,
                    max_position_embeddings=128,
                    use_cache=True,
                )
            )
            .eval()
            .cuda()
        )
        compiled = torch.compile(model, backend=counting_backend, fullgraph=True)

        ref_cache = DynamicCache(config=model.config)
        out_cache = DynamicCache(config=model.config)
        input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

        with torch.no_grad():
            ref = model(input_ids=input_ids, past_key_values=ref_cache, use_cache=True)
            out = compiled(
                input_ids=input_ids,
                past_key_values=out_cache,
                use_cache=True,
            )

        for _ in range(4):
            ref_next = int(ref.logits[0, -1].argmax().item())
            out_next = int(out.logits[0, -1].argmax().item())
            assert out_next == ref_next
            with torch.no_grad():
                ref = model(
                    input_ids=torch.tensor([[ref_next]], device="cuda"),
                    past_key_values=ref.past_key_values,
                    use_cache=True,
                )
                out = compiled(
                    input_ids=torch.tensor([[out_next]], device="cuda"),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )

        assert len(backend_invocations) == 3, (
            "Expected prefill/static decode/dynamic decode traces only once each, "
            f"got {len(backend_invocations)} backend invocations"
        )
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = prev_auto
        torch._dynamo.config.cache_size_limit = prev_cache_limit
        torch._dynamo.config.recompile_limit = prev_recompile_limit
        torch._dynamo.reset()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="R1 full-width 1-layer is too memory-heavy for CPU reference backend",
)
@pytest.mark.slow
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

    # Release any memory accumulated by previous tests in the same pytest
    # process — full-width R1 instantiation needs ~3 GB and the test runner's
    # GPU is shared with ~230 prior tests' allocations.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1")
    config.num_hidden_layers = 1
    # first_k_dense_replace=3 (default) makes the 1 layer dense, so we avoid
    # the 256-expert MoE path and the associated memory pressure.
    config.torch_dtype = torch.float32
    # Aggressively shrink the embedding / LM head / FFN dimensions while
    # preserving the MLA-specific knobs that the test is actually exercising
    # (q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim).
    # Full R1 has vocab=129280, intermediate=18432, hidden=7168 — at fp32 the
    # embedding + LM head alone is ~3.5 GB, which OOMs the 40 GB test runner
    # after prior tests' allocations. The MLA path is unchanged at vocab=256.
    config.vocab_size = 256
    config.intermediate_size = 512
    config.max_position_embeddings = 128
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
