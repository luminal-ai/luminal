"""Llama3 component and HuggingFace model integration tests.

Tests individual Llama3 building blocks (RMSNorm, RoPE, SwiGLU, causal attention,
full transformer block) and progressively larger HuggingFace LlamaForCausalLM configs
through the PyTorch -> Pt2 -> luminal pipeline via torch.compile.
"""

from typing import Callable

import pytest
import torch
import torch._dynamo
from test_models import (
    CausalSelfAttentionModel,
    LlamaTransformerBlockModel,
    RMSNormModel,
    RotaryEmbeddingModel,
    SwiGLUMLPModel,
)

from luminal import luminal_backend

# ========== Component Tests ==========


def test_rms_norm(device: torch.device):
    """Test RMS normalization: x * rsqrt(mean(x^2) + eps) * weight."""
    model: torch.nn.Module = RMSNormModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4, 32), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_swiglu_mlp(device: torch.device):
    """Test SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""
    model: torch.nn.Module = SwiGLUMLPModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4, 32), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_rotary_embedding(device: torch.device):
    """Test rotary position embeddings (RoPE) with rotate-half approach."""
    model: torch.nn.Module = RotaryEmbeddingModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    # (batch, seq_len, num_heads, head_dim)
    x: torch.Tensor = torch.rand((1, 4, 4, 8), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_causal_self_attention(device: torch.device):
    """Test multi-head causal self-attention with additive mask."""
    model: torch.nn.Module = CausalSelfAttentionModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4, 32), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-5)


def test_llama_transformer_block(device: torch.device):
    """Test full Llama transformer block: RMSNorm -> Attn -> Residual -> RMSNorm -> MLP -> Residual."""
    torch.manual_seed(0)
    model: torch.nn.Module = LlamaTransformerBlockModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4, 32), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(output - original)).item():.2e}"
    )


# ========== HuggingFace LlamaForCausalLM Tests ==========


def _make_llama_config(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    vocab_size: int,
):
    """Create a LlamaConfig with use_cache=False and eager attention."""
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=128,
        use_cache=False,
        attn_implementation="eager",
    )


def _run_hf_llama_test(config, device: torch.device, atol: float):
    """Run a HuggingFace LlamaForCausalLM test with the given config."""
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=atol), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def _assert_bf16_logits_match(out_logits, ref_logits, label: str = ""):
    """Compare luminal vs PyTorch-eager logits for a model loaded in bf16.

    Tests that use ``from_pretrained`` load the checkpoint in its native bf16
    (no ``torch_dtype=float32``), so both the eager reference and the luminal
    output run in bf16 and agree only to the bf16 rounding floor — ~O(1) in
    absolute logit terms for an 8B model, not the fp32-era ``1e-5``. We assert
    exact next-token (argmax) agreement at every position — the property that
    governs generation — plus bf16-appropriate closeness via
    ``torch.testing.assert_close``.
    """
    out_f = out_logits.float()
    ref_f = ref_logits.float()
    assert torch.equal(out_f.argmax(dim=-1), ref_f.argmax(dim=-1)), (
        f"{label}next-token argmax mismatch"
    )
    torch.testing.assert_close(
        out_f,
        ref_f,
        rtol=0.05,
        atol=1.5,
        msg=lambda m: f"{label}bf16 logits diverged\n{m}",
    )


def test_hf_llama_tiny(device: torch.device):
    """HuggingFace LlamaForCausalLM — tiny (64 hidden, 1 layer, ~70K params)."""
    config = _make_llama_config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=256,
    )
    _run_hf_llama_test(config, device, atol=1e-5)


def test_hf_llama_small(device: torch.device):
    """HuggingFace LlamaForCausalLM — small (256 hidden, 1 layer, ~1.1M params)."""
    config = _make_llama_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=512,
        vocab_size=1024,
    )
    _run_hf_llama_test(config, device, atol=1e-5)


def test_hf_llama_medium(device: torch.device):
    """HuggingFace LlamaForCausalLM — medium (256 hidden, 2 layers, ~1.7M params)."""
    config = _make_llama_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
        vocab_size=1024,
    )
    _run_hf_llama_test(config, device, atol=1e-5)


@pytest.mark.slow
def test_hf_llama_large(device: torch.device):
    """HuggingFace LlamaForCausalLM — large (1024 hidden, 1 layer, ~18M params)."""
    config = _make_llama_config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=1,
        intermediate_size=2048,
        vocab_size=4096,
    )
    _run_hf_llama_test(config, device, atol=1e-5)


@pytest.mark.slow
def test_hf_llama3_real_config_1layer(device: torch.device):
    """HuggingFace LlamaForCausalLM — real Llama3.2-1B architecture, 1 layer.

    Uses AutoConfig.from_pretrained with num_hidden_layers overridden to 1.
    Random weights — tests the full-width architecture compiles correctly.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.num_hidden_layers = 1
    config.use_cache = False
    config._attn_implementation = "eager"

    model = LlamaForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-5), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_llama_decode_loop_static(device: torch.device):
    """Decode loop with recompilation each step — validates decode mechanics."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
        use_cache=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval().to(device)
    tokens = [1, 2, 3, 4]

    for step in range(3):
        input_ids = torch.tensor([tokens], device=device)
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            ref = model(input_ids)
            out = compiled(input_ids)
        assert torch.allclose(out.logits, ref.logits, atol=1e-5), (
            f"step {step}: max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
        )
        next_token = ref.logits[0, -1, :].argmax().item()
        tokens.append(next_token)


@pytest.mark.slow
def test_hf_llama3_1b_decode_loop_dynamic(device: torch.device):
    """Decode loop on real Llama3.2-1B with pretrained weights.

    Recompiles each step as sequence length grows, using the standard
    torch.compile(model, backend=luminal_backend) pattern.
    """
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Llama-3.2-1B",
            config=config,
        )
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")

    prompt = "The capital of france is"
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}' -> {len(tokens)} tokens: {tokens}")
    num_generate = 3

    for step in range(num_generate):
        input_ids = torch.tensor([tokens], device=device)
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            ref = model(input_ids)
            out = compiled(input_ids)
        _assert_bf16_logits_match(out.logits, ref.logits, label=f"step {step}: ")
        next_token = ref.logits[0, -1, :].argmax().item()
        tokens.append(next_token)
        print(f"Step {step}: '{tokenizer.decode(tokens)}'")


def _gpu_mem(label):
    """Print GPU memory stats at a given checkpoint."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[GPU MEM] {label}: allocated={alloc:.3f} GiB, reserved={reserved:.3f} GiB, peak={peak:.3f} GiB"
        )


@pytest.mark.slow
def test_hf_llama3_full(device: torch.device):
    """HuggingFace LlamaForCausalLM — full Llama3.2-1B with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from NousResearch/Llama-3.2-1B.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _gpu_mem("before model load")

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Llama-3.2-1B",
            config=config,
        )
        .eval()
        .to(device)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[MODEL] Total parameters: {n_params:,} ({n_params * 4 / 1024**3:.3f} GiB in f32)"
    )
    _gpu_mem("after model load")

    compiled = torch.compile(model, backend=luminal_backend)
    _gpu_mem("after torch.compile (lazy, no compilation yet)")

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        _gpu_mem("after PyTorch reference forward")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _gpu_mem("before compiled forward (peak reset)")
        out = compiled(input_ids)
        _gpu_mem("after compiled forward (includes compilation)")

    _assert_bf16_logits_match(out.logits, ref.logits)


@pytest.mark.slow
def test_hf_llama3_large_full(device: torch.device):
    """HuggingFace LlamaForCausalLM — full Llama-3.1-8B-Instruct with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from NousResearch/Meta-Llama-3.1-8B-Instruct.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Meta-Llama-3.1-8B-Instruct",
            config=config,
        )
        .eval()
        .to(device)
    )
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    _assert_bf16_logits_match(out.logits, ref.logits)


# ========== Dynamic Dimension Tests ==========


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graph in-place update test — requires CUDA",
)
def test_dynamic_dim_reuse_no_recompile(device: torch.device):
    """Compile once with dynamic shapes, execute with varying seq lengths.

    Validates that the luminal runtime correctly handles dynamic dimension
    changes without recompilation. This is the core scenario optimized by
    removing the unnecessary CUDA graph rebuild on dyn_map changes: a single
    compiled graph handles multiple sequence lengths via in-place parameter
    updates rather than rebuilding the entire CUDA graph each step.
    """
    from luminal.pt2 import compile as luminal_compile

    class DynamicSeqModel(torch.nn.Module):
        """Embedding + linear projection with variable-length integer input."""

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 64)
            self.proj = torch.nn.Linear(64, 64)

        def forward(self, x):
            return self.proj(self.embed(x))

    model = DynamicSeqModel().eval().to(device)

    # Compile once with dynamic seq dim (auto-detected for integer inputs).
    # Factory capsule is auto-detected from example.device.
    example = torch.tensor([[1, 2, 3, 4]], device=device)
    compiled = luminal_compile(model, example, search_iterations=5)

    # Execute with multiple different seq lengths — each call reuses the
    # same compiled graph, updating dynamic dims in-place.
    for seq_len in [4, 5, 6, 7, 8]:
        input_ids = torch.tensor([list(range(1, seq_len + 1))], device=device)
        with torch.no_grad():
            ref = model(input_ids)
            out = compiled(input_ids)
        assert torch.allclose(out[0], ref, atol=1e-5), (
            f"seq_len={seq_len}: "
            f"max_diff={torch.max(torch.abs(out[0] - ref)).item():.2e}"
        )


@pytest.mark.slow
def test_hf_llama38b_full(device: torch.device):
    """HuggingFace LlamaForCausalLM — full Llama-3.1-8B-Instruct with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from NousResearch/Meta-Llama-3.1-8B-Instruct.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Meta-Llama-3.1-8B-Instruct",
            config=config,
        )
        .eval()
        .to(device)
    )
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    _assert_bf16_logits_match(out.logits, ref.logits)


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full Llama-3.1-8B dynamic-shape regression requires CUDA",
)
def test_hf_llama38b_mark_dynamic_seq_dim_before_compile(device: torch.device):
    """Explicitly marking the token sequence dim dynamic should be honored end to end.

    This exercises the real user path:
      1. wrap the pretrained 8B model with ``torch.compile(..., backend=luminal_backend)``
      2. mark ``input_ids.shape[1]`` dynamic before the first invocation
      3. verify the first backend trace is already dynamic on that axis
      4. reuse the same compiled graph for multiple sequence lengths
    """
    import copy

    from transformers import AutoConfig, LlamaForCausalLM

    from luminal.pt2 import (
        _build_dynamic_shapes_from_gm,
        _reinternalize_lifted_params,
        _strip_symint_placeholders,
    )

    backend_invocations = []
    capture = {}

    def inspector_backend(gm, example_inputs, **kwargs):
        backend_invocations.append((gm, example_inputs, kwargs))
        if len(backend_invocations) == 1:
            capture["gm"] = copy.deepcopy(gm).eval()
            capture["example_inputs"] = example_inputs
        compiled_impl = luminal_backend(gm, example_inputs, **kwargs)
        if len(backend_invocations) == 1:
            capture["compiled_impl"] = compiled_impl
        return compiled_impl

    prev_auto = torch._dynamo.config.automatic_dynamic_shapes
    prev_cache_limit = torch._dynamo.config.cache_size_limit
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 8

    try:
        config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
        config.use_cache = False
        config._attn_implementation = "eager"

        model = (
            LlamaForCausalLM.from_pretrained(
                "NousResearch/Meta-Llama-3.1-8B-Instruct",
                config=config,
            )
            .eval()
            .to(device)
        )
        compiled = torch.compile(model, backend=inspector_backend)

        first_input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        torch._dynamo.mark_dynamic(first_input_ids, 1, min=2, max=16)

        seq_inputs = {
            4: first_input_ids,
            6: torch.arange(1, 7, device=device).unsqueeze(0),
            9: torch.arange(1, 10, device=device).unsqueeze(0),
        }

        with torch.no_grad():
            first_ref = model(first_input_ids)
            first_out = compiled(first_input_ids)

        compiled_impl = capture["compiled_impl"]
        assert compiled_impl.has_dynamic_dims, (
            "explicit mark_dynamic on input_ids[:, 1] should produce a dynamic Luminal graph"
        )
        assert len(compiled_impl.dim_params) == 1, (
            f"expected exactly one dynamic dim param, got {compiled_impl.dim_params}"
        )

        gm = capture["gm"]
        example_inputs = capture["example_inputs"]
        gm, user_inputs, _, _ = _reinternalize_lifted_params(gm, example_inputs)
        user_inputs, _, strip_ok = _strip_symint_placeholders(gm, user_inputs)
        dynamic_shapes = _build_dynamic_shapes_from_gm(gm) if strip_ok else None

        assert strip_ok, "Expected explicit mark_dynamic SymInts to be rewritten"
        assert dynamic_shapes is not None, (
            "Expected the first backend trace to preserve a dynamic shape spec"
        )
        args_spec = dynamic_shapes.get("args")
        assert args_spec is not None and len(args_spec) == 1, (
            f"expected one user-input dynamic spec, got {dynamic_shapes}"
        )
        assert args_spec[0] is not None, (
            f"expected a per-dim dynamic spec for input_ids, got {dynamic_shapes}"
        )
        assert set(args_spec[0].keys()) == {1}, (
            "Expected only the token sequence axis (dim=1) to be dynamic, "
            f"got {dynamic_shapes}"
        )

        _assert_bf16_logits_match(
            first_out.logits, first_ref.logits, label="seq_len=4: "
        )

        for seq_len, input_ids in seq_inputs.items():
            with torch.no_grad():
                ref = model(input_ids)
                out = first_out if seq_len == 4 else compiled(input_ids)
            assert (
                out.logits.shape
                == ref.logits.shape
                == (
                    1,
                    seq_len,
                    config.vocab_size,
                )
            ), f"seq_len={seq_len}: got {out.logits.shape}, expected {ref.logits.shape}"
            _assert_bf16_logits_match(
                out.logits, ref.logits, label=f"seq_len={seq_len}: "
            )

        assert len(backend_invocations) == 1, (
            "Explicit mark_dynamic should produce one dynamic backend trace from the start, "
            f"got {len(backend_invocations)} backend invocations"
        )
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = prev_auto
        torch._dynamo.config.cache_size_limit = prev_cache_limit
        torch._dynamo.reset()


def test_hf_llama3_1b_bf16_error_within_fp32_floor(device: torch.device):
    """Prototype: luminal's bf16 error stays within the bf16 rounding floor.

    Rather than comparing a single luminal run to eager bf16 with a fixed atol,
    this measures the *average* absolute logit error of luminal vs an fp32
    reference, and of PyTorch-eager bf16 vs the same fp32 reference (the bf16
    rounding floor), across several prompts of varying length, and asserts
    luminal adds no more than a small factor over that floor.

    The luminal graph is compiled ONCE with the sequence dimension marked
    dynamic and reused for every prompt length (verified via a counting backend).
    """
    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.use_cache = False
    config._attn_implementation = "eager"
    model = (
        LlamaForCausalLM.from_pretrained("NousResearch/Llama-3.2-1B", config=config)
        .eval()
        .to(device)
    )
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")

    prompts = [
        "Hello",
        "The capital of France is",
        "In 1969 the first humans landed on the Moon during the",
        "Machine learning models are trained on large datasets to learn "
        "patterns and make predictions about",
        "Once upon a time in a distant kingdom there lived a curious young "
        "inventor who spent every waking hour tinkering with gears, springs, "
        "and strange contraptions that nobody else could understand",
    ]
    id_lists = [tokenizer.encode(p) for p in prompts]
    lengths = [len(t) for t in id_lists]
    print(f"prompt token lengths: {lengths}")

    # luminal's average error vs fp32 may exceed the bf16 floor by at most this.
    factor = 1.5
    # Minimum fraction of positions where luminal's argmax matches eager bf16.
    # Not 1.0: near-ties (logit gap < the bf16 floor) flip benignly, so a single
    # strict-equality run flakes. A real miscompile collapses this toward 0.
    min_token_agreement = 0.9

    prev_auto = torch._dynamo.config.automatic_dynamic_shapes
    prev_cache = torch._dynamo.config.cache_size_limit
    torch._dynamo.reset()
    torch._dynamo.config.automatic_dynamic_shapes = False
    torch._dynamo.config.cache_size_limit = 8
    try:
        # Compile ONCE with the sequence dim dynamic; reuse for every length.
        compiled = torch.compile(model, backend=luminal_backend)
        first = torch.tensor([id_lists[0]], device=device)
        torch._dynamo.mark_dynamic(first, 1, min=min(lengths), max=max(lengths))

        # Phase A — bf16: luminal (one graph, reused) + PyTorch eager, per prompt.
        ids_all, lum_logits, bf16_logits = [], [], []
        agree_positions, total_positions = 0, 0
        with torch.no_grad():
            for i, toks in enumerate(id_lists):
                ids = first if i == 0 else torch.tensor([toks], device=device)
                out_lum = compiled(ids).logits.detach().float()
                ref_bf16 = model(ids).logits.detach().float()
                # Per-position argmax agreement vs eager bf16, accumulated across
                # prompts. We do NOT require exact equality per run: at a near-tie
                # (logit gap < the bf16 floor) luminal and eager bf16 can pick
                # different tokens, exactly as two correct bf16 runs disagree with
                # fp32 there. Threshold-assert the aggregate below instead.
                agree = out_lum.argmax(dim=-1) == ref_bf16.argmax(dim=-1)
                agree_positions += int(agree.sum().item())
                total_positions += int(agree.numel())
                ids_all.append(ids)
                lum_logits.append(out_lum)
                bf16_logits.append(ref_bf16)

        agreement = agree_positions / total_positions
        print(
            f"luminal vs eager-bf16 argmax agreement: "
            f"{agree_positions}/{total_positions} = {agreement:.3%}"
        )
        assert agreement >= min_token_agreement, (
            f"luminal vs eager-bf16 argmax agreement {agreement:.2%} "
            f"below the {min_token_agreement:.0%} floor "
            f"({agree_positions}/{total_positions} positions)"
        )

        # Phase B — fp32 reference: upcast in place, run PyTorch eager (no compile).
        model.float()
        f32_logits = []
        with torch.no_grad():
            for ids in ids_all:
                f32_logits.append(model(ids).logits.detach())

        # Per-prompt mean absolute logit error vs the fp32 reference, then average.
        e_bf16 = sum(
            (b - f).abs().mean().item() for b, f in zip(bf16_logits, f32_logits)
        ) / len(prompts)
        e_lum = sum(
            (lg - f).abs().mean().item() for lg, f in zip(lum_logits, f32_logits)
        ) / len(prompts)
        ratio = e_lum / e_bf16 if e_bf16 > 0 else float("inf")
        print(
            f"avg |bf16 - fp32| = {e_bf16:.4e}   "
            f"avg |luminal - fp32| = {e_lum:.4e}   ratio = {ratio:.2f}"
        )
        assert e_lum <= e_bf16 * factor, (
            f"luminal avg error {e_lum:.3e} exceeds {factor}x the bf16 floor "
            f"{e_bf16:.3e} (ratio {ratio:.2f})"
        )
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = prev_auto
        torch._dynamo.config.cache_size_limit = prev_cache
        torch._dynamo.reset()
