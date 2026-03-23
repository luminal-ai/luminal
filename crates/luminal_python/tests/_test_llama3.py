"""Llama3 component and HuggingFace model integration tests.

Tests individual Llama3 building blocks (RMSNorm, RoPE, SwiGLU, causal attention,
full transformer block) and progressively larger HuggingFace LlamaForCausalLM configs
through the PyTorch -> ONNX -> luminal pipeline via torch.compile.
"""

from typing import Callable

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
    model: torch.nn.Module = LlamaTransformerBlockModel().to(device)
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)
    x: torch.Tensor = torch.rand((1, 4, 32), device=device)
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)
    assert torch.allclose(output, original, atol=1e-4)


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
    _run_hf_llama_test(config, device, atol=1e-4)


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
    _run_hf_llama_test(config, device, atol=1e-4)


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
    _run_hf_llama_test(config, device, atol=1e-3)


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
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
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
        assert torch.allclose(out.logits, ref.logits, atol=1e-4), (
            f"step {step}: max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
        )
        next_token = ref.logits[0, -1, :].argmax().item()
        tokens.append(next_token)


def test_hf_llama3_1b_decode_loop_dynamic():
    """Decode loop with dynamic shapes on real Llama3.2-1B — compile once, run with varying seq_len.

    This is the end-goal test: full 1B model with pretrained weights, CUDA backend,
    ONNX exported once with dynamic_axes for seq_len, then decoded autoregressively
    without recompilation.
    """
    import os
    import tempfile

    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

    import luminal

    backend = os.environ.get("LUMINAL_BACKEND", "cuda")

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.use_cache = False
    config._attn_implementation = "eager"
    print("Loaded config")
    model = LlamaForCausalLM.from_pretrained(
        "NousResearch/Llama-3.2-1B",
        config=config,
        torch_dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
    print("Loaded Model")
    # Export ONNX once with dynamic seq_len
    dummy = torch.tensor([[1, 2, 3, 4]])
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        torch.onnx.export(
            model,
            (dummy,),
            tmp_path,
            opset_version=20,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {1: "seq_len"}, "logits": {1: "seq_len"}},
        )
        print("Exported onnx")
        graph = luminal.process_onnx(tmp_path, backend)
    finally:
        os.unlink(tmp_path)
    print("Exported Model")
    assert graph.has_dynamic_dims, "Graph should have dynamic dims"
    assert "seq_len" in graph.dim_params, f"Expected 'seq_len' in {graph.dim_params}"

    prompt = "The capital of france is"
    tokens = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}' -> {len(tokens)} tokens: {tokens}")

    num_generate = 3
    for step in range(num_generate):
        seq_len = len(tokens)
        graph.set_dim("seq_len", seq_len)

        graph.set_input("input_ids", [float(t) for t in tokens])
        graph.run()

        output_shapes = graph.resolve_output_shapes()
        logits_data = graph.get_output("logits")
        logits = torch.tensor(logits_data, dtype=torch.float32).reshape(
            output_shapes[0]
        )

        # Compare against PyTorch reference
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            ref = model(input_ids)

        assert torch.allclose(logits, ref.logits, atol=1e-3), (
            f"step {step}: max_diff={torch.max(torch.abs(logits - ref.logits)).item():.2e}"
        )

        next_token = ref.logits[0, -1, :].argmax().item()
        tokens.append(next_token)
        print(f"Step {step}: '{tokenizer.decode(tokens)}'")


def test_hf_llama3_full(device: torch.device):
    """HuggingFace LlamaForCausalLM — full Llama3.2-1B with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from NousResearch/Llama-3.2-1B.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Llama-3.2-1B",
            config=config,
            torch_dtype=torch.float32,
        )
        .eval()
        .to(device)
    )
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_llama38b_full(device: torch.device):
    """HuggingFace LlamaForCausalLM — full Llama3.2-1B with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from NousResearch/Llama-3.2-1B.
    """
    from transformers import AutoConfig, LlamaForCausalLM

    config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        LlamaForCausalLM.from_pretrained(
            "NousResearch/Meta-Llama-3.1-8B-Instruct",
            config=config,
            torch_dtype=torch.float32,
        )
        .eval()
        .to(device)
    )
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_llama38b_cached_onnx():
    """Llama 3.1-8B via pre-generated ONNX + reference logits.

    Requires artifacts generated by:
        uv run python tests/generate_llama38b_artifacts.py
    """
    import os
    from pathlib import Path

    import luminal

    backend = os.environ.get("LUMINAL_BACKEND", "cuda")

    tests_dir = Path(__file__).resolve().parent
    onnx_path = tests_dir / "llama38b.onnx"
    logits_path = tests_dir / "llama38b_ref_logits.pt"

    assert onnx_path.exists(), (
        f"{onnx_path} not found. Run: uv run python tests/generate_llama38b_artifacts.py"
    )
    assert logits_path.exists(), (
        f"{logits_path} not found. Run: uv run python tests/generate_llama38b_artifacts.py"
    )

    ref_logits = torch.load(logits_path, weights_only=True)
    print(f"Loaded reference logits: {ref_logits.shape}")

    graph = luminal.process_onnx(str(onnx_path), backend)
    print("Compiled luminal graph")

    graph.set_input("input_ids", [float(t) for t in [1, 2, 3, 4]])
    graph.run()

    logits_data = graph.get_output("logits")
    logits_shape = graph.output_shapes[0]
    logits = torch.tensor(logits_data, dtype=torch.float32).reshape(logits_shape)
    print(f"Output logits shape: {logits.shape}")

    assert torch.allclose(logits, ref_logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(logits - ref_logits)).item():.2e}"
    )
