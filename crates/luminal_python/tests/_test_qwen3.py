"""Qwen3-8B HuggingFace model integration tests.

Tests progressively larger HuggingFace Qwen3ForCausalLM configs through the
PyTorch -> PT2 -> luminal pipeline via torch.compile. Qwen3 shares the same
architecture family as Llama (GQA, RoPE, SwiGLU MLP, RMSNorm).
"""

import torch
import torch._dynamo

from luminal import luminal_backend

# ========== HuggingFace Qwen3ForCausalLM Tests ==========


def _make_qwen3_config(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    vocab_size: int,
):
    """Create a Qwen3Config with use_cache=False and eager attention."""
    from transformers import Qwen3Config

    return Qwen3Config(
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


def _run_hf_qwen3_test(config, device: torch.device, atol: float):
    """Run a HuggingFace Qwen3ForCausalLM test with the given config."""
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=atol), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_qwen3_tiny(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- tiny (64 hidden, 1 layer, ~70K params)."""
    config = _make_qwen3_config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        intermediate_size=64,
        vocab_size=128,
    )
    _run_hf_qwen3_test(config, device, atol=1e-5)


def test_hf_qwen3_small(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- small (256 hidden, 1 layer, ~1.1M params)."""
    config = _make_qwen3_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=1,
        intermediate_size=512,
        vocab_size=1024,
    )
    _run_hf_qwen3_test(config, device, atol=1e-4)


def test_hf_qwen3_medium(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- medium (256 hidden, 2 layers, ~1.7M params)."""
    config = _make_qwen3_config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=512,
        vocab_size=1024,
    )
    _run_hf_qwen3_test(config, device, atol=1e-4)


def test_hf_qwen3_large(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- large (1024 hidden, 1 layer, ~18M params)."""
    config = _make_qwen3_config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=1,
        intermediate_size=2048,
        vocab_size=4096,
    )
    _run_hf_qwen3_test(config, device, atol=1e-3)


def test_hf_qwen3_real_config_1layer(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- real Qwen3-8B architecture, 1 layer.

    Uses AutoConfig.from_pretrained with num_hidden_layers overridden to 1.
    Random weights -- tests the full-width architecture compiles correctly.
    """
    from transformers import AutoConfig, Qwen3ForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    config.num_hidden_layers = 1
    config.use_cache = False
    config._attn_implementation = "eager"

    model = Qwen3ForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


def test_hf_qwen3_decode_loop_static(device: torch.device):
    """Decode loop with recompilation each step -- validates decode mechanics."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
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
    model = Qwen3ForCausalLM(config).eval().to(device)
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


def test_hf_qwen3_8b_full(device: torch.device):
    """HuggingFace Qwen3ForCausalLM -- full Qwen3-8B with real pretrained weights.

    No config alterations except use_cache=False and eager attention.
    Loads actual weights from Qwen/Qwen3-8B.
    """
    from transformers import AutoConfig, Qwen3ForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        Qwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
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
