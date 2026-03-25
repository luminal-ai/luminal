"""Qwen3-8B HuggingFace model integration tests.

Tests progressively larger HuggingFace Qwen3ForCausalLM configs through the
PyTorch -> ONNX -> luminal pipeline via torch.compile. Qwen3 shares the same
architecture family as Llama (GQA, RoPE, SwiGLU MLP, RMSNorm).
"""

from typing import Callable

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
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=256,
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


def test_hf_qwen3_decode_loop_dynamic():
    """Decode loop with dynamic shapes -- compile once, run with varying seq_len.

    Bypasses torch.compile to use luminal's dynamic dim support directly.
    Exports ONNX once with dynamic_axes, then calls set_dim/set_input/run/get_output.
    """
    import os
    import tempfile

    from transformers import Qwen3Config, Qwen3ForCausalLM

    import luminal

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
    model = Qwen3ForCausalLM(config).eval()

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

        graph = luminal.process_onnx(tmp_path, "native")
    finally:
        os.unlink(tmp_path)

    assert graph.has_dynamic_dims, "Graph should have dynamic dims"
    assert "seq_len" in graph.dim_params, f"Expected 'seq_len' in {graph.dim_params}"

    tokens = [1, 2, 3, 4]
    for step in range(3):
        seq_len = len(tokens)
        graph.set_dim("seq_len", seq_len)

        # Set input as float (luminal works with f32 internally)
        graph.set_input("input_ids", [float(t) for t in tokens])
        graph.run()

        # Get output and reshape using resolved shapes
        output_shapes = graph.resolve_output_shapes()
        logits_data = graph.get_output("logits")
        logits = torch.tensor(logits_data, dtype=torch.float32).reshape(
            output_shapes[0]
        )

        # Compare against PyTorch reference
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            ref = model(input_ids)

        assert torch.allclose(logits, ref.logits, atol=1e-4), (
            f"step {step}: max_diff={torch.max(torch.abs(logits - ref.logits)).item():.2e}"
        )

        next_token = ref.logits[0, -1, :].argmax().item()
        tokens.append(next_token)


def test_hf_qwen3_8b_decode_loop_dynamic():
    """Decode loop with dynamic shapes on real Qwen3-8B -- compile once, run with varying seq_len.

    Full 8B model with pretrained weights, ONNX exported once with dynamic_axes
    for seq_len, then decoded autoregressively without recompilation.
    """
    import os
    import tempfile

    from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

    import luminal

    backend = os.environ.get("LUMINAL_BACKEND", "cuda")

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    config.use_cache = False
    config._attn_implementation = "eager"
    print("Loaded config")
    model = Qwen3ForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        config=config,
        torch_dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
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
