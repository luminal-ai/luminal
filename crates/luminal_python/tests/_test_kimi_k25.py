"""Kimi-K2.5 / DeepseekV3 model integration tests.

Tests the DeepseekV3 text backbone (MoE + MLA attention with LoRA-compressed KV,
SwiGLU, YaRN RoPE) through the PyTorch -> ONNX -> luminal pipeline.

The model code requires trust_remote_code=True and uses custom HF modules from
moonshotai/Kimi-K2.5. Since torch.compile cannot trace the MoE routing (it uses
.numpy() and tensor indexing incompatible with dynamo), tests use manual ONNX
export + onnxsim simplification + luminal.process_onnx.
"""

import os
import tempfile
import warnings

import onnx
import onnxsim
import pytest
import torch

warnings.filterwarnings("ignore")


def _get_deepseek_v3_classes():
    """Import DeepseekV3Config and DeepseekV3ForCausalLM from the Kimi-K2.5 HF repo."""
    import importlib

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("moonshotai/Kimi-K2.5", trust_remote_code=True)
    tc = config.text_config
    DeepseekV3Config = type(tc)
    pkg = DeepseekV3Config.__module__.rsplit(".", 1)[0]
    modeling_mod = importlib.import_module(f"{pkg}.modeling_deepseek")
    return DeepseekV3Config, modeling_mod.DeepseekV3ForCausalLM


def _make_deepseek_v3_config(
    DeepseekV3Config,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    num_hidden_layers: int = 1,
    intermediate_size: int = 128,
    vocab_size: int = 256,
    kv_lora_rank: int = 16,
    q_lora_rank: int = 32,
    qk_nope_head_dim: int = 8,
    qk_rope_head_dim: int = 8,
    v_head_dim: int = 8,
    n_routed_experts: int = 4,
    num_experts_per_tok: int = 2,
    n_shared_experts: int = 1,
    moe_intermediate_size: int = 32,
    first_k_dense_replace: int = 1,
):
    """Create a small DeepseekV3Config for testing."""
    config = DeepseekV3Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=128,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        n_shared_experts=n_shared_experts,
        moe_intermediate_size=moe_intermediate_size,
        first_k_dense_replace=first_k_dense_replace,
        use_cache=False,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        rope_scaling={
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 32,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "rope_theta": 10000.0,
        },
        rope_theta=10000.0,
    )
    config._attn_implementation = "eager"
    return config


def _export_and_simplify(model, input_ids):
    """Export model to ONNX and simplify with onnxsim to constant-fold shape chains."""
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        torch.onnx.export(
            model,
            (input_ids,),
            tmp_path,
            opset_version=20,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamo=False,
        )
        m = onnx.load(tmp_path)
        m_sim, check = onnxsim.simplify(m)
        assert check, "onnxsim simplification failed"
        onnx.save(m_sim, tmp_path)
        return tmp_path
    except Exception:
        os.unlink(tmp_path)
        raise


def _run_deepseek_v3_test(config, DeepseekV3ForCausalLM, backend: str, atol: float):
    """Export DeepseekV3 to ONNX, simplify, run through luminal, compare."""
    import luminal

    model = DeepseekV3ForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    onnx_path = _export_and_simplify(model, input_ids)
    try:
        graph = luminal.process_onnx(onnx_path, backend)
        graph.set_input("input_ids", [1.0, 2.0, 3.0, 4.0])
        graph.run()
        logits_data = graph.get_output("logits")
        logits = torch.tensor(logits_data, dtype=torch.float32).reshape(
            1, 4, config.vocab_size
        )
    finally:
        os.unlink(onnx_path)

    with torch.no_grad():
        ref = model(input_ids)

    assert torch.allclose(logits, ref.logits, atol=atol), (
        f"max_diff={torch.max(torch.abs(logits - ref.logits)).item():.2e}"
    )


# ========== Tests ==========


def test_deepseek_v3_tiny_dense():
    """Tiny DeepseekV3 with dense MLP (no MoE): 64 hidden, 1 layer, MLA attention."""
    DeepseekV3Config, DeepseekV3ForCausalLM = _get_deepseek_v3_classes()
    config = _make_deepseek_v3_config(
        DeepseekV3Config,
        first_k_dense_replace=1,  # all layers use dense MLP
    )
    backend = os.environ.get("LUMINAL_BACKEND", "native")
    _run_deepseek_v3_test(config, DeepseekV3ForCausalLM, backend, atol=1e-5)


@pytest.mark.xfail(reason="MoE routing uses Int/F32 mixed ops not yet supported")
def test_deepseek_v3_tiny_moe():
    """Tiny DeepseekV3 with MoE: 64 hidden, 1 layer, 4 routed experts + 1 shared."""
    DeepseekV3Config, DeepseekV3ForCausalLM = _get_deepseek_v3_classes()
    config = _make_deepseek_v3_config(
        DeepseekV3Config,
        first_k_dense_replace=0,  # all layers use MoE
    )
    backend = os.environ.get("LUMINAL_BACKEND", "native")
    _run_deepseek_v3_test(config, DeepseekV3ForCausalLM, backend, atol=1e-5)


def test_deepseek_v3_small_dense():
    """Small DeepseekV3 with dense MLP: 256 hidden, 1 layer."""
    DeepseekV3Config, DeepseekV3ForCausalLM = _get_deepseek_v3_classes()
    config = _make_deepseek_v3_config(
        DeepseekV3Config,
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=512,
        vocab_size=1024,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        first_k_dense_replace=1,
    )
    backend = os.environ.get("LUMINAL_BACKEND", "native")
    _run_deepseek_v3_test(config, DeepseekV3ForCausalLM, backend, atol=1e-4)
