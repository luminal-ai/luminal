import pytest
import torch
import torch._dynamo
from luminal_cuda_lite import Compiler
from reference.tests.models.test_qwen3_moe import (
    _make_qwen3_moe_config,
)

luminal_backend = Compiler()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bf16 grouped_mm coverage requires CUDA",
)
def test_hf_qwen3_moe_tiny_bf16(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — tiny bf16 path on CUDA.

    Exercises the grouped-mm MoE lowering with bf16 weights/activations so we
    catch mixed-dtype compile regressions without paying the full 30B checkpoint
    cost. Like the full pretrained bf16 test below, this only asserts that the
    compiled path runs and stays numerically sane; tight bf16 equivalence is
    tracked separately.
    """
    from transformers import Qwen3MoeForCausalLM

    config = _make_qwen3_moe_config(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_experts=2,
        num_experts_per_tok=1,
        vocab_size=128,
    )

    model = Qwen3MoeForCausalLM(config).eval().to(dtype=torch.bfloat16, device=device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)

    ref_logits = ref.logits.float()
    out_logits = out.logits.float()
    ref_max = ref_logits.abs().max().item()
    out_max = out_logits.abs().max().item()
    n_nan = int(out_logits.isnan().sum().item())
    n_inf = int(out_logits.isinf().sum().item())

    assert n_nan == 0 and n_inf == 0, (
        f"compiled forward produced non-finite logits: {n_nan} NaNs, "
        f"{n_inf} Infs (eager max abs={ref_max:.2f}, compiled max abs={out_max:.2f})"
    )
    assert 0.1 * ref_max <= out_max <= 10.0 * ref_max, (
        f"compiled max abs={out_max:.2f} is out of band vs eager max abs={ref_max:.2f} "
        f"(>10x off in either direction); likely a numerical/scale bug"
    )
