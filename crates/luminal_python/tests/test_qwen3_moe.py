"""Qwen3-MoE HuggingFace model integration tests.

Tests progressively larger HuggingFace `Qwen3MoeForCausalLM` configs through
the PyTorch -> PT2 -> luminal pipeline via `torch.compile(..., backend=
luminal_backend)`. Qwen3-MoE shares the dense Qwen3 backbone but replaces
the FFN with a top-k router over `num_experts` independent expert MLPs —
which exercises code paths the dense tests don't:

  - `aten._grouped_mm.default`  (gather-then-matmul lowering, PR #298)
  - bf16 `KernelScatter`        (KV cache scatter on a non-F32 dtype)
  - `aten.empty_permuted` / `aten.histc` (MoE expert dispatch and
                                          tokens-per-expert counts)
  - clamp-on-Int dtype handling (router top-k indices flowing into
                                 `aten.clamp`)

The smaller configs run on GPU in seconds; the "real config" case loads
the actual `Qwen/Qwen3-30B-A3B` arch (128 experts, top-8) with
`num_hidden_layers` overridden to 1 so a full-width compile is
exercised on random weights.

Together these guard the regression-and-fix story that landed alongside:
the bf16 KernelScatter dtype-aware vec count, the `aten.empty(_permuted)`
/ `aten.histc` translator entries, and the
`maximum_f32`-on-Int casting fix.
"""

import torch
import torch._dynamo

from luminal import luminal_backend


# ────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────

def _make_qwen3_moe_config(
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_hidden_layers: int,
    intermediate_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    vocab_size: int,
):
    """Create a Qwen3MoeConfig with use_cache=False and eager attention.

    Shared helper so each test only specifies the scaling knobs that matter
    for that case.
    """
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        vocab_size=vocab_size,
        max_position_embeddings=128,
        use_cache=False,
        attn_implementation="eager",
    )


def _run_hf_qwen3_moe_test(config, device: torch.device, atol: float):
    """Run a HuggingFace Qwen3MoeForCausalLM test with the given config.

    Compiles the model with `luminal_backend`, runs both eager and compiled
    on the same input, asserts the logits match within `atol`.
    """
    from transformers import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=atol), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )


# ────────────────────────────────────────────────────────────────────────
#  Tests — progressively larger configs
# ────────────────────────────────────────────────────────────────────────

def test_hf_qwen3_moe_tiny(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — tiny: 2 experts, top-1 routing.

    Smallest config that still exercises the MoE expert dispatch
    (`aten._grouped_mm`). Top-1 routing keeps the test simple while still
    validating the gather-then-matmul lowering path.
    """
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
    _run_hf_qwen3_moe_test(config, device, atol=1e-5)


def test_hf_qwen3_moe_small(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — small: 4 experts, top-2 routing."""
    config = _make_qwen3_moe_config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=512,
    )
    _run_hf_qwen3_moe_test(config, device, atol=1e-4)


def test_hf_qwen3_moe_medium(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — medium: 8 experts, top-2, 2 layers.

    Two layers means the e-graph crosses a layer boundary, which is where
    the late-memory-analysis cleanup pass operates differently than
    single-layer cases.
    """
    config = _make_qwen3_moe_config(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=512,
    )
    _run_hf_qwen3_moe_test(config, device, atol=1e-4)


def test_hf_qwen3_moe_real_config_1layer(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — real Qwen3-30B-A3B architecture, 1 layer.

    Loads `Qwen/Qwen3-30B-A3B`'s AutoConfig (128 experts, top-8 routing,
    2048 hidden) and overrides `num_hidden_layers=1`. Random weights —
    tests that the production-shape MoE layer compiles end-to-end through
    luminal_backend.

    This is the regression test for the rust-side `qwen3_moe` example
    panic that was visible during 2026-05-07 on the same compile path.
    """
    from transformers import AutoConfig, Qwen3MoeForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    config.num_hidden_layers = 1
    config.use_cache = False
    config._attn_implementation = "eager"

    model = Qwen3MoeForCausalLM(config).eval().to(device)
    compiled = torch.compile(model, backend=luminal_backend)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids)
        out = compiled(input_ids)
    assert torch.allclose(out.logits, ref.logits, atol=1e-3), (
        f"max_diff={torch.max(torch.abs(out.logits - ref.logits)).item():.2e}"
    )
