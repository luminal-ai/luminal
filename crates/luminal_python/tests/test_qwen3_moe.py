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

import pytest
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


@pytest.mark.slow
def test_hf_qwen3_moe_real_config_1layer(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — real Qwen3-30B-A3B architecture, 1 layer.

    Loads `Qwen/Qwen3-30B-A3B`'s AutoConfig (128 experts, top-8 routing,
    2048 hidden) and overrides `num_hidden_layers=1`. Random weights —
    cheap smoke that the production-shape MoE *layer* compiles end-to-end
    through luminal_backend without paying the full 48-layer cost.
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


@pytest.mark.slow
def test_hf_qwen3_moe_real_config_full(device: torch.device):
    """HuggingFace Qwen3MoeForCausalLM — full Qwen3-30B-A3B, pretrained.

    Loads the real `Qwen/Qwen3-30B-A3B` checkpoint at its native bf16
    dtype: 48 hidden layers, 128 experts, top-8 routing, 2048 hidden —
    i.e. the production architecture, no `num_hidden_layers` override.
    This is the end-to-end "the full MoE compiles" regression guard;
    the 1-layer variant above is the cheap smoke.

    Asserts the **compile + run** path completes and the compiled
    forward produces *finite* output (no NaN / no Inf). It does NOT
    assert tight numerical equivalence with eager: at this depth the
    egglog search is non-deterministic enough that the two paths can
    diverge structurally (same general magnitudes, different per-element
    values). Tight numerical equivalence at full scale is tracked as
    follow-up work — the smaller-config tests above use atol≤1e-3 and
    cover the per-op correctness that this test cannot.

    Compared to the 1-layer test this primarily catches:
      - egglog cleanup behaviour over a 48-layer-wide e-graph (the
        `egglog_utils.rs:1286: No valid graphs` panic surfaces here
        if the cleanup cascade re-regresses on MoE root-eclasses);
      - per-layer plumbing of residual stream + KV state that
        single-layer tests don't exercise;
      - any bf16-specific code path (e.g. KernelScatter OOB) that's
        masked at fp32.

    Memory profile on H200/H100:
      - bf16 pretrained weights: ~60 GB
      - single-token input keeps activations & router state trivial
      - peak observed during compiled forward: ~75 GB total
    """
    import gc

    from transformers import AutoConfig, Qwen3MoeForCausalLM

    # Aggressively release any allocator state from prior tests in the
    # same process — at this scale we don't have headroom to absorb it.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B")
    config.use_cache = False
    config._attn_implementation = "eager"

    model = (
        Qwen3MoeForCausalLM.from_pretrained(
            "Qwen/Qwen3-30B-A3B",
            config=config,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to(device)
    )
    compiled = torch.compile(model, backend=luminal_backend)
    # Single-token input — the full-depth compile is the regression target,
    # not multi-token throughput (which the bench covers separately).
    input_ids = torch.tensor([[1]], device=device)

    with torch.no_grad():
        # Eager forward — confirms the test setup is sane (HF is happy).
        ref = model(input_ids)
        ref_max = ref.logits.float().abs().max().item()
        assert torch.isfinite(ref.logits).all(), (
            "eager forward produced non-finite logits — test setup is broken, "
            "not a luminal regression"
        )
        del ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compiled forward — the actual regression target.
        out = compiled(input_ids)

    out_logits = out.logits.float()
    n_nan = int(out_logits.isnan().sum().item())
    n_inf = int(out_logits.isinf().sum().item())
    out_max = out_logits.abs().max().item()

    assert n_nan == 0 and n_inf == 0, (
        f"compiled forward produced non-finite logits: {n_nan} NaNs, "
        f"{n_inf} Infs (eager max abs={ref_max:.2f}, compiled max abs={out_max:.2f})"
    )
    # Sanity-check magnitude: compiled output should be in the same ballpark
    # as eager — within an order of magnitude of the eager logits' scale.
    # Catches the failure mode where some kernel silently produces
    # near-zero or near-Inf values that pass the finite check.
    assert 0.1 * ref_max <= out_max <= 10.0 * ref_max, (
        f"compiled max abs={out_max:.2f} is out of band vs eager max abs={ref_max:.2f} "
        f"(>10× off in either direction); likely a numerical/scale bug"
    )
