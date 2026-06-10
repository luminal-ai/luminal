"""Regression test for the cublaslt_mixed_dtype bf16 miscompile.

Compiling the same bf16 Llama layer repeatedly must produce a correct result
EVERY time. Before the fix (mixed_dtype firing on beta!=0 matmuls), ~60-80% of
compiles were badly wrong because the perf-driven extraction search would pick
the unsound F32-output cuBLASLt candidate. This test fails if ANY compile breaks.

    LUMINAL_TEST_DEVICE=cuda uv run --group dev pytest tests/_bf16_nondeterminism.py -v -s
"""

import pytest
import torch
import torch._dynamo

from luminal import luminal_backend


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model(device):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=1,
        intermediate_size=11008,
        vocab_size=2048,
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
        head_dim=128,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval().to(device).to(torch.bfloat16)


@pytest.mark.slow
def test_nondeterminism():
    device = _device()
    if device.type != "cuda":
        pytest.skip("requires the CUDA backend")
    model = _model(device)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        ref = model(input_ids).logits.detach().clone()

    errs = []
    N = 20
    for i in range(N):
        torch._dynamo.reset()
        compiled = torch.compile(model, backend=luminal_backend)
        with torch.no_grad():
            out = compiled(input_ids).logits.detach().clone()
        err = (out.float() - ref.float()).abs().max().item()
        errs.append(err)
        # bf16 vs bf16 should agree to the rounding floor; a broken extraction
        # is off by O(1-10) or NaN.
        broken = err > 1 or err != err
        print(
            f"  compile {i:2d}: max_abs_err = {err:.3e}  {'BROKEN' if broken else 'ok'}"
        )

    n_broken = sum(1 for e in errs if e > 1 or e != e)
    assert n_broken == 0, (
        f"{n_broken}/{N} compiles produced a broken bf16 lowering "
        f"(min={min(errs):.3e} max={max(errs):.3e}) — cublaslt_mixed_dtype regressed"
    )
