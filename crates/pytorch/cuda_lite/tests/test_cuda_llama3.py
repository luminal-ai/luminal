import pytest
import torch
import torch._dynamo
from luminal_cuda_lite import Compiler
from reference.tests.models.test_llama3 import _assert_bf16_logits_match

luminal_backend = Compiler()


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
    from backend_test_utils import compile_for_test as luminal_compile

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

    from luminal_reference.export_helpers import (
        _build_dynamic_shapes_from_gm,
        _strip_symint_placeholders,
    )
    from transformers import AutoConfig, LlamaForCausalLM

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
        user_inputs = list(example_inputs)
        user_inputs, _, strip_ok = _strip_symint_placeholders(gm, user_inputs)
        dynamic_shapes = _build_dynamic_shapes_from_gm(gm) if strip_ok else None

        assert strip_ok, "Expected explicit mark_dynamic SymInts to be rewritten"
        assert dynamic_shapes is not None, (
            "Expected the first backend trace to preserve a dynamic shape spec"
        )
        args_spec = dynamic_shapes.get("args")
        assert args_spec is not None, f"expected an args spec, got {dynamic_shapes}"
        # Weights flow as inputs, so the spec covers every arg; exactly one
        # (input_ids) may be dynamic, and only on its sequence axis.
        dyn_specs = [spec for spec in args_spec if spec is not None]
        assert len(dyn_specs) == 1, (
            f"expected exactly one dynamic input, got {dynamic_shapes}"
        )
        assert set(dyn_specs[0].keys()) == {1}, (
            "Expected only the token sequence axis (dim=1) to be dynamic, "
            f"got {dyn_specs[0]}"
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
