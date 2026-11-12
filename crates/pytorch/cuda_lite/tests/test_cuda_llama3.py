import pytest
import torch
import torch._dynamo
from luminal_cuda_lite import Compiler
from reference.tests.models.test_llama3 import _assert_bf16_logits_match


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graph in-place update test — requires CUDA",
)
def test_dynamic_dim_reuse_no_recompile(device: torch.device):
    """Compile once over a declared sequence domain, execute at varying lengths.

    Validates that the luminal runtime correctly handles dynamic dimension
    changes without recompilation. This is the core scenario optimized by
    removing the unnecessary CUDA graph rebuild on dyn_map changes: a single
    compiled graph handles multiple sequence lengths via in-place parameter
    updates rather than rebuilding the entire CUDA graph each step.
    """
    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    class DynamicSeqModel(torch.nn.Module):
        """Embedding + linear projection with variable-length integer input."""

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(256, 64)
            self.proj = torch.nn.Linear(64, 64)

        def forward(self, x):
            return self.proj(self.embed(x))

    torch._dynamo.reset()
    model = DynamicSeqModel().eval().to(device)

    # One compile over the whole declared domain: the sequence axis is a
    # ShapeVar, so every length in [2, 16] runs the same program.
    seq = ShapeVar("seq", min=2, max=16, optimization_hint=4)
    compiler = Compiler(search_iterations=5)
    compiled = torch.compile(
        model,
        backend=compiler,
        fullgraph=True,
        isolate_recompiles=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([1, seq])}),
    )

    # Execute with multiple different seq lengths — each call reuses the
    # same compiled graph, updating dynamic dims in-place.
    for seq_len in [4, 5, 6, 7, 8, 16, 2]:
        input_ids = torch.tensor([list(range(1, seq_len + 1))], device=device)
        with torch.no_grad():
            ref = model(input_ids)
            out = compiled(input_ids)
        assert torch.allclose(out, ref, atol=1e-5), (
            f"seq_len={seq_len}: max_diff={torch.max(torch.abs(out - ref)).item():.2e}"
        )
    assert len(compiler.graphs) == 1, "a new sequence length recompiled the program"
    assert list(compiler.graphs[0].bounds.values()) == [(2, 16)]
    torch._dynamo.reset()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Full Llama-3.1-8B dynamic-shape regression requires CUDA",
)
def test_hf_llama38b_mark_dynamic_seq_dim_before_compile(device: torch.device):
    """Explicitly marking the token sequence dim dynamic should be honored end to end.

    This exercises the real user path:
      1. wrap the pretrained 8B model with ``torch.compile(..., backend=Compiler())``
      2. mark ``input_ids.shape[1]`` dynamic before the first invocation
      3. verify the first backend trace is already dynamic on that axis
      4. reuse the same compiled graph for multiple sequence lengths
    """
    from transformers import AutoConfig, LlamaForCausalLM

    compiler = Compiler()
    backend_invocations = []
    first_trace_axes = []

    def inspector_backend(gm, example_inputs, **kwargs):
        backend_invocations.append((gm, example_inputs, kwargs))
        if len(backend_invocations) == 1:
            # The symbolic axes of each tensor argument of the first graph
            # Dynamo traced, read off the graph itself.
            for node in gm.graph.nodes:
                value = node.meta.get("example_value")
                if node.op != "placeholder" or not isinstance(value, torch.Tensor):
                    continue
                symbolic = {
                    axis
                    for axis, size in enumerate(value.shape)
                    if not isinstance(size, int)
                }
                if symbolic:
                    first_trace_axes.append(symbolic)
        return compiler(gm, example_inputs, **kwargs)

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

        # Exactly one argument (input_ids) is dynamic, and only on its
        # sequence axis, from the first trace on.
        assert first_trace_axes == [{1}], (
            "Expected only the token sequence axis (dim=1) to be dynamic, "
            f"got {first_trace_axes}"
        )
        # The compiled program carries that one dimension over its
        # declared range.
        assert list(compiler.graphs[0].bounds.values()) == [(2, 16)], (
            f"expected one dynamic dimension over [2, 16], got "
            f"{compiler.graphs[0].bounds}"
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
