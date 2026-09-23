from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time

import pytest
import torch
from luminal_reference.artifact_cache import (
    CompiledArtifact,
)
from luminal_reference.region_compile import compile_region, load_region_artifact
from luminal_reference.region_export import export_region
from torch import fx


def _add_graph() -> fx.GraphModule:
    graph = fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    result = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output((result,))
    return fx.GraphModule(torch.nn.Module(), graph)


def _cuda_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        import luminal_cuda_lite

        luminal_cuda_lite.Compiler()
    except (ImportError, AttributeError, RuntimeError) as error:
        return f"luminal_cuda_lite is not available: {error}"
    return None


_CUDA_SKIP_REASON = _cuda_skip_reason()


def _cuda_sleep_ms(stream: torch.cuda.Stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        torch.cuda._sleep(1_000_000_000)
        end.record()
    end.synchronize()
    return start.elapsed_time(end)


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_non_static_writeback_accepts_changed_target():
    class Mutate(torch.nn.Module):
        def forward(self, target):
            target.add_(1)
            return (target,)

    module = fx.symbolic_trace(Mutate())
    inputs = [torch.zeros(2, device="cuda")]
    compiled = compile_region(export_region(module, inputs), device_type="cuda")
    first = torch.zeros(2, device="cuda")
    second = torch.zeros(2, device="cuda")
    compiled(first)
    compiled(second)
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.ones_like(second))


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_compile_region_from_fake_cuda_metadata() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    compiled = compile_region(region, device_type="cuda", search_iterations=1)

    real_inputs = [
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
    ]
    (actual,) = compiled(*real_inputs)
    torch.testing.assert_close(actual, real_inputs[0] + real_inputs[1])


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_region_artifact_round_trip_cuda() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    compiled = compile_region(region, device_type="cuda", search_iterations=1)
    artifact = compiled.serialize_artifact()
    payload = json.loads(artifact)
    assert payload["schema_version"] == CompiledArtifact.SCHEMA_VERSION
    assert payload["program"]
    loaded = load_region_artifact(
        artifact,
        input_indices=region.input_indices,
        output_spec=region.output_spec,
        device_index=region.device_index,
    )
    inputs = [torch.randn((2, 4), device="cuda", dtype=torch.float16) for _ in range(2)]

    (actual,) = loaded(*inputs)
    torch.testing.assert_close(actual, inputs[0] + inputs[1])


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_region_artifact_loads_in_fresh_process(tmp_path) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    artifact_path = tmp_path / "region.luminal"
    artifact_path.write_bytes(
        compile_region(
            region, device_type="cuda", search_iterations=1
        ).serialize_artifact()
    )
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import torch
        from torch import fx
        from torch._subclasses.fake_tensor import FakeTensorMode
        from luminal_reference.region_compile import load_region_artifact
        from luminal_reference.region_export import export_region

        graph = fx.Graph()
        left = graph.placeholder("left")
        right = graph.placeholder("right")
        result = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
        graph.output((result,))
        module = fx.GraphModule(torch.nn.Module(), graph)
        with FakeTensorMode():
            fake_inputs = [
                torch.empty((2, 4), device="cuda", dtype=torch.float16),
                torch.empty((2, 4), device="cuda", dtype=torch.float16),
            ]
            region = export_region(module, fake_inputs)
        model = load_region_artifact(
            Path(sys.argv[1]).read_bytes(),
            input_indices=region.input_indices,
            output_spec=region.output_spec,
            device_index=region.device_index,
        )
        inputs = [
            torch.randn((2, 4), device="cuda", dtype=torch.float16)
            for _ in range(2)
        ]
        (actual,) = model(*inputs)
        torch.testing.assert_close(actual, inputs[0] + inputs[1])
        """
    )

    subprocess.run(
        [sys.executable, "-c", script, str(artifact_path)],
        check=True,
        text=True,
    )


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_compile_region_uses_current_cuda_stream() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    compiled = compile_region(
        region, device_type="cuda", search_iterations=1, static_outputs=True
    )
    left = torch.empty((2, 4), device="cuda", dtype=torch.float16)
    right = torch.empty((2, 4), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()

    # Warm up one-time compilation and allocation work on this stream.
    with torch.cuda.stream(stream):
        (actual,) = compiled(left, right)
    stream.synchronize()

    # `_sleep` cycles do not have a portable wall-clock duration, so measure the
    # delay on this GPU before using it to test whether the host call blocks.
    sleep_ms = _cuda_sleep_ms(stream)
    assert sleep_ms > 10, f"CUDA delay is too short to test blocking: {sleep_ms} ms"

    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        left.fill_(1)
        right.fill_(2)
        call_start = time.perf_counter()
        (actual,) = compiled(left, right)
        call_ms = (time.perf_counter() - call_start) * 1_000

    assert call_ms < sleep_ms / 2, (
        "Luminal blocked on the borrowed CUDA stream "
        f"(call={call_ms:.3f} ms, queued_delay={sleep_ms:.3f} ms)"
    )
    stream.synchronize()
    torch.testing.assert_close(actual, torch.full_like(actual, 3))


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_compile_region_can_be_captured_by_pytorch() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(_add_graph(), fake_inputs)

    compiled = compile_region(
        region,
        search_iterations=1,
        static_outputs=True,
        external_cuda_graph=True,
    )
    left = torch.ones((2, 4), device="cuda", dtype=torch.float16)
    right = torch.full((2, 4), 2, device="cuda", dtype=torch.float16)

    # Prepare every Luminal resource before capture.
    compiled(left, right)
    torch.cuda.synchronize()

    # vLLM's capture buffers need not have the same addresses as warmup.
    capture_left = torch.full_like(left, 3)
    capture_right = torch.full_like(right, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (actual,) = compiled(capture_left, capture_right)

    capture_left.fill_(5)
    capture_right.fill_(6)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, torch.full_like(actual, 11))


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_static_writeback_uses_stable_target_on_current_stream() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    graph = fx.Graph()
    target = graph.placeholder("target")
    update = graph.placeholder("update")
    result = graph.call_function(torch.ops.aten.add_.Tensor, (target, update))
    graph.output(result)
    graph_module = fx.GraphModule(torch.nn.Module(), graph)

    with FakeTensorMode():
        fake_inputs = [
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
            torch.empty((2, 4), device="cuda", dtype=torch.float16),
        ]
        region = export_region(graph_module, fake_inputs)

    compiled = compile_region(
        region, device_type="cuda", search_iterations=1, static_outputs=True
    )
    target_value = torch.zeros((2, 4), device="cuda", dtype=torch.float16)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        compiled(target_value, torch.ones_like(target_value))
    stream.synchronize()
    torch.testing.assert_close(target_value, torch.ones_like(target_value))

    sleep_ms = _cuda_sleep_ms(stream)
    assert sleep_ms > 10, f"CUDA delay is too short to test blocking: {sleep_ms} ms"
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        call_start = time.perf_counter()
        compiled(target_value, torch.full_like(target_value, 2))
        call_ms = (time.perf_counter() - call_start) * 1_000

    assert call_ms < sleep_ms / 2, (
        "Luminal writeback blocked on the borrowed CUDA stream "
        f"(call={call_ms:.3f} ms, queued_delay={sleep_ms:.3f} ms)"
    )
    stream.synchronize()
    torch.testing.assert_close(target_value, torch.full_like(target_value, 3))

    with pytest.raises(ValueError, match="requires a contiguous CUDA tensor"):
        compiled(target_value.t(), torch.ones_like(target_value.t()))

    with pytest.raises(ValueError, match="target allocation changed"):
        compiled(torch.zeros_like(target_value), torch.ones_like(target_value))


@pytest.mark.skipif(
    _CUDA_SKIP_REASON is not None, reason=_CUDA_SKIP_REASON or "CUDA is unavailable"
)
def test_compile_region_enforces_dynamic_range() -> None:
    from torch._dynamo.source import LocalSource
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    tokens = shape_env.create_symintnode(
        shape_env.create_symbol(4, LocalSource("num_tokens")), hint=4
    )
    fake_mode = FakeTensorMode(shape_env=shape_env)
    with fake_mode:
        fake_left = torch.empty((tokens, 8), device="cuda", dtype=torch.float16)
        fake_right = torch.empty((tokens, 8), device="cuda", dtype=torch.float16)

    graph = fx.Graph()
    left = graph.placeholder("left")
    left.meta["example_value"] = fake_left
    right = graph.placeholder("right")
    right.meta["example_value"] = fake_right
    result = graph.call_function(torch.ops.aten.cat.default, ([left, right], 0))
    graph.output((result,))

    region = export_region(
        fx.GraphModule(torch.nn.Module(), graph),
        [fake_left, fake_right],
        dynamic_range=(1, 8),
    )
    compiled = compile_region(
        region, device_type="cuda", search_iterations=1, static_outputs=True
    )
    assert compiled._graph.output_shapes == [[16, 8]]

    output_ptrs = set()
    for size in (2, 3, 5, 8):
        left_value = torch.randn((size, 8), device="cuda", dtype=torch.float16)
        right_value = torch.randn((size, 8), device="cuda", dtype=torch.float16)
        (actual,) = compiled(left_value, right_value)
        output_ptrs.add(actual.data_ptr())
        assert actual.shape == (size * 2, 8)
        torch.testing.assert_close(actual, torch.cat((left_value, right_value)))
    assert len(output_ptrs) == 1

    for size in (1, 9):
        value = torch.randn((size, 8), device="cuda", dtype=torch.float16)
        with pytest.raises(ValueError, match="expected value in"):
            compiled(value, value)

    left_value = torch.randn((3, 8), device="cuda", dtype=torch.float16)
    right_value = torch.randn((4, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="inferred as both 3 and 4"):
        compiled(left_value, right_value)
