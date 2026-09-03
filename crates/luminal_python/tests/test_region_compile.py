from __future__ import annotations

import time

import pytest
import torch
from torch import fx

import luminal.region_compile as region_compile_module
from luminal.artifact_cache import (
    ArtifactCacheStats,
    CompiledArtifact,
    artifact_cache_stats,
    clear_artifact_cache,
    get_or_compile,
    region_artifact_key,
)
from luminal.compiled_model import CompiledModel
from luminal.region_compile import compile_region
from luminal.region_export import export_region


def _add_graph() -> fx.GraphModule:
    graph = fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    result = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    graph.output((result,))
    return fx.GraphModule(torch.nn.Module(), graph)


def test_compile_region_preserves_runtime_input_indices(monkeypatch) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inputs = [torch.randn(2, 4, device="cuda") for _ in range(2)]
        region = export_region(_add_graph(), inputs)
    sentinel = object()
    factory = object()
    received = {}

    monkeypatch.setattr(
        "luminal.luminal._cuda_lite_factory_capsule", lambda: factory, raising=False
    )

    def fake_save_and_compile(program, capsule, iterations, **kwargs):
        received.update(
            program=program,
            capsule=capsule,
            iterations=iterations,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(
        region_compile_module, "_save_and_compile", fake_save_and_compile
    )

    assert compile_region(region, search_iterations=3) is sentinel
    assert received == {
        "program": region.program,
        "capsule": factory,
        "iterations": 3,
        "user_indices": region.input_indices,
        "output_spec": region.output_spec,
        "input_device_ptrs": None,
        "device_index": 0,
        "use_current_stream": True,
        "static_outputs": False,
        "external_cuda_graph": False,
        "artifact_key": region_artifact_key(
            region.program,
            device_type="cuda",
            device_index=0,
            search_iterations=3,
            external_cuda_graph=False,
        ),
    }


def _key_program(input_name: str, op=torch.ops.aten.relu.default):
    from types import SimpleNamespace

    graph = fx.Graph()
    value = graph.placeholder(input_name)
    value.meta["val"] = torch.empty(2, 4)
    result = graph.call_function(op, (value,))
    result.meta["val"] = torch.empty(2, 4)
    graph.output((result,))
    return SimpleNamespace(
        constants={},
        state_dict={},
        range_constraints={},
        graph_module=fx.GraphModule(torch.nn.Module(), graph),
    )


def test_artifact_key_ignores_input_names() -> None:
    options = {"device_type": "cuda", "search_iterations": 1}

    assert region_artifact_key(_key_program("layer_0"), **options) == (
        region_artifact_key(_key_program("layer_1"), **options)
    )
    assert region_artifact_key(_key_program("layer_0"), **options) != (
        region_artifact_key(
            _key_program("layer_0", torch.ops.aten.sigmoid.default), **options
        )
    )


def test_artifact_cache_compiles_once() -> None:
    clear_artifact_cache()
    artifact = object()
    calls = 0

    def compile_artifact():
        nonlocal calls
        calls += 1
        return artifact

    assert get_or_compile("region", compile_artifact) is artifact
    assert get_or_compile("region", compile_artifact) is artifact
    assert calls == 1
    assert artifact_cache_stats() == ArtifactCacheStats(
        unique_artifacts=1,
        reuse_hits=1,
        searches=1,
    )
    clear_artifact_cache()


def test_shared_artifact_rebinds_inputs_between_models() -> None:
    from types import SimpleNamespace

    class CudaTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(2), False)

        @property
        def device(self):
            return torch.device("cuda:0")

        @property
        def is_cuda(self):
            return True

    registrations = []
    graph = SimpleNamespace(
        input_names=["x"],
        input_dtypes=[7],
        output_names=[],
        output_dtypes=[],
        output_shapes=[],
        writeback_outputs=[],
        has_dynamic_dims=False,
        device_type="cuda",
        device_index=0,
        supports_device_ptrs=True,
        set_input_device_ptr=lambda _, ptr, __: registrations.append(ptr),
        run=lambda: None,
    )
    artifact = CompiledArtifact(graph)
    first = artifact.bind()
    second = artifact.bind()
    first_input = CudaTensor()
    second_input = CudaTensor()

    first(first_input)
    second(second_input)
    first(first_input)

    assert registrations == [
        first_input.data_ptr(),
        second_input.data_ptr(),
        first_input.data_ptr(),
    ]


def test_compile_region_rejects_nonzero_device() -> None:
    from dataclasses import replace
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        inputs = [torch.randn(2, 4, device="cuda") for _ in range(2)]
        region = export_region(_add_graph(), inputs)

    with pytest.raises(ValueError, match="only logical CUDA device 0"):
        compile_region(replace(region, device_index=1))


def test_compiled_model_rejects_wrong_cuda_device() -> None:
    from types import SimpleNamespace

    class CudaOneTensor(torch.Tensor):
        @staticmethod
        def __new__(cls):
            return torch.Tensor._make_subclass(cls, torch.empty(2), require_grad=False)

        @property
        def device(self):
            return torch.device("cuda:1")

    graph = SimpleNamespace(
        input_names=["x"],
        input_dtypes=[7],
        output_names=[],
        output_shapes=[],
        writeback_outputs=[],
        has_dynamic_dims=False,
        device_type="cuda",
        device_index=0,
        supports_device_ptrs=True,
    )
    model = CompiledModel(graph)

    with pytest.raises(ValueError, match="compiled runtime uses logical device 0"):
        model(CudaOneTensor())


def test_compiled_model_passes_current_stream(monkeypatch) -> None:
    from types import SimpleNamespace

    calls = []
    graph = SimpleNamespace(
        input_names=[],
        input_dtypes=[],
        output_names=[],
        output_dtypes=[],
        output_shapes=[],
        writeback_outputs=[],
        has_dynamic_dims=False,
        device_type="cuda",
        device_index=0,
        supports_device_ptrs=True,
        run=lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=1234),
    )

    assert CompiledModel(graph, use_current_stream=True)() == ()
    assert calls == [(1234,)]


def test_compiled_model_restores_region_output_structure() -> None:
    from types import SimpleNamespace

    graph = SimpleNamespace(
        input_names=[],
        input_dtypes=[],
        output_names=["result"],
        output_dtypes=[7],
        output_shapes=[[1]],
        writeback_outputs=[],
        has_dynamic_dims=False,
        device_type="cpu",
        device_index=None,
        supports_device_ptrs=False,
        run=lambda: None,
        get_output_at=lambda position: [3.0],
    )
    _, output_spec = torch.utils._pytree.tree_flatten(torch.tensor(0))

    output = CompiledModel(graph, output_spec=output_spec)()

    assert torch.equal(output, torch.tensor([3.0]))


def _cuda_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        from luminal.luminal import _cuda_lite_factory_capsule

        _cuda_lite_factory_capsule()
    except (ImportError, AttributeError, RuntimeError) as error:
        return f"luminal_python was not built with CUDA support: {error}"
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
def test_non_static_writeback_accepts_changed_target() -> None:
    from types import SimpleNamespace

    copies = []
    graph = SimpleNamespace(
        input_names=["target"],
        input_dtypes=[7],
        output_names=["mutation"],
        output_dtypes=[7],
        output_shapes=[[2]],
        writeback_outputs=[(0, "target")],
        has_dynamic_dims=False,
        device_type="cuda",
        device_index=0,
        supports_device_ptrs=True,
        set_input_device_ptr=lambda *args: None,
        run=lambda *args: None,
        copy_outputs_to_device_ptrs_at=lambda value: copies.append(value),
    )
    model = CompiledModel(graph)
    first = torch.zeros(2, device="cuda")
    second = torch.zeros(2, device="cuda")

    model(first)
    model(second)

    assert [copy[0][1] for copy in copies] == [first.data_ptr(), second.data_ptr()]


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

    compiled = compile_region(region, search_iterations=1)

    real_inputs = [
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
        torch.randn((2, 4), device="cuda", dtype=torch.float16),
    ]
    (actual,) = compiled(*real_inputs)
    torch.testing.assert_close(actual, real_inputs[0] + real_inputs[1])


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

    compiled = compile_region(region, search_iterations=1, static_outputs=True)
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

    compiled = compile_region(region, search_iterations=1, static_outputs=True)
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
    compiled = compile_region(region, search_iterations=1, static_outputs=True)
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
