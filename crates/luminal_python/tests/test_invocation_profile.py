import importlib.util
import json
from pathlib import Path

import pytest
import torch
from luminal.compiled_model import CompiledModel, _cuda_input_binding_signature
from luminal.dtype_util import torch_dtype_code


class _ProfileTestGraph:
    has_dynamic_dims = False
    device_type = "cpu"
    supports_device_ptrs = False

    def __init__(self):
        self.input_names = ["x"]
        self.output_names = ["y"]
        self.writeback_outputs = []
        self.output_shapes = [[1]]
        self.input_dtypes = [7]
        self.output_dtypes = [7]
        self.invocation = None
        self.profile = None

    def set_input_from_ptr(self, *_args):
        pass

    def set_profile_invocation_id(self, invocation):
        self.invocation = invocation

    def run(self):
        self.profile = json.dumps(
            {
                "invocation": self.invocation,
                "backend": "test",
                "dynamic_dims": {},
                "bucket": 0,
                "timings_us": {"total": 4.0, "sync": 3.0},
                "counts": {"dirty_hlir": 1},
            }
        )

    def take_last_execution_profile_json(self):
        profile, self.profile = self.profile, None
        return profile

    def get_output(self, _name):
        return [3.0]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class _FakeCudaTensor:
    def __init__(self, ptr, n_bytes=16, device="cuda:0", dtype=torch.bfloat16):
        self._ptr = ptr
        self.n_bytes = n_bytes
        self.device = torch.device(device)
        self.dtype = dtype

    def data_ptr(self):
        return self._ptr


class _CudaBindingTestGraph:
    has_dynamic_dims = False
    device_type = "cuda"
    supports_device_ptrs = True

    def __init__(self):
        self.input_names = ["x"]
        self.output_names = ["y"]
        self.writeback_outputs = []
        self.output_shapes = [[1]]
        self.input_dtypes = [torch_dtype_code(torch.float32)]
        self.output_dtypes = [torch_dtype_code(torch.int32)]
        self.input_pointer_calls = []

    def set_input_device_ptr(self, name, ptr, n_bytes):
        self.input_pointer_calls.append((name, ptr, n_bytes))

    def run(self):
        pass

    def get_output_i32(self, _name):
        return [3]


def test_cuda_input_binding_cache_tracks_resource_relevant_metadata():
    first = _FakeCudaTensor(0x1000)
    signature = _cuda_input_binding_signature(first, first.n_bytes)

    # Contents may change at a stable address without changing the binding.
    alias = _FakeCudaTensor(0x1000)
    assert _cuda_input_binding_signature(alias, alias.n_bytes) == signature

    replacement = _FakeCudaTensor(0x2000)
    assert _cuda_input_binding_signature(replacement, replacement.n_bytes) != signature
    assert _cuda_input_binding_signature(first, 32) != signature
    assert (
        _cuda_input_binding_signature(
            _FakeCudaTensor(0x1000, dtype=torch.float16), first.n_bytes
        )
        != signature
    )
    assert (
        _cuda_input_binding_signature(
            _FakeCudaTensor(0x1000, device="cuda:1"), first.n_bytes
        )
        != signature
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_model_skips_unchanged_cuda_input_pointer_registration():
    graph = _CudaBindingTestGraph()
    model = CompiledModel(graph)
    tensor = torch.ones(1, device="cuda")

    assert model(tensor)[0].item() == 3
    assert len(graph.input_pointer_calls) == 1

    # A content mutation at the same address needs execution ordering, not a
    # new graph pointer binding.
    tensor.add_(1)
    assert model(tensor)[0].item() == 3
    assert len(graph.input_pointer_calls) == 1

    replacement = tensor.clone()
    assert model(replacement)[0].item() == 3
    assert len(graph.input_pointer_calls) == 2


def test_invocation_profile_is_captured_when_compiled_model_is_created(
    tmp_path, monkeypatch
):
    trace = tmp_path / "parallel" / "invocations.jsonl"
    monkeypatch.delenv("LUMINAL_PROFILE_JSONL", raising=False)
    first = CompiledModel(_ProfileTestGraph())
    assert torch.equal(first(torch.tensor([1.0]))[0], torch.tensor([3.0]))
    assert not trace.exists()

    monkeypatch.setenv("LUMINAL_PROFILE_JSONL", str(trace))
    second = CompiledModel(_ProfileTestGraph())
    assert torch.equal(first(torch.tensor([1.0]))[0], torch.tensor([3.0]))
    assert torch.equal(second(torch.tensor([1.0]))[0], torch.tensor([3.0]))
    records = _read_jsonl(trace)

    # Profiling configuration belongs to a compiled instance. The first model
    # remains unprofiled after the environment changes, while a model created
    # after the opt-in emits records without consulting getenv in __call__.
    assert len(records) == 1
    assert records[0]["call_index"] == 0
    assert records[0]["graph"] == second._profile_graph_id
    for record in records:
        assert record["kind"] == "luminal_invocation"
        assert record["runtime"]["invocation"] == record["invocation"]
        assert record["compiled_model"]["counts"]["input_bindings"] == 1
        assert record["compiled_model"]["counts"]["outputs"] == 1
        assert record["compiled_model"]["timings_us"]["total"] >= 0
        assert record["compiled_model"]["timings_us"]["graph_run"] >= 0


def test_invocation_profile_renderer_generates_heatmap(tmp_path):
    trace = tmp_path / "profile.jsonl"
    record = {
        "kind": "luminal_invocation",
        "invocation": 1,
        "graph": 1,
        "call_index": 1,
        "phase": "decode",
        "compiled_model": {
            "timings_us": {"total": 10.0, "setup": 1.0, "graph_run": 8.0},
            "counts": {},
        },
        "runtime": {
            "timings_us": {"total": 8.0, "prepare": 2.0, "sync": 5.0},
            "counts": {
                "resource_validations": 1,
                "resource_validation_inputs_changed": 1,
            },
        },
    }
    trace.write_text(json.dumps(record) + "\n")
    script = Path(__file__).parents[1] / "scripts" / "render_invocation_profile.py"
    spec = importlib.util.spec_from_file_location("render_invocation_profile", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = module.load_records(trace)
    rendered = module.render(records, trace)
    assert "Invocation heatmap" in rendered
    assert "Representative invocation waterfall" in rendered
    assert "Stream execution/wait" in rendered
    assert "Invocation counts and invalidation reasons" in rendered
    assert "Validation reason: resource input changed" in rendered
