"""Explicit runtime selection for translator and shared model parity tests."""

import importlib
import os

import pytest
import torch


def selected_backend():
    name = os.environ.get("LUMINAL_TEST_BACKEND", "reference")
    if name not in {"reference", "cuda_lite"}:
        raise ValueError(f"Unknown test backend: {name}")
    return name


def _backend(inputs):
    name = selected_backend()
    expected = "cuda" if name == "cuda_lite" else "cpu"
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.device.type != expected:
            raise ValueError(
                f"{name} test received a {value.device.type} input; expected {expected}"
            )
    return importlib.import_module(f"luminal_{name}")


def luminal_backend(gm, example_inputs, options=None):
    return _backend(example_inputs).Compiler(**(options or {}))(gm, example_inputs)


def compile_for_test(model, examples, *, dynamic_shapes=None, **options):
    """Internal PT2 translator harness; public API coverage uses Compiler directly."""
    from luminal_reference.export_utils import _drop_input_guards

    inputs = tuple(examples) if isinstance(examples, (tuple, list)) else (examples,)
    program = torch.export.export(
        model,
        inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    _drop_input_guards(program)
    program = program.run_decompositions()
    native = importlib.import_module(f"luminal_{selected_backend()}.backend")
    return native.compile_exported(program, inputs, **options)


@pytest.fixture
def device():
    name = "cuda" if selected_backend() == "cuda_lite" else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        pytest.fail("CUDA parity tests were requested but CUDA is unavailable")
    return torch.device(name)


@pytest.fixture(autouse=True)
def execution_device(device):
    # pytest_configure installs this default before collection, including for
    # module-level constants. Avoid a nested DeviceContext here: PyTorch cannot
    # temporarily disable nested default-device modes while capturing `.item()`.
    assert torch.get_default_device().type == device.type
    yield


@pytest.fixture
def stage_one_only(monkeypatch):
    """The real backend path, pinned to its first stage: the undecomposed
    graph must translate as it is. Re-decomposing the whole graph (the
    fallback for an unsupported op) fails the test instead of hiding a
    missing arm."""
    native = importlib.import_module(f"luminal_{selected_backend()}.backend")

    def refuse_decomposition():
        raise AssertionError("stage 1 refused the graph; it was re-decomposed")

    monkeypatch.setattr(native, "_decomp_table", refuse_decomposition)


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch.set_float32_matmul_precision("highest")
    cudnn_flags = None
    if selected_backend() == "cuda_lite":
        # Shared parity cases compare against eager CUDA. Keep that reference
        # at the authored FP32 precision instead of allowing TF32 input
        # conversion in either the matmul or convolution backend.
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        cudnn_flags = {
            "enabled": True,
            "allow_tf32": False,
            "deterministic": True,
            "benchmark": False,
        }
        if hasattr(torch.backends.cudnn, "fp32_precision"):
            cudnn_flags["fp32_precision"] = "ieee"
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.suppress_errors = False
    if cudnn_flags is None:
        yield
    else:
        # Some slow tests freeze global backend flags. The scoped API is the
        # supported way to override cuDNN policy after that point.
        with torch.backends.cudnn.flags(**cudnn_flags):
            yield
    torch._dynamo.reset()
