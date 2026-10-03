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
    # Bare tensor constructors in the shared cases must use the selected device,
    # rather than quietly exercising the reference runtime in a CUDA run.
    # PyTorch 2.14 Dynamo loses floating .item() dataflow inside a CPU
    # DeviceContext (even the eager FX backend receives return (None,)).
    # The existing CPU default already places bare constructors correctly;
    # avoid that redundant TorchFunctionMode while retaining the CUDA guard.
    if device.type == "cpu" and torch.get_default_device().type == "cpu":
        yield
    else:
        with torch.device(device):
            yield


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.suppress_errors = False
    yield
    torch._dynamo.reset()
