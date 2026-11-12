"""torch's ahead-of-time surface: ``aot_compile``, ``save_compiled_function``
and ``torch.compiler.load_compiled_function``.

Each test calls the compiled function several times, saves it, loads it, and
calls the loaded function several times. Weights are not part of the
artifact: a compiled module takes the model as its first argument, so one
loaded function serves any model of the same structure.

torch saves an ``aot_autograd`` backend through the ``OutputCode`` each of
its compilers returns. The reference backend's saved form is the graph the
translator reads and the plan search selected, so a loaded function runs
without searching. The CUDA runtime has no serialized form of a plan yet;
it refuses ``aot_compile`` by name, which is reported here as an expected
failure. Any other failure, at any stage, fails the test.
"""

import pytest
import torch
from backend_test_utils import luminal_backend, selected_backend

_CALLS = 3
_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}


def _aot_compile(compiled, example_args):
    """``compiled.aot_compile`` on positional example inputs.

    torch prepares the saved form of every graph while ``aot_compile``
    runs, so a backend that cannot save says so here, not at the save.
    """
    try:
        return compiled.aot_compile((example_args, {}))
    except NotImplementedError as error:
        if "cannot save a compiled graph yet" not in str(error):
            raise
        assert selected_backend() == "cuda_lite", (
            f"only the CUDA backend refuses to save; {selected_backend()} said: {error}"
        )
        pytest.xfail(
            "LUM-806, LUM-864: the CUDA runtime has no serialized form of a "
            "searched plan, so a graph it computes cannot be saved"
        )


def _save_and_load(aot, path):
    aot.save_compiled_function(str(path))
    assert path.stat().st_size > 0, "the saved artifact is empty"
    with open(path, "rb") as file:
        return torch.compiler.load_compiled_function(file, f_globals=globals())


def _affine_relu(x, w, b):
    return torch.relu(x @ w + b)


def test_aot_compiled_function_round_trips_through_save_and_load(
    device: torch.device, tmp_path
) -> None:
    def inputs():
        return (
            torch.randn(4, 16, device=device),
            torch.randn(16, 8, device=device),
            torch.randn(8, device=device),
        )

    torch.manual_seed(0)
    compiled = torch.compile(_affine_relu, backend=luminal_backend, fullgraph=True)
    aot = _aot_compile(compiled, inputs())
    for _ in range(_CALLS):
        args = inputs()
        torch.testing.assert_close(aot(*args), _affine_relu(*args), **_TOLERANCE)

    loaded = _save_and_load(aot, tmp_path / "affine_relu.pt")
    for _ in range(_CALLS):
        args = inputs()
        torch.testing.assert_close(loaded(*args), _affine_relu(*args), **_TOLERANCE)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


def test_aot_compiled_module_round_trips_and_follows_the_models_weights(
    device: torch.device, tmp_path
) -> None:
    torch.manual_seed(0)
    model = _Block().to(device).eval()
    compiled = torch.compile(model, backend=luminal_backend, fullgraph=True)
    with torch.no_grad():
        aot = _aot_compile(compiled.forward, (torch.randn(4, 16, device=device),))
        for _ in range(_CALLS):
            x = torch.randn(4, 16, device=device)
            torch.testing.assert_close(aot(model, x), model(x), **_TOLERANCE)

        loaded = _save_and_load(aot, tmp_path / "block.pt")
        for _ in range(_CALLS):
            x = torch.randn(4, 16, device=device)
            torch.testing.assert_close(loaded(model, x), model(x), **_TOLERANCE)

        # The artifact holds the program, not the weights: the same loaded
        # function computes another model's answer from that model's tensors.
        other = _Block().to(device).eval()
        x = torch.randn(4, 16, device=device)
        torch.testing.assert_close(loaded(other, x), other(x), **_TOLERANCE)


def _loss(x, w):
    return torch.tanh(x @ w).square().sum()


def test_aot_compiled_training_function_round_trips_with_its_gradients(
    device: torch.device, tmp_path
) -> None:
    """A function compiled for training saves its forward and its backward
    graph. The loaded function's output and gradients match eager, and no
    compiler runs after the load."""

    def inputs():
        return (
            torch.randn(4, 16, device=device),
            torch.randn(16, 8, device=device, requires_grad=True),
        )

    def check(fn):
        x, w = inputs()
        out, want = fn(x, w), _loss(x, w)
        torch.testing.assert_close(out, want, **_TOLERANCE)
        (grad,) = torch.autograd.grad(out, w)
        (reference,) = torch.autograd.grad(want, w)
        torch.testing.assert_close(grad, reference, **_TOLERANCE)

    torch.manual_seed(0)
    compiled = torch.compile(_loss, backend=luminal_backend, fullgraph=True)
    aot = _aot_compile(compiled, inputs())
    for _ in range(_CALLS):
        check(aot)

    loaded = _save_and_load(aot, tmp_path / "loss.pt")
    for _ in range(_CALLS):
        check(loaded)


def _scaled_rows(x, w):
    return torch.relu(x @ w) * x.shape[0]


def test_aot_compiled_function_with_a_dynamic_dimension_round_trips(
    device: torch.device, tmp_path
) -> None:
    """A dynamic dimension survives the save: the loaded function runs at
    sizes the example call never had, on the one saved program, and the
    size it reads as a value follows each call."""

    def inputs(rows):
        return (
            torch.randn(rows, 16, device=device),
            torch.randn(16, 8, device=device),
        )

    torch.manual_seed(0)
    compiled = torch.compile(_scaled_rows, backend=luminal_backend, fullgraph=True)
    example = inputs(4)
    torch._dynamo.mark_dynamic(example[0], 0, min=2, max=64)
    aot = _aot_compile(compiled, example)
    for rows in (4, 7, 2):
        args = inputs(rows)
        torch.testing.assert_close(aot(*args), _scaled_rows(*args), **_TOLERANCE)

    loaded = _save_and_load(aot, tmp_path / "scaled_rows.pt")
    for rows in (4, 9, 2, 33):
        args = inputs(rows)
        torch.testing.assert_close(loaded(*args), _scaled_rows(*args), **_TOLERANCE)
