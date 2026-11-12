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


def record_lowered_graphs(monkeypatch) -> list:
    """Record every graph the backend hands to the translator from here on,
    one ``LoweredGraph`` per compiled graph.

    The backend lowers each AOTAutograd graph once, in process, and hands
    the result to the translator. This observes that hand-off.
    """
    from luminal_torch_common import aot_graph

    lowered: list = []
    lower = aot_graph.lower_local_graph

    def recording_lower(gm, example_inputs):
        result = lower(gm, example_inputs)
        lowered.append(result[0])
        return result

    monkeypatch.setattr(aot_graph, "lower_local_graph", recording_lower)
    return lowered


class _Targets:
    """The ATen targets of each recorded graph, read when asked for."""

    def __init__(self, lowered):
        self._lowered = lowered

    def _sets(self):
        return [
            {
                str(node.target)
                for node in program.graph_module.graph.nodes
                if node.op == "call_function"
            }
            for program in self._lowered
        ]

    def __len__(self):
        return len(self._lowered)

    def __getitem__(self, index):
        return self._sets()[index]

    def __iter__(self):
        return iter(self._sets())

    def __eq__(self, other):
        return self._sets() == other

    def __repr__(self):
        return repr(self._sets())


def record_translated_targets(monkeypatch):
    """Record the ATen targets of every graph the backend hands to the
    translator from here on, one set per compiled graph.

    A test uses this to require that an op reached the translator itself
    rather than a decomposition of it, and that a call compiled exactly
    once.
    """
    return _Targets(record_lowered_graphs(monkeypatch))


def _fully_decomposed_backend(**options):
    """The selected backend's AOT compiler with every core decomposition
    applied, instead of the table that leaves the composites the translator
    lowers itself. The translator then sees only primitive ops."""
    name = selected_backend()
    aot = importlib.import_module(f"luminal_{name}.aot")
    base = aot.ReferenceAOTBackend if name == "reference" else aot.CudaAOTBackend

    def every_core_decomposition():
        return torch.export.default_decompositions().materialize()

    class FullyDecomposed(base):
        decompositions = staticmethod(every_core_decomposition)

    return FullyDecomposed(**options)


def _shapes_spec(model, inputs, dynamic_shapes):
    """``torch.compile``'s shape spec for dynamic dimensions written the way
    ``torch.export`` takes them: one ``{axis: Dim}`` mapping per positional
    input, or ``None`` for a static input."""
    import inspect

    from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec

    forward = model.forward if isinstance(model, torch.nn.Module) else model
    names = list(inspect.signature(forward).parameters)
    variables: dict[str, ShapeVar] = {}
    params = {}
    for name, value, axes in zip(names, inputs, dynamic_shapes):
        if not axes:
            continue
        sizes = []
        for axis, size in enumerate(value.shape):
            dim = axes.get(axis)
            if dim is None:
                sizes.append(size)
                continue
            # One Dim shared by two inputs is one dimension.
            sizes.append(
                variables.setdefault(
                    dim.__name__,
                    ShapeVar(
                        dim.__name__, min=dim.min, max=dim.max, optimization_hint=size
                    ),
                )
            )
        params[name] = TensorSpec(sizes)
    return ShapesSpec(params=params)


def compile_for_test(model, examples, *, dynamic_shapes=None, **options):
    """Compile ``model`` through the selected backend with every core
    decomposition applied, and return a callable that gives its outputs as
    one flat tuple.

    This is the translator harness for lowerings of primitive ops: the
    graph the translator sees has no composites left. Tests of the
    backend's own path, where composites reach the translator whole, use
    ``torch.compile(..., backend=luminal_backend)``.

    ``dynamic_shapes`` is written the way ``torch.export`` takes it: one
    ``{axis: Dim}`` mapping per positional input. Without one, every shape
    is static.
    """
    inputs = tuple(examples) if isinstance(examples, (tuple, list)) else (examples,)
    backend = _fully_decomposed_backend(**options)
    # A test may compile one module several times. Each compilation is its
    # own callable with its own recompile budget.
    if dynamic_shapes:
        compiled = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            isolate_recompiles=True,
            dynamic_shapes=_shapes_spec(model, inputs, dynamic_shapes),
        )
    else:
        compiled = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
            isolate_recompiles=True,
            dynamic=False,
        )

    def run(*args):
        leaves = torch.utils._pytree.tree_leaves(compiled(*args))
        return tuple(leaf for leaf in leaves if leaf is not None)

    return run


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
    with torch.device(device):
        yield


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch.set_float32_matmul_precision("highest")
    torch._dynamo.config.cache_size_limit = 1
    torch._dynamo.config.suppress_errors = False
    yield
    torch._dynamo.reset()
