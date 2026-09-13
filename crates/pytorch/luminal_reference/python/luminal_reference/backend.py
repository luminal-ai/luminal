"""The luminal_reference torch.compile backend.

The backend self-exports the incoming GraphModule with ``torch.export``,
saves the program to a temporary ``.pt2``, hands it to the Rust extension
for translation and reference-runtime search, and returns a callable that
binds caller tensors per invocation.
"""

import os
import tempfile
from typing import Any, Callable, Optional, Sequence

import torch
from torch.export import export

from . import _luminal

# torch._export.serde.schema.ScalarType codes we can round-trip today.
_PT2_TO_TORCH = {
    1: torch.uint8,
    2: torch.int8,
    3: torch.int16,
    4: torch.int32,
    5: torch.int64,
    6: torch.float16,
    7: torch.float32,
    8: torch.float64,
    12: torch.bool,
    13: torch.bfloat16,
}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    # Flatten first: a 0-dim tensor cannot be viewed as a wider dtype.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _output_tensor(raw: bytes, dtype_code: int, shape: Sequence[int]) -> torch.Tensor:
    dtype = _PT2_TO_TORCH.get(dtype_code)
    if dtype is None:
        raise RuntimeError(f"luminal_reference cannot materialize PT2 dtype code {dtype_code}")
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
    return tensor.reshape(tuple(shape)).clone()


class CompiledModel:
    """Callable wrapper around a compiled reference-backend graph."""

    def __init__(self, graph: Any, ep: Any):
        self._graph = graph
        self._ep = ep
        names = graph.input_names
        kinds = graph.input_kinds
        self._user_input_names = [
            name for name, kind in zip(names, kinds) if kind == "user_input"
        ]
        self._outputs = list(
            zip(
                graph.output_names,
                graph.output_dtypes,
                graph.output_shapes,
                graph.output_mutations,
                graph.output_returns,
            )
        )

    def __call__(self, *args: torch.Tensor) -> Any:
        if len(args) != len(self._user_input_names):
            raise RuntimeError(
                f"luminal_reference expected {len(self._user_input_names)} inputs, "
                f"got {len(args)}"
            )
        for name, value in zip(self._user_input_names, args):
            self._graph.set_input(name, _tensor_bytes(value))
        self._graph.execute()

        results = []
        for index, (_, dtype_code, shape, mutation, returned) in enumerate(self._outputs):
            tensor = _output_tensor(self._graph.output_bytes(index), dtype_code, shape)
            if mutation is not None:
                target = self._user_input_names.index(mutation)
                args[target].copy_(tensor)
                # A returned mutation IS the caller's tensor (same storage),
                # matching eager's aliasing semantics.
                if returned:
                    results.append(args[target])
                continue
            if returned:
                results.append(tensor)
        # Dynamo's backend contract: return the graph's output tree (a
        # sequence), even for one result. It unwraps single-tensor returns
        # for the user.
        return tuple(results)


def luminal_reference(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    options: Optional[dict] = None,
    search_iterations: Optional[int] = None,
) -> CompiledModel:
    """The torch.compile backend entry point."""
    if options:
        search_iterations = options.get("search_iterations", search_iterations)

    ep = export(gm, tuple(example_inputs), strict=False)
    with tempfile.TemporaryDirectory() as tmp:
        pt2_path = os.path.join(tmp, "model.pt2")
        torch.export.save(ep, pt2_path)
        graph = _luminal.compile(pt2_path)

    names = graph.input_names
    kinds = graph.input_kinds
    parameter_names = graph.parameter_names

    user_index = 0
    for name, kind, parameter_name in zip(names, kinds, parameter_names):
        if kind == "user_input":
            if user_index >= len(example_inputs):
                raise RuntimeError(
                    f"export declared more user inputs than example_inputs: {name!r}"
                )
            value = example_inputs[user_index]
            user_index += 1
        else:
            if parameter_name not in ep.state_dict:
                raise RuntimeError(
                    f"parameter {parameter_name!r} (graph input {name!r}) is not in "
                    "the exported state_dict"
                )
            value = ep.state_dict[parameter_name]
        graph.set_input(name, _tensor_bytes(value))

    if user_index != len(example_inputs):
        raise RuntimeError(
            f"export consumed {user_index} of {len(example_inputs)} example_inputs"
        )

    graph.search(search_iterations)
    return CompiledModel(graph, ep)


def register_backend() -> None:
    """Register ``"luminal_reference"`` so ``backend="luminal_reference"`` works."""
    if "luminal_reference" in torch._dynamo.list_backends():
        return
    torch._dynamo.register_backend(luminal_reference, name="luminal_reference")
