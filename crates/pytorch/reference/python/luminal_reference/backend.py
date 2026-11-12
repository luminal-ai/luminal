"""The luminal_reference local compiler.

AOTAutograd hands the backend one functional ATen graph per phase. Each is
lowered in process (see ``aot_graph``), translated and searched by the
reference runtime, and wrapped as a ``CompiledModel``, which binds the
caller's tensors and symbolic dimensions on each call.
"""

from collections.abc import Sequence
from typing import Any

import torch
from luminal_torch_common.aot_graph import (
    LoweredGraph,
    integer_argument,
    tensor_argument,
)
from luminal_torch_common.dimensions import profile_value

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
        raise RuntimeError(
            f"luminal_reference cannot materialize PT2 dtype code {dtype_code}"
        )
    if 0 in shape and not raw:
        return torch.empty(tuple(shape), dtype=dtype, device="cpu")
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
    return tensor.reshape(tuple(shape)).clone()


class CompiledModel:
    """One compiled graph on the reference runtime."""

    # Whether each call evaluates the runtime checks torch asserted in the
    # graph (a user's ``torch._check`` on a size). Binding does not depend
    # on them: set this to ``False``, on the class or on one model, and
    # calls run unchecked.
    runtime_assertions = True

    def __init__(self, graph: Any, program: bytes, held: dict | None = None):
        self._graph = graph
        # The graph as it was handed to the translator, kept so the model
        # can be saved.
        self._program = program
        # Graph input name -> the tensor constant the program holds.
        self._held = dict(held or {})
        # The program's boundary, as the translation states it. A call
        # passes one value per argument slot: `(kind, name, read, symbol)`.
        self._slots = list(graph.input_slots)
        self._user_input_names = [
            name for kind, name, read, _ in self._slots if kind == "tensor" and read
        ]
        self._output_dtypes = graph.output_dtypes
        self._output_mutations = graph.output_mutations

    def _bind(self, args: Sequence[Any]) -> list:
        """Bind one call's arguments, one per slot, and return the tensors
        the program reads. Nothing is filtered: an argument the program
        does not read is not looked at, and a value of the wrong type for a
        slot the program reads is refused by the slot's name."""
        if len(args) != len(self._slots):
            raise RuntimeError(
                f"luminal_reference expected {len(self._slots)} inputs, got {len(args)}"
            )
        inputs = []
        integers = []
        for (kind, name, read, symbol), value in zip(self._slots, args):
            if kind == "tensor" and read:
                value = tensor_argument(name, value)
                inputs.append(value)
                self._graph.set_input(name, _tensor_bytes(value), list(value.shape))
            elif kind == "int" and symbol is not None:
                integers.append((name, integer_argument(name, value)))
            # Any other argument is not looked at. A tensor no node reads
            # is never touched, and any other symbolic scalar is the
            # expression torch recorded for it: the program reads that,
            # never the call's value.
        # An integer argument that is a symbol defines that symbol, as it
        # does for torch, so it is bound after the tensor shapes.
        for name, value in integers:
            self._graph.bind_int(name, value)
        return inputs

    def rows(self, *args: Any) -> list:
        """Run one call and return the tensor the program wrote in each
        row: the caller's own tensor for a writeback, a fresh tensor
        otherwise."""
        inputs = self._bind(args)
        if self.runtime_assertions:
            self._graph.check_assertions()
        # Output shapes depend on the bound dims, so read them after the
        # inputs are staged rather than caching them at compile time.
        output_shapes = self._graph.output_shapes
        self._graph.execute()

        written = []
        outputs = zip(self._output_dtypes, output_shapes, self._output_mutations)
        for index, (dtype_code, shape, mutation) in enumerate(outputs):
            tensor = _output_tensor(self._graph.output_bytes(index), dtype_code, shape)
            if mutation is not None:
                # A mutation target is an argument of the call.
                if mutation not in self._user_input_names:
                    raise RuntimeError(
                        f"luminal_reference: mutation target {mutation!r} is not an "
                        "argument the program reads"
                    )
                destination = inputs[self._user_input_names.index(mutation)]
                destination.copy_(tensor)
                # A returned mutation IS the caller's tensor (same storage),
                # matching eager's aliasing semantics.
                tensor = destination
            written.append(tensor)
        return written


# The version of the saved form `save_compiled` writes.
SAVED_FORMAT = 1


def save_compiled(model: CompiledModel) -> dict:
    """A compiled graph as plain data: the graph the translator reads, the
    plan search selected for it, and the constants the program holds.
    Nothing in it has to be searched for again."""
    return {
        "format": SAVED_FORMAT,
        "program": model._program,
        "plan": bytes(model._graph.serialize_compiled()),
        "constants": dict(model._held),
    }


def load_compiled(saved: dict) -> CompiledModel:
    """The compiled model ``save_compiled`` described: the saved graph is
    translated again and the saved plan installed on it. No search runs."""
    if saved.get("format") != SAVED_FORMAT:
        raise RuntimeError(
            f"saved program has format {saved.get('format')!r}; this build of "
            f"luminal_reference reads format {SAVED_FORMAT}"
        )
    graph = _luminal.compile_graph(saved["program"])
    graph.load_compiled(saved["plan"])
    held = dict(saved["constants"])
    for name, value in held.items():
        graph.set_input(name, _tensor_bytes(value), list(value.shape))
    return CompiledModel(graph, saved["program"], held)


def _compile_local_graph(
    program: LoweredGraph,
    example_inputs: Sequence[Any],
    search_iterations: int | None = None,
    search_log: bool = False,
    max_intermediate_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
) -> CompiledModel:
    """Compile one lowered AOTAutograd graph with the native runtime.

    ``example_inputs`` holds one example value per argument slot: the data
    the search executes candidates on. An op the translator does not lower
    fails here, by name.
    """
    program_json = program.program_json()
    graph = _luminal.compile_graph(program_json)

    slots = graph.input_slots
    if len(slots) != len(example_inputs):
        raise RuntimeError(
            f"the graph has {len(slots)} arguments "
            f"({', '.join(name for _, name, _, _ in slots)}) and was given "
            f"{len(example_inputs)} example inputs"
        )
    examples = {name: value for (_, name, _, _), value in zip(slots, example_inputs)}

    # Every tensor the program reads is staged: an argument at its example
    # value, a constant the graph holds at the constant itself.
    held: dict = {}
    for name, kind, constant_name in zip(
        graph.input_names, graph.input_kinds, graph.parameter_names
    ):
        if kind == "user_input":
            value = examples[name]
        else:
            value = program.constants.get(constant_name)
            if value is None:
                raise RuntimeError(
                    f"constant {constant_name!r} (graph input {name!r}) is not "
                    "among the constants lifted from the graph"
                )
            held[name] = value
        graph.set_input(name, _tensor_bytes(value), list(value.shape))

    # An integer argument that defines a symbol is profiled at the value
    # torch traced it with.
    for (kind, name, _, symbol), value in zip(slots, example_inputs):
        if kind == "int" and symbol is not None:
            graph.bind_int(name, integer_argument(name, profile_value(value)))

    graph.search(
        search_iterations,
        search_log=search_log,
        max_intermediate_bytes=max_intermediate_bytes,
        memory_budget_bytes=memory_budget_bytes,
    )
    return CompiledModel(graph, program_json, held)
