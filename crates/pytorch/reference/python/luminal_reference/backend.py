"""The luminal_reference torch.compile backend.

The public entry point uses AOTAutograd for inference and training. Its local
compiler wraps each AOTAutograd graph as a PT2 ExportedProgram, without a second
trace (see ``aot_graph``), for translation and reference-runtime search, then
binds caller tensors and symbolic dimensions on each invocation.
"""

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import torch

from . import _luminal
from .aot_graph import LoweredGraph, integer_argument, tensor_argument
from .dimensions import profile_value
from .export_utils import _decomp_table, _drop_dead_data_dependent_ops, _lower_sym_sum

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
    """Callable wrapper around a compiled reference-backend graph."""

    # Whether each call evaluates the runtime checks torch asserted in the
    # graph (a user's ``torch._check`` on a size). Binding does not depend
    # on them: set this to ``False``, on the class or on one model, and
    # calls run unchecked.
    runtime_assertions = True

    def __init__(
        self,
        graph: Any,
        ep: Any,
        scalar_output_positions: Sequence[int] = (),
        held: dict | None = None,
    ):
        self._graph = graph
        self._ep = ep
        self._scalar_output_positions = frozenset(scalar_output_positions)
        # Graph input name -> the exported parameter/buffer tensor staged for
        # it, so a buffer mutation has somewhere to write back.
        self._held = dict(held or {})
        # The program's boundary, as the translation states it. A call
        # passes one value per argument slot: `(kind, name, read, symbol)`.
        # The output leaves are `(kind, source, index)`.
        self._slots = list(graph.input_slots)
        self._output_slots = list(graph.output_slots)
        self._user_input_names = [
            name for kind, name, read, _ in self._slots if kind == "tensor" and read
        ]
        self._output_names = graph.output_names
        self._output_dtypes = graph.output_dtypes
        self._output_mutations = graph.output_mutations
        self._output_returns = graph.output_returns

    def __copy__(self):
        model = type(self)(
            self._graph.fork(), self._ep, self._scalar_output_positions, self._held
        )
        for name, value in self._held.items():
            model._graph.set_input(name, _tensor_bytes(value), list(value.shape))
        return model

    @property
    def writeback_inputs(self):
        return {
            name: mutation
            for name, mutation in zip(
                self._graph.output_names, self._graph.output_mutations
            )
            if mutation is not None
        }

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
                # A mutation target is a graph input: a user input, or a
                # parameter/buffer held from the export's state_dict.
                if mutation in self._user_input_names:
                    destination = inputs[self._user_input_names.index(mutation)]
                elif mutation in self._held:
                    destination = self._held[mutation]
                else:
                    raise RuntimeError(
                        f"luminal_reference: mutation target {mutation!r} is neither a "
                        "user input nor a staged parameter/buffer"
                    )
                destination.copy_(tensor)
                # A returned mutation IS the caller's tensor (same storage),
                # matching eager's aliasing semantics.
                tensor = destination
            written.append(tensor)
        return written

    def __call__(self, *args: Any) -> Any:
        """The program's output leaves that are tensors or forwarded
        arguments, in order."""
        written = self.rows(*args)
        results = []
        for _, source, index in self._output_slots:
            if source == "program":
                value = written[index]
                # Scalar graph outputs were boxed into rank-zero tensors
                # before export; restore the Python scalar backend contract.
                if len(results) in self._scalar_output_positions:
                    value = value.item()
                results.append(value)
            elif source == "input":
                # The leaf is an argument: the caller's own object.
                results.append(args[index])
        # Dynamo's backend contract: return the graph's output tree (a
        # sequence), even for one result. It unwraps single-tensor returns
        # for the user.
        return tuple(results)


# The version of the saved form `save_compiled` writes.
SAVED_FORMAT = 1


def save_compiled(model: CompiledModel) -> dict:
    """A compiled AOTAutograd graph as plain data: the graph the translator
    reads, the plan search selected for it, and the constants the program
    holds. Nothing in it has to be searched for again."""
    program = model._ep
    if not isinstance(program, LoweredGraph):
        raise TypeError(
            "only a graph lowered in process is saved; this model was compiled "
            f"from a {type(program).__name__}"
        )
    return {
        "format": SAVED_FORMAT,
        "program": program.program_json(),
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
    return CompiledModel(graph, None, (), held)


def _compile_local_graph(
    program: LoweredGraph,
    example_inputs: Sequence[Any],
    options: dict | None = None,
    search_iterations: int | None = None,
    search_log: bool = False,
    max_intermediate_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
) -> CompiledModel:
    """Compile one lowered AOTAutograd graph with the native runtime.
    ``example_inputs`` holds one example value per argument slot."""
    if options:
        search_iterations = options.get("search_iterations", search_iterations)
        search_log = options.get("search_log", search_log)
        max_intermediate_bytes = options.get(
            "max_intermediate_bytes", max_intermediate_bytes
        )
        memory_budget_bytes = options.get("memory_budget_bytes", memory_budget_bytes)
    return compile_exported(
        program,
        example_inputs,
        search_iterations,
        search_log=search_log,
        max_intermediate_bytes=max_intermediate_bytes,
        memory_budget_bytes=memory_budget_bytes,
        decompose_on_rejection=False,
        symbolic_outputs=True,
    )


def compile_exported(
    ep,
    export_inputs,
    search_iterations=None,
    scalar_output_positions=(),
    *,
    search_log=False,
    max_intermediate_bytes=None,
    memory_budget_bytes=None,
    decompose_on_rejection=True,
    symbolic_outputs=False,
):
    """Compile a functional program with the native runtime: an
    ``ExportedProgram``, or an AOTAutograd graph lowered in process.

    ``decompose_on_rejection`` retries a translator rejection after running
    the core decompositions on the program. It serves programs captured by
    ``torch.export``; the AOTAutograd path decomposes once up front and
    passes ``False`` so an unsupported op fails by name.

    ``symbolic_outputs`` says the caller evaluates output leaves that are
    symbolic scalars itself, as the AOTAutograd adapter does. Without it a
    program with such a leaf is refused: the compiled model returns tensors.
    """

    def _save_and_compile(program: Any) -> Any:
        # An AOTAutograd graph is handed to the translator in process.
        if isinstance(program, LoweredGraph):
            return _luminal.compile_graph(program.program_json())
        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = os.path.join(tmp, "model.pt2")
            torch.export.save(program, pt2_path)
            return _luminal.compile(pt2_path)

    try:
        graph = _save_and_compile(ep)
    except RuntimeError as exc:
        # The translator lowers a fixed op set. Decomposing the exported graph
        # rewrites higher-level composites into primitives the translator
        # handles, but it also rewrites ops it already lowers directly (e.g.
        # ``aten.linear`` -> ``aten.addmm``). Gate the aggressive pass on an
        # actual translator rejection so the common case keeps its original,
        # un-decomposed graph.
        if not decompose_on_rejection or "unsupported ATen op" not in str(exc):
            raise
        ep = ep.run_decompositions(_decomp_table())
        _drop_dead_data_dependent_ops(ep.graph_module)
        _lower_sym_sum(ep)
        graph = _save_and_compile(ep)

    names = graph.input_names
    kinds = graph.input_kinds
    parameter_names = graph.parameter_names

    # One example value per argument slot, in call order.
    slots = graph.input_slots
    if len(slots) != len(export_inputs):
        raise RuntimeError(
            f"export declared {len(slots)} user inputs "
            f"({', '.join(name for _, name, _, _ in slots)}) and was given "
            f"{len(export_inputs)} example_inputs"
        )
    examples = {name: value for (_, name, _, _), value in zip(slots, export_inputs)}
    if not symbolic_outputs:
        for position, (kind, source, _) in enumerate(graph.output_slots):
            if source == "expression":
                raise RuntimeError(
                    f"output leaf {position} is a symbolic {kind} computed from "
                    "sizes; this entry point returns tensors only"
                )

    held: dict = {}
    for name, kind, parameter_name in zip(names, kinds, parameter_names):
        if kind == "user_input":
            value = examples[name]
        else:
            value = ep.state_dict.get(parameter_name)
            if value is None:
                value = ep.constants.get(parameter_name)
            if value is None:
                raise RuntimeError(
                    f"parameter {parameter_name!r} (graph input {name!r}) is not in "
                    "the exported state_dict or constants"
                )
            # The export's own tensor, not the caller's module buffer: mutated
            # state persists across calls on the input's runtime buffer.
            held[name] = value
        from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily

        if isinstance(value, FakeTensor):
            with unset_fake_temporarily():
                value = torch.ones(
                    tuple(profile_value(d) for d in value.shape), dtype=value.dtype
                )
        graph.set_input(name, _tensor_bytes(value), list(value.shape))

    # An integer argument that defines a symbol is profiled at the value
    # torch traced it with.
    for (kind, name, _, symbol), value in zip(slots, export_inputs):
        if kind == "int" and symbol is not None:
            graph.bind_int(name, integer_argument(name, profile_value(value)))

    graph.search(
        search_iterations,
        search_log=search_log,
        max_intermediate_bytes=max_intermediate_bytes,
        memory_budget_bytes=memory_budget_bytes,
    )
    return CompiledModel(graph, ep, scalar_output_positions, held)
