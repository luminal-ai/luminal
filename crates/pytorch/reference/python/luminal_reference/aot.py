"""AOTAutograd integration for local CPU programs.

AOTAutograd owns differentiation and hands this adapter one functional ATen
graph per phase: forward, backward, or inference. Each graph compiles whole
through PT2 translation, egglog search, and ReferenceRuntime; there is no
eager fallback for local computation. Symbolic CPU shapes are retained through
compilation and bound at execution: from the call's tensor shapes, and from a
symbolic integer argument by its value.

Collective operations are refused by name. A program that communicates must be
split by the application so that each compiled graph is collective-free.
"""

import dataclasses
from functools import partial
from typing import Any

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.output_code import OutputCode

from .aot_graph import (
    GraphRecord,
    LocalGraph,
    prepare_local_graph,
    refuse_collectives,
)
from .backend import _compile_local_graph, load_compiled, save_compiled
from .dimensions import profile_value
from .export_utils import _aot_decomposition_table

__all__ = ["GraphRecord", "ReferenceAOTBackend", "ReferenceOutputCode"]


@dataclasses.dataclass
class ReferenceOutputCode(OutputCode):
    """One AOTAutograd graph compiled for the reference runtime, as the
    object torch's AOT machinery holds, calls, saves and loads.

    Returning an ``OutputCode`` is what lets
    ``torch.compile(...).aot_compile()`` save a compiled function and
    ``torch.compiler.load_compiled_function`` load it. torch pickles each
    graph's ``OutputCode`` after ``prepare_for_serialization`` and calls
    ``post_compile`` on the unpickled object before its first call.

    ``local`` is the graph's boundary, ``compiled`` the compiled program
    (``None`` for a graph that only forwards inputs and sizes), and
    ``record`` the inspection record of the compile. The saved form is the
    boundary as plain data, the graph the translator reads, the plan search
    selected, and the constants: a loaded function translates and installs
    the plan without searching.
    """

    local: LocalGraph | None = None
    compiled: Any = None
    record: GraphRecord | None = None
    saved: dict | None = None

    def __post_init__(self) -> None:
        # torch calls a compiled graph with its arguments in one list. It
        # reads this flag off a wrapper that copies the instance's own
        # attributes, so it is set on the instance.
        self._boxed_call = True

    def __call__(self, inputs) -> Any:
        args = self.local.bind(inputs)
        rows = self.compiled.rows(*args) if self.compiled is not None else ()
        if self.record is not None:
            self.record.executions += 1
        return self.local.assemble(args, rows)

    def prepare_for_serialization(self) -> None:
        """Turn this object into its saved form. torch calls this on a
        shallow copy while ``aot_compile`` runs, and the original keeps
        serving calls, so every attribute is rebound and nothing shared is
        changed in place."""
        if self.compiled is not None:
            self.saved = save_compiled(self.compiled)
        self.local = self.local.portable()
        self.compiled = None
        self.record = None

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        """Rebuild the compiled program from the saved form. torch calls
        this once on a loaded object, before its first call, with no
        example inputs and no shape environment."""
        if self.compiled is None and self.saved is not None:
            self.compiled = load_compiled(self.saved)

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Part of the ``OutputCode`` interface; there is nothing to bundle."""


def _compile_graph(
    gm,
    inputs,
    record,
    search_iterations,
    search_log=False,
    max_intermediate_bytes=None,
    memory_budget_bytes=None,
):
    local = prepare_local_graph(gm, inputs, device_type="cpu")

    compiled = None
    if local.computes:
        record.input_shapes = tuple(
            tuple(v.shape) for v in local.inputs if isinstance(v, torch.Tensor)
        )
        record.targets = tuple(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        # The reference runtime stages input bytes, so compile against real
        # CPU tensors at the profiling shapes, isolated from the surrounding
        # AOT FakeTensor mode.
        from torch._guards import tracing
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily(), tracing(None):
            # One example per argument slot. Only what the program reads is
            # staged: a tensor at its profiling shape, an integer at its
            # traced value.
            examples = [
                torch.empty_strided(
                    tuple(profile_value(d) for d in v.shape),
                    tuple(profile_value(d) for d in v.stride()),
                    dtype=v.dtype,
                    device="cpu",
                ).fill_(1)
                if slot.kind == "tensor" and slot.read
                else profile_value(v)
                if slot.symbol is not None
                else None
                for slot, v in zip(local.input_slots, local.export_inputs)
            ]
            compiled = _compile_local_graph(
                local.program,
                examples,
                search_iterations=search_iterations,
                search_log=search_log,
                max_intermediate_bytes=max_intermediate_bytes,
                memory_budget_bytes=memory_budget_bytes,
            )
            record.bounds = compiled._graph.dim_bounds
        record.program = compiled

    return ReferenceOutputCode(local=local, compiled=compiled, record=record)


class ReferenceAOTBackend:
    """Callable torch.compile backend; inspect ``graphs``.

    The same compiler handles static/dynamic forward, backward, and
    inference. No local computation falls back to eager PyTorch, and a graph
    that contains collective operations is refused.
    """

    def __init__(
        self,
        *,
        search_iterations=1,
        log=False,
        max_intermediate_bytes=None,
        memory_budget_bytes=None,
    ):
        self.graphs: list[GraphRecord] = []
        self.search_iterations = search_iterations
        self.search_log = log
        self.compile_options = {
            "search_iterations": search_iterations,
            "search_log": log,
            "max_intermediate_bytes": max_intermediate_bytes,
            "memory_budget_bytes": memory_budget_bytes,
        }
        if not isinstance(search_iterations, int) or search_iterations < 1:
            raise ValueError("search_iterations must be a positive integer")
        for name, value in (
            ("max_intermediate_bytes", max_intermediate_bytes),
            ("memory_budget_bytes", memory_budget_bytes),
        ):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        self._backend = aot_autograd(
            fw_compiler=partial(self._compile, phase="forward"),
            bw_compiler=partial(self._compile, phase="backward"),
            inference_compiler=partial(self._compile, phase="inference"),
            # Decompose once, while AOTAutograd traces: the graph the local
            # compiler receives is already in the translator's vocabulary.
            decompositions=_aot_decomposition_table,
            # An input mutation stays in the graph as a trailing ``copy_``
            # and becomes a mutation output the runtime writes in place;
            # AOTAutograd's epilogue then has nothing to copy back.
            keep_inference_input_mutations=True,
        )

    def __call__(self, gm, example_inputs):
        return self._backend(gm, example_inputs)

    def _compile(self, gm, example_inputs, *, phase):
        refuse_collectives(gm, phase, "luminal_reference")
        record = GraphRecord(phase, gm.code)
        self.graphs.append(record)
        return _compile_graph(gm, example_inputs, record, **self.compile_options)
