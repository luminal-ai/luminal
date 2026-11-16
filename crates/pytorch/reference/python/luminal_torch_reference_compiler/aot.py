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

import json
from functools import partial

import torch
from luminal_torch_common.aot_graph import (
    GraphRecord,
    prepare_local_graph,
    refuse_collectives,
)
from luminal_torch_common.dimensions import profile_value
from luminal_torch_common.graph_prep import _aot_decomposition_table
from luminal_torch_reference_runtime import Code
from torch._dynamo.backends.common import aot_autograd

from . import _luminal
from .backend import _compile_local_graph

__all__ = ["Code", "GraphRecord", "ReferenceAOTBackend"]


class _Recorded(Code):
    """A code object that also counts its calls on the compiler's inspection
    record. It saves as a plain ``Code``: the record is the compiler's, and
    a saved program names the runtime alone."""

    def __init__(self, code: bytes, record: GraphRecord) -> None:
        super().__init__(code)
        self.record = record

    def __call__(self, inputs: list):
        self.record.executions += 1
        return super().__call__(inputs)


def _stand_in(value: torch.Tensor) -> torch.Tensor:
    """The data the search executes candidates on in place of an argument
    torch describes without data.

    AOTAutograd compiles ahead of the first call, so an argument has a
    shape, strides and a dtype, and no values. The search still executes
    every candidate, and the reference runtime refuses a read outside a
    tensor's extent, so the stand-in has to be data the program accepts.
    Integer and boolean tensors are the ones that index and select: zero
    is inside every extent that has an element, and an all-false mask
    selects nothing. A floating-point tensor is ones.
    """
    stand_in = torch.empty_strided(
        tuple(profile_value(d) for d in value.shape),
        tuple(profile_value(d) for d in value.stride()),
        dtype=value.dtype,
        device="cpu",
    )
    indexes_or_selects = not (value.dtype.is_floating_point or value.dtype.is_complex)
    return stand_in.fill_(0 if indexes_or_selects else 1)


def _compile_graph(
    gm,
    inputs,
    record,
    search_iterations,
    search_log=False,
    max_intermediate_bytes=None,
    memory_budget_bytes=None,
):
    local = prepare_local_graph(
        gm, inputs, device_type="cpu", boundary_plan=_luminal.boundary_plan
    )

    # The output leaves as plain rows: what the code object builds a call's
    # output from.
    leaves = local.leaf_rows()
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
                _stand_in(v)
                if slot.kind == "tensor" and slot.read
                else profile_value(v)
                if slot.symbol is not None
                else None
                for slot, v in zip(local.input_slots, local.example_inputs)
            ]
            code, graph = _compile_local_graph(
                local.program,
                examples,
                leaves,
                search_iterations=search_iterations,
                search_log=search_log,
                max_intermediate_bytes=max_intermediate_bytes,
                memory_budget_bytes=memory_budget_bytes,
            )
            record.bounds = graph.dim_bounds
            record.slots = tuple(graph.input_slots)
            record.checks = tuple(graph.runtime_assertions)
    else:
        # Nothing is computed: the graph forwards arguments, sizes and
        # constants, and its code object has no plan.
        code = bytes(
            _luminal.forwarding_code(local.program.program_json(), json.dumps(leaves))
        )

    # From here on a call needs only the code object. The lowered graph, the
    # translation and the search state are the compiler's.
    compiled = _Recorded(code, record)
    if local.computes:
        record.program = compiled
    return compiled


class ReferenceAOTBackend:
    """Callable torch.compile backend; inspect ``graphs``.

    The same compiler handles static/dynamic forward, backward, and
    inference. No local computation falls back to eager PyTorch, and a graph
    that contains collective operations is refused.
    """

    # The decomposition table AOTAutograd applies while it traces: the
    # translator's own vocabulary. A test that wants every core
    # decomposition applied overrides this on a subclass.
    decompositions = staticmethod(_aot_decomposition_table)

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
            decompositions=self.decompositions,
            # An input mutation stays in the graph as a trailing ``copy_``
            # and becomes a mutation output the runtime writes in place;
            # AOTAutograd's epilogue then has nothing to copy back.
            keep_inference_input_mutations=True,
        )

    def __call__(self, gm, example_inputs):
        return self._backend(gm, example_inputs)

    def _compile(self, gm, example_inputs, *, phase):
        refuse_collectives(gm, phase, "luminal_torch_reference_compiler")
        record = GraphRecord(phase, gm.code)
        self.graphs.append(record)
        return _compile_graph(gm, example_inputs, record, **self.compile_options)
