"""AOTAutograd integration for CUDA programs.

AOTAutograd owns differentiation and hands this adapter one functional ATen
graph per phase: forward, backward, or inference. Each graph compiles whole
through PT2 translation, egglog search, and the CUDA-lite runtime; there is no
eager fallback for local computation. Symbolic shapes are retained through
compilation and bound at execution.

Input mutations stay in the graph (``keep_inference_input_mutations``): the
trailing ``copy_`` AOTAutograd emits per mutated input becomes a mutation
output bound on that input's own device storage, so a KV-cache update or a
buffer increment is one in-place write rather than a result copied back.

Collective operations are refused by name. A program that communicates must be
split by the application so that each compiled graph is collective-free.
"""

import dataclasses
from functools import partial
from typing import Any

import torch
from luminal_reference.aot_graph import (
    GraphRecord,
    LocalGraph,
    prepare_local_graph,
    refuse_collectives,
)
from luminal_reference.export_utils import _aot_decomposition_table, private_graph_copy
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.output_code import OutputCode

from .backend import _compile_local_graph

__all__ = ["CudaAOTBackend", "CudaOutputCode", "GraphRecord"]


@dataclasses.dataclass
class CudaOutputCode(OutputCode):
    """One AOTAutograd graph compiled for the CUDA-lite runtime, as the
    object torch's AOT machinery holds and calls.

    ``local`` is the graph's boundary, ``compiled`` the compiled program
    (``None`` for a graph that only forwards inputs and sizes), and
    ``record`` the inspection record of the compile.

    torch prepares the saved form of every graph while ``aot_compile``
    runs. The CUDA runtime has no serialized form of a searched plan yet,
    so a graph it computes cannot be saved, and ``aot_compile`` is refused
    here, by name.
    """

    local: LocalGraph | None = None
    compiled: Any = None
    record: GraphRecord | None = None

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
        """Turn this object into its saved form, on the shallow copy torch
        makes while ``aot_compile`` runs. Only a graph with no compiled
        program has one: its boundary, as plain data."""
        if self.compiled is not None:
            raise NotImplementedError(
                "luminal_cuda_lite cannot save a compiled graph yet: the CUDA runtime "
                "has no serialized form of a searched plan (LUM-806, LUM-864)"
            )
        self.local = self.local.portable()
        self.record = None

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        """torch calls this once on a loaded object. A saved graph has no
        compiled program, so there is nothing to rebuild."""

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Part of the ``OutputCode`` interface; there is nothing to bundle."""


def _saved_activation_outputs(local, phase):
    """The output leaves that are the forward's activations saved for the
    backward graph and computed by the program.

    Nobody but this backend reads a saved activation again, so its layout
    is the compiler's to choose: row-major, which every plan can write,
    rather than the view layout eager happened to give it. The choice is
    reported to AOTAutograd, which restrides the backward's placeholders
    to match (as it does for Inductor). Best effort: a symbolic extent
    keeps eager's layout, since only literal strides are reported.
    """
    from torch._guards import TracingContext

    context = TracingContext.try_get()
    if (
        phase != "forward"
        or context is None
        or context.fw_metadata is None
        or context.output_strides is None
    ):
        return frozenset()
    leaves = range(len(local.output_slots))
    saved = set(leaves[context.fw_metadata.tensors_saved_for_backwards_slice])
    dense = set()
    for i in local.positions:
        value = local.leaf_values[i]
        if (
            i in saved
            and isinstance(value, torch.Tensor)
            and all(isinstance(size, int) for size in value.shape)
        ):
            dense.add(i)
    return frozenset(dense)


def _report_output_strides(local, compiled):
    """Tell AOTAutograd the strides every forward output is returned at:
    one entry per output of the AOT graph, read from the layout the output
    was bound at (eager's, row-major, or a view's), ``None`` for a
    forwarded input or a symbolic layout."""
    import sympy
    from torch._guards import TracingContext

    context = TracingContext.try_get()
    if context is None or context.output_strides is None:
        return
    strides = [None] * len(local.output_slots)
    for i in local.positions:
        # The row the leaf's value is written in is the plan's to say.
        binding = compiled._output_bindings[local.output_slots[i].index]
        spelled = [sympy.sympify(stride) for stride in binding.layout.strides]
        if all(stride.is_Integer for stride in spelled):
            strides[i] = tuple(int(stride) for stride in spelled)
    context.output_strides.extend(strides)


def _rebase_saved_activations(gm, inputs, phase):
    """Saved activations reach the backward graph as the views eager made,
    but this backend allocated every one of them fresh in the forward, at
    the strides it reported and at offset zero. AOTAutograd restrides the
    backward's EXAMPLE inputs to the reported strides and leaves the
    placeholders' own metadata as traced, so restate that metadata from
    the example: the strides torch applied, and offset zero. Only the
    saved-activation placeholders, which sit between the saved sizes and
    the tangents, are the forward's to describe."""
    from torch._guards import TracingContext

    context = TracingContext.try_get()
    metadata = context.fw_metadata if context is not None else None
    if (
        phase != "backward"
        or metadata is None
        or metadata.num_symints_saved_for_bw is None
    ):
        return gm, inputs
    gm = private_graph_copy(gm)
    inputs = list(inputs)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    start = metadata.num_symints_saved_for_bw
    stop = len(placeholders) - len(metadata.traced_tangents)
    for index in range(start, max(start, stop)):
        node = placeholders[index]
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        example = inputs[index] if isinstance(inputs[index], torch.Tensor) else value
        rebased = value.as_strided(value.shape, example.stride(), 0)
        node.meta["val"] = rebased
        inputs[index] = rebased
    return gm, inputs


def _compile_graph(
    gm,
    inputs,
    record,
    search_iterations,
    search_log=False,
    device_budget_bytes=None,
    max_intermediate_bytes=None,
    phase="inference",
):
    gm, inputs = _rebase_saved_activations(gm, inputs, phase)
    local = prepare_local_graph(gm, inputs, device_type="cuda")

    compiled = None
    if local.computes:
        dense = _saved_activation_outputs(local, phase)
        record.input_shapes = tuple(
            tuple(v.shape) for v in local.inputs if isinstance(v, torch.Tensor)
        )
        record.targets = tuple(
            str(n.target) for n in gm.graph.nodes if n.op == "call_function"
        )
        # The boundary reads dtypes, extents and strides off AOTAutograd's
        # fake inputs and addresses nothing until a call, so they are the
        # example inputs as they are. Compile outside the surrounding AOT
        # FakeTensor mode: constant lifting and serialization touch the
        # graph's real constants, which that mode refuses.
        from torch._guards import tracing
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily(), tracing(None):
            compiled = _compile_local_graph(
                local.program,
                local.export_inputs,
                search_iterations=search_iterations,
                search_log=search_log,
                device_budget_bytes=device_budget_bytes,
                max_intermediate_bytes=max_intermediate_bytes,
                dense_outputs=dense,
            )
        _report_output_strides(local, compiled)
        record.bounds = compiled._graph.dim_bounds
        record.program = compiled

    return CudaOutputCode(local=local, compiled=compiled, record=record)


class CudaAOTBackend:
    """Callable torch.compile backend; inspect ``graphs``.

    The same compiler handles static/dynamic forward, backward, and
    inference. No local computation falls back to eager PyTorch, and a graph
    that contains collective operations is refused.
    """

    def __init__(
        self,
        *,
        search_iterations=1,
        search_log=False,
        device_budget_bytes=None,
        max_intermediate_bytes=None,
    ):
        self.graphs: list[GraphRecord] = []
        self.compile_options = {
            "search_iterations": search_iterations,
            "search_log": search_log,
            "device_budget_bytes": device_budget_bytes,
            "max_intermediate_bytes": max_intermediate_bytes,
        }
        self._backend = aot_autograd(
            fw_compiler=partial(self._compile, phase="forward"),
            bw_compiler=partial(self._compile, phase="backward"),
            inference_compiler=partial(self._compile, phase="inference"),
            # Decompose once, while AOTAutograd traces: the graph the local
            # compiler receives is already in the translator's vocabulary.
            decompositions=_aot_decomposition_table,
            keep_inference_input_mutations=True,
        )

    def __call__(self, gm, example_inputs):
        return self._backend(gm, example_inputs)

    def _compile(self, gm, example_inputs, *, phase):
        refuse_collectives(gm, phase, "luminal_cuda_lite")
        record = GraphRecord(phase, gm.code)
        self.graphs.append(record)
        return _compile_graph(
            gm, example_inputs, record, phase=phase, **self.compile_options
        )
