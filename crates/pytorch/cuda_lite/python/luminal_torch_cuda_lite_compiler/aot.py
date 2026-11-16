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

import json
from functools import partial

import torch
from luminal_torch_common.aot_graph import (
    GraphRecord,
    prepare_local_graph,
    refuse_collectives,
)
from luminal_torch_common.graph_prep import _aot_decomposition_table, private_graph_copy
from luminal_torch_cuda_lite_runtime import Code
from torch._dynamo.backends.common import aot_autograd

from . import _luminal
from .backend import _compile_local_graph
from .boundary import layout_from_spec

__all__ = ["Code", "CudaAOTBackend", "GraphRecord"]


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


def _report_output_strides(local, graph):
    """Tell AOTAutograd the strides every forward output is returned at:
    one entry per output of the AOT graph, read from the layout the output
    was bound at (eager's, row-major, or a view's), ``None`` for a
    forwarded input or a symbolic layout."""
    import sympy
    from torch._guards import TracingContext

    context = TracingContext.try_get()
    if context is None or context.output_strides is None:
        return
    names = graph.output_names
    layouts = graph.output_layouts
    strides = [None] * len(local.output_slots)
    for i in local.positions:
        # The row the leaf's value is written in is the plan's to say.
        row = local.output_slots[i].index
        layout = layout_from_spec(names[row], *layouts[row])
        spelled = [sympy.sympify(stride) for stride in layout.strides]
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
    local = prepare_local_graph(
        gm, inputs, device_type="cuda", boundary_plan=_luminal.boundary_plan
    )

    # The output leaves as plain rows: what the code object builds a call's
    # output from.
    leaves = local.leaf_rows()
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
            code, graph = _compile_local_graph(
                local.program,
                local.example_inputs,
                leaves,
                search_iterations=search_iterations,
                search_log=search_log,
                device_budget_bytes=device_budget_bytes,
                max_intermediate_bytes=max_intermediate_bytes,
                dense_outputs=dense,
            )
        _report_output_strides(local, graph)
        record.bounds = graph.dim_bounds
        record.slots = tuple(graph.input_slots)
        record.checks = tuple(graph.runtime_assertions)
    else:
        # Nothing is computed: the graph forwards arguments, sizes and
        # constants, and its code object has no program.
        code = bytes(
            _luminal.forwarding_code(local.program.program_json(), json.dumps(leaves))
        )

    # From here on a call needs only the code object. The lowered graph, the
    # translation and the search state are the compiler's.
    compiled = _Recorded(code, record)
    if local.computes:
        record.program = compiled
    return compiled


class CudaAOTBackend:
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
            decompositions=self.decompositions,
            keep_inference_input_mutations=True,
        )

    def __call__(self, gm, example_inputs):
        return self._backend(gm, example_inputs)

    def _compile(self, gm, example_inputs, *, phase):
        refuse_collectives(gm, phase, "luminal_torch_cuda_lite_compiler")
        record = GraphRecord(phase, gm.code)
        self.graphs.append(record)
        return _compile_graph(
            gm, example_inputs, record, phase=phase, **self.compile_options
        )
