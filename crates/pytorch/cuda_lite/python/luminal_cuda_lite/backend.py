"""The luminal_cuda_lite torch.compile backend.

The backend self-exports the incoming GraphModule with ``torch.export``,
saves the program to a temporary ``.pt2``, hands it to the Rust extension
for translation, declares the boundary, and returns a callable that binds
caller tensors per invocation.

This is the GPU twin of the ``luminal_reference`` backend. The export
preparation (dynamic-shape handling, scalar-output boxing, decomposition
fallback, dead-op/guard cleanup) is backend-neutral, so it is imported
from the reference package rather than duplicated.

DEVICE MODEL: every boundary tensor is the caller's own device memory.
Its element strides are read once at compile time (``boundary.py``) and
declared to the runtime, which binds it on one buffer id; each call hands
that buffer the tensor's address. Which map a chain is — contiguous,
column-major, neither — is discovered in the e-graph, never decided here.
A user-visible output is bound at eager's exact strides — read from the
same fake value the input layouts are read from — and allocated per call
at the byte span the installed plan discloses for it, which exceeds the
tensor's own bytes when the plan elected a view of the escape cell it
writes through. The plan writes that storage in place either way, so the
tensor the caller receives is the one the runtime wrote rather than a copy
of it. Its layout must be the one eager gives it: an output whose elected
strides are not the declared ones is refused by name, because this backend
returns outputs in eager's layout. Nothing is copied to the host and no
layout is reinterpreted — a tensor the runtime cannot bind, one whose
storage offset is non-zero among them, is refused by name. Aliasing has
one spelling: two bindings naming one buffer id, which is how a writeback
and the input it mutates share a pointer.

STREAM AND ARENA: the runtime always launches captured CUDA graphs, which
the legacy default stream cannot host, so it runs on a dedicated
``torch.cuda.Stream`` borrowed through ``use_borrowed_stream`` and ordered
against the caller's stream with ``side.wait_stream(caller)`` before the
launch and ``caller.wait_stream(side)`` after. The intermediate-scratch
arena is PyTorch's, not the runtime's: ``arena_bytes()`` is the searched
plan set's requirement, the wrapper ``caching_allocator_alloc``s exactly
that against the side stream, binds it with ``set_arena``, and
``caching_allocator_delete``s it right after ``execute`` — so PyTorch
accounts for the bytes and can reuse the block on the next call.
"""

import concurrent.futures
import os
import tempfile
from collections.abc import Sequence
from typing import Any

import torch
from luminal_reference.export_utils import (
    _box_scalar_graph_outputs,
    _decomp_table,
    _drop_dead_data_dependent_ops,
    _drop_input_guards,
    _lower_sym_sum,
    _register_cache_serialization,
    private_graph_copy,
)
from torch.export import Dim, export

from . import _luminal
from .boundary import (
    Binding,
    UnsupportedBoundary,
    boundary_layout,
    boundary_shape,
    buffer_nbytes,
    call_dim_values,
    check_binding,
    declared_strides,
    layout_from_spec,
    layout_spec,
    storage_span,
)

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

# The output specs this backend declares to the runtime. Anything else
# (a buffer mutation, a gradient, a token) would reach the translator as an
# ordinary returned tensor, because the PT2 signature reader models only
# ``user_input_mutation``.
_BOUND_OUTPUT_KINDS = frozenset(
    {"USER_OUTPUT", "USER_INPUT_MUTATION", "BUFFER_MUTATION"}
)


def _torch_dtype(dtype_code: int) -> torch.dtype:
    dtype = _PT2_TO_TORCH.get(dtype_code)
    if dtype is None:
        raise RuntimeError(
            f"luminal_cuda_lite cannot materialize PT2 dtype code {dtype_code}"
        )
    return dtype


def _node_fakes(ep: Any) -> dict[str, Any]:
    """The fake example value behind every graph node, by node name.

    It is the exported program's OWN symbolic metadata — the symbols the
    translator reads back out of the saved ``.pt2`` — so a size or stride
    stated from it names a dimension the runtime knows, at whatever extent
    a later call gives it. One table for both sides of the boundary: an
    output's layout is read from the same metadata an input's is.
    """
    fakes: dict[str, Any] = {}
    for node in ep.graph_module.graph.nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            fakes[node.name] = value
    return fakes


def _boundary_tensors(
    ep: Any, export_inputs: Sequence[Any]
) -> list[tuple[str, str, torch.Tensor, Any]]:
    """(graph input name, kind, tensor, fake example value) for every graph
    input, in export order. The names are the exported program's
    placeholder names, which are what the translator reads back out of the
    saved ``.pt2``. The fake carries the program's symbolic sizes and
    strides for a user input; a parameter, buffer or constant is the
    concrete tensor the state dict holds, and states itself."""
    fakes = _node_fakes(ep)
    rows: list[tuple[str, str, torch.Tensor, Any]] = []
    user_index = 0
    for spec in ep.graph_signature.input_specs:
        name = getattr(spec.arg, "name", None)
        kind = spec.kind.name
        if name is None:
            raise UnsupportedBoundary(
                f"graph input {kind.lower()} {spec.target!r} is not a tensor"
            )
        if kind == "USER_INPUT":
            if user_index >= len(export_inputs):
                raise RuntimeError(
                    f"export declared more user inputs than example_inputs: {name!r}"
                )
            value = export_inputs[user_index]
            user_index += 1
        elif kind in ("PARAMETER", "BUFFER"):
            if spec.target not in ep.state_dict:
                raise RuntimeError(
                    f"{name}: {spec.target!r} is not in the exported state_dict"
                )
            value = ep.state_dict[spec.target]
        elif kind == "CONSTANT_TENSOR":
            if spec.target not in ep.constants:
                raise RuntimeError(
                    f"{name}: {spec.target!r} is not in the exported constants"
                )
            value = ep.constants[spec.target]
        else:
            raise UnsupportedBoundary(
                f"{name}: graph inputs of kind {kind.lower()} are not bound by luminal_cuda_lite"
            )
        if not isinstance(value, torch.Tensor):
            raise UnsupportedBoundary(
                f"{name}: graph input {spec.target!r} is not a tensor"
            )
        rows.append(
            (name, kind, value, fakes.get(name) if kind == "USER_INPUT" else None)
        )
    if user_index != len(export_inputs):
        raise RuntimeError(
            f"export consumed {user_index} of {len(export_inputs)} example_inputs"
        )
    return rows


def _refuse_unbound_outputs(ep: Any) -> None:
    """Refuse an output the boundary has no statement for."""
    for spec in ep.graph_signature.output_specs:
        kind = spec.kind.name
        if kind in _BOUND_OUTPUT_KINDS:
            continue
        name = getattr(spec.arg, "name", spec.target)
        raise UnsupportedBoundary(
            f"output {name!r}: {kind.lower()} outputs are not declared to the runtime"
        )


def _output_fake(name: str, fakes: dict[str, Any]) -> Any:
    """The traced example value an output's layout and extents are read
    from, refused by name when the program carries none."""
    fake = fakes.get(name)
    if fake is None:
        raise UnsupportedBoundary(
            f"output {name!r} carries no exported example value, so the layout eager "
            "gives it cannot be read"
        )
    return fake


def _output_layout_rows(ep: Any) -> list[tuple[str, str, list[str]]]:
    """One ``(graph name, layout tag, element strides)`` row per graph
    output the caller allocates: the element strides EAGER gives it, read
    from the traced fake value exactly like an input's.

    This is a LAYOUT STATEMENT PER NAME, not the output list — which
    outputs a program has, and in which order, is the translation's to say
    (`graph.output_names`). A writeback takes no row: it writes the storage
    of the input it mutates and is bound at that input's layout.
    """
    fakes = _node_fakes(ep)
    writebacks = {
        spec.arg.name
        for spec in ep.graph_signature.output_specs
        if spec.kind.name in ("USER_INPUT_MUTATION", "BUFFER_MUTATION")
    }
    rows: list[tuple[str, str, list[str]]] = []
    stated: set[str] = set()
    for position, spec in enumerate(ep.graph_signature.output_specs):
        if spec.kind.name != "USER_OUTPUT":
            continue
        name = getattr(spec.arg, "name", None)
        if name is None:
            raise UnsupportedBoundary(
                f"graph output {position} ({spec.arg!r}) is not a tensor, so no boundary "
                "layout states it"
            )
        # One value returned twice is one statement: the same name, at the
        # same layout, stated once.
        if name in writebacks or name in stated:
            continue
        fake = _output_fake(name, fakes)
        stated.add(name)
        tag, strides = layout_spec(boundary_layout(name, fake))
        rows.append((name, tag, list(strides)))
    return rows


def _output_alias_rows(ep: Any) -> list[tuple[str, str, int]]:
    """One ``(output, owner)`` row per user output whose traced example value
    shares its STORAGE with an earlier boundary tensor: a graph input, a
    writeback, or an earlier output. Eager hands the caller a view of that
    tensor, so the runtime binds the output on the owner's buffer and the
    call returns a view. This is read off the fake tensors torch traced —
    which storage each value lives in — and states it; nothing is derived
    from op names."""
    fakes = _node_fakes(ep)
    names: list[str] = []
    for spec in list(ep.graph_signature.input_specs) + list(
        ep.graph_signature.output_specs
    ):
        name = getattr(spec.arg, "name", None)
        if name is not None and name not in names:
            names.append(name)
    owner_by_storage: dict[int, str] = {}
    outputs = {
        getattr(spec.arg, "name", None)
        for spec in ep.graph_signature.output_specs
        if spec.kind.name == "USER_OUTPUT"
    }
    # Rows carry the view's element offset RELATIVE to its owner, so the
    # returned view addresses the owner's storage where eager's does.
    rows: list[tuple[str, str, int]] = []
    for name in names:
        fake = fakes.get(name)
        if not isinstance(fake, torch.Tensor):
            continue
        storage = fake.untyped_storage()._cdata
        owner = owner_by_storage.setdefault(storage, name)
        if owner != name and name in outputs:
            offset = int(fake.storage_offset()) - int(fakes[owner].storage_offset())
            rows.append((name, owner, offset))
    return rows


def _refuse_overlapping_writebacks(
    named: Sequence[tuple[str, torch.Tensor]], writebacks: frozenset
) -> None:
    """Two boundary tensors whose device storage overlaps are refused when
    either one is written through.

    Read-only aliasing — tied weights, a tensor passed twice — is two
    External buffers carrying one address, which is a fact about the
    caller's memory and harmless. A writeback is not: the runtime knows
    one buffer per binding, so a write through one of an overlapping pair
    would land in the other's reads with no edge ordering them. Refused by
    name rather than bound.

    This is checked per call as well as at compile: Dynamo guards tensor
    identity, not storage overlap, so a program compiled on distinct
    tensors can be called as ``fn(x, x[:])``.
    """
    spans = [(name, *storage_span(tensor)) for name, tensor in named]
    for index, (name, start, stop) in enumerate(spans):
        for other, other_start, other_stop in spans[index + 1 :]:
            if start >= other_stop or other_start >= stop:
                continue
            target = next((row for row in (name, other) if row in writebacks), None)
            if target is None:
                continue
            raise UnsupportedBoundary(
                f"{name!r} and {other!r} share device storage and {target!r} is "
                "written back into; one buffer per binding cannot order a write "
                "against the other's reads"
            )


def _allocate_output(
    binding: Binding,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    span: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """One output's storage, plus the address and byte count its buffer is
    bound at.

    The plan discloses the span its backing buffer covers, which exceeds
    the output's own bytes when the plan elected a view of the escape cell
    it writes through. Allocating the span is what keeps that write inside
    storage PyTorch owns; the tensor handed back is a view of it at the
    declared strides, and reaches no further than the span the runtime
    computed from the same layout.
    """
    dense = torch.empty_strided(shape, strides, dtype=binding.dtype, device=device)
    if span <= dense.untyped_storage().nbytes():
        return dense, dense.data_ptr(), buffer_nbytes(dense)
    base = torch.empty(span, dtype=torch.uint8, device=device)
    # A span that is not a whole number of elements is viewed as the whole
    # elements it holds; the trailing bytes stay the runtime's to write.
    element = dense.element_size()
    whole = base[: span - span % element].view(binding.dtype)
    return whole.as_strided(shape, strides), base.data_ptr(), span


class CompiledModel:
    """Callable wrapper around a compiled CUDA-lite graph.

    Executions are zero-copy and the intermediate arena is PyTorch-owned:

    * every boundary tensor is bound by BUFFER ID to the caller's device
      pointer; a writeback and the input it mutates are one buffer and one
      pointer;
    * parameters and buffers are addressed once at compile time, user inputs
      and freshly allocated outputs once per call;
    * the intermediate-scratch arena is ``caching_allocator_alloc``'d for the
      duration of the call and ``caching_allocator_delete``'d right after, so
      PyTorch accounts for it and can reuse the block next call.
    """

    def __init__(
        self,
        graph: Any,
        ep: Any,
        input_bindings: Sequence[Binding],
        output_bindings: Sequence[Binding],
        scalar_output_positions: Sequence[int] = (),
        held_tensors: dict[str, torch.Tensor] | None = None,
        held_bindings: Sequence[Binding] = (),
        output_aliases: dict[str, tuple[str, int]] | None = None,
    ):
        self._graph = graph
        # Output name -> (the boundary tensor whose storage it is a view of,
        # the view's element offset relative to that tensor).
        self._output_aliases: dict[str, tuple[str, int]] = dict(output_aliases or {})
        self._output_names = list(graph.output_names)
        self._ep = ep
        self._input_bindings = list(input_bindings)
        self._input_names = [binding.name for binding in self._input_bindings]
        self._output_bindings = list(output_bindings)
        self._scalar_output_positions = frozenset(scalar_output_positions)
        # Parameters and buffers, by graph input name; a buffer writeback
        # lands in them. Their declared bindings are kept because a held
        # tensor is checked and re-addressed every call: `p.data = q` moves
        # a parameter's storage without changing any metadata Dynamo guards
        # on, so an address taken once at compile can go stale under a
        # program that still passes every guard.
        self._held: dict[str, torch.Tensor] = dict(held_tensors or {})
        self._held_bindings = list(held_bindings)
        # Fixed once a plan set is searched; the per-execution arena sizes to it.
        self._arena_bytes = graph.arena_bytes()
        # The runtime always launches captured CUDA graphs, which the legacy
        # default stream cannot host, so it runs on a dedicated side stream
        # ordered against the caller's stream with events.
        self._side_stream: torch.cuda.Stream | None = None
        self._region_execution_state = {}
        self._output_mutations = graph.output_mutations
        self._output_returns = graph.output_returns
        # The graph inputs this program writes back into. While it is
        # empty, overlapping boundary tensors are read-only aliasing and
        # need no per-call check.
        self._writebacks = frozenset(
            mutation for mutation in self._output_mutations if mutation is not None
        )

    def _boundary_tensor(
        self,
        name: str,
        inputs: Sequence[torch.Tensor],
        out_tensors: Sequence[torch.Tensor | None],
    ) -> torch.Tensor:
        """The live tensor a boundary name stands for on this call: a user
        input, a held parameter/buffer, a writeback's destination, or an
        output allocated this call."""
        if name in self._input_names:
            return inputs[self._input_names.index(name)]
        if name in self._held:
            return self._held[name]
        if name in self._output_names:
            index = self._output_names.index(name)
            mutation = self._output_mutations[index]
            if mutation is not None:
                return self._mutation_destination(mutation, inputs)
            tensor = out_tensors[index]
            if tensor is not None:
                return tensor
        raise RuntimeError(
            f"luminal_cuda_lite: {name!r} is not a boundary tensor this call holds"
        )

    def _mutation_destination(
        self, mutation: str, inputs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """The tensor a writeback landed in: the call's user input, or the
        held parameter/buffer (PT2 ``buffer_mutation``) whose buffer the
        sink shares."""
        if mutation in self._input_names:
            return inputs[self._input_names.index(mutation)]
        if mutation in self._held:
            return self._held[mutation]
        raise RuntimeError(
            f"luminal_cuda_lite: mutation target {mutation!r} is neither a user "
            "input nor a held parameter/buffer"
        )

    def configure_region(self, *, static_outputs=False, external_cuda_graph=False):
        """Give a region its own persistent PyTorch-owned boundary storage."""
        if external_cuda_graph and not static_outputs:
            raise ValueError("external CUDA graph capture requires static_outputs=True")
        self._region_current = True
        self._region_static = static_outputs
        self._region_external_capture = external_cuda_graph
        self._region_allocations = {}
        self._region_mutations = {}
        self._region_arena = None

    @property
    def writeback_inputs(self):
        return {
            name: mutation
            for name, mutation in zip(
                self._graph.output_names, self._graph.output_mutations
            )
            if mutation is not None
        }

    def __call__(self, *args: torch.Tensor) -> Any:
        # Under dynamic shapes Dynamo's wrapper passes the graph's symbolic
        # shape values alongside the tensor inputs (as SymInt or int). The
        # compiled program folds those symbols into `sym_size`, so it has no
        # scalar inputs: keep tensors, drop the scalars.
        inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if len(inputs) != len(self._input_bindings):
            raise RuntimeError(
                f"luminal_cuda_lite expected {len(self._input_bindings)} inputs, "
                f"got {len(inputs)}"
            )
        if not inputs:
            raise RuntimeError("luminal_cuda_lite requires at least one tensor input")
        device = inputs[0].device
        for value in inputs:
            if value.device != device:
                raise RuntimeError(
                    "luminal_cuda_lite requires all inputs on one device, got "
                    f"{device} and {value.device}"
                )
        stream = torch.cuda.current_stream(device)
        if getattr(self, "_region_current", False):
            side = stream
        else:
            if self._side_stream is None:
                self._side_stream = torch.cuda.Stream(device=device)
            side = self._side_stream
        static = getattr(self, "_region_static", False)
        signature = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in inputs)
        execution_key = (id(self), signature, side.cuda_stream)
        warmed = self._region_execution_state.get("last") == execution_key
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and (not static or not warmed):
            raise RuntimeError(
                "warm up each region shape with static_outputs=True before CUDA capture"
            )
        if static:
            for binding, value in zip(self._input_bindings, inputs):
                if binding.name in self._writebacks:
                    previous = self._region_mutations.setdefault(
                        binding.name, value.data_ptr()
                    )
                    if previous != value.data_ptr():
                        raise ValueError(
                            "static region mutation target allocation changed"
                        )

        # CHECK EVERYTHING BEFORE ADDRESSING ANYTHING: every tensor this
        # call binds — inputs, held tensors, freshly allocated outputs — is
        # checked here, and the pointers go in afterwards inside the `try`
        # whose `finally` clears them, so no refusal can leave one standing.
        # A tensor whose dtype, rank, extents or element strides are not the
        # declared ones is refused by name. The dimension map is the whole
        # call's, so a stride declared over a dimension ANOTHER input
        # carries is still checked here.
        bound = list(zip(self._input_bindings, inputs)) + [
            (binding, self._held[binding.name]) for binding in self._held_bindings
        ]
        dims = call_dim_values(bound)
        for binding, value in bound:
            check_binding(binding, value, dims)
        if self._writebacks:
            # Dynamo guards tensor identity, not storage overlap, so a
            # program compiled on distinct tensors can still be called with
            # two views of one allocation.
            _refuse_overlapping_writebacks(
                list(zip(self._input_names, inputs)) + list(self._held.items()),
                self._writebacks,
            )

        # The dimensions THIS call states, read off its every input at once:
        # a compound extent is checked against them, so reading input by
        # input would check it against what the call before left behind.
        # Refused here, before any pointer is set.
        self._graph.bind_input_shapes(
            [(binding.name, list(value.shape)) for binding, value in bound]
        )

        # Output shapes depend on this call's dims, so they are read after
        # the input shapes are bound rather than cached at compile time.
        output_shapes = self._graph.output_shapes

        # Allocate every output that is not a writeback, AT THE STRIDES IT
        # IS BOUND AT: the declared layout at this call's dimensions, which
        # is what eager gives this output. A writeback's buffer IS its
        # target input's. Allocate under the side stream so the caching
        # allocator records the stream that will write them.
        with torch.cuda.stream(side):
            out_tensors: list[torch.Tensor | None] = []
            # What each output's buffer is addressed with: the base of its
            # storage and the byte span the plan writes through it, which is
            # the tensor's own span only when the plan writes it densely.
            out_spans: list[tuple[int, int] | None] = []
            for index, binding in enumerate(self._output_bindings):
                if self._output_mutations[index] is not None:
                    out_tensors.append(None)
                    out_spans.append(None)
                    continue
                name = binding.name
                shape = tuple(output_shapes[index])
                # The plan was retargeted onto the caller's buffer at search;
                # a plan that writes this output somewhere else is a runtime
                # invariant broken, not a call this caller can fix.
                backing = self._graph.output_backing_buffer(name)
                if backing != binding.buffer:
                    raise RuntimeError(
                        f"luminal_cuda_lite: output {name!r} is bound on buffer "
                        f"{binding.buffer}, but the installed plan writes it into "
                        f"buffer {backing}"
                    )
                declared = tuple(declared_strides(binding, shape, dims))
                elected = tuple(self._graph.output_elected_strides(name))
                if elected != declared:
                    raise RuntimeError(
                        f"luminal_cuda_lite: output {name!r} is elected at element "
                        f"strides {elected}, and eager's are {declared}. This backend "
                        "returns outputs in eager's layout and the plan elected a "
                        "different one (LUM-829 covers making this a toggle)."
                    )
                # A view of another boundary tensor has that tensor's storage:
                # nothing to allocate or address, the owner's binding does it.
                if name in self._output_aliases:
                    out_tensors.append(None)
                    out_spans.append(None)
                    continue
                # The span the plan writes through this buffer, at this call's
                # dimensions: the allocation is sized to it, never to the
                # tensor alone.
                span = self._graph.output_span_bytes(name)
                allocation_key = (index, shape, declared, span)
                allocation = (
                    self._region_allocations.get(allocation_key) if static else None
                )
                if allocation is None:
                    allocation = _allocate_output(
                        binding, shape, declared, span, device
                    )
                    if static:
                        self._region_allocations[allocation_key] = allocation
                tensor, base_ptr, base_bytes = allocation
                # The allocation is checked against the binding the runtime
                # writes through — rank, extents, element strides — so a
                # disagreement is refused by name rather than written past.
                check_binding(binding, tensor, dims)
                out_tensors.append(tensor)
                out_spans.append((base_ptr, base_bytes))

        # Order the side stream after everything the caller enqueued: this is
        # what makes reading the caller's inputs safe without a host sync.
        if side != stream:
            side.wait_stream(stream)

        # Per-execution intermediate arena from PyTorch's caching allocator,
        # associated with the stream that uses it.
        arena_bytes = max(self._arena_bytes, 1)
        if static:
            if self._region_arena is None:
                self._region_arena = torch.empty(
                    arena_bytes, dtype=torch.uint8, device=device
                )
            self._region_arena.record_stream(side)
            arena = self._region_arena.data_ptr()
        else:
            arena = torch.cuda.caching_allocator_alloc(arena_bytes, device, side)
        self._graph.use_borrowed_stream(side.cuda_stream)
        self._graph.set_arena(arena, arena_bytes)
        # EVERY CHECK IS BEHIND US: the addresses go in here and come out in
        # `finally`, so no refusal of this call can leave one standing.
        per_call: list[int] = []
        try:
            # Every input is bound as it is, on the buffer it was declared on.
            for binding, value in zip(self._input_bindings, inputs):
                self._graph.set_device_ptr(
                    binding.buffer, value.data_ptr(), buffer_nbytes(value)
                )
                per_call.append(binding.buffer)

            # Held tensors are re-addressed rather than cleared: the address a
            # parameter has now is the one this execution reads, and leaving it
            # set is what makes a skipped user-input binding — never a held one
            # — the thing an execute refuses by name.
            for binding in self._held_bindings:
                tensor = self._held[binding.name]
                self._graph.set_device_ptr(
                    binding.buffer, tensor.data_ptr(), buffer_nbytes(tensor)
                )

            for binding, tensor, spanned in zip(
                self._output_bindings, out_tensors, out_spans
            ):
                if tensor is None:
                    continue
                base_ptr, base_bytes = spanned
                self._graph.set_device_ptr(binding.buffer, base_ptr, base_bytes)
                per_call.append(binding.buffer)

            if static and warmed:
                for value in [*inputs, *self._held.values(), *out_tensors]:
                    if value is not None:
                        value.record_stream(side)
                self._graph.execute_async()
            else:
                self._graph.execute()
            self._region_execution_state["last"] = execution_key
        finally:
            if not static:
                torch.cuda.caching_allocator_delete(arena)
            # These addresses belong to this call only: forget them, so an
            # execute that skipped a binding refuses by name instead of
            # reading storage the caller has released.
            for buffer in per_call:
                self._graph.clear_device_ptr(buffer)

        # Hand ordering back to the caller's stream for the returned tensors.
        if stream != side:
            stream.wait_stream(side)

        results = []
        for index, mutation in enumerate(self._output_mutations):
            returned = self._output_returns[index]
            if mutation is not None:
                # The write already landed in the caller's tensor.
                if returned:
                    results.append(self._mutation_destination(mutation, inputs))
                continue
            if returned:
                # Scalar positions index the RETURNED tree, which is what
                # Dynamo hands back; rows that are not returned (writebacks)
                # sit before it and do not count.
                position = len(results)
                tensor = out_tensors[index]
                binding = self._output_bindings[index]
                if binding.name in self._output_aliases:
                    # Eager returns a view of a tensor the caller already
                    # holds; so does this call, at the strides eager gives it.
                    owner_name, offset = self._output_aliases[binding.name]
                    owner = self._boundary_tensor(owner_name, inputs, out_tensors)
                    shape = tuple(output_shapes[index])
                    strides = tuple(declared_strides(binding, shape, dims))
                    tensor = owner.as_strided(
                        shape, strides, owner.storage_offset() + offset
                    )
                # Scalar graph outputs were boxed into rank-zero tensors before
                # export; restore the Python scalar backend contract here.
                if position in self._scalar_output_positions:
                    results.append(tensor.item())
                else:
                    results.append(tensor)
        # Dynamo's backend contract: return the graph's output tree (a
        # sequence), even for one result. It unwraps single-tensor returns
        # for the user.
        return tuple(results)


def _is_dynamic(size: Any) -> bool:
    """A dim is dynamic only if it is a SymInt that is not a literal.

    Static dims can also surface as ``SymInt('8')``; those are numbers and
    must NOT be marked dynamic.
    """
    return isinstance(size, torch.SymInt) and not size.node.expr.is_number


def _dynamic_export(gm: torch.fx.GraphModule, example_inputs: Sequence[Any]) -> Any:
    """Export a Dynamo GraphModule, preserving its symbolic dimensions.

    Dynamo hands each free symbolic dimension to the backend as an explicit
    ``SymInt`` graph input (``view``/``reshape`` take integer shape arguments,
    so the symbol has to be a scalar input), and ``torch.export`` rejects a raw
    ``SymInt``. We therefore:

    1. read the dynamic dims from the *fake* tensor metadata **before**
       materialising any hint (materialising a hint specializes the ShapeEnv,
       and every later shape comes back concrete);
    2. erase unused ``SymInt`` placeholders and rewrite used ones to
       ``aten.sym_size.int(tensor, dim)``, which carries the same symbol on
       the tensor's own dimension — no scalar input survives;
    3. re-export with a ``dynamic_shapes`` tree rebuilt from the fake metadata.
    """
    # Work on a copy: Dynamo keeps the original GraphModule and checks its own
    # guards against it after the backend returns, so mutating it in place
    # trips "Guard failed on the same frame it was created". Only the GRAPH is
    # private; the module's weights are shared (see private_graph_copy).
    gm = private_graph_copy(gm)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    records: list[tuple[str, torch.fx.Node, Any]] = []
    tensor_dims: dict[Any, tuple[torch.fx.Node, int]] = {}
    for node, value in zip(placeholders, example_inputs):
        if isinstance(value, torch.SymInt):
            records.append(("sym", node, value))
            continue
        shape = getattr(node.meta.get("example_value"), "shape", None)
        if shape is None:
            shape = getattr(value, "shape", ())
        dims = {dim: Dim.AUTO for dim, size in enumerate(shape) if _is_dynamic(size)}
        for dim, size in enumerate(shape):
            if _is_dynamic(size):
                tensor_dims.setdefault(size.node.expr, (node, dim))
        records.append(("tensor", node, dims))

    # Pass 2: rewrite used SymInts to `sym_size`, drop unused ones.
    erased: set[int] = set()
    for kind, node, value in records:
        if kind != "sym":
            continue
        if not node.users:
            gm.graph.erase_node(node)
            erased.add(id(node))
            continue
        source = tensor_dims.get(value.node.expr)
        if source is None:
            raise RuntimeError(
                f"cannot locate the tensor dimension for symbolic input {value}"
            )
        tensor_node, dim = source
        with gm.graph.inserting_after(tensor_node):
            size = gm.graph.call_function(
                torch.ops.aten.sym_size.int, args=(tensor_node, dim)
            )
        node.replace_all_uses_with(size)
        gm.graph.erase_node(node)
        erased.add(id(node))
    if erased:
        gm.graph.lint()
        gm.recompile()

    inputs: list[Any] = []
    specs: list[Any] = []
    any_dynamic = False
    for (kind, node, info), value in zip(records, example_inputs):
        if id(node) in erased:
            continue
        inputs.append(value)
        if kind == "tensor" and info:
            specs.append(info)
            any_dynamic = True
        else:
            specs.append(None)

    dynamic_shapes = {"args": tuple(specs)} if any_dynamic else None

    # `torch.export` runs its own Dynamo pass. Running that inside the caller's
    # compile pollutes the caller's guard manager (the inner frame's `args`
    # guards leak into the outer sanity check), so isolate the nested compile
    # on its own thread with a fresh Dynamo compile context.
    def _export():
        return export(gm, tuple(inputs), dynamic_shapes=dynamic_shapes, strict=False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        ep = pool.submit(_export).result()
    return ep, inputs


def _compile_graph(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    options: dict | None = None,
    search_iterations: int | None = None,
    search_log: bool = False,
    device_budget_bytes: int | None = None,
    max_intermediate_bytes: int | None = None,
) -> CompiledModel:
    """The torch.compile backend entry point."""
    if options:
        search_iterations = options.get("search_iterations", search_iterations)
        search_log = options.get("search_log", search_log)
        device_budget_bytes = options.get("device_budget_bytes", device_budget_bytes)
        max_intermediate_bytes = options.get(
            "max_intermediate_bytes", max_intermediate_bytes
        )

    # HF DynamicCache must be pytree-registered before torch.export capture so
    # use_cache=True models can export. Idempotent.
    _register_cache_serialization()

    # Canonicalize scalar (SymInt/SymFloat/SymBool) graph outputs into rank-zero
    # tensors before the export capture. Work on a private copy: Dynamo holds
    # onto the original graph module for guard installation and retracing, and
    # mutating it here would corrupt that bookkeeping. The copy shares the
    # module's weights (see private_graph_copy).
    gm = private_graph_copy(gm)
    scalar_output_positions = _box_scalar_graph_outputs(gm)

    # The graph-module preprocessing above runs first; `_dynamic_export` then
    # rewrites the SymInt placeholders onto `sym_size` and runs the nested
    # `torch.export`, so the exported program keeps its symbolic dims.
    ep, export_inputs = _dynamic_export(gm, example_inputs)
    # LUM-499: drop dynamo-emitted input guards before run_decompositions calls
    # ep.module(), which would otherwise emit a `_guards_fn` containing
    # data-dependent .item() calls and unresolved `L[...]` references.
    _drop_input_guards(ep)
    _drop_dead_data_dependent_ops(ep.graph_module)
    # Functionalise WITHOUT decomposing: every in-place op becomes its
    # functional form plus a USER_INPUT_MUTATION spec naming the input storage
    # it writes, and a mutation through a view becomes an explicit scatter
    # (select_scatter / slice_scatter). The empty table decomposes nothing, so
    # composites the translator lowers directly (aten.linear) survive. The
    # translator reads mutations from the specs and refuses in-place ops.
    try:
        ep = ep.run_decompositions({})
    except AssertionError as exc:
        # torch's export functionalisation cannot express a write to an
        # input that shares storage with another input (its alias wrapper
        # returns a callable where export needs a graph); the program is
        # refused by name, not translated.
        if "expected compiled_fn to be GraphModule" not in str(exc):
            raise
        raise UnsupportedBoundary(
            "graph inputs share device storage and the program writes one of them: "
            "the export cannot functionalise a write through aliased inputs"
        ) from exc
    _drop_dead_data_dependent_ops(ep.graph_module)
    # Serde gap workaround; must run before save. See _lower_sym_sum.
    _lower_sym_sum(ep)

    return compile_exported(
        ep,
        export_inputs,
        search_iterations,
        scalar_output_positions,
        search_log=search_log,
        device_budget_bytes=device_budget_bytes,
        max_intermediate_bytes=max_intermediate_bytes,
    )


def compile_exported(
    ep,
    export_inputs,
    search_iterations=None,
    scalar_output_positions=(),
    *,
    search_log=False,
    device_budget_bytes=None,
    max_intermediate_bytes=None,
):
    """Compile a functional ExportedProgram with the native runtime."""

    def _save_and_compile(program: Any) -> Any:
        # Read every boundary tensor's element strides and declare them with
        # the program: the runtime binds what the caller has, or refuses it.
        _refuse_unbound_outputs(program)
        rows = _boundary_tensors(program, export_inputs)
        layouts = {
            name: boundary_layout(name, value, fake) for name, _, value, fake in rows
        }
        shapes = {name: boundary_shape(value, fake) for name, _, value, fake in rows}
        declared = []
        for name, _, _, _ in rows:
            tag, strides = layout_spec(layouts[name])
            declared.append((name, tag, list(strides)))
        # A user-visible output is declared at the strides EAGER gives it,
        # read from the traced fake value exactly like an input: what the
        # caller receives has the strides the uncompiled program hands back,
        # never a contiguous substitute.
        declared_outputs = _output_layout_rows(program)
        # An output that is a view of another boundary tensor is bound on
        # that tensor's buffer; the caller receives a view of what it holds.
        aliases = _output_alias_rows(program)
        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = os.path.join(tmp, "model.pt2")
            torch.export.save(program, pt2_path)
            graph = _luminal.compile(
                pt2_path,
                declared,
                declared_outputs,
                [(name, owner) for name, owner, _ in aliases],
            )
        tensors = {name: value for name, _, value, _ in rows}
        return (
            graph,
            tensors,
            layouts,
            shapes,
            {name: (owner, offset) for name, owner, offset in aliases},
        )

    try:
        graph, tensors, layouts, shapes, aliases = _save_and_compile(ep)
    except RuntimeError as exc:
        # The translator lowers a fixed op set. Decomposing the exported graph
        # rewrites higher-level composites into primitives the translator
        # handles, but it also rewrites ops it already lowers directly (e.g.
        # ``aten.linear`` -> ``aten.addmm``). Gate the aggressive pass on an
        # actual translator rejection so the common case keeps its original,
        # un-decomposed graph.
        if "unsupported ATen op" not in str(exc):
            raise
        ep = ep.run_decompositions(_decomp_table())
        _drop_dead_data_dependent_ops(ep.graph_module)
        _lower_sym_sum(ep)
        graph, tensors, layouts, shapes, aliases = _save_and_compile(ep)

    # Every boundary row, not just the user inputs: a parameter tied to
    # another parameter is read-only aliasing and stays two buffers on one
    # address, but a writeback whose storage overlaps another binding is
    # refused by name.
    writebacks = frozenset(
        mutation for mutation in graph.output_mutations if mutation is not None
    )
    _refuse_overlapping_writebacks(list(tensors.items()), writebacks)

    # Seed the symbolic dims from the declared shapes, address the parameter
    # and buffer pointers once (they outlive every call), and keep one
    # `Binding` per user input for the per-call check.
    held: dict[str, torch.Tensor] = {}
    held_bindings: list[Binding] = []
    input_bindings: list[Binding] = []
    for name in graph.input_names:
        if name not in tensors:
            raise RuntimeError(
                f"the translated program names an input {name!r} the export signature "
                f"does not declare (declared: {sorted(tensors)})"
            )
    graph.bind_input_shapes(
        [(name, list(tensors[name].shape)) for name in graph.input_names]
    )
    for name, kind, buffer in zip(
        graph.input_names, graph.input_kinds, graph.input_buffers
    ):
        value = tensors[name]
        binding = Binding(name, buffer, value.dtype, shapes[name], layouts[name])
        if kind == "user_input":
            input_bindings.append(binding)
            continue
        graph.set_device_ptr(buffer, value.data_ptr(), buffer_nbytes(value))
        held_bindings.append(binding)
        held[name] = value

    graph.search(
        search_iterations,
        search_log=search_log,
        device_budget_bytes=device_budget_bytes,
        max_intermediate_bytes=max_intermediate_bytes,
    )

    # WHICH OUTPUTS THE PROGRAM HAS IS THE TRANSLATION'S TO SAY, not the
    # export signature's: the translator resolves a returned alias of a
    # mutated value onto the writeback itself and may give one value two
    # names, so the outputs and their order are read off the translation
    # and their example values looked up by graph name.
    #
    # The runtime states the layout every output was bound at: a
    # user-visible one at eager's exact strides, a writeback at the layout
    # of the input it mutates. The declared extents are the exported
    # program's own — an ``int`` per literal axis, the program's symbol per
    # dynamic one — so a call at a new extent is checked against what the
    # program says, not against what the example call happened to have.
    fakes = _node_fakes(ep)
    output_bindings = [
        Binding(
            name,
            buffer,
            _torch_dtype(dtype_code),
            shapes[mutation]
            if mutation is not None
            else boundary_shape(_output_fake(name, fakes)),
            layout_from_spec(name, tag, strides),
        )
        for name, buffer, dtype_code, mutation, (tag, strides) in zip(
            graph.output_names,
            graph.output_buffers,
            graph.output_dtypes,
            graph.output_mutations,
            graph.output_layouts,
        )
    ]
    return CompiledModel(
        graph,
        ep,
        input_bindings,
        output_bindings,
        scalar_output_positions,
        held,
        held_bindings,
        aliases,
    )
