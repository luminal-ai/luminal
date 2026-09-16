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
Its layout is recognized once at compile time (``boundary.py``) and
declared to the runtime, which binds it on one buffer id; each call hands
that buffer the tensor's address. Nothing is copied to the host and no
layout is reinterpreted — a tensor the runtime cannot bind is refused by
name. Aliasing has one spelling: two bindings naming one buffer id, which
is how a writeback and the input it mutates share a pointer.

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
import copy
import os
import tempfile
from typing import Any, Optional, Sequence

import torch

from .boundary import (
    Binding,
    RowMajor,
    UnsupportedBoundary,
    boundary_layout,
    boundary_shape,
    buffer_nbytes,
    check_binding,
    layout_spec,
    storage_span,
)
from torch.export import Dim, export

from luminal_reference.export_utils import (
    _box_scalar_graph_outputs,
    _decomp_table,
    _drop_dead_data_dependent_ops,
    _drop_input_guards,
    _lower_sym_sum,
    _register_cache_serialization,
)

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

# The output specs this backend declares to the runtime. Anything else
# (a buffer mutation, a gradient, a token) would reach the translator as an
# ordinary returned tensor, because the PT2 signature reader models only
# ``user_input_mutation``.
_BOUND_OUTPUT_KINDS = frozenset({"USER_OUTPUT", "USER_INPUT_MUTATION"})


def _torch_dtype(dtype_code: int) -> torch.dtype:
    dtype = _PT2_TO_TORCH.get(dtype_code)
    if dtype is None:
        raise RuntimeError(f"luminal_cuda_lite cannot materialize PT2 dtype code {dtype_code}")
    return dtype


def _placeholder_fakes(ep: Any) -> dict[str, Any]:
    """The fake example value of every graph input, by placeholder name.

    It is the exported program's OWN symbolic metadata — the symbols the
    translator reads back out of the saved ``.pt2`` — so a stride stated
    from it names a dimension the runtime knows, at whatever extent a
    later call gives it.
    """
    fakes: dict[str, Any] = {}
    for node in ep.graph_module.graph.nodes:
        if node.op != "placeholder":
            continue
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
    fakes = _placeholder_fakes(ep)
    rows: list[tuple[str, str, torch.Tensor, Any]] = []
    user_index = 0
    for spec in ep.graph_signature.input_specs:
        name = getattr(spec.arg, "name", None)
        kind = spec.kind.name
        if name is None:
            raise UnsupportedBoundary(f"graph input {kind.lower()} {spec.target!r} is not a tensor")
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
                raise RuntimeError(f"{name}: {spec.target!r} is not in the exported constants")
            value = ep.constants[spec.target]
        else:
            raise UnsupportedBoundary(
                f"{name}: graph inputs of kind {kind.lower()} are not bound by luminal_cuda_lite"
            )
        if not isinstance(value, torch.Tensor):
            raise UnsupportedBoundary(f"{name}: graph input {spec.target!r} is not a tensor")
        rows.append((name, kind, value, fakes.get(name) if kind == "USER_INPUT" else None))
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
        held_tensors: Optional[dict[str, torch.Tensor]] = None,
    ):
        self._graph = graph
        self._ep = ep
        self._input_bindings = list(input_bindings)
        self._input_names = [binding.name for binding in self._input_bindings]
        self._output_bindings = list(output_bindings)
        self._scalar_output_positions = frozenset(scalar_output_positions)
        # Parameters and buffers, by graph input name, whose device pointers
        # the runtime holds for its life; a buffer writeback lands in them.
        self._held: dict[str, torch.Tensor] = dict(held_tensors or {})
        # Fixed once a plan set is searched; the per-execution arena sizes to it.
        self._arena_bytes = graph.arena_bytes()
        # The runtime always launches captured CUDA graphs, which the legacy
        # default stream cannot host, so it runs on a dedicated side stream
        # ordered against the caller's stream with events.
        self._side_stream: Optional[torch.cuda.Stream] = None
        self._output_mutations = graph.output_mutations
        self._output_returns = graph.output_returns
        # The graph inputs this program writes back into. While it is
        # empty, overlapping boundary tensors are read-only aliasing and
        # need no per-call check.
        self._writebacks = frozenset(
            mutation for mutation in self._output_mutations if mutation is not None
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
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=device)
        side = self._side_stream

        # CHECK EVERY INPUT BEFORE ADDRESSING ANY: a refusal on input i must
        # not leave inputs before it holding this call's pointers. A tensor
        # whose dtype, rank, extents or element strides are not the declared
        # ones is refused by name.
        for binding, value in zip(self._input_bindings, inputs):
            check_binding(binding, value)
        if self._writebacks:
            # Dynamo guards tensor identity, not storage overlap, so a
            # program compiled on distinct tensors can still be called with
            # two views of one allocation.
            _refuse_overlapping_writebacks(
                list(zip(self._input_names, inputs)) + list(self._held.items()),
                self._writebacks,
            )

        # Every input is bound as it is, on the buffer it was declared on.
        per_call: list[int] = []
        for binding, value in zip(self._input_bindings, inputs):
            self._graph.bind_input_shape(binding.name, list(value.shape))
            self._graph.set_device_ptr(binding.buffer, value.data_ptr(), buffer_nbytes(value))
            per_call.append(binding.buffer)

        # Output shapes depend on the bound dims, so read them after the
        # inputs are bound rather than caching them at compile time.
        output_shapes = self._graph.output_shapes

        # Allocate every output that is not a writeback and bind it. A
        # writeback's buffer IS its target input's, already addressed above.
        # Allocate under the side stream so the caching allocator records the
        # stream that will write them.
        with torch.cuda.stream(side):
            out_tensors: list[Optional[torch.Tensor]] = []
            for index, binding in enumerate(self._output_bindings):
                if self._output_mutations[index] is not None:
                    out_tensors.append(None)
                    continue
                tensor = torch.empty(
                    tuple(output_shapes[index]), dtype=binding.dtype, device=device
                )
                self._graph.set_device_ptr(
                    binding.buffer, tensor.data_ptr(), buffer_nbytes(tensor)
                )
                per_call.append(binding.buffer)
                out_tensors.append(tensor)

        # Order the side stream after everything the caller enqueued: this is
        # what makes reading the caller's inputs safe without a host sync.
        side.wait_stream(stream)

        # Per-execution intermediate arena from PyTorch's caching allocator,
        # associated with the stream that uses it.
        arena_bytes = max(self._arena_bytes, 1)
        arena = torch.cuda.caching_allocator_alloc(arena_bytes, device, side)
        self._graph.use_borrowed_stream(side.cuda_stream)
        self._graph.set_arena(arena, arena_bytes)
        try:
            self._graph.execute()
        finally:
            torch.cuda.caching_allocator_delete(arena)
            # These addresses belong to this call only: forget them, so an
            # execute that skipped a binding refuses by name instead of
            # reading storage the caller has released.
            for buffer in per_call:
                self._graph.clear_device_ptr(buffer)

        # Hand ordering back to the caller's stream for the returned tensors.
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
                tensor = out_tensors[index]
                # Scalar graph outputs were boxed into rank-zero tensors before
                # export; restore the Python scalar backend contract here.
                if index in self._scalar_output_positions:
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
    # trips "Guard failed on the same frame it was created".
    gm = copy.deepcopy(gm)
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


def luminal_cuda_lite(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    options: Optional[dict] = None,
    search_iterations: Optional[int] = None,
) -> CompiledModel:
    """The torch.compile backend entry point."""
    if options:
        search_iterations = options.get("search_iterations", search_iterations)

    # HF DynamicCache must be pytree-registered before torch.export capture so
    # use_cache=True models can export. Idempotent.
    _register_cache_serialization()

    # Canonicalize scalar (SymInt/SymFloat/SymBool) graph outputs into rank-zero
    # tensors before the export capture. Work on a private copy: Dynamo holds
    # onto the original graph module for guard installation and retracing, and
    # mutating it here would corrupt that bookkeeping.
    gm = copy.deepcopy(gm)
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
    # Serde gap workaround; must run before save. See _lower_sym_sum.
    _lower_sym_sum(ep)

    def _save_and_compile(program: Any) -> Any:
        # Recognize every boundary tensor's layout and declare it with the
        # program: the runtime binds what the caller has, or refuses it.
        _refuse_unbound_outputs(program)
        rows = _boundary_tensors(program, export_inputs)
        layouts = {name: boundary_layout(name, value, fake) for name, _, value, fake in rows}
        shapes = {name: boundary_shape(value, fake) for name, _, value, fake in rows}
        declared = []
        for name, _, _, _ in rows:
            tag, strides = layout_spec(layouts[name])
            declared.append((name, tag, list(strides)))
        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = os.path.join(tmp, "model.pt2")
            torch.export.save(program, pt2_path)
            graph = _luminal.compile(pt2_path, declared)
        tensors = {name: value for name, _, value, _ in rows}
        return graph, tensors, layouts, shapes

    try:
        graph, tensors, layouts, shapes = _save_and_compile(ep)
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
        _lower_sym_sum(ep)
        graph, tensors, layouts, shapes = _save_and_compile(ep)

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
    input_bindings: list[Binding] = []
    for name, kind, buffer in zip(graph.input_names, graph.input_kinds, graph.input_buffers):
        if name not in tensors:
            raise RuntimeError(
                f"the translated program names an input {name!r} the export signature "
                f"does not declare (declared: {sorted(tensors)})"
            )
        value = tensors[name]
        graph.bind_input_shape(name, list(value.shape))
        if kind == "user_input":
            input_bindings.append(
                Binding(name, buffer, value.dtype, shapes[name], layouts[name])
            )
            continue
        graph.set_device_ptr(buffer, value.data_ptr(), buffer_nbytes(value))
        held[name] = value

    graph.search(search_iterations)

    # A writeback is declared at its target's layout; every other output is a
    # tensor this wrapper allocates, so it is row-major by construction.
    output_bindings = [
        Binding(
            name,
            buffer,
            _torch_dtype(dtype_code),
            tuple(shape),
            layouts[mutation] if mutation is not None else RowMajor(),
        )
        for name, buffer, dtype_code, shape, mutation in zip(
            graph.output_names,
            graph.output_buffers,
            graph.output_dtypes,
            graph.output_shapes,
            graph.output_mutations,
        )
    ]
    return CompiledModel(
        graph, ep, input_bindings, output_bindings, scalar_output_positions, held
    )


def register_backend() -> None:
    """Register ``"luminal_cuda_lite"`` so ``backend="luminal_cuda_lite"`` works."""
    if "luminal_cuda_lite" in torch._dynamo.list_backends():
        return
    torch._dynamo.register_backend(luminal_cuda_lite, name="luminal_cuda_lite")
