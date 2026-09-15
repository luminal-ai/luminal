"""The luminal_cuda_lite torch.compile backend.

The backend self-exports the incoming GraphModule with ``torch.export``,
saves the program to a temporary ``.pt2``, hands it to the Rust extension
for translation and CUDA-lite search, and returns a callable that binds
caller tensors per invocation.

This is the GPU twin of the ``luminal_reference`` backend. The export
preparation (dynamic-shape handling, scalar-output boxing, decomposition
fallback, dead-op/guard cleanup) is backend-neutral, so it is imported
from the reference package rather than duplicated; see DESIGN.md for the
plan to lift it into a shared, backend-neutral module.

DEVICE MODEL (interim): the runtime currently owns its CUDA context,
stream and arena, and stages every input H2D / output D2H. Caller
tensors are therefore moved with ``.cpu()`` before staging and outputs
are rebuilt on the caller's device. The zero-copy + PyTorch-caching-
allocator design is in DESIGN.md.
"""

import concurrent.futures
import copy
import os
import tempfile
from typing import Any, Optional, Sequence

import torch
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


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # Interim host-staged path: the runtime does its own H2D, so the caller's
    # device tensor is copied to host here. The zero-copy path will hand the
    # runtime `tensor.data_ptr()` instead (DESIGN.md).
    tensor = tensor.detach().cpu().contiguous()
    # Flatten first: a 0-dim tensor cannot be viewed as a wider dtype.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _torch_dtype(dtype_code: int) -> torch.dtype:
    dtype = _PT2_TO_TORCH.get(dtype_code)
    if dtype is None:
        raise RuntimeError(f"luminal_cuda_lite cannot materialize PT2 dtype code {dtype_code}")
    return dtype


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class CompiledModel:
    """Callable wrapper around a compiled CUDA-lite graph.

    Executions are zero-copy and the intermediate arena is PyTorch-owned:

    * every user input and weight is handed to the runtime as a device pointer
      (``set_input_ptr``) wherever it is contiguous;
    * every output (and mutation sink) is a PyTorch tensor bound with
      ``set_output_ptr``, so the producing op writes it in place and it is
      returned without a D2H;
    * the intermediate-scratch arena is ``caching_allocator_alloc``'d for the
      duration of the call and ``caching_allocator_delete``'d right after, so
      PyTorch accounts for it and can reuse the block next call.
    """

    def __init__(
        self,
        graph: Any,
        ep: Any,
        scalar_output_positions: Sequence[int] = (),
        held_tensors: Sequence[torch.Tensor] = (),
    ):
        self._graph = graph
        self._ep = ep
        self._scalar_output_positions = frozenset(scalar_output_positions)
        # Device copies of parameters/buffers whose pointers the runtime holds.
        self._held = list(held_tensors)
        # Fixed once a plan set is searched; the per-execution arena sizes to it.
        self._arena_bytes = graph.arena_bytes() if hasattr(graph, "arena_bytes") else 0
        # The runtime always launches captured CUDA graphs, which the legacy
        # default stream cannot host, so it runs on a dedicated side stream
        # ordered against the caller's stream with events.
        self._side_stream: Optional[torch.cuda.Stream] = None
        names = graph.input_names
        kinds = graph.input_kinds
        self._user_input_names = [
            name for name, kind in zip(names, kinds) if kind == "user_input"
        ]
        self._output_names = graph.output_names
        self._output_dtypes = graph.output_dtypes
        self._output_mutations = graph.output_mutations
        self._output_returns = graph.output_returns

    def __call__(self, *args: torch.Tensor) -> Any:
        # Under dynamic shapes Dynamo's wrapper passes the graph's symbolic
        # shape values alongside the tensor inputs (as SymInt or int). The
        # compiled program folds those symbols into `sym_size`, so it has no
        # scalar inputs: keep tensors, drop the scalars.
        inputs = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if len(inputs) != len(self._user_input_names):
            raise RuntimeError(
                f"luminal_cuda_lite expected {len(self._user_input_names)} inputs, "
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

        # Bind inputs. Contiguous tensors go zero-copy; anything else falls
        # back to host staging (a copy) so a stride/offset the plan does not
        # model can never be read as dense.
        zero_copy_inputs = all(value.is_contiguous() for value in inputs)
        for name, value in zip(self._user_input_names, inputs):
            if zero_copy_inputs:
                self._graph.set_input_ptr(
                    name, value.data_ptr(), _nbytes(value), list(value.shape)
                )
            else:
                self._graph.set_input(name, _tensor_bytes(value), list(value.shape))

        # Output shapes depend on the bound dims, so read them after the
        # inputs are bound rather than caching them at compile time.
        output_shapes = self._graph.output_shapes

        # Allocate every final output as a PyTorch tensor and bind it. A
        # mutation sink IS the caller's input tensor (eager aliasing). Allocate
        # under the side stream so the caching allocator records the stream that
        # will write them.
        with torch.cuda.stream(side):
            out_tensors: list[torch.Tensor] = []
            for index, (dtype_code, shape, mutation) in enumerate(
                zip(self._output_dtypes, output_shapes, self._output_mutations)
            ):
                if mutation is not None:
                    target = self._user_input_names.index(mutation)
                    tensor = inputs[target]
                else:
                    tensor = torch.empty(
                        tuple(shape), dtype=_torch_dtype(dtype_code), device=device
                    )
                self._graph.set_output_ptr(index, tensor.data_ptr(), _nbytes(tensor))
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

        # Hand ordering back to the caller's stream for the returned tensors.
        stream.wait_stream(side)

        results = []
        for index, mutation in enumerate(self._output_mutations):
            returned = self._output_returns[index]
            if mutation is not None:
                # The write already landed in the caller's tensor.
                if returned:
                    results.append(inputs[self._user_input_names.index(mutation)])
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
        if "unsupported ATen op" not in str(exc):
            raise
        ep = ep.run_decompositions(_decomp_table())
        _lower_sym_sum(ep)
        graph = _save_and_compile(ep)

    names = graph.input_names
    kinds = graph.input_kinds
    parameter_names = graph.parameter_names

    # The backend is CUDA-only: pick the device from the example inputs.
    device = next(
        (
            value.device
            for value in export_inputs
            if isinstance(value, torch.Tensor) and value.is_cuda
        ),
        None,
    )
    if device is None:
        raise RuntimeError(
            "luminal_cuda_lite requires CUDA example inputs (the model must be on "
            "a CUDA device)"
        )
    # Parameters/buffers are bound once, zero-copy, to persistent device copies
    # this wrapper holds; only user inputs are rebound per call. The pointer
    # binding must happen AFTER search: the runtime resolves a graph tensor to
    # its plan buffer only once a plan is installed.
    params: list[tuple[str, torch.Tensor]] = []
    zero_copy = hasattr(graph, "set_input_ptr")

    user_index = 0
    for name, kind, parameter_name in zip(names, kinds, parameter_names):
        if kind == "user_input":
            if user_index >= len(export_inputs):
                raise RuntimeError(
                    f"export declared more user inputs than example_inputs: {name!r}"
                )
            value = export_inputs[user_index]
            user_index += 1
            # Seed the symbolic dims from the example; the real binding happens
            # per call. Host-staged so no dangling pointer survives compile.
            graph.set_input(name, _tensor_bytes(value), list(value.shape))
            continue
        if parameter_name not in ep.state_dict:
            raise RuntimeError(
                f"parameter {parameter_name!r} (graph input {name!r}) is not in "
                "the exported state_dict"
            )
        value = ep.state_dict[parameter_name]
        if not value.is_cuda:
            value = value.to(device)
        value = value.contiguous()
        graph.set_input(name, _tensor_bytes(value), list(value.shape))
        params.append((name, value))

    if user_index != len(export_inputs):
        raise RuntimeError(
            f"export consumed {user_index} of {len(export_inputs)} example_inputs"
        )

    # Every final output is bound to a caller tensor per call, so the arena
    # planner must exclude output buffers from the slab. This must precede
    # search so the finalist budget and the installed plan agree.
    if hasattr(graph, "set_external_outputs"):
        graph.set_external_outputs(True)

    graph.search(search_iterations)

    held: list[torch.Tensor] = []
    if zero_copy:
        for name, value in params:
            graph.set_input_ptr(
                name, value.data_ptr(), _nbytes(value), list(value.shape)
            )
            held.append(value)
    return CompiledModel(graph, ep, scalar_output_positions, held)


def register_backend() -> None:
    """Register ``"luminal_cuda_lite"`` so ``backend="luminal_cuda_lite"`` works."""
    if "luminal_cuda_lite" in torch._dynamo.list_backends():
        return
    torch._dynamo.register_backend(luminal_cuda_lite, name="luminal_cuda_lite")
