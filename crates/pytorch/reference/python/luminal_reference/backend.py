"""The luminal_reference torch.compile backend.

The public entry point uses AOTAutograd for inference and training. Its local
compiler exports each ATen region to PT2 for translation and reference-runtime
search, then binds caller tensors and symbolic dimensions on each invocation.
"""

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import torch
from torch.export import export

from . import _luminal
from .dimensions import export_specs, remap_buckets
from .export_utils import (
    _box_scalar_graph_outputs,
    _decomp_table,
    _drop_dead_data_dependent_ops,
    _drop_input_guards,
    _lower_sym_sum,
    _register_cache_serialization,
    private_graph_copy,
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
        names = graph.input_names
        kinds = graph.input_kinds
        self._user_input_names = [
            name for name, kind in zip(names, kinds) if kind == "user_input"
        ]
        self._output_names = graph.output_names
        self._output_dtypes = graph.output_dtypes
        self._output_mutations = graph.output_mutations
        self._output_returns = graph.output_returns

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
        if len(inputs) != len(self._user_input_names):
            raise RuntimeError(
                f"luminal_reference expected {len(self._user_input_names)} inputs, "
                f"got {len(inputs)}"
            )
        for name, value in zip(self._user_input_names, inputs):
            self._graph.set_input(name, _tensor_bytes(value), list(value.shape))
        # Output shapes depend on the bound dims, so read them after the
        # inputs are staged rather than caching them at compile time.
        output_shapes = self._graph.output_shapes
        self._graph.execute()

        results = []
        outputs = zip(
            self._output_names,
            self._output_dtypes,
            output_shapes,
            self._output_mutations,
            self._output_returns,
        )
        for index, (_, dtype_code, shape, mutation, returned) in enumerate(outputs):
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
                if returned:
                    results.append(destination)
                continue
            if returned:
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


def _dynamic_export(
    gm: torch.fx.GraphModule, example_inputs: Sequence[Any], symbol_buckets=None
) -> Any:
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

    all_shapes = [
        getattr(n.meta.get("example_value", v), "shape", ())
        for n, v in zip(placeholders, example_inputs)
    ]
    axis_specs = export_specs(all_shapes)
    records: list[tuple[str, torch.fx.Node, Any]] = []
    tensor_dims: dict[Any, tuple[torch.fx.Node, int]] = {}
    for node, value, dim_spec in zip(placeholders, example_inputs, axis_specs):
        if isinstance(value, torch.SymInt):
            records.append(("sym", node, value))
            continue
        shape = getattr(node.meta.get("example_value"), "shape", None)
        if shape is None:
            shape = getattr(value, "shape", ())
        dims = dim_spec
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

    # The AOT local compiler isolates FakeTensor and TracingContext before
    # entering here. Keep export on the owning thread: native reference graphs
    # are unsendable, and cyclic GC in an export worker could destroy them.
    ep = export(gm, tuple(inputs), dynamic_shapes=dynamic_shapes, strict=False)
    original_shapes = [
        getattr(n.meta.get("example_value", v), "shape", ())
        for (_, n, _), v in zip(records, example_inputs)
        if id(n) not in erased
    ]
    return ep, inputs, remap_buckets(ep, original_shapes, symbol_buckets)


def _prepare_local_graph(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    symbol_buckets=None,
) -> tuple:
    """Export one local ATen graph after AOT partitioning."""

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
    ep, export_inputs, dim_buckets = _dynamic_export(gm, example_inputs, symbol_buckets)
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
        raise RuntimeError(
            "graph inputs share device storage and the program writes one of them: "
            "the export cannot functionalise a write through aliased inputs"
        ) from exc
    _drop_dead_data_dependent_ops(ep.graph_module)
    # Serde gap workaround; must run before save. See _lower_sym_sum.
    _lower_sym_sum(ep)

    return ep, export_inputs, scalar_output_positions, dim_buckets


def _compile_local_graph(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    options: dict | None = None,
    search_iterations: int | None = None,
    search_log: bool = False,
    max_intermediate_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
    symbol_buckets=None,
) -> CompiledModel:
    """Compile one local ATen graph after AOT partitioning."""
    if options:
        search_iterations = options.get("search_iterations", search_iterations)
        search_log = options.get("search_log", search_log)
        max_intermediate_bytes = options.get(
            "max_intermediate_bytes", max_intermediate_bytes
        )
        memory_budget_bytes = options.get("memory_budget_bytes", memory_budget_bytes)
    ep, export_inputs, scalar_output_positions, dim_buckets = _prepare_local_graph(
        gm, example_inputs, symbol_buckets
    )
    return compile_exported(
        ep,
        export_inputs,
        search_iterations,
        scalar_output_positions,
        search_log=search_log,
        max_intermediate_bytes=max_intermediate_bytes,
        memory_budget_bytes=memory_budget_bytes,
        dim_buckets=dim_buckets,
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
    dim_buckets=None,
    artifact=None,
):
    """Compile a functional ExportedProgram with the native runtime."""

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
        _drop_dead_data_dependent_ops(ep.graph_module)
        _lower_sym_sum(ep)
        graph = _save_and_compile(ep)

    names = graph.input_names
    kinds = graph.input_kinds
    parameter_names = graph.parameter_names

    user_index = 0
    held: dict = {}
    for name, kind, parameter_name in zip(names, kinds, parameter_names):
        if kind == "user_input":
            if user_index >= len(export_inputs):
                raise RuntimeError(
                    f"export declared more user inputs than example_inputs: {name!r}"
                )
            value = export_inputs[user_index]
            user_index += 1
        else:
            if parameter_name not in ep.state_dict:
                raise RuntimeError(
                    f"parameter {parameter_name!r} (graph input {name!r}) is not in "
                    "the exported state_dict"
                )
            value = ep.state_dict[parameter_name]
            # The export's own tensor, not the caller's module buffer: mutated
            # state persists across calls on the input's runtime buffer.
            held[name] = value
        from torch._subclasses.fake_tensor import FakeTensor, unset_fake_temporarily

        if isinstance(value, FakeTensor):
            with unset_fake_temporarily():
                value = torch.ones(
                    tuple(int(d) for d in value.shape), dtype=value.dtype
                )
        graph.set_input(name, _tensor_bytes(value), list(value.shape))

    if user_index != len(export_inputs):
        raise RuntimeError(
            f"export consumed {user_index} of {len(export_inputs)} example_inputs"
        )

    if artifact is None:
        graph.search(
            search_iterations,
            search_log=search_log,
            max_intermediate_bytes=max_intermediate_bytes,
            memory_budget_bytes=memory_budget_bytes,
            dim_buckets=dim_buckets,
        )
    else:
        graph.load_compiled(artifact, memory_budget_bytes=memory_budget_bytes)
    return CompiledModel(graph, ep, scalar_output_positions, held)
