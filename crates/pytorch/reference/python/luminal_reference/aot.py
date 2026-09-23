"""AOTAutograd integration for local CPU programs and DTensor SPMD.

DTensor owns sharding; AOTAutograd owns differentiation. This adapter compiles
local ATen regions and lowers functional collectives to CPU/Gloo point-to-point
communication schedules after differentiation. Symbolic CPU shapes are retained
through compilation and bound at execution; communication is synchronous.
"""

from dataclasses import dataclass, field
from functools import partial

import torch
from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_unflatten

from .backend import _compile_local_graph
from .communication import SUPPORTED, communication_scope, lower_collective
from .dimensions import normalize_buckets, profile_value, resolve_policies
from .export_utils import private_graph_copy


@dataclass
class RegionRecord:
    """Inspection evidence for a compiled local region (no retained tensors)."""

    phase: str
    input_shapes: tuple[tuple[int, ...], ...]
    targets: tuple[str, ...]
    executions: int = 0
    buckets: dict = field(default_factory=dict)


@dataclass
class GraphRecord:
    phase: str
    code: str
    collectives: tuple[str, ...]


def _collective(node):
    target = str(node.target)
    if target.startswith(("_c10d_functional.", "_c10d_functional_autograd.")):
        if target.split(".")[1] not in SUPPORTED:
            raise RuntimeError(
                f"unsupported communication op in AOT reference graph: {target}"
            )
        return True
    if "c10d" in target:
        raise RuntimeError(
            f"unsupported communication op in AOT reference graph: {target}"
        )
    return False


def _compile_region(
    gm,
    inputs,
    phase,
    records,
    search_iterations,
    search_log=False,
    max_intermediate_bytes=None,
    memory_budget_bytes=None,
    dim_buckets=None,
):
    graph = private_graph_copy(gm)
    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    input_positions = {node: i for i, node in enumerate(placeholders)}
    output = next(n for n in graph.graph.nodes if n.op == "output")
    leaves, spec = tree_flatten(output.args[0])
    # AOT can carry opaque DTensor metadata through forward/backward, as well
    # as saved inputs and missing gradients. Only computed tensor outputs go
    # through PT2. Forwarded inputs keep their original identity and aliases.
    passthrough = {
        i: input_positions[leaf]
        for i, leaf in enumerate(leaves)
        if isinstance(leaf, torch.fx.Node) and leaf in input_positions
    }
    symbolic_outputs = {
        i: leaf.meta.get("val", leaf.meta.get("example_value"))
        for i, leaf in enumerate(leaves)
        if isinstance(leaf, torch.fx.Node)
        and isinstance(
            leaf.meta.get("val", leaf.meta.get("example_value")), torch.SymInt
        )
        and i not in passthrough
    }
    from torch._guards import detect_fake_mode

    input_mode = detect_fake_mode(inputs)
    shape_env = input_mode.shape_env if input_mode is not None else None
    shape_positions = [
        i
        for i, value in enumerate(inputs)
        if isinstance(value, (torch.Tensor, torch.SymInt))
    ]
    shape_metadata = [inputs[i] for i in shape_positions]
    positions = [
        i
        for i, leaf in enumerate(leaves)
        if isinstance(leaf, torch.fx.Node)
        and i not in passthrough
        and i not in symbolic_outputs
    ]
    output.args = (tuple(leaves[i] for i in positions),)
    graph.graph.eliminate_dead_code()
    indices = []
    for i, node in enumerate(placeholders):
        if node.users:
            indices.append(i)
        else:
            graph.graph.erase_node(node)
    graph.recompile()
    local_inputs = [inputs[i] for i in indices]
    # AOT inputs are metadata, never storage. Do not specialize symbolic dims
    # behind Dynamo's guards or let opaque metadata enter local computation.
    for value in local_inputs:
        if isinstance(value, torch.SymInt):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "AOT reference computation requires tensor inputs; opaque values may only pass through"
            )
        if value.device.type != "cpu":
            raise RuntimeError("AOT reference requires CPU tensors")
        if hasattr(value, "placements"):
            raise RuntimeError(
                "AOT reference expected local tensors, received a DTensor"
            )

    # A backward graph may save only a symbolic size (e.g. sum backward),
    # with no tensor carrying that dimension. Encode scalar shape inputs as
    # zero-storage tensors at the PT2 boundary; their sizes remain symbolic.
    carriers = set()
    live_placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    for position, (node, value) in enumerate(zip(live_placeholders, local_inputs)):
        if not isinstance(value, torch.SymInt):
            continue
        with input_mode:
            carrier = torch.empty((value, 0), dtype=torch.uint8, device="cpu")
        users = list(node.users)
        with graph.graph.inserting_after(live_placeholders[-1]):
            size = graph.graph.call_function(torch.ops.aten.sym_size.int, (node, 0))
            size.meta["val"] = value
        for user in users:
            user.replace_input_with(node, size)
        node.meta = {**node.meta, "val": carrier, "example_value": carrier}
        local_inputs[position] = carrier
        carriers.add(position)
    graph.graph.lint()
    graph.recompile()

    record = None
    compiled = None
    if positions:
        shapes = tuple(
            tuple(v.shape) for v in local_inputs if isinstance(v, torch.Tensor)
        )
        record = RegionRecord(
            phase,
            shapes,
            tuple(str(n.target) for n in graph.graph.nodes if n.op == "call_function"),
        )
        # Isolate synthetic storage from the surrounding AOT FakeTensor mode.
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        for node, value in zip(live_placeholders, local_inputs):
            node.meta["example_value"] = value
        from torch._guards import tracing

        with unset_fake_temporarily(), tracing(None):
            examples = [
                torch.empty_strided(
                    tuple(profile_value(d) for d in v.shape),
                    tuple(profile_value(d) for d in v.stride()),
                    dtype=v.dtype,
                    device="cpu",
                ).fill_(1)
                if isinstance(v, torch.Tensor)
                else v
                for v in local_inputs
            ]
            compiled = _compile_local_graph(
                graph,
                examples,
                search_iterations=search_iterations,
                search_log=search_log,
                max_intermediate_bytes=max_intermediate_bytes,
                memory_budget_bytes=memory_budget_bytes,
                symbol_buckets=resolve_policies(shape_env, dim_buckets or {}),
            )
            record.buckets = compiled._graph.dim_buckets
        records.append(record)

    def run(*args):
        bound_inputs = [
            torch.empty((args[index], 0), dtype=torch.uint8, device="cpu")
            if position in carriers
            else args[index]
            for position, index in enumerate(indices)
        ]
        values = compiled(*bound_inputs) if compiled is not None else ()
        if len(values) != len(positions):
            raise RuntimeError("local compiler changed the AOT output arity")
        result = list(leaves)
        for position, value in zip(positions, values):
            result[position] = value
        for position, index in passthrough.items():
            result[position] = args[index]
        if symbolic_outputs:
            bindings = (
                shape_env.bind_symbols(
                    shape_metadata, [args[i] for i in shape_positions]
                )
                if shape_env is not None
                else {}
            )
            for position, symbol in symbolic_outputs.items():
                expression = symbol.node.expr.subs(bindings)
                if expression.free_symbols:
                    raise RuntimeError(f"unbound AOT output dimension: {expression}")
                result[position] = int(expression)
        if record is not None:
            record.executions += 1
        return tree_unflatten(result, spec)

    return run


class ReferenceAOTBackend:
    """Callable torch.compile backend; inspect ``graphs`` and ``regions``.

    The same compiler handles static/dynamic forward, backward, and
    inference. No local computation falls back to eager PyTorch.
    """

    def __init__(
        self,
        *,
        search_iterations=1,
        search_log=False,
        max_intermediate_bytes=None,
        memory_budget_bytes=None,
        dim_buckets=None,
    ):
        self.dim_buckets = normalize_buckets(dim_buckets)
        self.regions: list[RegionRecord] = []
        self.graphs: list[GraphRecord] = []
        self.communications = []
        self.search_iterations = search_iterations
        self.search_log = search_log
        self.compile_options = {
            "search_iterations": search_iterations,
            "search_log": search_log,
            "max_intermediate_bytes": max_intermediate_bytes,
            "memory_budget_bytes": memory_budget_bytes,
            "dim_buckets": self.dim_buckets,
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
        )

    def __call__(self, gm, example_inputs):
        from torch._guards import detect_fake_mode

        mode = detect_fake_mode(example_inputs)
        policies = (
            resolve_policies(mode.shape_env, self.dim_buckets)
            if self.dim_buckets
            else {}
        )
        axes = {}
        for index, node in enumerate(
            n for n in gm.graph.nodes if n.op == "placeholder"
        ):
            value = node.meta.get("example_value")
            if isinstance(value, torch.Tensor):
                for axis, size in enumerate(value.shape):
                    if isinstance(size, torch.SymInt):
                        expr = size.node.shape_env.replace(size.node.expr)
                        if expr in policies:
                            axes.setdefault(expr, []).append((index, axis))
        compiled = self._backend(gm, example_inputs)
        if not axes:
            return compiled

        def run(*args):
            for symbol, positions in axes.items():
                values = [args[index].shape[axis] for index, axis in positions]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError(
                        f"axes sharing ShapeVar {symbol} must have equal sizes"
                    )
                if not any(
                    bucket.min <= values[0] <= bucket.max for bucket in policies[symbol]
                ):
                    raise ValueError(
                        f"dimension {symbol}={values[0]} is outside its configured buckets"
                    )
            return compiled(*args)

        return run

    def _compile(self, gm, example_inputs, *, phase):
        communication = [
            n for n in gm.graph.nodes if n.op == "call_function" and _collective(n)
        ]
        self.graphs.append(
            GraphRecord(phase, gm.code, tuple(str(n.target) for n in communication))
        )
        if not communication:
            return make_boxed_func(
                _compile_region(
                    gm,
                    example_inputs,
                    phase,
                    self.regions,
                    **self.compile_options,
                )
            )

        # Contiguous topological regions keep collective ordering explicit.
        # This is a frontend boundary, not an extracted-Luminal graph rewrite.
        partition = -1
        previous = None
        kinds = {}

        def assign(node):
            nonlocal partition, previous
            is_comm = _collective(node)
            if previous is None or is_comm != previous:
                partition += 1
                previous = is_comm
                kinds[partition] = is_comm
            return partition

        split = split_module(gm, gm, assign, keep_original_order=True)
        compiled_regions = {}
        communication_plans = {}
        for name, module in split.named_children():
            index = int(name.removeprefix("submod_"))
            if kinds[index]:
                plans = {}
                for node in module.graph.nodes:
                    if (
                        node.op != "call_function"
                        or str(node.target).split(".")[1] == "wait_tensor"
                    ):
                        continue
                    plan = lower_collective(
                        node,
                        module,
                        phase,
                        lambda graph, inputs: _compile_region(
                            graph,
                            inputs,
                            phase,
                            self.regions,
                            **self.compile_options,
                        ),
                    )
                    plans[node.name] = plan
                    self.communications.append(plan)
                communication_plans[name] = plans
                continue
            inputs = [
                n.meta["val"] for n in module.graph.nodes if n.op == "placeholder"
            ]
            compiled_regions[name] = _compile_region(
                module,
                inputs,
                phase,
                self.regions,
                **self.compile_options,
            )

        class CommunicationExecutor(torch.fx.Interpreter):
            def __init__(self, module, plans, bindings):
                super().__init__(module)
                self.plans = plans
                self.bindings = bindings

            def run_node(self, node):
                if node.name in self.plans:
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    value = args[0] if args else kwargs["input"]
                    return self.plans[node.name](value, self.bindings)
                # wait_tensor remains valid for externally produced async inputs;
                # outputs of our schedules have already completed their P2P work.
                return super().run_node(node)

        class Executor(torch.fx.Interpreter):
            def __init__(self, module, bindings):
                super().__init__(module)
                self.bindings = bindings

            def call_module(self, target, args, kwargs):
                if target in compiled_regions:
                    return compiled_regions[target](*args, **kwargs)
                if kwargs:
                    raise RuntimeError(
                        "unexpected keyword arguments in communication region"
                    )
                return CommunicationExecutor(
                    self.fetch_attr(target), communication_plans[target], self.bindings
                ).run(*args)

        from torch._guards import detect_fake_mode

        mode = detect_fake_mode(example_inputs)
        shape_env = mode.shape_env if mode is not None else None
        symbolic_positions = [
            i
            for i, v in enumerate(example_inputs)
            if isinstance(v, (torch.Tensor, torch.SymInt))
        ]
        metadata = [example_inputs[i] for i in symbolic_positions]

        def run(*args):
            # Bind the original AOT signature, including saved scalar dimensions.
            bindings = (
                shape_env.bind_symbols(metadata, [args[i] for i in symbolic_positions])
                if shape_env is not None
                else {}
            )
            with communication_scope():
                return Executor(split, bindings).run(*args)

        return make_boxed_func(run)
