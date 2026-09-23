"""Post-AOT collective decomposition into ordered CPU/Gloo P2P schedules.

Schedules are effectful programs, not ordinary pure FX expressions. Their sends
and waits execute even if the resulting tensor has no users. Calls must follow
the same collective order on every participating rank. Concurrent compiled
communication invocations in one process are rejected. Gloo P2P tags in
[2**30, 2**31) are reserved for this adapter on each participating group.
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _resolve_process_group

SUPPORTED = {
    "all_reduce",
    "all_gather_into_tensor",
    "reduce_scatter_tensor",
    "all_to_all_single",
    "broadcast",
    "wait_tensor",
}
_execution_lock = threading.RLock()
_sequences = WeakKeyDictionary()


@contextmanager
def communication_scope():
    if not _execution_lock.acquire(blocking=False):
        raise RuntimeError("concurrent reference communication calls are not supported")
    try:
        yield
    finally:
        _execution_lock.release()


@contextmanager
def _pending_transfers():
    pending = []
    try:
        yield pending
    finally:
        # On failure, drain every posted request before dropping its storage.
        # The process-group timeout bounds failed-peer waits; do not retry a plan.
        error = None
        for work, _tensor in pending:
            try:
                work.wait()
            except RuntimeError as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error


@dataclass(frozen=True)
class Send:
    value: str
    peer: int
    channel: int = 0


@dataclass(frozen=True)
class Recv:
    output: str
    peer: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    channel: int = 0


@dataclass(frozen=True)
class Wait:
    """Complete every outstanding transfer and release its retained buffers."""


@dataclass(frozen=True)
class Local:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    run: object
    symbols: tuple = ()


@dataclass
class P2PPlan:
    collective: str
    phase: str
    group: object
    rank: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    steps: tuple
    output: str
    executions: int = 0

    def __call__(self, value, bindings=None):
        def resolve(dimension):
            if not isinstance(dimension, torch.SymInt):
                return dimension
            expression = dimension.node.expr.subs(bindings or {})
            if expression.free_symbols:
                raise RuntimeError(f"unbound communication shape: {expression}")
            return int(expression)

        if (
            value.device.type != "cpu"
            or tuple(value.shape) != tuple(resolve(d) for d in self.shape)
            or value.dtype != self.dtype
        ):
            raise ValueError("P2P input does not match its CPU shape/dtype contract")
        with communication_scope(), _pending_transfers() as pending:
            sequence = _sequences.get(self.group, 0)
            tag = (1 << 30) + 2 * sequence
            if tag + 1 >= 1 << 31:
                raise RuntimeError("reference P2P message sequence exhausted")
            _sequences[self.group] = sequence + 1
            values = {"input": value}
            unreadable = set()
            for step in self.steps:
                if isinstance(step, Send):
                    if step.value in unreadable:
                        raise RuntimeError("send attempted before receive completion")
                    packed = values[step.value].contiguous()
                    if packed.numel():
                        work = dist.isend(
                            packed,
                            group=self.group,
                            group_dst=step.peer,
                            tag=tag + step.channel,
                        )
                        pending.append((work, packed))
                elif isinstance(step, Recv):
                    tensor = torch.empty(
                        tuple(resolve(d) for d in step.shape),
                        dtype=step.dtype,
                        device="cpu",
                    )
                    values[step.output] = tensor
                    unreadable.add(step.output)
                    if tensor.numel():
                        work = dist.irecv(
                            tensor,
                            group=self.group,
                            group_src=step.peer,
                            tag=tag + step.channel,
                        )
                        pending.append((work, tensor))
                elif isinstance(step, Wait):
                    # Holding both work and tensor keeps packed send storage alive.
                    for work, _tensor in pending:
                        work.wait()
                    pending.clear()
                    unreadable.clear()
                elif isinstance(step, Local):
                    if unreadable.intersection(step.inputs):
                        raise RuntimeError(
                            "local computation attempted before receive completion"
                        )
                    result = step.run(
                        *(values[name] for name in step.inputs),
                        *(resolve(s) for s in step.symbols),
                    )
                    if len(result) != len(step.outputs):
                        raise RuntimeError(
                            "local communication computation changed output arity"
                        )
                    values.update(zip(step.outputs, result))
                else:
                    raise TypeError(f"unknown communication instruction: {step!r}")
            if pending or unreadable:
                raise RuntimeError("communication schedule omitted its final wait")
            self.executions += 1
            # Functional collectives never expose the input as mutable output,
            # including a one-rank group and the root of a broadcast.
            return values[self.output].clone()


def lower_collective(node, module, phase, compile_local):
    """Build a rank-specific schedule from a captured functional collective."""
    name = str(node.target).split(".")[1]
    arguments = dict(zip((a.name for a in node.target._schema.arguments), node.args))
    arguments.update(node.kwargs)
    example = arguments["input"].meta["val"]
    if example.device.type != "cpu":
        raise RuntimeError("P2P lowering requires CPU tensors")
    for key in ("input_split_sizes", "output_split_sizes"):
        if key in arguments:
            arguments[key] = [
                v.meta["val"] if isinstance(v, torch.fx.Node) else v
                for v in arguments[key]
            ]
    shape, dtype = tuple(example.shape), example.dtype
    group = arguments["group_name"]
    if isinstance(group, torch.fx.Node) and group.op == "get_attr":
        group = getattr(module, group.target)
    group = _resolve_process_group(group) if isinstance(group, str) else group
    if not isinstance(group, dist.ProcessGroup) or dist.get_backend(group) != "gloo":
        raise RuntimeError(
            "reference P2P lowering requires a static Gloo process group"
        )
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    if rank < 0:
        raise RuntimeError("this rank is not a member of the collective group")
    if "group_size" in arguments and arguments["group_size"] != size:
        raise ValueError("collective group_size does not match the process group")
    steps = []

    def recv(output, peer, recv_shape=shape, channel=0):
        steps.append(Recv(output, peer, recv_shape, dtype, channel))

    def local(inputs, shapes, outputs, build):
        # Lift symbolic slice boundaries before FX sees them as constants.
        from torch.utils._pytree import tree_map

        symbols = []
        symbolic_nodes = {}

        def lift(value):
            if not isinstance(value, torch.SymInt):
                return value
            expr = value.node.expr
            if expr.is_number:
                return int(expr)
            if expr not in symbolic_nodes:
                first_op = next((n for n in graph.nodes if n.op != "placeholder"), None)
                with (
                    graph.inserting_before(first_op)
                    if first_op is not None
                    else graph.inserting_after(list(graph.nodes)[-1])
                ):
                    node = graph.placeholder(f"size_{len(symbols)}")
                node.meta["val"] = value
                symbols.append(value)
                symbolic_nodes[expr] = node
            return symbolic_nodes[expr]

        class LocalGraph(torch.fx.Graph):
            def call_function(self, target, args=(), kwargs=None, type_expr=None):
                return super().call_function(
                    target, tree_map(lift, args), tree_map(lift, kwargs), type_expr
                )

        graph = LocalGraph()
        placeholders = [graph.placeholder(f"arg_{i}") for i in range(len(inputs))]
        graph.output(tuple(build(graph, placeholders)))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        metadata = [example.new_empty(s) for s in shapes] + symbols
        steps.append(
            Local(
                tuple(inputs),
                tuple(outputs),
                compile_local(gm, metadata),
                tuple(symbols),
            )
        )

    def concat(inputs, shapes, output):
        local(
            inputs,
            shapes,
            [output],
            lambda g, xs: [g.call_function(torch.ops.aten.cat.default, (xs, 0))],
        )

    if name in {"all_reduce", "reduce_scatter_tensor"}:
        reduction = arguments["reduce_op"]
        operations = {
            "sum": torch.ops.aten.add.Tensor,
            "avg": torch.ops.aten.add.Tensor,
            "product": torch.ops.aten.mul.Tensor,
            "min": torch.ops.aten.minimum.default,
            "max": torch.ops.aten.maximum.default,
        }
        if reduction not in operations:
            raise ValueError(f"unsupported P2P reduction: {reduction}")
        if reduction == "avg" and not dtype.is_floating_point:
            raise ValueError("P2P average requires a floating-point dtype")
        scatter = name == "reduce_scatter_tensor"
        if scatter and (not shape or shape[0] % size):
            raise ValueError(
                "reduce-scatter requires a leading dimension divisible by group size"
            )
        output_shape = (shape[0] // size, *shape[1:]) if scatter else shape
        if rank == 0:
            names = ["input"]
            for peer in range(1, size):
                names.append(f"from_{peer}")
                recv(names[-1], peer)
            steps.append(Wait())

            def reduce_graph(g, xs):
                result = xs[0]
                for tensor in xs[1:]:
                    result = g.call_function(operations[reduction], (result, tensor))
                if reduction == "avg":
                    result = g.call_function(torch.ops.aten.div.Scalar, (result, size))
                if scatter:
                    width = output_shape[0]
                    return [
                        g.call_function(
                            torch.ops.aten.slice.Tensor,
                            (result, 0, peer * width, (peer + 1) * width),
                        )
                        for peer in range(size)
                    ]
                return [result]

            outputs = (
                [f"result_{peer}" for peer in range(size)] if scatter else ["result"]
            )
            local(names, [shape] * size, outputs, reduce_graph)
            for peer in range(1, size):
                steps.append(Send(outputs[peer] if scatter else "result", peer, 1))
            steps.append(Wait())
            output = outputs[0]
        else:
            steps.extend([Send("input", 0), Wait()])
            recv("result", 0, output_shape, 1)
            steps.append(Wait())
            output = "result"
    elif name == "all_gather_into_tensor":
        if not shape:
            raise ValueError("all-gather requires a tensor with a leading dimension")
        names = []
        for peer in range(size):
            names.append("input" if peer == rank else f"from_{peer}")
            if peer != rank:
                recv(names[-1], peer)
                steps.append(Send("input", peer))
        steps.append(Wait())
        concat(names, [shape] * size, "result")
        output = "result"
    elif name == "all_to_all_single":
        if not shape:
            raise ValueError("all-to-all requires a leading dimension")
        ins, outs = arguments["input_split_sizes"], arguments["output_split_sizes"]
        if any(not isinstance(n, (int, torch.SymInt)) or n < 0 for n in [*ins, *outs]):
            raise ValueError("all-to-all requires nonnegative integer split sizes")
        if (
            len(ins) != size
            or len(outs) != size
            or sum(ins) != shape[0]
            or ins[rank] != outs[rank]
        ):
            raise ValueError("invalid all-to-all split sizes")
        offsets = [0]
        for count in ins:
            offsets.append(offsets[-1] + count)
        chunks = [f"to_{peer}" for peer in range(size)]
        local(
            ["input"],
            [shape],
            chunks,
            lambda g, xs: [
                g.call_function(
                    torch.ops.aten.slice.Tensor,
                    (xs[0], 0, offsets[peer], offsets[peer + 1]),
                )
                for peer in range(size)
            ],
        )
        names = []
        shapes = [(n, *shape[1:]) for n in outs]
        for peer in range(size):
            names.append(chunks[rank] if peer == rank else f"from_{peer}")
            if peer != rank:
                recv(names[-1], peer, shapes[peer])
                steps.append(Send(chunks[peer], peer))
        steps.append(Wait())
        concat(names, shapes, "result")
        output = "result"
    elif name == "broadcast":
        src = arguments["src"]
        if not isinstance(src, int) or not 0 <= src < size:
            raise ValueError("broadcast source must be a group-relative rank")
        if rank == src:
            steps.extend(Send("input", peer) for peer in range(size) if peer != rank)
            output = "input"
        else:
            recv("result", src)
            output = "result"
        steps.append(Wait())
    else:
        raise RuntimeError(f"unsupported P2P collective: {name}")
    return P2PPlan(name, phase, group, rank, shape, dtype, tuple(steps), output)
