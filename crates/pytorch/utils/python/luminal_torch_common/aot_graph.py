"""AOTAutograd graph preparation shared by the PyTorch backends.

AOTAutograd owns differentiation and hands a backend one functional ATen graph
per phase: forward, backward, or inference. Every backend feeds each graph
through the same two steps here before its native runtime sees it.

1. ``lower_local_graph`` writes the whole graph, in process, as the JSON
   document the Rust translator reads: the graph module of a PT2 program,
   with every placeholder and every output leaf. No ``ExportedProgram`` is
   built and nothing is saved to disk. torch's own serializer writes the
   nodes and their value metadata; the inputs, the signature and the symbol
   ranges are written here, because torch's serializer has no spelling for a
   symbolic float or bool as a graph input. An in-graph input mutation
   (AOTAutograd's trailing ``copy_`` when a backend asks for
   ``keep_inference_input_mutations``) becomes the ``user_input_mutation``
   output spec ``torch.export`` would have written.
2. ``prepare_local_graph`` asks the Rust side for the graph's boundary,
   through the ``boundary_plan`` function of the backend's own extension,
   and keeps it as slots. torch calls a compiled graph with one value per
   placeholder and expects one value per output leaf, in order; a slot is
   one of those positions, and its kind is the type of that value: a
   tensor, an ``int``, a ``float``, a ``bool``, or any other object. Which
   arguments the program reads, which symbol an integer argument defines,
   and where each output leaf comes from are all read off the serialized
   graph in Rust; nothing is classified here.

Collective operations are refused by name: a program that communicates must be
split by the application so that each compiled graph is collective-free.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .graph_prep import (
    _drop_dead_data_dependent_ops,
    _lower_sym_sum,
    private_graph_copy,
)


@dataclass
class GraphRecord:
    """Inspection evidence for one compiled AOT graph.

    ``program`` is the native compiled program behind the graph (``None``
    when the graph only forwards inputs and sizes); it owns the runtime and
    the program's constants, and nothing else here retains a tensor.
    """

    phase: str
    code: str
    input_shapes: tuple[tuple[int, ...], ...] = ()
    targets: tuple[str, ...] = ()
    executions: int = 0
    bounds: dict = field(default_factory=dict)
    program: Any = None


def _collective(node):
    return node.op == "call_function" and "c10d" in str(node.target)


def refuse_collectives(gm, phase, backend):
    found = sorted({str(n.target) for n in gm.graph.nodes if _collective(n)})
    if found:
        raise RuntimeError(
            f"{backend} does not compile collective operations; the "
            f"{phase} graph contains {', '.join(found)}. Split the program at "
            "its collectives so that each compiled graph is collective-free."
        )


def _value(node):
    return node.meta.get("val", node.meta.get("example_value"))


# The symbolic scalar types of an AOT graph. Each has one Python type at call
# time: torch passes and expects an ``int``, a ``float`` or a ``bool``.
_SYMBOLIC_SCALARS = (torch.SymInt, torch.SymFloat, torch.SymBool)


# The kinds of slot: torch's value types at a graph boundary.
TENSOR, INT, FLOAT, BOOL, OBJECT = "tensor", "int", "float", "bool", "object"

# Where an output slot's value comes from.
PROGRAM, INPUT, EXPRESSION, LITERAL = "program", "input", "expression", "literal"


def _shape_env(values):
    """The shape environment a graph's symbols live in, read off the first
    symbolic value among ``values``: a symbolic scalar, or a tensor's size
    or stride. A graph whose only arguments are integers has no fake tensor
    to find a fake mode through, and its symbols still have ranges."""
    for value in values:
        items = (
            (*value.shape, *value.stride())
            if isinstance(value, torch.Tensor)
            else (value,)
        )
        for item in items:
            if isinstance(item, _SYMBOLIC_SCALARS):
                return item.node.shape_env
    return None


def _python_scalar(kind, expression):
    """The call-time value of a symbolic output leaf, in the Python type
    torch's backend contract gives a leaf of that kind."""
    if kind == BOOL:
        return bool(expression)
    if kind == FLOAT:
        return float(expression)
    return int(expression)


def integer_argument(name: str, value: Any) -> int:
    """The value a call passes in an integer slot. torch passes a Python
    ``int`` there; anything else is refused by the slot's name."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"argument {name!r} is an integer slot and was called with a "
            f"{type(value).__name__}"
        )
    return value


def tensor_argument(name: str, value: Any) -> torch.Tensor:
    """The value a call passes in a tensor slot, refused by the slot's name
    when it is not a tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"argument {name!r} is a tensor slot and was called with a "
            f"{type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class InputSlot:
    """One placeholder of a graph: the position of one call argument.

    ``read`` says whether a node reads the argument. A tensor that is read
    is storage the program binds; one nothing reads keeps its slot and is
    never touched. ``symbol`` is the dimension an integer argument defines:
    the call's value binds it. Every other symbolic scalar is the
    expression torch recorded for it, and its call-time value is not read.
    """

    name: str
    kind: str
    read: bool
    symbol: str | None = None


@dataclass(frozen=True)
class OutputSlot:
    """One output leaf of a graph, and where its value comes from.

    * ``PROGRAM``: the compiled program computes it; ``index`` is the row
      it writes.
    * ``INPUT``: the leaf is a placeholder, so the caller's own object is
      returned; ``index`` is that input slot.
    * ``EXPRESSION``: a symbolic ``int``, ``float`` or ``bool`` computed
      from sizes; ``value`` is the sympy expression torch recorded for it,
      evaluated per call.
    * ``LITERAL``: ``value`` itself, such as the ``None`` of a missing
      gradient.
    """

    kind: str
    source: str
    index: int | None = None
    value: Any = None


def input_slots(rows) -> tuple:
    """The Rust plan's argument rows as slots."""
    return tuple(
        InputSlot(name, kind, read, symbol) for kind, name, read, symbol in rows
    )


def _symbol_sources(inputs) -> dict:
    """Where a call states each symbol, as ``(argument position, axis)``:
    an integer argument that is the symbol (``axis`` is ``None``), else a
    tensor axis that is. An integer argument comes first, as it does for
    torch."""
    sources: dict = {}
    for position, value in enumerate(inputs):
        if isinstance(value, torch.SymInt) and value.node.expr.is_Symbol:
            sources.setdefault(value.node.expr, (position, None))
    for position, value in enumerate(inputs):
        if isinstance(value, torch.Tensor):
            for axis, size in enumerate(value.shape):
                if isinstance(size, torch.SymInt) and size.node.expr.is_Symbol:
                    sources.setdefault(size.node.expr, (position, axis))
    return sources


@dataclass
class LocalGraph:
    """One AOTAutograd graph: the program a native runtime compiles, and its
    boundary as slots.

    ``program`` is the whole graph as the translator reads it.
    ``input_slots`` has one entry per placeholder and ``output_slots`` one
    per output leaf, in torch's order; ``written`` has one row per tensor
    the program writes, ``(name, mutation target, returned)``. All three are
    the Rust plan's. ``example_inputs`` holds the example value of every
    placeholder and ``leaf_values`` the traced value of every output leaf.
    """

    program: Any
    example_inputs: list
    input_slots: tuple
    output_slots: tuple
    leaf_values: list
    written: tuple
    spec: Any
    sources: dict
    scalar_leaves: frozenset
    device: torch.device

    @property
    def inputs(self) -> list:
        """The example value of each argument the program reads."""
        return [
            value
            for slot, value in zip(self.input_slots, self.example_inputs)
            if slot.read or slot.symbol is not None
        ]

    @property
    def positions(self) -> list:
        """The output leaves the program computes."""
        return [
            position
            for position, slot in enumerate(self.output_slots)
            if slot.source == PROGRAM
        ]

    @property
    def computes(self) -> bool:
        """Whether the native runtime has anything to do: a tensor to
        write, whether a returned value or an input written back into. A
        graph that only forwards inputs and sizes runs in Python alone."""
        return bool(self.written)

    def portable(self) -> "LocalGraph":
        """The boundary alone, as plain data: the slots, the output tree and
        where a call states each symbol. It holds no graph, no example
        value and no shape environment, so it pickles, and it is everything
        a call needs beside the compiled program."""
        return LocalGraph(
            program=None,
            example_inputs=[],
            input_slots=self.input_slots,
            output_slots=self.output_slots,
            leaf_values=[],
            written=self.written,
            spec=self.spec,
            sources=self.sources,
            scalar_leaves=self.scalar_leaves,
            device=self.device,
        )

    def bind(self, args) -> list:
        """One AOT call's arguments, one per slot."""
        if len(args) != len(self.input_slots):
            raise RuntimeError(
                f"AOT graph takes {len(self.input_slots)} arguments, one per "
                f"placeholder, and was called with {len(args)}"
            )
        return list(args)

    def _scalar(self, slot, args):
        """A symbolic scalar leaf at this call: torch's expression with
        each symbol read from the argument that states it."""
        expression = slot.value
        values = {}
        for symbol in expression.free_symbols:
            source = self.sources.get(symbol)
            if source is not None:
                position, axis = source
                value = args[position]
                values[symbol] = value if axis is None else value.shape[axis]
        expression = expression.subs(values)
        if expression.free_symbols:
            raise RuntimeError(f"unbound AOT output dimension: {expression}")
        return _python_scalar(slot.kind, expression)

    def assemble(self, args, rows) -> Any:
        """The AOT output tree: one value per output slot, in order.
        ``rows`` are the tensors the program wrote, by row."""
        result = []
        for position, slot in enumerate(self.output_slots):
            if slot.source == PROGRAM:
                value = rows[slot.index]
                # A Python scalar the graph returns was boxed into a
                # rank-zero tensor for the program.
                if position in self.scalar_leaves:
                    value = value.item()
            elif slot.source == INPUT:
                value = args[slot.index]
            elif slot.source == LITERAL:
                value = slot.value
            else:
                value = self._scalar(slot, args)
            result.append(value)
        return tree_unflatten(result, self.spec)


def prepare_local_graph(
    gm, inputs, *, device_type: str, boundary_plan: Callable[[bytes], tuple]
) -> LocalGraph:
    """Lower one AOTAutograd graph and state its boundary as slots.

    ``boundary_plan`` is the backend extension's function of that name: it
    reads the lowered program and returns its argument slots, output leaves
    and written rows. Every backend's extension builds it from the same
    Rust code, so the slots do not depend on the backend.

    The graph is written whole, and the Rust side reads its boundary off
    that: one slot per placeholder and per output leaf. Nothing here decides
    a slot. A tensor the program reads is checked for ``device_type`` and
    nothing is read from it; an argument the program does not read is not
    looked at.
    """
    program, example_inputs, scalar_leaves = lower_local_graph(gm, inputs)
    arguments, leaves, written = boundary_plan(program.program_json())
    slots = input_slots(arguments)
    if len(slots) != len(inputs):
        raise RuntimeError(
            f"graph has {len(inputs)} placeholders but its boundary states "
            f"{len(slots)} arguments"
        )

    device = torch.device(device_type)
    for slot, value in zip(slots, inputs):
        if slot.kind != TENSOR or not slot.read:
            continue
        if value.device.type != device_type:
            raise RuntimeError(
                f"AOT graph input {slot.name!r} is on {value.device}; this backend "
                f"compiles {device_type} programs"
            )
        if hasattr(value, "placements"):
            raise RuntimeError(
                "AOT compilation expected local tensors, received a DTensor"
            )
        device = value.device

    output = next(n for n in gm.graph.nodes if n.op == "output")
    fx_leaves, spec = tree_flatten(output.args[0])
    if len(fx_leaves) != len(leaves):
        raise RuntimeError(
            f"graph has {len(fx_leaves)} output leaves but its boundary states "
            f"{len(leaves)}"
        )
    output_slots = []
    for leaf, (kind, source, index) in zip(fx_leaves, leaves):
        value = None
        if source == EXPRESSION:
            value = _value(leaf).node.expr
        elif source == LITERAL:
            value = leaf
        output_slots.append(OutputSlot(kind, source, index, value))

    return LocalGraph(
        program=program,
        example_inputs=example_inputs,
        input_slots=slots,
        output_slots=tuple(output_slots),
        leaf_values=[
            _value(leaf) if isinstance(leaf, torch.fx.Node) else leaf
            for leaf in fx_leaves
        ],
        written=tuple(written),
        spec=spec,
        sources=_symbol_sources(inputs),
        scalar_leaves=frozenset(scalar_leaves),
        device=device,
    )


def _placeholder_argument(node: torch.fx.Node, value: Any) -> Any:
    """The graph-signature argument describing one placeholder.

    A tensor or a symbolic scalar is named. A Python constant states
    itself. Any other object has no spelling: it keeps its position as an
    object nothing reads, and a graph that reads it is refused by name.
    """
    from torch.export.graph_signature import (
        ConstantArgument,
        SymBoolArgument,
        SymFloatArgument,
        SymIntArgument,
        TensorArgument,
    )

    if isinstance(value, torch.Tensor):
        return TensorArgument(name=node.name)
    if isinstance(value, torch.SymInt):
        return SymIntArgument(name=node.name)
    if isinstance(value, torch.SymFloat):
        return SymFloatArgument(name=node.name)
    if isinstance(value, torch.SymBool):
        return SymBoolArgument(name=node.name)
    if node.users:
        raise TypeError(
            f"AOT graph input {node.name!r} is a {type(value).__name__} that the "
            "graph reads; AOT compilation requires tensor inputs, and opaque "
            "values may only pass through"
        )
    if not isinstance(value, (int, float, bool, str, type(None))):
        value = None
    node.meta["val"] = value
    return ConstantArgument(name=node.name, value=value)


def _box_plain_scalar_leaves(gm, placeholders) -> tuple:
    """Box each output leaf a node computes as a plain Python scalar into a
    rank-zero tensor, and return those leaves' positions.

    The program computes tensors. A symbolic scalar leaf is an expression
    over sizes and is not touched here; a placeholder leaf is the caller's
    own object. What is left is a number a node produced with no symbol
    behind it, which the program returns as a rank-zero tensor and the
    adapter unboxes with ``.item()``.
    """
    from torch._guards import detect_fake_mode

    output = next(node for node in gm.graph.nodes if node.op == "output")
    leaves, spec = tree_flatten(output.args[0])
    arguments = set(placeholders)
    dtypes = ((bool, torch.bool), (int, torch.int64), (float, torch.float64))
    boxed = []
    for position, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.fx.Node) or leaf in arguments:
            continue
        value = _value(leaf)
        dtype = next((dtype for kind, dtype in dtypes if isinstance(value, kind)), None)
        if dtype is None or isinstance(value, _SYMBOLIC_SCALARS):
            continue
        with gm.graph.inserting_before(output):
            box = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default, (leaf,), {"dtype": dtype}
            )
        fake_mode = detect_fake_mode([_value(node) for node in placeholders])
        if fake_mode is not None:
            with fake_mode:
                box.meta["val"] = torch.empty((), dtype=dtype)
        leaves[position] = box
        boxed.append(position)
    if boxed:
        output.args = (tree_unflatten(leaves, spec),)
        gm.graph.lint()
        gm.recompile()
    return tuple(boxed)


def _symbol_argument(leaf: Any) -> Any:
    """The graph-signature argument describing one placeholder or output leaf."""
    from torch.export.graph_signature import (
        ConstantArgument,
        SymBoolArgument,
        SymFloatArgument,
        SymIntArgument,
        TensorArgument,
    )

    if not isinstance(leaf, torch.fx.Node):
        return ConstantArgument(name="", value=leaf)
    value = _value(leaf)
    if isinstance(value, torch.SymInt):
        return SymIntArgument(name=leaf.name)
    if isinstance(value, torch.SymFloat):
        return SymFloatArgument(name=leaf.name)
    if isinstance(value, torch.SymBool):
        return SymBoolArgument(name=leaf.name)
    return TensorArgument(name=leaf.name)


def _argument_json(arg: Any) -> dict:
    """One signature argument in the PT2 spelling."""
    from torch.export.graph_signature import (
        ConstantArgument,
        SymBoolArgument,
        SymFloatArgument,
        SymIntArgument,
        TensorArgument,
    )

    if isinstance(arg, TensorArgument):
        return {"as_tensor": {"name": arg.name}}
    if isinstance(arg, SymIntArgument):
        return {"as_sym_int": {"as_name": arg.name}}
    if isinstance(arg, SymFloatArgument):
        return {"as_sym_float": {"as_name": arg.name}}
    if isinstance(arg, SymBoolArgument):
        return {"as_sym_bool": {"as_name": arg.name}}
    if isinstance(arg, ConstantArgument):
        value = arg.value
        if value is None:
            return {"as_none": True}
        # ``bool`` before ``int``: a Python bool is an int.
        for key, kind in (
            ("as_bool", bool),
            ("as_int", int),
            ("as_float", float),
            ("as_string", str),
        ):
            if isinstance(value, kind):
                return {key: value}
    raise TypeError(f"graph signature argument {arg!r} has no PT2 spelling")


def _signature_json(signature: Any) -> dict:
    """The graph signature in the PT2 spelling, for the spec kinds an
    AOTAutograd graph has: user inputs, lifted tensor constants, user
    outputs and user input mutations."""
    inputs = []
    for spec in signature.input_specs:
        kind = spec.kind.name
        if kind == "USER_INPUT":
            inputs.append({"user_input": {"arg": _argument_json(spec.arg)}})
        elif kind == "CONSTANT_TENSOR":
            inputs.append(
                {
                    "tensor_constant": {
                        "arg": {"name": spec.arg.name},
                        "tensor_constant_name": spec.target,
                    }
                }
            )
        else:
            raise TypeError(f"graph input kind {kind} has no PT2 spelling here")
    outputs = []
    for spec in signature.output_specs:
        kind = spec.kind.name
        if kind == "USER_OUTPUT":
            outputs.append({"user_output": {"arg": _argument_json(spec.arg)}})
        elif kind == "USER_INPUT_MUTATION":
            outputs.append(
                {
                    "user_input_mutation": {
                        "arg": {"name": spec.arg.name},
                        "user_input_name": spec.target,
                    }
                }
            )
        else:
            raise TypeError(f"graph output kind {kind} has no PT2 spelling here")
    return {"input_specs": inputs, "output_specs": outputs}


def _graph_json(gm: torch.fx.GraphModule, signature: Any) -> dict:
    """The graph in the PT2 spelling: inputs, nodes, outputs, and the
    metadata of every value.

    torch's serializer writes each node. Its class is final, so it is driven
    here node by node instead of subclassed. The one node it cannot write
    is a placeholder holding a symbolic float or bool, which is written with
    the serializer's own value writers, exactly as it writes a symbolic
    integer placeholder.
    """
    from torch._export.serde import serialize as serde

    serializer = serde.GraphModuleSerializer(signature, [])
    state = serializer.graph_state
    for node in gm.graph.nodes:
        value = node.meta.get("val")
        try:
            if node.op == "placeholder" and isinstance(value, torch.SymFloat):
                state.inputs.append(
                    serde.Argument.create(
                        as_sym_float=serializer.serialize_sym_float_output(
                            node.name, value
                        )
                    )
                )
            elif node.op == "placeholder" and isinstance(value, torch.SymBool):
                state.inputs.append(
                    serde.Argument.create(
                        as_sym_bool=serializer.serialize_sym_bool_output(
                            node.name, value
                        )
                    )
                )
            else:
                getattr(serializer, f"handle_{node.op}")(node)
        except Exception as error:
            raise RuntimeError(
                f"cannot serialize graph node {node.format_node()}: {error}"
            ) from error
    graph = serde.Graph(
        inputs=state.inputs,
        nodes=state.nodes,
        tensor_values=state.tensor_values,
        sym_int_values=state.sym_int_values,
        sym_float_values=state.sym_float_values,
        sym_bool_values=state.sym_bool_values,
        custom_obj_values=state.custom_obj_values,
        outputs=state.outputs,
        is_single_tensor_return=state.is_single_tensor_return,
    )
    return serde._dataclass_to_dict(graph)


@dataclass
class LoweredGraph:
    """One graph in the form the Rust translator reads.

    ``graph_module`` is the graph as serialized: constants lifted to
    placeholders, input mutations as outputs. ``graph_signature`` is torch's
    signature object used as plain data; ``constants`` holds each lifted
    tensor constant by name. The fields carry the names an
    ``ExportedProgram`` gives the same things.
    """

    graph_module: torch.fx.GraphModule
    graph_signature: Any
    constants: dict
    range_constraints: dict
    _json: bytes | None = field(default=None, repr=False)

    def program_json(self) -> bytes:
        """The program as the JSON document of a PT2 archive, without the
        archive: the graph module and the range of every symbol. Written
        once."""
        if self._json is None:
            self._json = self._write_json()
        return self._json

    def _write_json(self) -> bytes:
        from torch._export.serde import serialize as serde

        program = {
            "graph_module": {
                "graph": _graph_json(self.graph_module, self.graph_signature),
                "signature": _signature_json(self.graph_signature),
            },
            "range_constraints": serde._dataclass_to_dict(
                serde.serialize_range_constraints(self.range_constraints)
            ),
        }
        return json.dumps(program, cls=serde.EnumEncoder, allow_nan=False).encode()


def _lift_input_mutations(gm) -> list:
    """Turn AOTAutograd's in-graph input mutations into mutation outputs.

    With ``keep_inference_input_mutations`` AOTAutograd ends the graph with
    one ``copy_(input, value)`` per mutated input and returns the ``copy_``
    itself where the caller receives the mutated input. ``torch.export``
    spells the same program as a ``user_input_mutation`` output per input,
    emitted before the user outputs, and the translator binds that output on
    the input's own storage. Rewrite to that spelling: drop each ``copy_``,
    read its uses from the written value, and return ``(placeholder, value)``
    pairs in graph order for the caller to prepend to the outputs.
    """
    output = next(node for node in gm.graph.nodes if node.op == "output")
    mutations: list[tuple[torch.fx.Node, torch.fx.Node]] = []
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not torch.ops.aten.copy_.default
        ):
            continue
        target, value = node.args[0], node.args[1]
        if not isinstance(target, torch.fx.Node) or target.op != "placeholder":
            raise RuntimeError(
                f"in-place copy {node.name!r} writes {target!r}, which is not a graph "
                "input; only input mutations are compiled in place"
            )
        node.replace_all_uses_with(value)
        gm.graph.erase_node(node)
        mutations.append((target, value))
    if mutations:
        leaves, _ = tree_flatten(output.args[0])
        output.args = ((*(value for _, value in mutations), *leaves),)
        gm.graph.lint()
        gm.recompile()
    return mutations


def lower_local_graph(gm, example_inputs) -> tuple:
    """Lower one AOTAutograd graph to the form the Rust translator reads,
    without re-tracing and without an ``ExportedProgram``.

    AOTAutograd already hands the backend a functional ATen graph whose nodes
    carry FakeTensor ``val`` metadata with the program's symbolic shapes, and
    whose operators were decomposed once by the table the backend passed to
    ``aot_autograd``. The translator needs three more things: a graph
    signature, lifted tensor constants, and the range of every shape symbol.
    All three are read off that metadata, so no second ``torch.export``
    capture runs and no decomposition retry is needed.

    Returns ``(program, inputs, scalar_leaves)``: ``program`` is a
    ``LoweredGraph`` holding the whole graph, ``inputs`` the example value
    of every placeholder, and ``scalar_leaves`` the output leaves that were
    boxed from plain Python scalars.
    """
    from torch._export.passes.lift_constants_pass import (
        ConstantAttrMap,
        lift_constants_pass,
    )
    from torch._guards import detect_fake_mode
    from torch.export.exported_program import _common_getitem_elimination_pass
    from torch.export.graph_signature import (
        ExportGraphSignature,
        InputKind,
        InputSpec,
        OutputKind,
        OutputSpec,
        TensorArgument,
    )

    # Work on a copy: the caller keeps the original graph for its own
    # bookkeeping. The copy shares the module's attributes (see
    # private_graph_copy), which is what the constant lifting below reads.
    gm = private_graph_copy(gm)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        raise RuntimeError(
            f"graph has {len(placeholders)} placeholders but "
            f"{len(example_inputs)} example inputs"
        )
    for node, value in zip(placeholders, example_inputs):
        if "val" not in node.meta:
            node.meta["val"] = node.meta.get("example_value", value)
    # torch.export spells the module type in ``nn_module_stack`` as a string
    # while it traces; Dynamo leaves the class object, which the PT2
    # serializer refuses. Spell it the same way so provenance survives.
    for node in gm.graph.nodes:
        stack = node.meta.get("nn_module_stack")
        if stack:
            node.meta["nn_module_stack"] = {
                key: (
                    path,
                    ty if isinstance(ty, str) else f"{ty.__module__}.{ty.__qualname__}",
                )
                for key, (path, ty) in stack.items()
            }

    # Leaf positions count the graph's own output leaves, so box before the
    # mutation outputs take the leading positions; the written values are
    # outputs before any dead-code pass runs.
    scalar_leaves = _box_plain_scalar_leaves(gm, placeholders)
    mutations = _lift_input_mutations(gm)
    _drop_dead_data_dependent_ops(gm)
    # An in-graph mutation was just made an output, and an assertion is
    # impure, so both survive this.
    gm.graph.eliminate_dead_code()

    output = next(node for node in gm.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output.args[0])
    user_leaves = leaves[len(mutations) :]
    signature = ExportGraphSignature(
        input_specs=[
            InputSpec(
                kind=InputKind.USER_INPUT,
                arg=_placeholder_argument(node, node.meta["val"]),
                target=None,
            )
            for node in placeholders
        ],
        output_specs=[
            OutputSpec(
                kind=OutputKind.USER_INPUT_MUTATION,
                arg=TensorArgument(name=value.name),
                target=target.name,
            )
            for target, value in mutations
        ]
        + [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT, arg=_symbol_argument(leaf), target=None
            )
            for leaf in user_leaves
        ],
    )
    # Graph-internal literals (``torch.tensor([...])`` in the function body)
    # arrive as ``get_attr`` nodes; the PT2 schema wants them as
    # CONSTANT_TENSOR inputs with their values in ``constants``. The lifting
    # pass names a constant after the module that first used it, read from
    # ``nn_module_stack``; AOTAutograd graphs carry no module stack, and an
    # empty one names the constant after itself.
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            node.meta.setdefault("nn_module_stack", {})
    constants = lift_constants_pass(gm, signature, ConstantAttrMap())
    # Dynamo spells a literal as ``lift_fresh_copy`` of the constant. The
    # tracer handles that op itself, outside the decomposition table, and
    # the translator has no lowering for it. Its functional meaning once the
    # constant is a graph input is a fresh copy.
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.lift_fresh_copy.default
        ):
            node.target = torch.ops.aten.clone.default
    gm.recompile()

    symbols: set = set()
    for node in gm.graph.nodes:
        for value in torch.utils._pytree.tree_leaves(node.meta.get("val")):
            if isinstance(value, torch.Tensor):
                value = (*value.shape, *value.stride())
            else:
                value = (value,)
            for item in value:
                if isinstance(item, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                    symbols.update(item.node.expr.free_symbols)
    fake_mode = detect_fake_mode([node.meta["val"] for node in placeholders])
    shape_env = (
        fake_mode.shape_env
        if fake_mode is not None
        else _shape_env(
            value
            for node in gm.graph.nodes
            for value in torch.utils._pytree.tree_leaves(node.meta.get("val"))
        )
    )
    range_constraints = {}
    if shape_env is not None:
        for symbol in symbols:
            if symbol in shape_env.var_to_range:
                range_constraints[symbol] = shape_env.var_to_range[symbol]

    # The one graph rewrite an ``ExportedProgram`` applies when it is built:
    # two ``getitem`` nodes reading one element become one.
    _common_getitem_elimination_pass(gm, signature, [])
    program = LoweredGraph(
        graph_module=gm,
        graph_signature=signature,
        constants=constants,
        range_constraints=range_constraints,
    )
    # Serializer gap workaround; must run before the graph is written. See
    # _lower_sym_sum.
    _lower_sym_sum(program)
    return program, list(example_inputs), scalar_leaves
