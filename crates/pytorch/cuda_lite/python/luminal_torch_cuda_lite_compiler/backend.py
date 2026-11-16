"""Compiling one AOTAutograd graph for the CUDA-lite runtime.

The public entry point uses AOTAutograd for inference and training. Its
local compiler lowers each AOTAutograd graph in process
(``luminal_torch_common.aot_graph``, shared by every backend), hands it to
the Rust extension for translation, declares the boundary, searches, and
returns the graph's code object as bytes. Running a call is the runtime
package's (``luminal_torch_cuda_lite_runtime``); nothing here does.
Parameters and buffers are AOTAutograd inputs, bound per call like any
other; an input the program mutates is written in place through a mutation
output bound on the input's own storage. An activation the forward saves for
the backward graph is nobody's but this backend's to read again, so it is
allocated row-major, which every plan can write, and the strides are
reported to AOTAutograd, which hands the backward example inputs at those
strides.

DEVICE MODEL: every boundary tensor is the caller's own device memory.
Its element strides are read once at compile time (``boundary.py``) and
declared to the runtime, which binds it on one buffer id; each call hands
that buffer the tensor's address. Which map a chain is — contiguous,
column-major, neither — is discovered in the e-graph, never decided here.
A user-visible output is bound at eager's exact strides — read from the
same fake value the input layouts are read from — and the code object
states them, so the runtime allocates the output at those strides on every
call. Aliasing has one spelling: two bindings naming one buffer id, which
is how a writeback and the input it mutates share a pointer.
"""

import json
from collections.abc import Sequence
from typing import Any

import torch
from luminal_torch_common.aot_graph import LoweredGraph, integer_argument
from luminal_torch_common.dimensions import profile_value
from torch.export.graph_signature import TensorArgument

from . import _luminal
from .boundary import (
    UnsupportedBoundary,
    boundary_layout,
    buffer_nbytes,
    layout_spec,
)

# The output specs this backend declares to the runtime: the two kinds an
# AOTAutograd graph has. Anything else is refused by name.
_BOUND_OUTPUT_KINDS = frozenset({"USER_OUTPUT", "USER_INPUT_MUTATION"})


def _node_fakes(program: Any) -> dict[str, Any]:
    """The fake example value behind every graph node, by node name.

    It is the program's OWN symbolic metadata — the symbols the translator
    reads out of the program it is handed — so a size or stride stated from
    it names a dimension the runtime knows, at whatever extent a later call
    gives it. One table for both sides of the boundary: an
    output's layout is read from the same metadata an input's is.
    """
    fakes: dict[str, Any] = {}
    for node in program.graph_module.graph.nodes:
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            fakes[node.name] = value
    return fakes


def _boundary_tensors(
    program: Any, example_inputs: Sequence[Any], unread: frozenset
) -> list[tuple[str, str, torch.Tensor, Any]]:
    """(graph input name, kind, tensor, fake example value) for every graph
    input that is storage the program binds, in graph order. An argument
    named in ``unread`` is a tensor no node reads: nothing states it, so
    nothing looks at it. The names are the program's placeholder names,
    which are what the translator reads. The fake carries the program's
    symbolic sizes and strides for an argument; a constant the graph holds
    is a concrete tensor and states itself."""
    fakes = _node_fakes(program)
    rows: list[tuple[str, str, torch.Tensor, Any]] = []
    user_index = 0
    for spec in program.graph_signature.input_specs:
        name = getattr(spec.arg, "name", None)
        kind = spec.kind.name
        if name is None:
            raise UnsupportedBoundary(
                f"graph input {kind.lower()} {spec.target!r} is not a tensor"
            )
        if kind == "USER_INPUT":
            if user_index >= len(example_inputs):
                raise RuntimeError(
                    f"the graph has more arguments than example inputs: {name!r}"
                )
            value = example_inputs[user_index]
            user_index += 1
            # Only a tensor is storage to bind. A symbolic scalar is the
            # expression torch recorded for it, and an integer that is a
            # symbol is bound by value.
            if not isinstance(spec.arg, TensorArgument) or name in unread:
                continue
        elif kind == "CONSTANT_TENSOR":
            if spec.target not in program.constants:
                raise RuntimeError(
                    f"{name}: {spec.target!r} is not among the constants lifted "
                    "from the graph"
                )
            value = program.constants[spec.target]
        else:
            raise UnsupportedBoundary(
                f"{name}: graph inputs of kind {kind.lower()} are not bound by luminal_torch_cuda_lite_compiler"
            )
        if not isinstance(value, torch.Tensor):
            raise UnsupportedBoundary(
                f"{name}: graph input {spec.target!r} is not a tensor"
            )
        rows.append(
            (name, kind, value, fakes.get(name) if kind == "USER_INPUT" else None)
        )
    if user_index != len(example_inputs):
        raise RuntimeError(
            f"the graph consumed {user_index} of {len(example_inputs)} example inputs"
        )
    return rows


def _refuse_unbound_outputs(program: Any) -> None:
    """Refuse an output the boundary has no statement for."""
    for spec in program.graph_signature.output_specs:
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
            f"output {name!r} carries no traced example value, so the layout eager "
            "gives it cannot be read"
        )
    return fake


def _output_layout_rows(
    program: Any, dense_names: frozenset = frozenset()
) -> list[tuple[str, str, list[str]]]:
    """One ``(graph name, layout tag, element strides)`` row per graph
    output the caller allocates: the element strides EAGER gives it, read
    from the traced fake value exactly like an input's. An output named in
    ``dense_names`` is declared row-major instead: a value only this
    backend reads again (an activation saved for the backward graph), whose
    layout the forward compiler chooses and reports to AOTAutograd.

    This is a LAYOUT STATEMENT PER NAME, not the output list — which
    outputs a program has, and in which order, is the translation's to say
    (`graph.output_names`). A writeback takes no row: it writes the storage
    of the input it mutates and is bound at that input's layout.
    """
    fakes = _node_fakes(program)
    writebacks = {
        spec.arg.name
        for spec in program.graph_signature.output_specs
        if spec.kind.name == "USER_INPUT_MUTATION"
    }
    # An output that is a view of a boundary tensor is bound on that
    # tensor's storage, as it lies; every other output is allocated fresh,
    # so the offset eager's view had inside an intermediate is not a fact
    # about the storage the caller provides.
    aliased = {name for name, _, _ in _output_alias_rows(program)}
    arguments = {
        spec.arg.name
        for spec in program.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT" and isinstance(spec.arg, TensorArgument)
    }
    rows: list[tuple[str, str, list[str]]] = []
    stated: set[str] = set()
    for spec in program.graph_signature.output_specs:
        if spec.kind.name != "USER_OUTPUT":
            continue
        # Only a tensor the program computes is storage the caller
        # allocates. A leaf that is an argument is the caller's own object,
        # and a symbolic scalar or a literal has no storage.
        if not isinstance(spec.arg, TensorArgument) or spec.arg.name in arguments:
            continue
        name = spec.arg.name
        # One value returned twice is one statement: the same name, at the
        # same layout, stated once.
        if name in writebacks or name in stated:
            continue
        fake = _output_fake(name, fakes)
        stated.add(name)
        layout = boundary_layout(
            name,
            fake,
            allocated=name not in aliased,
            dense=name in dense_names and name not in aliased,
        )
        tag, strides = layout_spec(layout)
        rows.append((name, tag, list(strides)))
    return rows


def _output_alias_rows(program: Any) -> list[tuple[str, str, int]]:
    """One ``(output, owner)`` row per user output whose traced example value
    shares its STORAGE with an earlier boundary tensor: a graph input, a
    writeback, or an earlier output. Eager hands the caller a view of that
    tensor, so the runtime binds the output on the owner's buffer and the
    call returns a view. This is read off the fake tensors torch traced —
    which storage each value lives in — and states it; nothing is derived
    from op names."""
    fakes = _node_fakes(program)
    names: list[str] = []
    for spec in list(program.graph_signature.input_specs) + list(
        program.graph_signature.output_specs
    ):
        name = getattr(spec.arg, "name", None)
        if name is not None and name not in names:
            names.append(name)
    owner_by_storage: dict[int, str] = {}
    outputs = {
        getattr(spec.arg, "name", None)
        for spec in program.graph_signature.output_specs
        if spec.kind.name == "USER_OUTPUT"
    }
    # Rows carry the view's element offset RELATIVE to its owner, so the
    # returned view addresses the owner's storage where eager's does. Two
    # outputs that are views of one INTERMEDIATE share a storage too, but
    # neither is the other's base: a view is only bound on an earlier
    # output when it lies inside that output's own element span, and
    # otherwise gets a buffer of its own.
    rows: list[tuple[str, str, int]] = []
    for name in names:
        fake = fakes.get(name)
        if not isinstance(fake, torch.Tensor):
            continue
        storage = fake.untyped_storage()._cdata
        owner = owner_by_storage.setdefault(storage, name)
        if owner == name or name not in outputs:
            continue
        offset = int(fake.storage_offset()) - int(fakes[owner].storage_offset())
        if owner in outputs:
            inside = _element_span(fake)
            holds = _element_span(fakes[owner])
            if inside is None or holds is None or offset < 0 or offset + inside > holds:
                continue
        rows.append((name, owner, offset))
    return rows


def _element_span(fake: torch.Tensor) -> int | None:
    """Elements a tensor reaches from its own first element: the last
    element its strides address, plus one. ``None`` when an extent or
    stride is symbolic, so a caller stays conservative."""
    span = 1
    for size, stride in zip(fake.shape, fake.stride()):
        if isinstance(size, torch.SymInt) or isinstance(stride, torch.SymInt):
            return None
        if size == 0:
            return 0
        span += (int(size) - 1) * int(stride)
    return span


def _span_bytes(tensor: torch.Tensor) -> bytes:
    """The bytes a constant's strides reach from its first element, in
    storage order: what the code object holds for it, and what the runtime
    copies back to the device under the same shape and strides."""
    elements = buffer_nbytes(tensor) // tensor.element_size()
    flat = tensor.detach().as_strided((elements,), (1,)).cpu().contiguous()
    return flat.view(torch.uint8).numpy().tobytes()


def _compile_local_graph(
    program: LoweredGraph,
    example_inputs: Sequence[Any],
    leaves: list,
    search_iterations: int | None = None,
    search_log: bool = False,
    device_budget_bytes: int | None = None,
    max_intermediate_bytes: int | None = None,
    dense_outputs: frozenset = frozenset(),
) -> tuple[bytes, Any]:
    """Compile one lowered AOTAutograd graph with the CUDA-lite runtime, and
    return its code object's bytes beside the searched graph they were made
    from. The bytes are all a call needs; the graph is the compiler's, kept
    only for what it reports about the compile.

    ``leaves`` is the output leaf table (``LocalGraph.leaf_rows``).

    ``example_inputs`` holds one example value per argument slot. They are
    AOTAutograd's fake tensors: the boundary reads dtypes, extents and
    strides from them and addresses nothing until a call.
    ``dense_outputs`` are the output leaves whose layout this backend
    chooses (row-major) rather than eager's. An op the translator does not
    lower fails here, by name.
    """
    leaf_names = [
        getattr(spec.arg, "name", None)
        for spec in program.graph_signature.output_specs
        if spec.kind.name == "USER_OUTPUT"
    ]
    dense_names = frozenset(
        leaf_names[i] for i in dense_outputs if i < len(leaf_names) and leaf_names[i]
    )

    # Read every boundary tensor's element strides and declare them with
    # the program: the runtime binds what the caller has, or refuses it.
    _refuse_unbound_outputs(program)
    # The program states its boundary before anything else is read: a
    # tensor argument no node reads is not looked at, so it may live
    # anywhere and lie any way.
    program_json = program.program_json()
    arguments, _, _ = _luminal.boundary_plan(program_json)
    unread = frozenset(
        name for kind, name, read, _ in arguments if kind == "tensor" and not read
    )
    rows = _boundary_tensors(program, example_inputs, unread)
    tensors = {name: value for name, _, value, _ in rows}
    layouts = {
        name: boundary_layout(name, value, fake) for name, _, value, fake in rows
    }
    declared = []
    for name, _, _, _ in rows:
        tag, strides = layout_spec(layouts[name])
        declared.append((name, tag, list(strides)))
    # A user-visible output is declared at the strides EAGER gives it,
    # read from the traced fake value exactly like an input: what the
    # caller receives has the strides the uncompiled program hands back,
    # never a contiguous substitute.
    declared_outputs = _output_layout_rows(program, dense_names)
    # An output that is a view of another boundary tensor is bound on
    # that tensor's buffer; the caller receives a view of what it holds.
    alias_rows = _output_alias_rows(program)
    graph = _luminal.compile_graph(
        program_json,
        declared,
        declared_outputs,
        [(name, owner) for name, owner, _ in alias_rows],
    )

    for name in graph.input_names:
        if name not in tensors:
            raise RuntimeError(
                f"the translated program names an input {name!r} the graph signature "
                f"does not declare (declared: {sorted(tensors)})"
            )
    slots = graph.input_slots
    if len(slots) != len(example_inputs):
        raise RuntimeError(
            f"the graph has {len(slots)} arguments "
            f"({', '.join(name for _, name, _, _ in slots)}) and was given "
            f"{len(example_inputs)} example inputs"
        )
    # Seed the symbolic dims from the declared shapes. An integer argument
    # that defines a symbol is profiled at the value torch traced it with.
    graph.bind_input_shapes(
        [
            (name, [profile_value(size) for size in tensors[name].shape])
            for name in graph.input_names
        ],
        [
            (name, integer_argument(name, profile_value(value)))
            for (kind, name, _, symbol), value in zip(slots, example_inputs)
            if kind == "int" and symbol is not None
        ],
    )
    graph.search(
        search_iterations,
        search_log=search_log,
        device_budget_bytes=device_budget_bytes,
        max_intermediate_bytes=max_intermediate_bytes,
    )

    # The code object: the searched plan lowered to what the device runs,
    # with the boundary tables a call reads beside it. A constant the graph
    # holds goes in as its bytes; the runtime puts it back on the device.
    constants = [
        (
            name,
            list(tensors[name].shape),
            list(tensors[name].stride()),
            _span_bytes(tensors[name]),
        )
        for name, kind in zip(graph.input_names, graph.input_kinds)
        if kind != "user_input"
    ]
    code = bytes(graph.code(json.dumps(leaves), constants, alias_rows))
    return code, graph
