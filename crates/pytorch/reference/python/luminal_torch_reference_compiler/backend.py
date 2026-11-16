"""The luminal_torch_reference_compiler local compiler.

AOTAutograd hands the backend one functional ATen graph per phase. Each is
lowered in process (see ``aot_graph``), translated and searched by the
reference runtime, and handed over as a code object: the bytes
``luminal_torch_reference_runtime`` builds every executable from.

Everything in this module runs at compile time. A call runs in the runtime
package, which imports none of it.
"""

import json
from collections.abc import Sequence
from typing import Any

import torch
from luminal_torch_common.aot_graph import LoweredGraph, integer_argument
from luminal_torch_common.dimensions import profile_value

from . import _luminal


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    # Flatten first: a 0-dim tensor cannot be viewed as a wider dtype.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _compile_local_graph(
    program: LoweredGraph,
    example_inputs: Sequence[Any],
    leaves: list,
    search_iterations: int | None = None,
    search_log: bool = False,
    max_intermediate_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
) -> tuple[bytes, Any]:
    """Compile one lowered AOTAutograd graph with the native runtime, and
    return its code object's bytes beside the searched graph they were made
    from. The graph is the compiler's: nothing a call needs is left in it.

    ``example_inputs`` holds one example value per argument slot: the data
    the search executes candidates on. ``leaves`` is the graph's output leaf
    table (``LocalGraph.leaf_rows``). An op the translator does not lower
    fails here, by name.
    """
    graph = _luminal.compile_graph(program.program_json())

    slots = graph.input_slots
    if len(slots) != len(example_inputs):
        raise RuntimeError(
            f"the graph has {len(slots)} arguments "
            f"({', '.join(name for _, name, _, _ in slots)}) and was given "
            f"{len(example_inputs)} example inputs"
        )
    examples = {name: value for (_, name, _, _), value in zip(slots, example_inputs)}

    # Every tensor the program reads is staged: an argument at its example
    # value, a constant the graph holds at the constant itself. A constant's
    # elements then go into the code object.
    for name, kind, constant_name in zip(
        graph.input_names, graph.input_kinds, graph.parameter_names
    ):
        if kind == "user_input":
            value = examples[name]
        else:
            value = program.constants.get(constant_name)
            if value is None:
                raise RuntimeError(
                    f"constant {constant_name!r} (graph input {name!r}) is not "
                    "among the constants lifted from the graph"
                )
        graph.set_input(name, _tensor_bytes(value), list(value.shape))

    # An integer argument that defines a symbol is profiled at the value
    # torch traced it with.
    for (kind, name, _, symbol), value in zip(slots, example_inputs):
        if kind == "int" and symbol is not None:
            graph.bind_int(name, integer_argument(name, profile_value(value)))

    graph.search(
        search_iterations,
        search_log=search_log,
        max_intermediate_bytes=max_intermediate_bytes,
        memory_budget_bytes=memory_budget_bytes,
    )
    return bytes(graph.code(json.dumps(leaves))), graph
