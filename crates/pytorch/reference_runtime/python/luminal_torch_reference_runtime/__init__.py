"""Runtime for programs compiled by the Luminal reference backend.

A compiled graph is a *code object*: plain bytes that hold everything a call
needs and nothing the compiler needed. ``Code`` is that code object as the
object torch holds, calls, saves and loads.

This package is everything a saved program needs at load. It imports no
compiler: not the parser, not the translator, not search. Its import path is
written into every saved program, so the name ``luminal_torch_reference_runtime
.Code`` is a compatibility contract.
"""

import json
import threading
from typing import Any

from torch._inductor.output_code import OutputCode

from . import _runtime

__all__ = ["FORMAT", "Code", "describe"]

# The version of the code object's byte form this runtime reads.
FORMAT: int = _runtime.FORMAT


class Code(OutputCode):
    """One compiled graph.

    ``code`` is the code object's bytes, which is all of it: pickling,
    copying and loading carry those bytes and nothing else. A call drops
    straight into the runtime, which reads torch's argument list itself, runs
    the program and builds the outputs.

    A compiled graph is an ordinary Python object. It may be called on any
    thread and freed on any thread: torch calls a backward graph from its
    autograd thread, for one. It holds one executable, built from the bytes
    on the first call, and its calls run one at a time: a call from another
    thread waits for the one in progress.
    The interpreter is released while the program computes.
    """

    # Whether each call evaluates the size conditions torch asserted in the
    # graph (a user's ``torch._check`` on a size). Binding does not depend on
    # them: set this to ``False``, on the class or on one code object, and
    # calls run unchecked.
    runtime_assertions = True

    def __init__(self, code: bytes) -> None:
        self.code = bytes(code)
        # torch calls a compiled graph with its arguments in one list. It
        # reads this flag off a wrapper that copies the instance's own
        # attributes, so it is set on the instance.
        self._boxed_call = True
        self._executable = None
        self._building = threading.Lock()

    def _built(self):
        """The one executable, built from the bytes by whichever call is
        first."""
        executable = self._executable
        if executable is None:
            with self._building:
                executable = self._executable
                if executable is None:
                    executable = self._executable = _runtime.Executable(self.code)
        return executable

    def __call__(self, inputs: list) -> Any:
        return self._built()(inputs, self.runtime_assertions)

    def __reduce__(self):
        # Always as this class and these bytes: a saved program names the
        # runtime, whatever built it.
        return (Code, (self.code,))

    def prepare_for_serialization(self) -> None:
        """torch calls this on a copy before it saves. The copy already is
        its saved form."""

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        """torch calls this once on a loaded object. There is nothing to
        rebuild: the executable is built on the first call."""

    def set_triton_bundle(self, triton_bundle: Any) -> None:
        """Part of the ``OutputCode`` interface; there is nothing to bundle."""


def describe(code: "Code | bytes") -> dict:
    """A code object's tables, without its plan: what a call reads its
    arguments against and builds its outputs from."""
    raw = code.code if isinstance(code, Code) else code
    return json.loads(_runtime.describe(raw))
