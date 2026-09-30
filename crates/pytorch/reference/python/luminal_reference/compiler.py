"""Configurable reference compiler for torch.compile."""

from .aot import ReferenceAOTBackend


class Compiler(ReferenceAOTBackend):
    """Compile through AOTAutograd with options fixed at construction.

    Example::

        compiler = Compiler(search_iterations=10, log=True,
                            memory_budget_bytes=8 * 1024**3)
        model = torch.compile(model, backend=compiler)

    ``None`` memory limits select the runtime defaults (2 GiB per intermediate,
    8 GiB live payload). ``graphs`` and ``regions`` retain compilation
    diagnostics. One compiler can compile multiple graphs.
    Each local graph compiles once over its PyTorch-declared shape bounds.
    Applications can use ``torch.compile(..., isolate_recompiles=True)`` to create independently bounded callables and
    dispatch among them. ``search_iterations`` selects the number of candidates
    to profile for each local graph.
    """
