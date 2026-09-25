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
    ``dim_buckets`` maps PyTorch ``ShapeVar`` objects to sequences of
    ``DimBucket`` values. Reuse those objects in ``torch.compile``'s
    ``dynamic_shapes=ShapesSpec(...)``; bounds come from PyTorch.
    ``search_iterations`` selects one candidate per iteration for each
    local graph and each dimension-bucket combination. Each candidate
    executes a warmup and a timed profile run.
    """
