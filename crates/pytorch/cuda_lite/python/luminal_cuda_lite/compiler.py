"""Configurable CUDA-lite compiler for torch.compile."""

from dataclasses import dataclass

from .backend import _compile_graph


@dataclass(frozen=True)
class Compiler:
    """Pass an instance as ``torch.compile(..., backend=compiler)``.

    Memory limits are bytes; None selects the CUDA runtime's device budget.
    Logging is off by default. CUDA-lite retains its existing compilation path.
    """

    search_iterations: int = 1
    search_log: bool = False
    device_budget_bytes: int | None = None
    max_intermediate_bytes: int | None = None

    def __post_init__(self):
        if not isinstance(self.search_iterations, int) or self.search_iterations < 1:
            raise ValueError("search_iterations must be a positive integer")
        for name in ("device_budget_bytes", "max_intermediate_bytes"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")

    def __call__(self, gm, example_inputs):
        return _compile_graph(
            gm,
            example_inputs,
            search_iterations=self.search_iterations,
            search_log=self.search_log,
            device_budget_bytes=self.device_budget_bytes,
            max_intermediate_bytes=self.max_intermediate_bytes,
        )
