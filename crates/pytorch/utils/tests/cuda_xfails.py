"""Known CUDA-lite failures in the shared PT2 parity suite.

The CUDA Modal workflow was unable to launch until PR 615 repaired its Modal
volume configuration.  The first complete, isolated run after that repair
(PR 615 run 42458080021) is the baseline recorded in ``cuda_xfails.txt``.
Every listed case still runs and
is marked non-strict xfail: a backend fix therefore appears as XPASS and can be
removed from the file without hiding a regression in supported coverage.

Keys are ``<test file basename>::<parameterized test name>`` so they do not
depend on pytest's resolved root directory.
"""

from pathlib import Path

import backend_test_utils
import pytest


def _key(item: pytest.Item) -> str:
    return f"{item.path.name}::{item.name}"


def _known_failures() -> set[str]:
    path = Path(__file__).with_name("cuda_xfails.txt")
    return {
        line
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    }


def apply_cuda_xfails(items: list[pytest.Item]) -> None:
    """Mark the exact observed CUDA-lite backlog, while continuing to run it."""
    if backend_test_utils.selected_backend() != "cuda_lite":
        return
    failures = _known_failures()
    reason = (
        "known CUDA-lite PT2 parity gap from the first isolated Modal baseline; "
        "the case still runs and reports XPASS when support lands"
    )
    for item in items:
        if _key(item) in failures:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
