"""The collective wire contract between a front end and the translator.

A compiler front end that honours PyTorch's own sharding sees functional
collectives (``_c10d_functional.all_reduce`` and friends) in the exported
graph. Resolving a process-group name to its member ranks is only possible in
Python, so the front end's pre-pass rewrites every collective into this one
custom op plus ordinary ATen ops, and the translator records it as the logical
all-gather:

``all_gather(x, group, width, sequence)`` stacks ``x`` from every member of the
ordered group along a new leading axis, ``[width, *x.shape]``, member ``i``'s
tensor at index ``i``. ``group`` is the communicator descriptor, a JSON string
over logical ranks (``{"world": W, "groups": [[..], ..]}``) that is identical on
every member; ``width`` is the group's size; ``sequence`` is the effect
identity that keeps two gathers over equal operands distinct. An all-reduce is
the sum over axis 0, an all-gather the reshape merging the two leading axes, a
reduce-scatter the rank's slice of the sum.

The eager body refuses: the op only has meaning inside a compiled program whose
backend lowers it to communication.
"""
from __future__ import annotations

import torch

OP_NAME = "luminal_collective::all_gather"


@torch.library.custom_op(OP_NAME, mutates_args=())
def all_gather(input: torch.Tensor, group: str, width: int, sequence: int) -> torch.Tensor:
    raise RuntimeError(
        "luminal_collective.all_gather has no eager implementation: it stands for a native "
        "collective inside a compiled program"
    )


@all_gather.register_fake
def _all_gather_fake(input: torch.Tensor, group: str, width: int, sequence: int) -> torch.Tensor:
    if width < 1:
        raise ValueError("luminal_collective.all_gather needs a nonempty group")
    return input.new_empty((width, *input.shape))


__all__ = ["OP_NAME", "all_gather"]
