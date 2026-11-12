"""The torch.compile frontend the Luminal PyTorch backends share.

A backend is a ``torch.compile`` backend built on AOTAutograd. This package
holds the part that is the same for every one of them and knows nothing
about any runtime:

- ``aot_graph``: lowers one AOTAutograd graph, in process, to the program
  the Rust translator reads, and describes its boundary as slots (one per
  placeholder and per output leaf).
- ``graph_prep``: the decomposition table AOTAutograd applies while it
  traces, and the small graph cleanups applied before lowering.
- ``dimensions``: symbol ranges and profiling values.

A backend supplies its own native extension, its own compiled-model class,
and its own ``OutputCode`` (how a compiled graph is called, saved and
loaded). It passes its extension's ``boundary_plan`` function to
``aot_graph.prepare_local_graph``; nothing here imports a backend.
"""

from . import _torch_version as _torch_version
