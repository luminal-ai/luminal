# PyTorch backends

This directory contains backend-specific Python packages and what they share.
There is no `luminal` Python package. `utils/` holds the shared parts: a Rust
crate that translates the program a backend hands it, the Python package
`luminal_torch_common` with the `torch.compile` frontend, and the shared
tests. A backend depends on `utils/`, never on another backend.

```python
import torch
from luminal_reference import Compiler

model = torch.nn.Linear(4, 4).eval()
compiled = torch.compile(model, backend=Compiler(), fullgraph=True)
with torch.inference_mode():
    output = compiled(torch.randn(2, 4))
```

Use `from luminal_cuda_lite import Compiler` for CUDA. Configure search and
memory options on the compiler, for example `Compiler(search_log=True)`.
`Compiler` is the only public compilation entry point.
PyTorch 2.14.0 or newer is required; package imports enforce the minimum before
loading native extensions.

For AOTAutograd details, see [reference/README.md](reference/README.md).

## Development

Each backend owns its Python environment and `pyproject.toml`; there is no
project or test suite at the `crates/pytorch` root. From this directory:

```sh
uv sync --project reference --group dev
uv run --project reference --group dev maturin develop --manifest-path reference/Cargo.toml
uv run --project reference --group dev pytest -c reference/pyproject.toml reference/tests/test_aot.py
```

Test ownership:

- `reference/tests`: CPU runtime, host-buffer boundaries, and AOTAutograd.
- `cuda_lite/tests`: CUDA runtime, layouts, streams, capture, and CUDA-specific
  model regressions.
- `utils/tests`: PT2 translator operation, dtype, shape, and composite-lowering
  parity against eager PyTorch. Python cases execute through a native runtime;
  Rust translator tests are run with `cargo test -p luminal_pytorch_utils`.
- `reference/tests/models`: shared model parity cases. The reference runner
  executes them on CPU; the CUDA runner explicitly includes them on CUDA.

`./reference/run_tests.sh` builds reference and runs its tests plus translator
parity. `./cuda_lite/run_tests.sh` builds CUDA and runs CUDA tests,
translator parity, and shared model cases. Extra pytest arguments are forwarded,
for example `-m 'not slow'`.

Shared tests use `LUMINAL_TEST_BACKEND=reference` or `cuda_lite`, set explicitly
by the runner. Tensor device never silently selects a different backend. To run
one translator test directly:

```sh
LUMINAL_TEST_BACKEND=reference uv run --project reference --group dev pytest -c reference/pyproject.toml utils/tests/test_hlir_ops.py::test_add
```

The migrated corpus still exposes unsupported native-runtime operations;
collection does not imply all cases pass. `reference/modal_pytest_runner.py` runs CPU coverage on Modal;
`cuda_lite/modal_pytest_runner.py` runs CUDA coverage and requires `--gpu`.
Both use their backend environment and share setup/profiling code in
`ci/modal_pytorch_tests.py`. Explicit pytest paths are relative to
`crates/pytorch`. Each backend owns its scripts in `examples/`. The Whisper
examples select their backend explicitly and download the sample WAV on first
run into an ignored `examples/assets/` directory.

```sh
uv run --project reference --group dev modal run reference/modal_pytest_runner.py
uv run --project cuda_lite --group dev modal run cuda_lite/modal_pytest_runner.py --gpu A100
```

Python environments and caches are generated, ignored files. Only backend
`.venv` directories are used; none is needed at the PyTorch root.

## Reduction and scan dtypes

The PT2 translator owns PyTorch's implicit dtype conversions. For `sum`,
`prod`, `cumsum`, and `cumprod`, an omitted or `None` `dtype` promotes Boolean
and integer inputs to `torch.int64`; floating inputs keep their dtype. An
explicit `dtype` overrides that rule and converts each input element **before**
accumulation, including rank-zero inputs. PT2 output metadata records the
resolved dtype even when the exported ATen node contains no cast or dtype
argument; the translator uses that metadata to insert the conversion.

`sum`/`prod` retain the existing F32 opmath path for FP16/BF16, followed by one
conversion to the declared output dtype. With explicit Boolean output they
convert each element to Boolean (including NaN to true), reduce its 0/1
indicator, and project the result. `cumsum`/`cumprod` retain the existing scan at the resolved
dtype, without adding generic opmath widening. These rules do not apply to
index results such as `argmax`: its Int64 output does not change its input
dtype. Luminal's core operators and strict declared-dtype checks are unchanged.

Translation coverage does not imply every runtime has the corresponding
kernel. In particular, the reference runtime currently lacks Int64 reductions
and scans and FP16/BF16 storage; Int32 sums require value-bound proofs that the
PT2 test harness does not currently supply. Their dtype contracts can still be
checked during translation and saturation. PyTorch CPU also rejects non-scalar
Boolean scans with explicit `dtype=torch.bool`.

## Adding a backend

Every backend enters through the same point: it is a `torch.compile` backend
built on AOTAutograd, and it compiles one functional ATen graph per phase.
The frontend is shared and the rest is the backend's own.

Shared, in `utils/`:

- `luminal_torch_common.aot_graph` lowers a graph in process to the program
  the translator reads, and states its boundary as slots: one per placeholder
  and one per output leaf, as PyTorch calls it.
- `luminal_torch_common.graph_prep` holds the decomposition table AOTAutograd
  applies while it traces.
- The Rust crate `luminal_pytorch_utils` translates that program to a Luminal
  graph and computes the boundary plan.

Owned by each backend, with no shared base class:

- A native extension that exposes `compile_graph` and `boundary_plan`, both
  built on `luminal_pytorch_utils`.
- `backend.py`: the compiled-model class, which binds a call's arguments to
  the runtime and returns the rows the program wrote.
- `aot.py`: the AOTAutograd backend and its `OutputCode`, which says how a
  compiled graph is called, saved and loaded. Saving is the backend's own
  format.

`reference/python/luminal_reference` is the smallest complete example. A
backend passes its extension's `boundary_plan` to
`aot_graph.prepare_local_graph`; nothing in `utils/` imports a backend.
