# PyTorch backends

This directory contains backend-specific Python packages and what they share.
There is no `luminal` Python package. `utils/` holds the shared parts: a Rust
crate that translates the program a backend hands it, the Python package
`luminal_torch_common` with the `torch.compile` frontend, and the shared
tests. A backend depends on `utils/`, never on another backend.

```python
import torch
from luminal_torch_reference_compiler import Compiler

model = torch.nn.Linear(4, 4).eval()
compiled = torch.compile(model, backend=Compiler(), fullgraph=True)
with torch.inference_mode():
    output = compiled(torch.randn(2, 4))
```

Use `from luminal_torch_cuda_lite_compiler import Compiler` for CUDA. Configure search and
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
- `utils/tests`: everything that holds for every backend, run against the
  selected one. PT2 translator operation, dtype, shape, and
  composite-lowering parity against eager PyTorch, and the backend-neutral
  facts of a compiled graph: its code object (bytes, torch's side of a call,
  running without the compiler), saving and loading, outputs a caller holds,
  and what is refused by name. A new backend's gate is this suite. Python
  cases execute through a native runtime; Rust translator tests are run with
  `cargo test -p luminal_pytorch_utils`.
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

- The Rust crate `luminal_torch_slots` (`slots/`) names what kind of value
  each argument slot holds, and reads a value out of a slot or creates a
  tensor torch owns. It calls torch's stable C entry points, found by name
  in the libtorch that is already loaded, so it needs no build against
  torch. It depends on no compiler code.

- The Rust crate `luminal_torch_code` (`code/`) holds what every backend's
  code object states the same way: the boundary tables (argument slots and
  what a call does with each, output leaves, torch's runtime checks,
  dimension values), reading a code object from its bytes, and the steps of
  a call that read only those tables. A backend's own inputs, rows and
  program stay in its runtime crate. It depends on no compiler code.

Owned by each backend, with no shared base class, in two halves:

- **The compiler.** A native extension built on `luminal_pytorch_utils`
  (`compile_graph`, `boundary_plan`) and the AOTAutograd backend in `aot.py`.
  It parses, translates and searches, and its product is a *code object*:
  plain bytes holding everything a call needs and nothing the compiler
  needed.
- **The runtime.** A package of its own that holds the code object type, the
  call path and the `OutputCode` subclass torch calls, saves and loads. Its
  code object is the shared boundary tables beside its own inputs, rows and
  program. It reads torch's argument list through `luminal_torch_code` and
  `luminal_torch_slots`, reads each tensor its own way, and imports no
  compiler. Its import path is written into every saved program.

The compiler depends on the runtime, never the reverse. The first call in
process and a call after a save and a load run the same code object, built
from the same bytes.

A compiled graph is an ordinary Python object across threads. It holds one
executable, and it can be called on any thread and freed on any thread:
torch calls a backward graph from its autograd thread, for one. Its calls
run one at a time, so nothing a compiled graph holds has to be safe for two
calls at once; a call from another thread waits for the one in progress,
without holding the interpreter. The reference runtime releases the
interpreter while a plan computes, as torch does around its own kernels.

Both backends have this shape:

| Backend | Compiler | Runtime |
| --- | --- | --- |
| reference (CPU) | `luminal_torch_reference_compiler` (`reference/`) | `luminal_torch_reference_runtime` (`reference_runtime/`) |
| CUDA-lite | `luminal_torch_cuda_lite_compiler` (`cuda_lite/`) | `luminal_torch_cuda_lite_runtime` (`cuda_lite_runtime/`) |

What a code object's program is differs by backend. The reference one is
the reference runtime's saved plan. The CUDA one is the searched plan
lowered to what the device runs: a list of kernel launches, copies and
cuBLASLt calls, with every kernel already compiled to PTX. Loading it
parses, translates, searches and compiles nothing. Each runtime's
`describe(code)` returns a code object's tables as plain data.

The CUDA runtime issues a program step by step on torch's current stream;
it captures no CUDA graph and issues nothing on a stream of its own. The
stream is read at the top of every call, as Inductor's compiled code reads
it (`torch._C._cuda_getCurrentRawStream`), so a call made on another thread
or under `torch.cuda.stream(...)` runs where the caller is. It binds every boundary
tensor in place and allocates each output through torch. A call returns
once its program is queued and does not wait for the device, as eager torch
does not: every output's shape is computed on the host from the call's
arguments before anything is issued, so nothing is read back. The scratch
memory a program uses in between is a block from torch's allocator for the
device, taken for the call and released before the call returns, sized to
the largest dimensions the program was compiled for. torch accounts for it
like any tensor's memory and keeps the released block in its cache, where
a later allocation on the same stream reuses it behind the queued program.
An executable holds no device memory between calls.

Two things are refused by name while compiling. A size computed from tensor
data (`nonzero`, `x[mask]`, a size read with `.item()`) reaches a backend
only under `fullgraph=True`, and no compiled program can state one yet
(LUM-1580); the shared compiler refuses it in a temporary pass
(`luminal_pytorch_utils::data_dependent`). On CUDA, a dynamic dimension
with no declared maximum that sizes a tensor the program reads or writes
is refused, because the device's memory is planned for the largest sizes.

`reference/python/luminal_torch_reference_compiler` is the smallest complete example. A
backend passes its extension's `boundary_plan` to
`aot_graph.prepare_local_graph`; nothing in `utils/` imports a backend.
