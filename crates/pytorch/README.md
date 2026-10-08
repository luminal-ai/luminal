# PyTorch backends

This directory contains backend-specific Python packages and shared validation.
There is no `luminal` Python package. `utils/` is a Rust crate used to translate
PT2 exports; Python does not import it.

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

For AOTAutograd and DTensor SPMD, see [reference/README.md](reference/README.md)
and the inference-only [SPMD example](reference/examples/spmd.py).

## Development

Each backend owns its Python environment and `pyproject.toml`; there is no
project or test suite at the `crates/pytorch` root. From this directory:

```sh
uv sync --project reference --group dev
uv run --project reference --group dev maturin develop --manifest-path reference/Cargo.toml
uv run --project reference --group dev pytest -c reference/pyproject.toml reference/tests/test_aot.py
uv run --project reference torchrun --standalone --nproc-per-node=2 reference/examples/spmd.py
```

Test ownership:

- `reference/tests`: CPU runtime, host-buffer boundaries, export/region APIs,
  AOTAutograd, and SPMD.
- `cuda_lite/tests`: CUDA runtime, layouts, streams, capture, and CUDA-specific
  model regressions.
- `utils/tests`: PT2 translator operation, dtype, shape, and composite-lowering
  parity against eager PyTorch. Python cases execute through a native runtime;
  Rust translator tests are run with `cargo test -p luminal_pytorch_utils`.
- `reference/tests/models`: shared model parity cases. The reference runner
  executes them on CPU; the CUDA runner explicitly includes them on CUDA.

`./reference/run_tests.sh` builds reference and runs its tests plus translator
parity. `./cuda_lite/run_tests.sh` builds both extensions and runs CUDA tests,
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

## Internal region artifacts

Region export, compilation, serialization, and artifact caching are internal
implementation helpers tested by the backend suites. Users compile models with
`torch.compile(model, backend=Compiler(...))`, which preserves PyTorch output
structures.
