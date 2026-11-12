# Reference PyTorch backend

PyTorch **2.14.0 or newer** is required and checked at import time. The package
dependency enforces the same minimum during installation. On Linux/Windows,
the configured CUDA wheel source is CUDA 13.0; the CUDA 12.8 index does not
provide PyTorch 2.14 wheels.

## Backend API

Construct a compiler with its options, then pass that instance to PyTorch.
Reference compilation uses AOTAutograd for inference and training:

```python
import torch
from luminal_reference import Compiler

compiler = Compiler(
    search_iterations=10,
    log=True,
    max_intermediate_bytes=2 * 1024**3,
    memory_budget_bytes=8 * 1024**3,
)
compiled = torch.compile(model, backend=compiler, fullgraph=True, dynamic=True)
with torch.no_grad():
    output = compiled(inputs)
```

`log` defaults to `False`. When enabled, actual searches print `Start`,
`Faster`, and `Slower` to stderr. `Start` appears after the first candidate finishes
profiling, not when compilation begins. Cached graphs do not rerun search or print
search progress. The existing core `SEARCH_LOG`/`LUMINAL_LOG` environment settings
still take precedence when set.

The compiler exposes `graphs`: one record per compiled AOT graph with its phase,
FX source, local input shapes, ATen targets, execution count, and dimension bounds.

AOTAutograd traces each graph once with a fixed decomposition table: the core
ATen decompositions, minus the composites the translator lowers directly (for
example `native_layer_norm`, `silu`, SDPA). The functional graph it produces is
wrapped as a PT2 `ExportedProgram` from its own metadata, with no second
`torch.export` capture; an operator outside both the table and the translator
fails compilation by name.
One compiler instance can compile multiple graphs. `Compiler` is the only public
compilation entry point; pass it to `torch.compile`.

CUDA users import `Compiler` from `luminal_cuda_lite`. It accepts `search_iterations`,
`search_log`, `device_budget_bytes`, and `max_intermediate_bytes`, and compiles
through the same AOTAutograd adapter; `graphs` holds its records.

## Named dimensions and application bucketing

Each local program compiles over its PyTorch-declared bounds. Reuse `ShapeVar`
objects to tie input axes together; the name is only diagnostic. Profiling uses
the tracing or optimization hint, or a small valid size if neither exists.
There is no implicit 4096 limit. `compiler.graphs[i].bounds` reports the local
exported symbol intervals. Data-dependent sizes remain unsupported.

Applications can compile separate domains with isolated Dynamo caches and
choose which callable to invoke:

```python
import torch
from torch.fx.experimental.dynamic_spec import ShapeVar, ShapesSpec, TensorSpec
from luminal_reference import Compiler

def model(x):
    return x.sin()

callables = []
for lo, hi, hint in [(2, 16, 8), (17, 64, 32)]:
    s = ShapeVar("s", min=lo, max=hi, optimization_hint=hint)
    compiled = torch.compile(
        model, backend=Compiler(), fullgraph=True, isolate_recompiles=True,
        dynamic_shapes=ShapesSpec(params={"x": TensorSpec([s])}),
    )
    callables.append((lo, hi, compiled))

def dispatch(x):
    for lo, hi, compiled in callables:
        if lo <= x.shape[0] <= hi:
            return compiled(x)
    raise ValueError("no application bucket covers input")
```

The application owns overlap/gap policy, dispatch, and each compiler's budget.
The backend searches one domain per graph. CUDA obtains intermediate scratch
through PyTorch; its explicit static arena/capture mode remains available for
one program. A GPU program must fit its entire declared domain; a reference
program also checks its live memory budget at execution.

The [Whisper example](examples/whisper.py) uses separate decoder callables for
short and long token sequences and dispatches at the application layer.

## Collectives

Collective operations are not compiled. A forward, backward, or inference
graph that contains a `c10d` functional collective (for example `all_reduce`
or `wait_tensor`) is refused with an error naming the operations. Programs
that communicate must be split by the application so that each
`torch.compile` region is collective-free; PyTorch then executes the
collectives between compiled regions. Unsupported local operations also fail
compilation; there is no eager compute fallback. The same rules apply to the
CUDA backend, which shares this adapter (`luminal_reference.aot_graph`).

Compilation uses synthetic tensors derived from local FakeTensor metadata and
never reads fake storage. Real parameters bind on every invocation.

Each graph is handed to the Rust translator in process, as the JSON document
of a PT2 program. No `ExportedProgram` is built and nothing is written to
disk. PyTorch's own serializer writes the nodes; the inputs, the signature and
the symbol ranges are written by the adapter, because PyTorch's serializer has
no spelling for a symbolic float or bool as a graph input. The whole graph is
written: every placeholder and every output leaf.

The Rust side reads the graph's boundary off that document as slots. PyTorch
calls a compiled graph with one value per placeholder and expects one value per
output leaf, in order. A slot is one of those positions, and its kind is the
type of the value: a tensor, an `int`, a `float`, a `bool`, or any other
object. A call passes one value per slot, and nothing is dropped by type.

- A tensor a node reads is bound to the caller's tensor. A tensor no node
  reads keeps its slot and is never touched: nothing states or checks its
  device, its layout or its storage.
- A symbolic scalar is the expression PyTorch recorded for it. A node that
  reads one reads that expression, and the call's value is not read. This is
  how a bool saved for the backward graph (`x.shape[0] > 3`) is read.
- An integer argument that is a symbol defines that symbol by value, as it
  does for PyTorch; no tensor has to carry it. A negative value is refused by
  the argument's name, because a symbol's value is unsigned in the runtimes
  today.
- A symbolic float has no expression the translator reads yet. A tensor op
  that reads one is refused, naming the value and its expression.
- An output slot is a tensor the program computes, the caller's own object for
  a leaf that is an argument, a scalar evaluated from the call's sizes, or a
  literal such as the `None` of a missing gradient.

PyTorch asserts some conditions inside the graph (`_assert_scalar`). A
declared range arrives this way and is also guarded by PyTorch before the
graph is called. A check the user wrote, such as
`torch._check(x.shape[0] != 5)`, is asserted nowhere else: the compiled
program evaluates it at each call and raises PyTorch's own message. These
checks are separate from binding. Set `CompiledModel.runtime_assertions =
False`, on the class or on one compiled model, to run without them.

### Saving a compiled function

Each compiler returns an `OutputCode`, the object PyTorch's AOT machinery
holds for a compiled graph. That is what `torch.compile(...).aot_compile(...)`
needs to save a compiled function and `torch.compiler.load_compiled_function`
to load it. On the reference backend the saved form of a graph is the JSON
document above, the plan search selected for it, and the constants it holds.
A loaded function translates the graph, installs the plan, and runs: no search
runs after a load. Forward, backward and inference graphs are all saved, and a
dynamic dimension keeps working at new sizes.

The CUDA runtime has no serialized form of a searched plan yet, so
`aot_compile` on the CUDA backend is refused by name.

Both backends keep input mutations in the graph
(AOTAutograd's `keep_inference_input_mutations`), so a mutated input (a KV
cache, a module buffer) is written in place on its own storage by the runtime
rather than returned and copied back by AOTAutograd's epilogue.
The reference binding copies computed outputs into independently owned tensors,
so later forwards do not overwrite saved activations.

Static and input-backed symbolic CPU shapes are supported in forward, backward,
and inference, including input dimensions declared with `ShapesSpec`.
Use `dynamic=True` or `torch._dynamo.mark_dynamic` to request
symbolic dimensions; default PyTorch automatic dynamism is also supported.
Shape-only scalar outputs are evaluated from runtime shape bindings and returned
in the Python type torch expects for the leaf: an `int`, a `float` or a `bool`.
A saved size enters the backward graph as an integer argument, so backward does
not need to retain an activation merely to recover its dimensions.

A symbolic integer can be used wherever a Python integer can: as a count or an
amount (`repeat`, `tile`, `pad`, `roll`, `full`, `topk`) and as a value in
tensor arithmetic. Integer arithmetic on one follows torch: `//` floors and `%` takes
the sign of the divisor. A comparison of symbolic integers (`x.shape[0] > 3`)
is a symbolic bool, which reads as 1 or 0 where a program uses it as a number.
A symbolic count or pad amount must be nonnegative on the whole declared range
of its dimensions; one that may go negative is refused at compile time. A
symbolic `k` for `topk` must likewise stay inside the sorted axis on that
range. A symbolic exponent (`x ** n`) is not translated.

Data-dependent unbacked dimensions and changing tensor rank are outside this
contract. Shape-dependent branches may require separate guarded graphs.
The integration uses private PyTorch AOT/FX APIs; the end-to-end tests are the
compatibility gate (validated on PyTorch 2.14.0).

## Runtime memory limits

The reference runtime executes a depth-first dependency schedule, including all
anti-dependencies. It allocates destinations when their producer runs and releases
intermediates after their last reader. Only outputs survive execution. The existing
ops and kernel bodies are unchanged; the runtime moves intermediate operands into
the kernel context and restores them when later consumers still need them.

Defaults are **2 GiB per intermediate** and **8 GiB of live tensor payloads**.
The live budget includes staged inputs, live intermediates and outputs, temporary
operand copies, and conservative reservations for the kernels' existing scratch
arrays. Budget checks precede allocations and run again for every dynamic shape.
Oversized search candidates are refused before their offending allocation.

Rust callers configure `CompileOptions::max_intermediate_bytes` and
`CompileOptions::memory_budget_bytes`, or the matching `ReferenceRuntime::set_*`
methods. Native Python graphs accept those two keyword arguments to `graph.search`.
The normal Python compile APIs use these defaults; no environment variable is read.
`ReferenceRuntime::peak_live_bytes()` reports the accounted high-water mark,
including reserved scratch (which can exceed actual usage).

These are payload budgets, not an operating-system RSS cap: Python/model copies,
compiler state, allocator overhead, and rank-sized kernel metadata are excluded.

## Validation

From the repository root:

```sh
bash crates/pytorch/reference/run_tests.sh -q -k "aot or dynamic_shapes or symbolic_shapes"
```

Single-process AOT tests exercise two outstanding forwards before backward,
output structure, parameter rebinding, and the collective refusal. Dynamic tests
assert graph reuse across changing sizes, including backward graphs that save
only dimensions.
