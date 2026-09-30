# Reference PyTorch backend

PyTorch **2.14.0 or newer** is required and checked at import time. The package
dependency enforces the same minimum during installation. On Linux/Windows,
the configured CUDA wheel source is CUDA 13.0; the CUDA 12.8 index does not
provide PyTorch 2.14 wheels.

## Run a minimal SPMD example

From the repository root:

```sh
cd crates/pytorch/reference
uv run --group dev maturin develop
uv run python examples/spmd.py
```

[`examples/spmd.py`](examples/spmd.py) runs one inference-only sharded matrix
multiply on two CPU ranks. It launches both ranks automatically, using a temporary
file for rendezvous and the local loopback interface for Gloo communication
(`lo0` on macOS, `lo` on Linux), without hostname discovery. It checks the result against dense PyTorch and prints
the output and local shard shapes. Local computation executes through
ReferenceRuntime, while PyTorch executes the collectives.

## Backend API

Construct a compiler with its options, then pass that instance to PyTorch.
Reference compilation uses AOTAutograd for inference, training, and DTensor:

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

The compiler exposes `graphs` (phase, FX source, collective targets) and `regions`
(local shapes, ATen targets, execution count).
One compiler instance can compile multiple graphs. `Compiler` is the only public
compilation entry point; pass it to `torch.compile`.

CUDA users import `Compiler` from `luminal_cuda_lite`. It accepts `search_iterations`,
`log`, `device_budget_bytes`, and `max_intermediate_bytes`; it retains the
CUDA backend's existing compilation path.

## Named dimensions and application bucketing

Each local program compiles over its PyTorch-declared bounds. Reuse `ShapeVar`
objects to tie input axes together; the name is only diagnostic. Profiling uses
the tracing or optimization hint, or a small valid size if neither exists.
There is no implicit 4096 limit. `compiler.regions[i].bounds` reports the local
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
The backend searches one domain per region. CUDA obtains intermediate scratch
through PyTorch; its explicit static arena/capture mode remains available for
one program. A GPU program must fit its entire declared domain; a reference
program also checks its live memory budget at execution.

The [Whisper example](examples/whisper.py) uses separate decoder callables for
short and long token sequences and dispatches at the application layer.

## Initial SPMD contract

Use one CPU process per rank, a Gloo process group, and a CPU `DeviceMesh`.
DTensor placements specify the sharding plan. AOTAutograd traces DTensor's
local operations and creates the forward and backward programs. For example,
with a one-dimensional mesh:

```python
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

# All ranks start with the same full tensors. Setup and optimizer stay eager.
x = distribute_tensor(full_x, mesh, [Shard(1)]).requires_grad_()
w = torch.nn.Parameter(distribute_tensor(full_w, mesh, [Shard(0)]))

def model(x, w):
    partial = x @ w
    return partial.redistribute(placements=[Replicate()]).to_local()

compiled = torch.compile(model, backend=compiler, fullgraph=True, dynamic=True)
compiled(x, w).square().sum().backward()
```

The backend partitions the AOT FX graph at communication boundaries. Every
compute region goes through PT2 translation, egglog search, and ReferenceRuntime;
there is no eager compute fallback. PyTorch executes functional collectives and
`wait_tensor` in their original FX order. Communication does not enter Luminal's
algebraic rewrites or reference plans. The process group and PyTorch own the
collective algorithms, synchronization, and transport.

Compilation uses synthetic tensors derived from local FakeTensor metadata,
never reads fake storage, and never executes a collective. Real parameters and
shards bind on every invocation. The adapter preserves AOT output slots,
constants, missing gradients, and forwarded inputs/opaque DTensor metadata.
The reference binding copies computed outputs into independently owned tensors,
so later forwards do not overwrite saved activations.

When a multi-rank Gloo process group is active, each participating rank exports
its local regions for an AOT compilation round. The group's rank zero searches
each distinct region request and sends the selected reference plans to their owning
ranks. Followers translate their local PT2 boundary metadata and install the
received plans without saturation, extraction, search, or profiling. The plans
include the complete dimension bounds and remap boundary slots to the follower's
local graph values. The cache key keeps graph contents, declared bounds, constants,
input specs, and compile options while ignoring process-local node provenance IDs.
The compiler's
`leader_compilations` count is positive only on the rank that actually searches.
Artifact exchange and boundary synchronization use a separate Gloo process group,
so they do not interleave with PyTorch's model collectives.
Participating ranks must enter compilation rounds in the same order; a failed
rank or process group must be restarted.

Static and input-backed symbolic CPU shapes are supported in forward, backward,
and inference, including input dimensions declared with `ShapesSpec`.
Use `dynamic=True` or `torch._dynamo.mark_dynamic` to request
symbolic dimensions; default PyTorch automatic dynamism is also supported.
Shape-only scalar outputs are evaluated from runtime shape bindings. Saved scalar
dimensions enter local PT2 graphs through zero-storage shape carriers, so backward
does not need to retain an activation merely to recover its dimensions.
PyTorch resolves communication shapes and split sizes on every invocation, while
group membership, rank topology, and communication order stay fixed.

Data-dependent unbacked dimensions and changing tensor rank are outside this
contract. Shape-dependent branches may require separate guarded graphs. CUDA,
communication overlap, coalesced collectives, and automatic sharding selection
remain outside this scope. Unsupported local operations fail compilation.
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
bash crates/pytorch/reference/run_tests.sh -q -k "aot or spmd or collectives or dynamic_shapes or symbolic_shapes"
```

The two-rank Gloo test uses real DTensors and checks forward results, input and
weight gradients, and optimizer updates against both eager DTensor and dense
PyTorch, including two reductions interleaved with local computation. It asserts
local shard dimensions and ReferenceRuntime execution in
both AOT phases. Single-process tests also exercise two outstanding forwards
before backward, output structure, and parameter rebinding.

The three-rank collective test covers reductions, repeated
execution, all-gather/reduce-scatter/all-to-all gradients, uneven and empty peer
payloads, subgroups, and a singleton group.

Dynamic tests assert graph reuse across changing sizes, including DTensor
inference, collective gradients, symbolic all-to-all splits, and backward graphs
that save only dimensions. The minimal SPMD example uses the default module backend.
