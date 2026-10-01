# Application-level bucketing

This change makes one bounded program the unit of core/backend compilation.
Bucketing is optional application composition. Every search has its own memory
budget; applications may choose equal or different budgets. PyTorch bounds come
from ShapesSpec/exported programs, with no backend override. The old bucket APIs
are removed rather than retained as compatibility wrappers.

## Ownership and dependencies

```text
Rust application                     PyTorch application
  BucketSpec / BucketSet               ShapesSpec + isolated torch.compile calls
  optional SharedArenaPlan             application dispatch
          |                                      |
          +---------- one compilation -----------+
                                 |
            graph + complete bounds + profile + options
                                 |
                 egglog -> selection -> DPS -> bufferization
                                 |
                       one selected program
                                 |
                  independent execution binding
```

The core crate exposes `luminal::bucketing` and `luminal::memory` as optional
application composition utilities. Search and runtime compilation do not use
these modules: they still handle one bounded program. The utilities have no
backend dependencies and keep their unit tests at the bottom of each module.
There is no separate application crate or integration-test directory for them.

## Concrete API changes

### Core symbol bounds

`src/shape/bounds.rs` introduces:

- `DimensionRange::new(min, max)` and `exact(value)`: checked inclusive intervals
  with `min <= max <= i64::MAX`. Accessors, membership, and checked intersection
  do not carry any representative or bucket identity.
- `SymbolBounds::new`, `from_ranges`, and `exact`: deterministic complete
  symbol maps. Duplicate symbols are rejected. `project` selects a subset;
  `validate_symbols` rejects missing or unused bounds; `validate_values` checks
  all required concrete assignments while allowing unrelated application values.
- `program_dimensions`: collects `IntVar` references from the parsed bound
  program before saturation, including metadata, indexing, strides, and contracts.
- `egglog_seeds`: ordinary lower/upper function sets, with escaped symbol names.

Symbol names occupy a flat namespace. Applications own name qualification and
reuse the same symbol across inputs when their dimensions should be shared.
Shapes may contain compound `IntExpr` values, such as `prefix + query`. Bounds
are supplied for the free symbols, and interval analysis derives bounds for the
expression. `SymbolBounds` does not add separate expression constraints or
relations between independently supplied symbols.

The former `graph::DimBucket` and its prelude export are gone. `Graph::set_dim`
remains a concrete-value convenience for graph users and tests; it is not an
implicit compiler specialization. Singleton bounds express specialization.

### Runtime compilation

All three runtime facades expose the same call structure, with their own payload
and options types:

```rust
runtime.search(&bounds, &profile_dims, &input_data, &options)?;
```

A search checks the original symbol inventory and profiling assignment, emits
one bounds map, saturates once, selects operations, converts to DPS, bufferizes,
and measures candidates at the supplied profiling values. Failure diagnostics
may re-run saturation to identify a contract error. A successful search does
not enumerate domains or repeat a successful saturation for bucket validation.

The facade retains one selected plan and its complete bounds. `set_dim` supplies
execution values; it does not alter compilation. Execution guards include exact
and metadata-only dimensions even when specialization has eliminated them from
the selected plan. Invalid values are rejected before device work or mutation.
`plan()` and GPU `arena_bytes()` describe this program only.

Reference execution retains dynamic allocation and live-budget enforcement for
broad domains. GPU arena sizing uses the full declared domain, including interval
bounds of compound expressions. A profiling hint never substitutes for capacity.
GPU preparation now rejects supplied profiling storage that is too short instead
of resizing it. The profiling harness explicitly prepares synthetic zeros for
absent benchmark inputs. Executables never fabricate, initialize, upload, or
read back tensor payloads. Profiling uploads are outside timed execution.

The migration keeps the runtime facade and existing `SearchOutcome` statistics
rather than adding another compiler entry point. GPU `best_plan`, `best_genome`,
and `best_nanos` now identify the actually finalized candidate. `finalist_rank`
and `finalist_rejections` explain fallback; `ranked` retains the measured order.
The lower-level GPU search takes a separate `ShapeEnv`, and `CompileOptions.shapes`
is removed as a competing source of shape configuration.

| Removed API/state | Replacement |
| --- | --- |
| `bind_dyn_range`, `bind_dim_buckets` | Complete `SymbolBounds` argument to each search |
| Reference `search_buckets` | Ordinary single-program `search` |
| `search_with_profile_inputs` and profile override lists | One profiling assignment and input map |
| `BucketAssembly`, bucket rendering/search functions | One ordinary assembly with bounds seeds |
| Backend `BucketPlan`, `bucket_plans`, selection helpers | One plan per runtime; application owns collections |
| Backend `lattice.rs`, `select_finalist_set` | Linear per-program finalist validation |
| `SearchOutcome.lattice_rejections` | `finalist_rejections`, plus selected `finalist_rank` |
| Device `install(Vec<...>)` | One `(plan, bounds)` installation |
| Device `execute(bucket_index, ...)` | Execute this installed program |
| Reference `load_plan(plan)` | `load_plan(plan, bounds)` |

Finalist retention and device warmup remain. A failed candidate advances to the
next measured rank; no backend constraint or choice spans several programs.
A joint candidate optimizer would be a separate application feature and is not
part of this change.

### Devices, executions, and memory

`CudaDevice` and `MetalDevice` hold live device resources and code caches.
`device.executable()` creates an installed-program/launch cache, without owning
tensor storage. `Runtime::with_device` attaches a context; the default facade
creates one lazily for search/execution. These are live runtime objects, not
portable serialized programs. Device handles are needed while loading modules
and executing, but need not be part of creating or deserializing a portable plan.
Full program serialization remains a separate feature.

`Runtime::fork` copies the selected plan and metadata, shares code caches, and
starts with fresh launch state. It does not copy application storage.

CUDA execution takes `CudaArena` (an exclusive borrowed device span) and
`&mut CudaStaging` (caller-owned pinned host storage for dimension parameters only). `arena_bytes()` and
`staging_bytes()` report the requirements over the full declared domain.
Applications allocate tensor storage with native device APIs and pass a borrowed
span using unsafe `CudaArena::from_raw(ptr, bytes)`. `cuda_stream()` exposes the
execution dependency for cudarc callers. `allocate_staging()` continues returning
caller-owned dimension parameter storage; its behavior is unchanged.
`CudaAllocation`, `MetalAllocation`, and runtime `allocate_arena()` helpers are
removed. Tensor allocation and transfer helpers live in tests and applications,
with independent private helpers inside each profiler.
The raw caller must supply memory on the execution device; pointer provenance
is not inferred from the address. Capacity and base alignment are checked.
CUDA graph caches are rebuilt when the slab address or host-staging allocation
identity changes. Recycled addresses do not disguise replacement host storage.
Asynchronous callers must retain all storage and streams through GPU completion
and must not reuse the storage while work is in flight. Outer CUDA graph captures
require those bindings to remain valid for replay, or must be recaptured.

Metal takes `&metal::Buffer` each call. `execute_external` also borrows a map of
`ExternalBuffer { buffer, offset, bytes }`; the runtime retains neither. Device,
capacity, and range checks happen before commands execute. `metal_device()` and
`metal_queue()` expose the native dependencies to callers. Dimension parameter
staging is allocated for each synchronous execution and released after completion.

GPU `set_data`, `fetch`, and typed `get_*` APIs are removed. `Placement::Arena`
means device storage at a planned offset; it never implies a host transfer.
`memory_plan`, `input_arena_range`, `output_arena_range`, and `output_layout`
expose storage/layout metadata without accessing tensor contents. Applications
perform explicit uploads before execution and explicit readbacks after completion.
Tests implement these steps in their own support utilities. Chat applications
upload step inputs, execute, then read logits for CPU sampling.

The physical schedule contains device operations only. Caller boundary storage
is present at entry and escaping outputs remain valid until return. An early
output can no longer be overwritten on the assumption that an automatic readback
has saved it; a late input cannot be uploaded into space used earlier in the call.

The core and CUDA `resident` modules, resident input bindings, one-time upload
state, and `set_arena`/`clear_arena` APIs are deleted. Persistent weights and
mutable state are application-owned external buffers. Each executable preserves
its own boundary layout and access contracts. Profiling owns its temporary
allocations outside the executor and explicitly restores profiling inputs before each timed trial.
Those allocation and upload operations are private to `profile.rs`; search invokes
its candidate measurement/warmup functions rather than preparing device data itself.

### Application utilities

`luminal::bucketing` (`src/bucketing.rs`) provides `BucketSpec` and `BucketSet<T>`.
A spec carries checked complete bounds, an explicit profiling assignment, and an
optional label.
The set verifies equal symbol inventories and pairwise multidimensional overlap:
domains overlap only when every axis intersects. Unsorted domains and gaps are
legal. `select`, `select_mut`, and `select_index` report missing dimensions or
uncovered inputs without compiling anything.

`luminal::arena::plan_slab` is the low-level public planner:

```rust
plan_slab(&dag, &buffers, capacity, alignment, SlabAlgorithm::FirstFit)?
// -> Vec<(Id, usize)> // ID and offset, in request order
```

Each `SlabBuffer { id, bytes, uses }` lists its DAG use nodes. The planner returns
only offsets and preserves all existing parallelism. Two buffers can overlap
only if every use of one strictly precedes every use of the other; independent
branches and operands used at one node remain disjoint. `buffer_lifetimes`
extracts this information directly from `BufferIrGraph` using its node IDs.
Callers must include transfer and external uses that extend a lifetime.
Alignment is one setting for the whole invocation. First fit is deterministic
but not optimal; failure means this algorithm found no placement within the
capacity, not a proof that no placement exists.

CUDA/Metal's existing serial schedule adapter still returns its execution steps;
its optimized interval packing is safe because those runtimes enforce that
order. It does not change the public DAG planner's concurrency contract.

`luminal::memory` provides a thin convenience for serial application dispatch:
`PersistentBinding { resource, buffer, bytes }`, `ProgramMemory`, and
`SharedArenaPlan::build(programs, capacity)`. It overlays maximum scratch capacity
and reserves the maximum declared byte capacity for each application resource.
It delegates placement to the slab planner and checks no tensor shape, dtype,
contiguity, or access equality. Applications supply sharing identities and own
their meaning. It owns no allocations. Concurrent invocations need disjoint
scratch or application synchronization.

The [CUDA chat](../../examples/llm_chat_cuda/src/backend.rs),
[Metal chat](../../examples/llm_chat_metal/src/backend.rs), and
[Metal example](../../crates/luminal_metal/examples/llama_1b.rs) show allocation,
binding, dispatch, and reset. Each application shares a device/code cache across
independently compiled programs, builds the shared arena from their requirements,
uploads persistent resources once, and binds the per-program boundary homes.
The allocation must outlive all programs that use it. CUDA callers must order
work on borrowed streams before updating or releasing shared memory.

## Consumer migration

`llm_chat_cuda` and `llm_chat_metal` build complete decode/prefill domains, including the context interval
in each. It searches them independently on one device, using valid model-specific
profiling inputs. Weights, RoPE tables and KV state use explicit shared resource
identities. The application owns the arena, uploads persistent data once, chooses
a runtime in `step`, and resets shared state once. Its `BucketPlan` diagnostics
are application records. Inspection and benchmark tools use these records.

Metal's `llama_1b` preserves its model, prompt, checkpoint, and cache readback
semantics. Its application explicitly builds the former dimension combinations,
compiles each, shares weights/cache inputs and scratch, and dispatches at each
step. Ordinary examples, test harnesses, and operation tests pass explicit empty,
exact, or ranged dimension maps according to their existing intended domain.
Historical validation snapshots remain historical records, not live API clients.

## PyTorch behavior

The reference compiler removes `DimBucket`, normalization/remapping helpers,
`Compiler(dim_buckets=...)`, and bucket arguments throughout AOT, local export,
native search, and distributed compilation. `RegionRecord.bounds` reports one
local domain. Forward/backward and DTensor regions keep their exported dimension
relationships, including tied and affine axes. Both Rust bridges use the same
exported-bounds conversion; CUDA's implicit `[1,4096]` domain is removed.
CUDA re-export preserves those bounds and shared/derived symbols. Profiling
shapes are evaluated without adding equality guards to the original symbolic
metadata. Dynamic CUDA consumers explicitly declare finite domains for arena
capacity planning.

Applications call `torch.compile(..., isolate_recompiles=True)` for separately
bounded callables and dispatch themselves. The Whisper example follows this
pattern. PyTorch ordinary CUDA calls still supply intermediate scratch through
its caching allocator per call and pass it, plus a wrapper-owned host-staging
object, into Rust. Existing static-arena mode retains a PyTorch byte tensor in
the Python wrapper. That allocation covers the bounded dynamic sizes, with
output allocations retained separately per concrete shape.

External PyTorch CUDA graph capture is not currently functional: the runtime
uses `cuGraphLaunch`, which CUDA rejects inside an active stream capture. This
pre-existing limitation requires a separate graph-composition implementation;
ordinary execution through Luminal's own CUDA graphs works independently of it.

A retained byte tensor is sufficient for a static Torch-owned arena; an outer
Inductor compile is unnecessary. `torch.cuda.MemPool` / `use_mem_pool` route
allocations but do not promise one stable slab. CUDA graph pools retain captured
allocations, while Inductor CUDA graph trees may record different graphs for
different concrete shapes. Nesting `torch.compile` around an opaque custom
backend is not a general arena-ownership API. See [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html#graph-memory-management)
and [CUDA graph trees](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html).

Region artifacts cache compilation but `RegionModel` copies now fork native
execution state. Bound callables share neither runtime input/output state nor
CUDA capture state. Domain bounds remain part of the artifact cache key, so equal
profiling shapes with different domains do not collide.

Distributed reference compilation hashes/transfers the exported program,
profiling specs, and options without bucket policies. The exported program
already carries the domain. The leader searches one program; followers install
its native reference artifact and rebind local boundary slots without search.

## Serialization scope

The existing native reference artifact is schema 2: complete bounds, input/output
slot mappings, and one selected plan. Loading performs no search, extraction, or
profiling. Old multi-plan artifacts fail with a precise schema-version error;
no compatibility importer is added.

Complete executable serialization remains a separate feature. Its intended
foundation is Rust serialization of completed plans and compiled kernels, plus
library-call descriptors for operations such as cuBLASLt. Loading should restore
a ready-to-run program, with external libraries available as required.

The existing Python region wire format still stores an exported program and
recompiles on load. This is an existing limitation, not the target contract and
not an implementation requirement added by bucketing. This refactor does not
create a new CUDA/Metal binary artifact format or broaden recompile-on-load.

## Validation

Coverage belongs at the owning layer:

- Core: checked intervals, escaped symbols, complete dimension inventories,
  multidimensional overlap/gaps, shared resource identities and arena sizing.
  Bucketing and memory tests live inline in their modules.
- Reference: multidimensional bounds, exact and metadata-only guards, domain
  capacity pruning, live budgets, and one-program artifact round trips.
- GPU backends: symbolic execution across a domain, capacity and pruning,
  profiling payloads, linear finalist fallback, application-owned state, output lifetimes,
  graph replay and library-call behavior.
- Application consumers: shared scratch/state/reset through the existing chat
  backend tests.
- PyTorch: isolated callables, declared ranges beyond 4096, tied/affine axes,
  zero/one shapes, forward/backward, distributed artifacts, domain cache identity,
  and independent region bindings.

Core CI runs the inline utility tests as part of `luminal`. CUDA execution
validation requires an NVIDIA GPU; device-feature compilation on another host
is not a substitute for those tests.
