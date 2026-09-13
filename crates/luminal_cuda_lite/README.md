## luminal_cuda_lite

This crate contains the CUDA backend for Luminal.

The backend can be broken down into several main types of ops. Starting from the highest level and going lower:

#### Host Ops

Host ops are opaque operations executed from the host (can execute on device, simply launched in an opaque manner). cuBLAS is a good example of this type of op. Luminal can't assume much about these operations since they are so opaque. These ops implement the `HostOp` trait.

#### Kernel Ops

Kernel ops are operations encoded as a kernel and launch parameters. Luminal can put these into CUDA graphs. Cutlass kernels are good examples of these. These ops implement the `KernelOp` trait.

#### Block Ops

Block ops are operations encoded on the threadblock level, which implement an operation that runs for a duration within a single threadblock. These are required to use a fixed number of threads per threadblock (or gate unused threads out), and are given a fixed-size shared memory scratchpad. Luminal can fuse these operations into megakernels. These ops impelement the `BlockOp` trait.

#### Warp Ops

Warp ops are not yet merged. Stay tuned!

#### Thread Ops

Thread ops are not yet merged. Stay tuned!

### Architecture

`luminal_cuda_lite` can model a joint search space that smoothly searches through various mixed configurations of these ops. At compile time, a waterfall process takes place to iteratively raise each op to the level above, resulting in all host-level ops in the final runtime graph. For instance, block ops get combined into megakernels, implemented as kernel ops. Kernel ops get combined into cuda graphs, implemented as host ops.

### Semantic search contract

Backend rewrites add legal implementations with `union`; they do not remove a
legal implementation merely because another implementation is usually faster.
The profiling search, rather than cleanup, chooses between alternatives such as
generic kernels, specialized kernels, and host-library calls.

That includes GenericMatmul/cuBLASLt/GEMV, direct/decomposed Conv2D,
materialized/absorbed fusion and casts, copying/no-copy scatter, and
materialized/fused RoPE-scatter paths. These alternatives are matched in
egglog; selected LLIR is not rewritten into a different operator pattern after
extraction.

Cleanup may remove only representations that are not executable plans: cycles,
malformed shape/stride metadata, unsupported type/layout combinations, and
proven alias or ownership violations. Candidate resource checks may reject a
plan that cannot fit or launch on the target device. The intermediate-memory
cap applies to the peak planned bucket arena. Bucket dispatch drops the active
arena before allocating another, so bucket arenas peak rather than coexist. The
device-memory check separately includes that peak arena, persistent host-op state
retained by all compiled buckets, the peak transient host-op allocation, and
deduplicated shared workspaces. That check is a necessary planned-capacity bound,
not an available-memory guarantee: external allocations, CUDA context and
allocator overhead, and pool reservations are not observable in the plan. Arena
growth likewise drops the synchronized old arena before allocating its
replacement, so replacement itself does not introduce an old-plus-new peak. The
intermediate-memory and synchronous-NVRTC source budgets are reported as resource
rejections and can be adjusted independently of rewrite semantics. Otherwise, a
plan that is legal but merely expensive remains available for measured search.

Choice-set validation detects correlated e-class cycles before LLIR loading.
Random initial genomes repair only those reachable cycles; later mutations may
still produce them, in which case candidate filtering discards them without
profiling and continues searching the remaining legal alternatives.

### Representative profiling workloads

Search can rank equivalent programs on explicit graph-input examples instead of
its legacy runtime-bound synthetic inputs. This API is shared by Lite and the
full CUDA runtime and has no model-specific types or scheduling rules.

Build the search space first, then install a workload before search:

```rust,ignore
use luminal_cuda_lite::runtime::{ProfileInputs, ProfileWorkload};

let workload = ProfileWorkload::new()
    .shared_input(coefficients)
    .case("sample-a", dims_a, ProfileInputs::new().input(input, values_a), 0.6)
    .case("sample-b", dims_b, ProfileInputs::new().input(input, values_b), 0.4);
runtime.set_profile_workload(&graph, workload)?;
runtime = graph.search_with_rng(runtime, search_options, &mut rng);
```

Each case supplies exact dimensions and physical input bytes. Declare every
graph input in each case or explicitly share it read-only. `.mirrored_input`
keeps host metadata coherent with device data. `.view` binds byte ranges from a
shared `ProfileStorage` when inputs alias. Graph strides and views remain part
of the graph. Contents must satisfy the application's index/layout contracts;
byte-length and dtype-schema checks cannot establish arbitrary semantic validity.

`runtime.capture_profile_inputs(&[tensor_a, tensor_b], &dims)` produces the same
`ProfileInputs` from current bindings, preserving overlapping storage and existing
host mirrors. Capture before execution, after installing inputs and state.
Snapshots are host-backed; search allocates one reusable set of device buffers
sized for the cases. Shared read-only inputs are not duplicated. Original caller
bindings, contents of replayed inputs, and output registrations are protected
from profiling and restored on completion or unwind.

Every case is restored before preparation and again before timed execution, so
warmup, candidate ordering and in-place updates cannot evolve the next trial's
inputs. Kernel alias effects and `HostOp::profile_mutated_inputs` prevent writes
to declared shared inputs. A destructive alternative is also rejected if its
cross-input aliases invalidate the graph's exclusive-use proof. Custom HostOps
with semantic state outside tensors implement `capture_profile_state`; its
`ProfileState` restores that state before trials and after evaluation, including
failure. The default hook declares no hidden semantic state. Operations that
cannot be replayed must return an error from that hook. RNG state can be supplied
as an ordinary tensor or included in an opaque-state snapshot.

Positive case weights describe relative invocation frequency over the entire
workload. Search evaluates all assigned cases in both exploration and CUDA-graph
finalist reranking. It sums globally weighted latency contributions when choosing
a resource-compatible bucket set; per-case measurements are available through
`runtime.profile_evaluations()`. Incomplete workloads that exceed their execution
budget are rejected, rather than scored using only the measured subset. A
single-case early-stop threshold is not applied to a multi-case aggregate.

The default metric uses wall-clock timing, including ordinary runtime
preparation, parameter updates, dispatch and completion. Select
`.timing_method(luminal::op::TimingMethod::DeviceTimestamp)` for device timestamps. Both modes exclude
sample restoration and untimed warmup. These measure steady repeated invocations,
not application queueing, cold compilation or a state-evolving multi-step trace.
No artificial dimension changes are imposed on explicit workload trials.

Each case gets one warmup and `CompileOptions::trials` measured executions
(default three), averaged for its score, subject to execution timeouts. Set
`LUMINAL_CUDA_PROFILE_EXEC=1` to print coarse `SEARCH_PHASE`, `SEARCH_EXEC`,
and `SEARCH_PROFILE_BEGIN/END` records. These separate search/extraction,
compilation and installation, direct preparation or CUDA graph materialization,
state restoration, and launch through completion. The profile records include
actual trial counts and metric sums for warmup versus measured executions.
Metric sums overlap execution wall time; do not add them to wall phases.

Samples do not change semantic bucket ranges or prove content-dependent rewrite
legality. Missing bucket coverage, duplicate IDs/bindings, invalid weights,
inconsistent byte lengths, stale search spaces and conflicting `profile_dims`
overrides are rejected. Static graphs use empty dimension maps. If rebuilding
the search space, reinstall the workload against the new space.

Run the complete non-LLM example with:

```sh
cargo run --release -p luminal_cuda_lite --example representative_profiles
```

The example captures two matrix workloads, searches on them, and checks an unseen
valid shape. This initial implementation provides in-memory workloads and CUDA
replay; on-disk corpus serialization, application-specific collectors, multi-step
traces and replay for other device backends remain separate extensions.

For large stateful workloads, `.device_snapshots(true)` caches immutable sample
backing on the GPU and resets trials with device-to-device copies. Shared
`ProfileStorage` objects are uploaded only once across cases. This trades extra
VRAM for lower replay overhead; allocation failure leaves caller bindings intact.
The default continues to replay host snapshots without retaining device copies.
