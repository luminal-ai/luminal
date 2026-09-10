# CUDA Lite execution

CUDA Lite has one execution path: a CUDA graph launch. Serving, candidate warmup,
and profiling use `CudaDevice::execute`. Graph compilation and instantiation are
outside timed trials. An execution stages inputs, launches the graph, waits for
completion, and returns owned host output bytes.

## Compilation and memory

Egglog selects all operations. Bucket searches extract from range-seeded e-graphs,
so implementation guards must hold throughout the interval. Representative
shapes are used for timing, without freezing the installed plan's geometry.

The storage planner evaluates conservative interval capacities for each buffer.
It respects the bufferizer's allocation/free events and data/anti-dependencies.
Each bucket has fixed offsets into one runtime-owned device arena containing:

- The dynamic-dimension parameter block.
- Temporary buffers, with reuse after their last access.
- Private device copies of boundary inputs and escaping outputs.
- Shared HostOp scratch, sized to the largest host operation's requirement.

The arena allocation is the **maximum** requirement across installed buckets.
Pinned host staging is shared across buckets too. Execution is serialized on one
nonblocking stream; buckets cannot run concurrently against these shared ranges.
Returned output data owns its host memory and survives later launches.

Installing a new plan set synchronizes and destroys old executables before any
allocation can move. Search releases candidate graphs, staging, and arena memory
between candidates, while retaining the CUDA context and compiled module cache.

## Dynamic updates

| Change | Work before graph launch |
| --- | --- |
| Input contents | Fill the existing pinned staging ranges. |
| Dimension used only inside kernels | Write the parameter block; no graph node updates. |
| Copy length | Patch dependent copy nodes; disable zero-length copies. |
| Explicit kernel launch geometry | Patch dependent kernel nodes. |
| HostOp capture dimensions | Reuse a cached capture, or prepare and capture that HostOp. |
| Compatible child graph topology | Update the existing executable's child node. |
| Incompatible child graph topology | Rebind a cached parent executable, or instantiate a new one. |
| Bucket switch | Select that bucket's graph using the same arena and staging allocations. |

Dimension-to-node dependencies are built once. Host captures and parent topology
variants have bounded caches (eight each). Parents retain the preparations
referenced by their source and executable nodes, even after capture-cache
eviction. A failed partial update invalidates
the affected compiled plan before another launch can use it.

## Operation interfaces

`KernelOp::codegen` returns CUDA source plus symbolic launch metadata. The kernel
ABI is input pointers, output pointer, then `const long long* params`. Dimension
identifiers returned by `symbolic::variable` are defined as entries in `params`.
Default launches cover the bucket capacity and kernels guard against the live
extent. `KernelSource::launch` can instead specify symbolic grid, block, and
shared-memory geometry through `KernelLaunch`.

`HostOp::prepare` resolves host descriptors and algorithms outside capture.
`PreparedHostOp::record` submits GPU work during capture; its Rust body is not
called on replay. The executor treats the resulting child graph as opaque.
`workspace_bytes` reserves device scratch in the shared arena. `capture_dims`
may narrow invalidation to the dimensions that affect captured work; its default
conservatively depends on all dimensions.

cuBLASLt uses this interface for all DPS forms. Geometry changes rebuild its
public library descriptors and capture a new child graph. The executor never
inspects or edits cuBLASLt's private kernel arguments.

During bucket profiling, dynamic payloads supplied at another size are truncated
or zero-extended to the representative size for timing. Serving requires exact
live payload sizes. Profiling includes staging and readback, matching serving.

`CudaRuntime::graph_stats` exposes counters and arena/staging footprint for
checking reuse. GPU regressions live in `tests/dynamic_graphs.rs` and
`tests/execution_traits.rs`.
