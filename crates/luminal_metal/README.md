# Metal backend

`luminal_metal` uses the native logical graph, layout extraction, DPS conversion,
and bufferization path, following the same runtime boundary as CUDA Lite. Its
operation structs, egglog matchers, bindings, search, and MSL kernels live in this
crate. It does not depend on CUDA Lite or the reference runtime for execution.

```rust
use luminal::prelude::*;
use luminal_metal::{MetalRuntime, harness_search_options};

let mut graph = Graph::new();
let x = graph.tensor(3, DType::F32);
let out = x + 2.;
let mut runtime = MetalRuntime::load(&graph)?;
let data = [(x.id, vec![1f32, 2., 3.].into())].into_iter().collect();
runtime.search(&data, &harness_search_options())?;
runtime.set_data(x.id, vec![1f32, 2., 3.]);
runtime.execute()?;
assert_eq!(runtime.get_f32(out.id)?, vec![3., 4., 5.]);
# Ok::<(), anyhow::Error>(())
```

Loading, shape binding, saturation and code generation work without a GPU.
Search and execution require macOS with Metal. Search always measures
synchronized execution, including staging and readback, and ranks random
candidates and mutations by their measured times. There is no byte-cost
ranking, byte-cost seed, or profiling toggle.
Profiling and finalist validation honor resident input bindings. Warmup uploads
residents once per candidate. Timed trials stage only transient inputs; writable
resident state resets from the supplied payloads before each trial, outside its
timer and timeout budget. Resident mutation outputs stay on the device.
`algebra_match_budget` bounds cumulative matches per associativity/distributivity
rule (4096 by default); `None` requests exhaustive algebra saturation. Layout
propagation, substitution, and contract checks still run to completion. This
keeps address-expression optimization bounded on full model graphs.
`load_with_registry` and `metal_registry_filtered` configure the operation set.
External operations implement `KernelOp` and expose `MetalOpInterface` through
their DPS operation's `runtime_interface`.

Use `bind_dyn_range` or `bind_dim_buckets` before search, then `set_dim` before
execution. Buckets share one arena sized for the largest selected plan;
`device_budget_bytes` bounds this arena. The limit does not include shared host
staging, host payloads, or cached pipelines. Commands follow the buffer plan's
data and anti-dependencies, and preserve outputs before recycling their storage.

Before extraction, `CompileOptions::serialized_graph_passes` can edit the
received serialized e-graph. Metal owns these passes, their context/report
types, and the memory policy in `egraph_postpass`; core does not prune it.
A mandatory memory pass removes materializations larger
than the entire arena limit and all their producer implementations, while
preserving views with smaller backing storage. It sizes tensors over the full
bucket bounds and refuses to delete required boundaries. The arena limit is
capped by Metal's maximum buffer length; full candidate allocations must fit too.

The boundary is a binding, not a model annotation. `load` binds every input
read-only on its own buffer and every leaf read-write on its own;
`load_with(graph, bindings, registry)` takes a `MetalBindings` the caller
builds, which is how a non-leaf read, an aliased output and device residency
are stated. `MetalBindings::resident(value)` keeps a static input in the arena
after its initial upload; a later `set_data` explicitly updates it. An output
bound on an input's buffer (`output_on`) shares one `BufferLit`, so the
mutation writes the resident range in place; it is not copied to host and is
not available through `fetch`. Dimension values may change between executions,
but bounds, buckets and re-searches are refused once the arena is installed.
Resident ranges count toward the arena budget and are shared across every
bucket. Physical lifetimes and resident allocation use core's planner, also
used by CUDA Lite.

`fetch` returns an owned backing payload and its elected layout. For views, use
`layouts::dense_f32` to interpret that layout; `get_f32` returns backing elements.
Inputs must have the declared dtype and exact live byte length. Supported
storage types are F32, F16, Int, Int64, and byte booleans. Other dtypes fail
explicitly. Arithmetic obeys the core's integer proof gates.

The primitive registry covers arithmetic, comparisons, casts, reductions,
gather/scatter, iota, and materialized or folded index maps. A fused F32
multiply/reduce matcher avoids broadcast-product temporaries in matrix products;
matching stays in egglog. This port uses MSL kernels, without MPS-specific
matmul or the retired HLIR fusion passes. Pipelines are cached; command buffers
are encoded for each execution.

Run `cargo test -p luminal_metal` on a Metal Mac and
`cargo clippy -p luminal_metal --all-targets -- -D warnings` for validation.
The mini model smoke tests cover execution; numerical tests compare the runtime
to independent scalar results or `ReferenceRuntime`. Mini Flux retains the
core suite's documented adaLN rejoin-divergence search blocker.

The `llama_1b` example retains its checkpoint, model, and prompt and uses the
native runtime API. It stages KV cache updates through host readback. Run it
with `cargo run --release -p luminal_metal --example llama_1b`.

For a shared model-zoo chat runner with resident weights and KV state, see
[`llm_chat`](../../examples/llm_chat/README.md).
