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
runtime.search(&DimensionBounds::default(), &DynMap::default(), &data, &harness_search_options())?;
// The application allocates and initializes Metal storage directly.
let arena = runtime.metal_device()?.new_buffer(
    runtime.arena_bytes()? as u64, metal::MTLResourceOptions::StorageModeShared);
let input = runtime.input_arena_range(x.id)?;
let values = [1f32, 2., 3.];
assert_eq!(input.bytes, std::mem::size_of_val(&values));
unsafe {
    std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(),
        arena.contents().cast::<u8>().add(input.offset), input.bytes);
}
runtime.execute(&arena)?; // synchronous: results can now be read by the application
let output = runtime.output_arena_range(out.id)?;
let values = unsafe { std::slice::from_raw_parts(
    arena.contents().cast::<u8>().add(output.offset).cast::<f32>(), output.bytes / 4) };
let layout = runtime.output_layout(out.id)?;
assert_eq!(luminal_metal::layouts::dense_f32(values, &layout.layout)?, vec![3., 4., 5.]);
# Ok::<(), anyhow::Error>(())
```

Loading, shape binding, saturation and code generation work without a GPU.
Search and execution require macOS with Metal. Search always measures
synchronized device execution, excluding tensor uploads and readbacks, and ranks random
candidates and mutations by their measured times. There is no byte-cost
ranking, byte-cost seed, or profiling toggle.
Profiling owns candidate storage outside the executor. Host stand-ins are
explicitly uploaded before each timed trial; external application storage is borrowed during serving.
`algebra_match_budget` bounds cumulative matches per associativity/distributivity
rule (4096 by default); `None` requests exhaustive algebra saturation. Layout
propagation, substitution, and contract checks still run to completion. This
keeps address-expression optimization bounded on full model graphs.
`load_with_registry` and `metal_registry_filtered` configure the operation set.
External operations implement `KernelOp` and expose `MetalOpInterface` through
their DPS operation's `runtime_interface`.

Pass a complete `DimensionBounds` map and a concrete profiling `DynMap` to
`search`, then use `set_dim` for execution values within that domain. Each
search selects one program under its own `device_budget_bytes`. That limit
includes scratch and boundary storage in the supplied arena, and excludes external storage, host staging,
host payloads and cached pipelines. Profiling values never narrow the domain.

Applications can independently compile multiple domains and dispatch with
[`luminal::bucketing`](../../src/bucketing.rs), with optional shared storage from
[`luminal::memory`](../../src/buffer/memory.rs). `MetalDevice` shares the device and
pipeline cache; each runtime/executable owns its launch state. An application
can bind its own arena and external buffers to share weights and mutable state
across serially executed programs.

Before extraction, `CompileOptions::serialized_graph_passes` can edit the
received serialized e-graph. Metal owns these passes, their context/report
types, and the memory policy in `egraph_postpass`; core does not prune it.
A mandatory memory pass removes materializations larger
than the entire arena limit and all their producer implementations, while
preserving views with smaller backing storage. It sizes tensors over the full
declared bounds and refuses to delete required boundaries. The arena limit is
capped by Metal's maximum buffer length; full candidate allocations must fit too.

The boundary is a binding, not a model annotation. `load` binds every input
read-only on its own buffer and every leaf read-write on its own;
`load_with(graph, bindings, registry)` takes a `MetalBindings` the caller
builds, which declares non-leaf reads, aliased outputs and external storage.
`MetalBindings::input_external` marks application-owned device storage.
Pass its `ExternalBuffer` mapping to `execute_external` each call, along with
a borrowed scratch arena. Binding an output on that input's buffer (`output_on`)
updates application storage in place without a host readback. The application
owns initial uploads and resets. No executor allocation persists between calls.

`input_arena_range` and `output_arena_range` disclose offsets and live byte lengths;
`output_layout` discloses the elected layout without reading tensor contents.
Applications explicitly upload inputs and download results when needed. Tests own
their allocation and transfer helpers in `tests/support`; applications keep their
own helpers. The runtime provides no tensor allocation, upload or readback API. For views, use `layouts::dense_f32` to
interpret the disclosed layout. Supplied storage must satisfy the declared layout. Supported
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

For a shared model-zoo chat runner with application-owned weights and KV state, see
[`llm_chat`](../../examples/llm_chat/README.md).
