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
let out = (x + 2.).output();
let mut runtime = MetalRuntime::load(&graph)?;
runtime.search(&Default::default(), &harness_search_options())?;
runtime.set_data(x.id, vec![1f32, 2., 3.]);
runtime.execute()?;
assert_eq!(runtime.get_f32(out.id)?, vec![3., 4., 5.]);
# Ok::<(), anyhow::Error>(())
```

Planning works without a GPU. Execution and measured search require macOS with
Metal. `CompileOptions::profile_on_device` measures synchronized execution,
including staging and readback; the default ranks by estimated byte movement.
Search starts with a byte-cost baseline and explores mutations and random plans;
traffic is counted once per operation in the executable DAG.
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
