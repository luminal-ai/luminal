# Production cuBLASLt admission

`approved_rules.egg` proves `cublaslt-approved-call` only after joining a
`cublaslt-approved-configuration` fact for the complete `CublasLtConfigurationOf`
tuple with every logical and layout premise. Two final emission rules consume
that same proof: `LayoutTensorOpCublasLtCall` uses an independent destination;
`LayoutTensorOpCublasLtAccumulateInPlace` reads and overwrites C.
Recognition, layout readings, C and bias cannot emit calls independently.
An empty whitelist emits neither executable form, including
from graphs containing C, bias, or ReLU. Removing one row removes that
configuration and leaves the others available.

Names follow the cuBLASLt API: `D = alpha * op(A) * op(B) + beta * C`,
followed by the selected epilogue. C is an optional full matrix with the output's
dimensions; the epilogue bias is a separate vector. Rule variables keep `?m`,
`?n`, and `?k` for the product dimensions. Shared layout proofs use
`?LeadingDimension` for the API's leading dimension in elements. Matrix-specific
proofs use `?D_LeadingDimension`; the in-place emission rule uses one
`?C_and_D_LeadingDimension` variable to require equal C/D leading dimensions.
Matrix descriptors use names such as `?A_matrix_descriptor`.

`approved_whitelist.egg` contains 48 configurations:

| A / B storage | Logical multiply / reduce | C / bias / D | Epilogue |
| --- | --- | --- | --- |
| F32 / F32 | F32 / F32 | F32 | default or bias |
| BF16 / BF16 | F32 / F32 | F32 | default or bias |

The tuple also specifies compute type, scale type, bias type or absence, host
alpha/beta, A/B/D order, and both transpose flags. Alpha is exactly 1; beta is
0 without C and 1 with C. C has D's order and dimensions but may have a
different leading dimension. The in-place form requires identical C/D descriptor
geometry and the same layout. It carries one shared C/D descriptor and declares
an unconditional `Sharing::Must` tie from C (operand 2) to result 0. Its inputs
are A, B, C and optional bias, with no extra destination operand. The independent
form appends a write-only destination during DPS conversion and grants no input
alias permissions. Neither form has a conditional C/D alias flag.
Bufferization may reuse writable, dead C storage or repair a conflict by copying
the old C contents before executing the in-place form on the copy.
Bias has the vendor's row length and column-ordered D. The rules transpose a
complete expression to implement normal feature bias, preserving its operands
and selected output layout.

Only explicit F32 input casts authorize direct BF16 storage readings. Arithmetic
precision comes from the logical operands and product. A BF16 product followed
by an F32 output cast cannot become an F32-compute call. Casts between multiply,
reduce, C addition, or bias addition stop fusion at that rounding boundary.
Earlier narrowing casts remain in the graph, and final output casts remain
separate operations. C and bias widening is materialized as F32. F32 reduction
reassociation and FMA contraction retain the existing compiler policy.

ReLU has no approved configurations: vendor ReLU normalizes negative zero,
whereas the logical operation preserves it. ReLU executes through the generic
path. F16 arithmetic, TF32, FP8, mixed A/B storage, narrow outputs, batched
calls, arbitrary scalars, and other epilogues have no approved rows.

ElementStride proofs use cons lists and relation facts, with no egglog
containers. They retain exact alternatives independently and grant `Any` only
on a proven singleton axis. A callable matrix needs positive rank-two extents,
a unit minor stride, and a leading dimension at least the minor extent.
Authored boundary strides supply whole shape-owned certificates. Overlapping,
negative, broadcast, and nonunit-minor layouts cannot produce a reading.
Unproved layouts remain available through generic operations/materialization.

Extraction decodes the elected tuple and descriptors. Dispatch evaluates their
integer expressions, checks backing capacities and vendor availability, and
passes them unchanged to cuBLASLt. There is no Rust pattern selector, transpose
bridge, destination rebinding, or legacy emitter. Device heuristics can still
refuse a configuration on a particular GPU or geometry.

Boundary value delivery uses `BufferCopyValue` with an explicit unchanged
address span; both backing capacities must cover it. This permits dense output
in a larger shared allocation while preserving unused tail bytes. A rejected
read-modify-write tie whose C operand is a folded view uses the same value-span
copy to initialize its tied result, without copying the larger parent allocation
into a separate buffer. Ordinary
`BufferCopy {src,dst}` retains exact-size whole-storage semantics.
Reference compiled artifacts use schema version 3 to encode ElementStride
layouts and value-copy spans; older artifacts must be regenerated.

## Regression coverage

The exploratory whitelist and historical marker emitters were removed after
moving their useful checks to the production vocabulary:

| Invariant from the former marker boards | Production coverage |
| --- | --- |
| Complete admission, one-row removal, every configuration witnessed | `cublaslt_production_whitelist` |
| Input/output cast placement and every arithmetic rounding stage | `cublaslt_precision_matches`, `precision_contract` |
| Singleton, square, transposed and duplicate operands; distinct sites; chains; bounded growth | `cublaslt_production_proofs` |
| Whole shape certificates, invalid pitches, correlated symbolic bounds, singleton specialization | `cublaslt_production_proofs`, core `element_stride` unit tests |
| Sliced reduction extent, exact descriptor/layout address agreement | `cublaslt_production_proofs` |
| C/bias ordering, destination descriptors and the distinct accumulating form | `cublaslt_production_proofs`, `cublaslt_bias_premise`, `cublaslt_in_place` |
| Every table row dispatched numerically, in-place execution, signed-zero fallback, shared padded boundaries | `cublaslt_production_device`, `cublaslt_in_place` |
| Dynamic geometry and graph replay | `dynamic_graphs`, `precision_contract` |
| Buffer bounds, singleton reach, overflow, separate C/D pitches | `cublaslt_contracts_cpu`, `cublaslt_contracts` |
| Folded view identity, lifetime repair, transpose termination | retained `test_runtime` fold/bufferizer/transpose suites |
| Value-copy bounds, spare capacity and artifact round trip | reference `value_copy` |

Tests about obsolete constructor child positions, old specification parsing,
transpose bridges, destination rebinding, and diagnostic-only candidate counts
were retired with those implementations. No old emitter remains registered as
a test fixture.

## Accumulation performance comparison

Run the synthetic residual-chain benchmark on an otherwise idle CUDA device:

```sh
LUMINAL_CUDA_HEADERS_DIR=/usr/include cargo bench -p luminal_cuda_lite --features device --bench cublaslt_accumulation
LUMINAL_CUDA_HEADERS_DIR=/usr/include cargo bench -p luminal_cuda_lite --features device --bench cublaslt_accumulation -- --allow-materialization
```

The benchmark compares separate-destination calls, in-place calls, and search
with both forms available. All calls pass the production egglog proofs and
whitelist. Generic add/reduce implementations are excluded so they cannot hide
a missing accumulation match. The first command also excludes layout
materialization, isolating C/D behavior on already-callable native layouts;
the second includes the production copy/materialization candidates. Both keep
bufferizer repair and boundary-delivery copies. These are controlled subsets
of the production registry, rather than full-model benchmarks.

Workloads use four or eight F32 residual additions with a shared dense-storage
weight matrix. Independent branch inputs exercise reuse of C; `self_residual`
uses `x @ W + x`, and `live_intermediates` binds every residual as an output.
Read-only and writable initial C, decode-sized matrices, and larger prefill
matrices are covered. Writable cases permit overwriting caller-owned C;
the final output has its own boundary buffer. Synthetic weights have three
nonzeros per column, enabling an independent sparse arithmetic reference for
the ordinary dense GEMM dispatch.

Each mode uses the default search budget, seeds 0/1/2, five additional warmups,
and 31 timed replays per seed. Search order rotates between seeds; replay order
rotates between modes. Inputs are restored before every warmup and replay.
Execution timing covers the serving path's parameter binding, CUDA graph launch
and synchronization; input transfers and output readback are outside it.
Search time includes assembly, saturation, candidate profiling and finalist
validation after runtime loading. Modes share one device and compiled-module
cache. TSV output preserves every replay sample and per-seed plan statistics.

Copy counts include bufferizer copies, boundary delivery and selected layout
materializations; copied bytes count payload once per transfer. Peak live bytes
and required slab bytes are reported separately by the arena planner, including
boundary tensors and the 32 MiB cuBLASLt workspace. They exclude host memory,
CUDA context/library allocations and the other comparison modes' arenas.
Numerical outputs are checked before and after timing, including all retained
residuals. `--quick` uses one seed and seven replays; `--case <name>` selects a
single workload.

Measured on 2026-10-26 with an A100-SXM4-40GB, driver 570.195.03 and Rust
1.97.1, using the default 8 generations, 8 candidates per generation and 3
profiling trials. Each table cell lists **separate / in-place / both**;
values are medians across the three seeds, with replay latency first reduced
to its per-seed median. Memory describes the elected plans and can vary with
their layouts and lifetimes; search ranks time rather than minimizing memory.

With native layouts:

| Workload | Replay (us) | Peak live (MiB) | Copies | Search (ms) |
| --- | --- | --- | --- | --- |
| decode_read_only | 52.5 / 56.1 / 52.6 | 36.031 / 36.024 / 36.031 | 0 / 2 / 0 | 880 / 766 / 746 |
| prefill_read_only | 81.1 / 85.6 / 81.1 | 34.750 / 34.500 / 35.000 | 0 / 2 / 0 | 1428 / 1448 / 1353 |
| prefill_donated | 80.9 / 83.6 / 81.2 | 35.000 / 34.500 / 34.750 | 0 / 1 / 0 | 1363 / 1402 / 1133 |
| deep_donated | 149.4 / 156.5 / 149.4 | 35.750 / 35.500 / 36.000 | 0 / 3 / 0 | 2386 / 2479 / 1871 |
| self_residual | 81.2 / 90.6 / 81.3 | 34.000 / 33.750 / 34.000 | 0 / 4 / 0 | 1237 / 1353 / 1172 |
| live_intermediates | 83.5 / 94.9 / 83.7 | 35.500 / 35.750 / 35.500 | 1 / 6 / 1 | 1655 / 1761 / 1354 |
| large_donated | 151.0 / 153.6 / 151.4 | 41.000 / 39.000 / 41.000 | 0 / 1 / 0 | 1539 / 1568 / 1435 |

With layout materialization candidates available:

| Workload | Replay (us) | Peak live (MiB) | Copies | Search (ms) |
| --- | --- | --- | --- | --- |
| decode_read_only | 42.0 / 46.9 / 45.6 | 40.031 / 40.024 / 36.031 | 1 / 3 / 1 | 655 / 636 / 628 |
| prefill_read_only | 81.5 / 90.7 / 81.2 | 35.000 / 35.750 / 35.000 | 0 / 4 / 0 | 1183 / 1132 / 1041 |
| prefill_donated | 81.5 / 89.9 / 81.4 | 35.000 / 35.750 / 34.750 | 0 / 3 / 0 | 1169 / 1131 / 1032 |
| deep_donated | 150.1 / 159.8 / 151.6 | 37.000 / 36.750 / 37.000 | 3 / 5 / 3 | 1351 / 1142 / 1248 |
| self_residual | 85.5 / 94.4 / 86.6 | 35.000 / 34.750 / 34.000 | 1 / 5 / 3 | 897 / 835 / 661 |
| live_intermediates | 83.6 / 100.5 / 83.4 | 35.500 / 36.750 / 35.500 | 1 / 9 / 1 | 1159 / 985 / 1078 |
| large_donated | 151.1 / 156.1 / 151.9 | 41.000 / 40.000 / 40.000 | 0 / 2 / 0 | 1354 / 1291 / 1300 |

The native-layout experiment shows the cost of forcing C/D reuse: writable
prefill C saves 0.5 MiB of peak live storage but adds one 0.25 MiB delivery
copy. `x @ W + x` requires four repairs (1 MiB total) and is about 12% slower;
retaining intermediate residuals increases the copy cost further. The large
case saves 2 MiB in each seed at roughly 2% replay cost. Keeping both forms
usually avoids these extra copies and provides additional memory choices.

Use paired per-seed changes for relative comparisons, since clock and layout
differences affect absolute medians. With native layouts, the median paired
latency change for both forms versus separate-only lies between -0.33% and
+0.53% across workloads. With materialization admitted it lies between +0.54%
and +3.62%; individual seeds range from -6.81% to +8.76%. These measurements
support retaining both op forms, with no universal speedup claim. A larger
candidate vocabulary still needs reliable search coverage: the default budget
can miss a faster separate-destination plan, and fewer copies alone do not
establish a faster GEMM layout. Mixed search's median wall time was below the
separate-only median in every workload here; this is a bounded experiment,
not a search-scaling guarantee.

All 126 selected programs passed independent reference checks before and after
3,906 timed replays, with maximum absolute error zero. Search profiled 4,974
candidates, deduplicated within each search, with zero extraction, plan-build
or execution refusals and zero finalist rejections. The benchmark does not pin
GPU clocks, and the synthetic F32 graphs do not establish model-level or BF16
performance.
