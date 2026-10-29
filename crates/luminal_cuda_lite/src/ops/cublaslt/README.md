# Production cuBLASLt admission

The rules describe complete calls to `D = alpha * op(A) * op(B) + beta * C`,
followed by an optional epilogue. Arithmetic recognition, layout proofs, approved
API configurations, and executable emission are separate stages. Rust decodes
the elected call and dispatches its descriptors unchanged.

## Rule ownership and style

| File | Responsibility |
| --- | --- |
| `approved_constructors.egg` | Flat vendor layout data, storage descriptors, optional arguments, configuration and expression-frame types |
| `approved_rules.egg` | Rank-two expression recognition, layout readings, precision proofs, and complete admission |
| `scalar_rules.egg` | Authored alpha/beta scaling and exact FP32 scalar storage/pointer proofs |
| `call_inputs.egg` | Assemble approved matrix/scalar reads in one canonical order |
| `strided_batch_rules.egg` | One explicit batch axis, batch layout proofs, and complete product/accumulation admission |
| `approved_whitelist.egg` | The 2,176 complete configurations allowed for execution |
| `separate_destination_call.egg` | Emit an independently allocated destination from an approved call |
| `accumulate_in_place_call.egg` | Emit a required C/D storage tie after proving identical descriptions and layouts |

Write rules as named, local proof steps in `:ruleset backend`. Recognition facts
name existing logical expressions; they do not imply permission to execute.
Layout readings prove how one particular storage tensor is addressed. Complete
admission joins these facts with one exact whitelist tuple. Only the final
emission rules construct executable operations. There is no decorator that can
bypass admission, and an empty whitelist emits neither executable form.

Use the vendor's names and units. `?m`, `?n`, and `?k` describe the product.
`?LeadingDimension` and `?BatchStrideInElements` are measured in elements.
Variables such as `?A_matrix_descriptor` identify complete descriptors. Share a
variable only when equality is required; independent C/D pitches have distinct
variables. Keep an entire expression, its operands, and its call frame together.
Never harvest descriptors from unrelated sites with matching dimensions.

## Descriptor structure

`CublasLtMatrixLayoutLit` is a flat value with seven fields: storage dtype,
matrix order, rows, columns, leading dimension, batch count, and batch stride.
`CublasLtMatrixDescriptorOf` pairs a storage `LayoutTensor` with that complete
vendor layout. Logical tensors and storage are not inferred from one another in
Rust. Both the rank-two and batched proofs construct this same descriptor type.
Rank-two descriptors explicitly use batch count 1 and stride 0.

C absence is structural: `CublasLtCMatrixAbsent` and
`CublasLtNoCMatrixDataType` carry no fictitious storage input. Present C has its
own descriptor, dtype, and operand position. Dispatch still supplies the valid
placeholder C descriptor required by the vendor when beta is zero.

Both forms use inputs A, B, optional device alpha/beta, optional bias, then
optional C. Host scalar literals are metadata; device scalars are ordinary
FP32 read operands. The separate-destination form appends its write-only D
operand during DPS conversion. The in-place form requires C and ties its final
read operand to result 0 with unconditional `Sharing::Must`. Its emission rule
requires the same complete vendor layout and underlying layout for C and D.
There is no conditional alias flag or Rust destination rebinding. Scalar values,
including beta zero, never change the selected storage tie.

## Approved configurations and precision

The table contains **2,176 complete configurations**: 1,088 for host scalar
pointers and 1,088 for device scalar pointers. Each pointer mode admits:

- 896 FP32-output rows: ordinary FP32 computation with matching F32, BF16 or
  FP16 A/B storage, plus explicit TF32 computation over F32 storage. Each has
  16 A/B order/transpose combinations, absent/present F32 C, and seven
  destination/epilogue combinations (default in either D order; bias, ReLU,
  ReLU+bias, GELU or GELU+bias with column-major D).
- 192 narrow-output rows: matching BF16 or FP16 A/B storage, FP32 arithmetic,
  and an authored final cast to the matching narrow dtype. Each has 16 A/B
  order/transpose combinations, absent/present matching narrow C, and default
  in either D order or ReLU with column-major D. Native narrow bias and GELU
  epilogues have no approved rows.

Scalars and accumulation remain FP32. Alpha and beta are general FP32 values;
absent C still requires literal beta zero. Storage never chooses arithmetic
precision. Only an authored exact widening permits direct BF16/FP16 readings.
Casts between multiplication,
reduction, C addition, bias addition and activation remain boundaries; earlier
narrowing is preserved. A separate output-storage proof recognizes an authored
final cast of the complete FP32 expression. Pure-view transposition carries that
whole rounded expression into the API frame and back to the caller's output.

F32 reduction reassociation and FMA contraction follow the existing compiler
policy. ReLU accepts a difference in the sign of zero; the FP32-output GELU rows
accept the vendor tanh approximation. These are explicit admission policies,
not claims of bitwise equivalence. FP8 and mixed A/B storage have no approved
configurations.

### General alpha and beta

`scalar_rules.egg` recognizes one FP32 scalar multiplication of the completed
product and one of C before addition: `alpha * (A @ B) + beta * C`, followed by
optional bias/activation and authored output rounding. Scaling is not moved
across casts or distributed over C, bias, activation, or multiple scalar
multiplications. Rank-two and strided-batch calls share these scalar proofs.
Both authored operand orders of a scalar multiplication and the product/C
addition are recognized locally within this contract. Admission does not depend
on global floating-point commutation or reassociate a chain of additions.

The vendor's [dtype table](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatmul)
requires **FP32 alpha/beta** for all currently admitted modes: ordinary FP32
computation over F32/FP16/BF16 storage and explicit TF32 computation over F32
storage. Half matrix storage does not authorize half scalar pointers. An
explicit scalar cast to F32 remains an executable operation when needed.

The configuration records `CublasLtHostScalarPointerMode` or
`CublasLtDeviceScalarPointerMode`; actual values/storage live in separate flat
`CublasLtScalarArguments`. Host arguments require literal FP32 graph constants.
Tensor coefficients require rank-zero FP32 storage and use GPU pointers. If a
literal is paired with a tensor coefficient, the literal can be materialized as
a GPU scalar so both pointers use DEVICE mode. There is no GPU-to-CPU scalar
readback. The prepared CUDA graph reads updated device coefficients on every
replay, including coefficients produced by preceding GPU operations.

Only the vendor's ordinary scalar HOST and DEVICE [pointer modes](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltpointermode-t)
are admitted. Row vectors, per-batch coefficient arrays, and mixed host/device
pointer modes are not inferred from tensor shapes. The selected vendor
algorithm's pointer-mode capability is checked during preparation; unsupported
geometries/modes are refused without substituting another precision contract.

Scaling follows the practical cuBLAS numerical contract, including FMA and
reduction-order differences. A zero coefficient may suppress a matrix read,
so NaNs in a zero-weighted matrix need not propagate as they would through
separate elementwise multiplication. A present C remains structurally present,
including when beta is zero; this does not grant extra alias permissions or
remove the scalar's graph dependency. Tests compare finite numerical inputs
and retain explicit cast boundaries.

### Narrow C, bias and output storage

`cublaslt-f32-storage-reading` proves identity F32 storage or exact BF16/FP16
widening. Ordinary A/B readings, C and bias can use that proof. TF32's permission
to read pre-cast storage remains confined to its computation-specific relation.
C still shares D's dtype, shape, order and count, with independent pitches.
In-place emission requires identical complete C/D descriptions and layouts.

A packed narrow feature-bias vector is consumed directly through **read-only
broadcast C**, with leading dimension zero. Both row and column broadcasts are
proved from complete shape-owned stride certificates. No matrix-sized broadcast
or FP32 widening buffer is needed. This is the vendor's documented
[C broadcast representation](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatmul).
The zero-leading-dimension permission applies only to C: A, B and all
destinations retain their ordinary non-overlapping proofs. A broadcast C cannot
become an in-place destination. This consumes C's argument slot, so a separate
residual C plus feature bias still uses the FP32-output path.

Native narrow bias/GELU epilogues can round an intermediate result before the
authored final cast. On CUDA 12.8/A100, cancellation tests exposed infinity where
the FP32 graph remains finite: `65504 + 32 - 65504` should produce FP16 32, but
rounding after the C addition overflows first. Other layouts also exposed this
in bias-only epilogues. A BF16 GELU case distinguished activation of an unrounded
product from activation after BF16 rounding. These tuples remain deliberately
unwhitelisted. Their graphs retain FP32 calls and explicit final casts; adding
cast tolerance does not authorize a new arithmetic boundary.

For narrow outputs, dispatch restricts split-K reductions to FP32 workspace
partials or no split-K reduction, excluding reduction through the narrow output
type. The selected bias argument carries an explicit dtype and operand index;
dispatch configures its vendor type and checks capacities using its real width.
The reference runtime has native two-byte FP16/BF16 payloads, exact widening,
explicit narrowing, and bit-preserving payload movement for these graphs.

### TF32 precision contract

A logical `F32 -> TF32 -> F32` round trip means the executing device's TF32
input conversion followed by exact widening. Storage remains F32; multiply,
reduction, C/bias addition and final casts keep their authored rounding sites.
TF32 uses eight exponent bits and ten fraction bits. Exact tie handling,
subnormal behavior and NaN encodings follow the device/library; they are not
additional portable compiler guarantees. The reference uses nearest-even as
one representative implementation. NVIDIA's
[compute-mode documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t)
describes nearest-even conversion, while the selected CUDA 12.8/A100 kernels
were observed to choose ties away from zero. Both are accepted by this policy.

`CublasLtTensorFloat32Computation` maps to `CUBLAS_COMPUTE_32F_FAST_TF32`.
Both operands must prove the exact authored cast chain, including through pure
views, and join the same computation variant and complete whitelist row. Its
448 rows across host/device scalar modes use F32 A/B storage, F32 accumulation
and F32 output. The call reads
pre-cast F32 storage and delegates input conversion to cuBLASLt. Earlier
narrowing remains in that storage source. Casts after multiplication/reduction
cannot authorize this mode. The proof composes with rank-two epilogues,
strided batches, separate destinations, in-place accumulation and subsequent
output casts. Batched bias/activation remain outside the existing batch grammar.

Operand storage proofs are parameterized by computation mode. A global fact
allowing rounded operands to read pre-cast storage would also let ordinary F32
calls skip rounding. Ordinary F32 calls instead consume materialized rounded
values. CUDA's `Tf32RoundTrip` materializes the complete cast chain into F32
storage using native TF32 conversion on Ampere and later, or software
nearest-even on earlier devices. Decomposed operations and final round trips
therefore remain executable. This adds no packed 19-bit CUDA storage ABI.

FAST_TF32 alone selected a full-F32 algorithm for a 1x1 product on the A100.
Dispatch constrains the numerical implementation mask to actual TF32 dot-product
inputs using the vendor's
[numerical implementation flags](https://docs.nvidia.com/cuda/cublas/index.html#cublasltnumericalimplflags-t).
If no algorithm supports a descriptor, preparation refuses that candidate;
ordinary F32 calls over materialized rounded operands remain available to
production search. The agreed environment assumption is that
`NVIDIA_TF32_OVERRIDE` is unset. No runtime environment check is added.

## Element strides and strided batches

Core represents coefficients with `ElementStrideExact IntExpr` and
`ElementStrideAny`. Any is freedom on a statically proved singleton axis;
it never unions global coefficients or turns conflicting exact proofs into
freedom. Complete shape-owned certificates are validated before publishing
backend-facing readings. Authored boundary strides preserve factors that index
simplification can erase at singleton extents.

The batched recognizer matches `A[B,m,k]` and `B[B,k,n]` broadcast into the exact
`[B,m,n,k]` multiply domain followed by reduction over k. Both maps must use the
batch coordinate, even when a matrix extent happens to equal B. A static B=1
also permits its coordinate to be literal zero. Arithmetic remains F32, with
explicit narrow-input widening handled by the shared storage-reading proofs. An
authored final BF16/FP16 cast can select the matching narrow D and C storage.

Batch descriptors require positive rank-three extents and one shared batch
count that fits positive int32. Every matrix has unit minor stride and a legal
leading dimension. A nonzero live batch stride must cover the **full padded matrix**:
`rows * ld` in row order or `columns * ld` in column order. Covering only the
last touched element is insufficient. A singleton batch may choose its
canonical padded pitch from Any. Overlapping nonzero batch strides, negative
strides, and unproved geometry cannot become matrix readings.

A and B may **independently share one matrix** across that batch axis. The
frontend states this with an explicit broadcast, for example:

```rust,ignore
let a = graph.tensor([g, m, k], DType::F32);
let b = graph.tensor([k, n], DType::F32).expand_dim(0, g);
let y = a.matmul(b); // y[g] = a[g] @ b
```

Broadcasting A instead gives `y[g] = a @ b[g]`; broadcasting both repeats the
same product, optionally adding a different C for each batch. Expanding an
existing singleton batch axis is also supported. These are pure views: the
shared operand needs only one physical matrix, including its row/column pitch.
No repeated matrix buffer is required. This feature does not insert frontend
broadcasts or flatten multiple batch axes.

`cublaslt-strided-batch-operand-reading` admits the existing disjoint readings
or a shape-owned **exact zero batch stride**, with the same unit minor stride
and legal leading dimension. This relation is used only for A and B. Every
vendor descriptor still carries the common logical batch count; sharing is
represented by stride zero, not a descriptor count of one. The independent
[layout attributes](https://docs.nvidia.com/cuda/archive/12.8.0/cublas/index.html#cublasltmatrixlayoutattribute-t)
are dispatched unchanged. Precision/storage proofs and the complete 2,176-row
configuration whitelist remain shared with ordinary batches.

Read sharing never publishes a disjoint matrix or destination proof. C and D
retain the full padded, non-overlapping batch footprint, including in-place
calls. Thus a shared A/B reading cannot authorize an overlapping output or a
shared C/D write. Partial matrix overlap remains unsupported.

Batched admission currently covers a product and product+C, without
bias or activation. C and D share shape, dtype, order and count but may have
independent leading dimensions and batch strides. In-place emission additionally
requires identical complete descriptions and layouts. Multiple batch axes and
arbitrary flattening are not part of this grammar. Unproved cases retain generic
operations/materialization.

Experimental literal-expression discovery is separately scheduled through
`element-stride-discovery`; production schedules do not run it. Native stride
composition handles permutations, broadcasts, zero selections and diagonals.
Unsupported arithmetic view maps deliberately stall.

## Extending the backend

1. State the logical arithmetic and rounding contract, including exceptional
   values, optional arguments, and allowed approximations.
2. Add local egglog recognition and complete shape-owned layout proofs. Reuse
   the flat vendor descriptor; give every new geometry field an explicit proof.
3. Join the complete configuration whitelist at admission. Add only supported
   rows, preserving the distinction between absent and present C/bias.
4. Let the existing final emitters consume the approval, or add an emitter whose
   storage contract is unconditional and proved before extraction.
5. Extend Rust decoding only for new data variants. Dispatch may evaluate
   expressions, validate backing capacities, and check vendor availability; it
   must not match graphs, flip transposes, select layouts, or repair admission.
6. Test whitelist ablation, every selectable descriptor's addresses, precision
   boundaries, symbolic/singleton geometry, ties/repair, and numerical execution.
   Use semantic references and fixed seeds rather than relying on a search win.

Boundary value delivery and folded writing-tie repair use `BufferCopyValue` with
an explicit unchanged address span, preserving spare allocation capacity.
`BufferCopy` retains exact-size whole-storage semantics. Reference compiled
artifacts use schema version 3 for element-stride layouts and value-copy spans.

## Recovery provenance and validation

The six `.egg` files at recovery commit `0adfb8d` were recovered byte for byte
from two surviving libraries
built on 2026-10-27 before the container rebuild; their SHA-256 values agree.
The recovery bundle is `.recovery/cublaslt-2026-10-27/`, locally excluded from Git
because it also preserves the binaries. Its manifest records archive sections,
source bytes, and hashes. Subsequent TF32 changes extend that recovered rule vocabulary; the manifest
continues to record the original recovery bytes. The core element-stride rule
files did not require replacement during recovery.

The small [recovery records](../../../../../docs/recovery/cublaslt-2026-10-27/)
are also committed: the ELF extraction script, original artifact locations and
timestamps, and the section/byte/hash manifest. The large original binaries
remain in the local bundle. To repeat extraction from those preserved archives:

```sh
python3 docs/recovery/cublaslt-2026-10-27/extract_egg.py .recovery/cublaslt-2026-10-27
```

The accompanying Rust wiring and missing batch test source were reconstructed
from the surviving baseline, the exact rule types, and binary debug information;
they are not claimed to be verbatim recovered Rust source. The preserved original
batch board passes 14 CPU tests and 4 device tests on the A100. Source validation
is recorded separately in the recovery bundle.

Restored-source validation on the A100 passes all 672 separate-destination and
336 in-place configuration executions, 288 batched dispatch witnesses, dynamic
batch-count replay, precision boundaries, bounds checks, and storage-tie repair.
The CPU admission suites pass, as do core, neural-network, reference-runtime,
CUDA Lite and test-runtime suites. The transpose-roundtrip regression evaluates
the elected vendor descriptors over the original input storage and compares the
folded output layout against an independent dense product. It retains zero-copy
and caller-ownership checks; complete expression transposition may validly add
folded views. Batch regressions also cover false batch coordinates at equal
extents, independent C/D pitches, cast boundaries, and C reuse/copy repair across
all batches. `cargo fmt --all -- --check` and device-enabled CUDA Lite and
test-runtime Clippy with `-D warnings` pass.

At the TF32 checkpoint, the extension expanded execution coverage to all **896 separate-destination
configurations**, **448 in-place configurations**, and **384 batched dispatch
witnesses**. Its regressions cover mode-specific raw-storage permission,
whitelist ablation, views, earlier narrowing, final rounding, native cast
materialization, decomposed execution, epilogues and batched C reuse/repair.
Comparisons use operands away from halfway cases when checking GEMM against
the reference; cast tests accept either nearest tie rather than imposing the
reference codec's tie choice on CUDA. The five-package landing sweep passes
663 tests with 54 existing ignores; the targeted device-enabled cuBLASLt suites
pass 23 tests. Formatting and device-enabled CUDA Lite/test-runtime Clippy pass.

At the narrow-storage checkpoint, validation executed all **1,088
separate-destination configurations**, **544 in-place configurations**, and **576 batched dispatch
witnesses** on the A100. Regressions cover exact widening and authored final
rounding, retained earlier narrowing, packed broadcast-C bias without
materialization, independent C/D pitches, changing batch counts, C reuse and
read-only repair, FP32 split-K partial sums, and finite cancellation/GELU
fallbacks for the deliberately excluded native narrow epilogues.
At that checkpoint, the 14 narrow-storage regressions passed with device
execution enabled. The five-package landing checks passed 673 tests with 54 existing ignores, including
the updated frontend proof that checks native-bias and broadcast-C operand
addresses. Formatting and device-enabled CUDA Lite/reference/test-runtime
Clippy with `-D warnings` passed.

General alpha/beta validation executes all **2,176 configurations**, **1,088
in-place configurations**, and **1,152 batched dispatch witnesses** on the A100
with non-unit coefficients. The nine scalar regressions cover host literals,
GPU inputs, mixed literal/tensor arguments, GPU-produced coefficients, changing
values on replay, zero/negative coefficients, precision boundaries and exact
whitelist ablation. Device-enabled cuBLASLt suites pass 52 tests; the five
required package suites cover 678 passing tests with 54 existing ignores.
Scalar dtype/capacity rejection and descriptor address/storage-tie proofs pass
for both pointer modes. Formatting and device-enabled CUDA
Lite/reference/test-runtime Clippy with `-D warnings` pass.

Explicit broadcast-batch validation executes **3,456 shared-batch dispatch
witnesses** on the A100, covering shared A, shared B and both across every
currently admitted batched configuration and both executable forms. Descriptor
addresses, exact single-matrix capacities, expanded singleton axes, padded
row/column storage, positive symbolic counts, whitelist ablation and rejection
of partial overlap/shared destinations are checked independently. Reference
comparisons exercise F32, FP16/BF16 storage and explicit TF32, non-unit host/GPU
scalars, retained C, and batch-count/scalar changes on replay. Execution tests
exclude generic cast/view materializers so they must consume shared storage
and fused rounding directly; production search still has those legal alternatives.
All 17 device-enabled batch regressions pass, including the original 1,152
disjoint-batch witnesses. Formatting and device-enabled CUDA Lite Clippy with
`-D warnings` pass. The five-package landing sweep passes 681 tests with 54
existing ignores.

The benchmark below is historical: it measures the older 48-row foundation on
`77d3060f`, before this restored 672-row vocabulary and batched support. Its
numbers do not measure the restored code.

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
