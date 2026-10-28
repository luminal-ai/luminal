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
| `strided_batch_rules.egg` | One explicit batch axis, batch layout proofs, and complete product/accumulation admission |
| `approved_whitelist.egg` | The 672 complete configurations allowed for execution |
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

The separate-destination form uses inputs A, B, optional C, optional bias, and a
write-only destination appended by DPS conversion. It grants no input alias
permission. The in-place form uses A, B, optional bias, then C as its destination;
C is operand 2 without bias and operand 3 with bias. Its unconditional
`Sharing::Must` tie connects that final operand to result 0. Its emission rule
requires the same complete vendor layout and the same underlying layout for C
and D. There is no conditional alias flag or Rust destination rebinding.

## Approved configurations and precision

The table contains **672 configurations**:

- Matching F32/F32, BF16/BF16, or F16/F16 A/B storage: 3 choices.
- Independent A/B matrix orders and transpose flags: 16 choices.
- Absent C with beta 0 or F32 C with beta 1: 2 choices.
- Default epilogue with either D order, or bias, ReLU, ReLU+bias, GELU,
  GELU+bias with column-major D: 7 choices.

Compute, scalar, D, and any present C or bias types are F32. Alpha is 1.
Only explicit F32 input casts permit direct BF16 or F16 storage readings.
Storage does not choose arithmetic precision. Casts between multiply, reduction,
C addition, or bias addition remain rounding boundaries. Earlier narrowing and
final output casts are retained. C and bias widening is materialized as F32.
F32 reduction reassociation and FMA contraction follow the existing compiler
policy. ReLU accepts a difference in the sign of zero; both frontend GELU
spellings may use the vendor tanh approximation. These are explicit admission
policies, not claims of bitwise equivalence.

Bias has the vendor's row length and packed storage. The rank-two rules
transpose a complete expression to implement normal feature bias. C is added
before bias and activation. Existing core addition commutativity remains
available; floating reassociation or crossing casts does not authorize another
fused expression. TF32, FP8, mixed A/B storage, narrow
outputs, and arbitrary alpha/beta values have no approved configurations.

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
explicit narrow-input widening handled by the shared storage-reading proofs.

Batch descriptors require positive rank-three extents and one shared batch
count that fits positive int32. Every matrix has unit minor stride and a legal
leading dimension. A live batch stride must cover the **full padded matrix**:
`rows * ld` in row order or `columns * ld` in column order. Covering only the
last touched element is insufficient. A singleton batch may choose its
canonical padded pitch from Any. Live zero/broadcast strides, overlapping
batches, negative strides, and unproved geometry cannot become matrix readings.

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

These six `.egg` files were recovered byte for byte from two surviving libraries
built on 2026-10-27 before the container rebuild; their SHA-256 values agree.
The recovery bundle is `.recovery/cublaslt-2026-10-27/`, locally excluded from Git
because it also preserves the binaries. Its manifest records archive sections,
source bytes, and hashes. The core element-stride rule files match the recovered
core library exactly and did not require replacement.

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
