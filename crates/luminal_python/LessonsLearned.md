# Lessons Learned

This file documents hard bugs encountered in this codebase, their root causes, and principles
to prevent similar issues in the future.

---

## 2026-02-24 — Intermittent CUDA Backend Failures: Embed False Match + Batched Matmul Dimension Drop

### Background: Why the Failures Were Intermittent

Both bugs only appeared on roughly 50% of test runs. The source of non-determinism is
`FxHashMap` (a fixed-seed hash map). The egglog optimizer's `SerializedEGraph::new` builds
`Vec<NodeId>` orderings for each e-class by iterating a `FxHashMap`, producing non-deterministic
node orderings. `random_initial_choice()` in `src/egglog_utils/mod.rs` then randomly picks one
e-node per e-class as the starting representation for the profiling phase. The combination means
some runs pick a correct kernel and some pick a broken one from the same e-class.

**Lesson**: When a test fails intermittently at a roughly 50% rate, suspect the egglog extractor
choosing between two e-nodes in the same e-class — one correct, one broken. The fix is always in
the broken e-node's rewrite rule.

---

### Bug 1: `test_gather_elements` — KernelEmbed and RowEmbed False Match

**Files changed**:
- `crates/luminal_cuda/src/kernel/hlir.rs` (KernelEmbed, 4 rules)
- `crates/luminal_cuda/src/block/ops.rs` (RowEmbed, 4 rules)

#### What happened

`gather_elements` (axis-aware gather) decomposes into a flat gather by computing:

```
flat_idx = Add(
    Mul(indices, stride[axis]),
    Mul(Expand(Iota(dim_size)), stride[non_axis])
)
```

`KernelEmbed` and `RowEmbed` are optimized embedding lookup kernels. A genuine embedding
lookup produces:

```
flat_idx = Add(
    Mul(Cast(token_ids), embed_dim),
    Iota(embed_dim)              ← bare Iota, the position within an embedding row
)
```

The egglog rewrite rules for both ops matched `Add(?mul_result, ?iota_result)` where
`?iota_result` was **unconstrained** — it could bind to anything, including
`Mul(Expand(Iota(n)), stride)` from `gather_elements`. This created a `KernelEmbed`/`RowEmbed`
node in the same e-class as the `Gather` node. When the extractor picked it, `build_payload`
called `flatten_mul_strides(range, token_stride)` which asserted `range.len() == token_stride.len()`:
- `range` came from `RemoveNthFromEnd(idx_shape, 0)` → length 1
- `token_stride` came from the indices strides → length 2
- Assertion failed → panic.

#### The fix

Add `(= ?iota_result (Iota ?iota_expr ?iota_range))` to all 8 rules, requiring the positional
component to be a bare `Iota` node:

```egglog
(= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
(= ?iota_result (Iota ?iota_expr ?iota_range))   ← added
(= ?mul_result (Mul ...))
```

#### Investigation note

The initial plan correctly identified `KernelEmbed` as faulty, but missed `RowEmbed`. The two
ops are structurally identical but live in different parts of the codebase (`kernel/` vs
`block/`). The second bug was only discovered when the backtrace pointed to
`RowEmbed::build_payload` instead of `KernelEmbed::compile`. Always search for sibling
implementations when fixing a pattern-matching bug in one op.

---

### Bug 2: `test_matmul_batched` — CuBlasLt Drops Batch Dimension

**Files changed**:
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_RmRm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_RmCm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_CmRm_rewrite.egg`
- `crates/luminal_cuda/src/host/cublaslt/cublaslt_CmCm_rewrite.egg`

#### What happened

The luminal frontend decomposes `(2,3,4) @ (2,4,5)` into:

```rust
let w = rhs.permute((0, 2, 1));       // (2,4,5) → (2,5,4)
let mul = self.expand_dim(2, d)        // (2,3,4) → (2,3,5,4)
        * w.expand_dim(1, b);          // (2,5,4) → (2,3,5,4)
mul.sum(3)                             // → (2,3,5), correct out_shape
```

All four cublaslt rewrite rules extracted `m` and `n` from the output shape using
`nth_from_end`, which succeeds for any rank:

```egglog
(= ?m (nth_from_end ?out_shape 1))
(= ?n (nth_from_end ?out_shape 0))
```

For `out_shape = [2, 3, 5]`: `?m = 3`, `?n = 5`. The batch dim `2` is never extracted or
stored. The rules also validated stride patterns using `nth_from_end` on the stride arrays —
but for this batched case, **all stride checks coincidentally passed** because the last three
strides of the 4D expanded tensors happened to satisfy the 2D row/column-major patterns.

The resulting `CuBlasLt` node had `output_size() = m * n = 15`. The batch dimension was
silently discarded. The runtime allocated a 15-element output buffer, cuBLAS wrote a 3×5
result, and the test got back 15 values instead of 30.

#### The fix

Add `(= (len ?out_shape) 2)` to all 4 rules:

```egglog
(= (len ?out_shape) 2)    ← added: cuBLAS is 2D only
(= ?m (nth_from_end ?out_shape 1))
(= ?n (nth_from_end ?out_shape 0))
```

`len` counts elements in the `ECons`-list shape. With this constraint, any `Sum` node with a
3D+ output shape (batched matmul) is not matched by cuBLAS rules and falls through to
`KernelSumReduce + KernelMul` (or the tiling block ops), which correctly use
`out_shape.iter().product()` for their output sizes.

Note: `TileMatmulSplitK` and `TileMatmulFullSplit` do NOT need this fix — their `output_size()`
already returns `untiled_range.iter().product()` which includes all dimensions.

---

### General Principle: Always Constrain Shape Rank in Egglog Rules

Both bugs share the same structural cause: **egglog rewrite rules that used `nth_from_end` to
extract dimensions from a shape list without constraining the list's length.** Since
`nth_from_end` silently succeeds for any list with enough trailing elements, rules written for
2D tensors accidentally matched higher-rank tensors.

**Rule for writing egglog rewrite rules in this codebase**:

> If a rule is designed for a specific tensor rank, always add an explicit
> `(= (len ?shape) N)` constraint. If a rule is designed to handle arbitrary ranks but an
> op's output only covers a subset of dimensions (like cuBLAS covering only the last 2),
> that is a correctness bug — either implement strided batched cuBLAS or add the rank
> constraint and fall back to a kernel that handles all dimensions.

---

### Debugging Intermittent CUDA Failures: Effective Approach

The investigation used extensive `eprintln!` debug logging to trace which kernels were compiled
vs. skipped. Key observations:

1. **In the passing case**: `KernelSumReduce::compile()` was called, kernels were allocated.
2. **In the failing case**: `KernelSumReduce::compile()` was never called, yet output was produced.

This asymmetry pointed to a `HostOp` path (cuBLAS) executing instead of the `KernelOp` path,
which narrowed the search to cublaslt rewrite rules. The HLIR-level `SumReduce::to_egglog` log
confirmed the correct HLIR node existed — the bug was in the e-graph optimization choosing
a different (broken) e-node from the same e-class.

**Effective debug strategy for egglog non-determinism bugs**:
1. Add logging at compile time for each kernel type (`KernelFoo::compile`, `HostFoo::execute`)
2. Compare passing vs. failing runs to see which kernels are/aren't invoked
3. The missing kernel's e-class contains a broken alternative — find it via the egglog rewrite rules
4. Check the op that *is* executing — its `output_size()` reveals what's wrong with the false match

---

## 2026-02-25 — OneHot Test Panic: Cast(Int→F32) Produces Int Output

### What the symptom was

`test_onehot` panicked at `src/hlir.rs:1625` in `get_f32()`: the output buffer was
`NativeData::Int` instead of the expected `NativeData::F32`.

### What the actual root cause was

The Cast parser's `* 1.0` workaround for `Int → F32` casts used `input * one_expanded`
(Int GraphTensor on the left, F32 constant on the right). However, `Mul for GraphTensor`
always uses `self.dtype` (the **left** operand's dtype) for the result, and the native
runtime's `Mul::execute` dispatches on the **first** input's `NativeData` variant. So
`Int * F32` produced `DType::Int` / `NativeData::Int` — the exact opposite of the intended
F32 output.

### Why it was hard to find

1. **The OneHot parser was a red herring**: The initial plan assumed the OneHot ONNX node
   was being parsed, but `torch.onnx.export` decomposes `one_hot` into
   `Unsqueeze → Equal → Cast(Bool→Int) → Cast(Int→F32)`. The OneHot parser was never called.
2. **The `* 1.0` workaround looked correct**: It was used successfully in many other parsers,
   but those all had F32 inputs (where `F32 * F32 = F32`). The Int→F32 case was the only
   path where the left operand was Int.
3. **Operand order matters silently**: Nothing warns about mixed-dtype Mul — it just takes
   the left operand's dtype.

### The fix

In `ops_parse/unary.rs` `parse_cast_node`, split the combined condition into two cases:
- **No-op cast** (`cast_result.id == input.id`): `input * one_expanded` — preserves dtype
- **Int source** (`input.dtype == DType::Int`): `one_expanded * input` — F32 on the left
  ensures F32 output

### General principle

**In luminal, binary op dtype is always the LEFT operand's dtype.** When constructing
`GraphTensor * constant_float(1.0)` for type materialization, always put the operand
whose dtype you want to preserve on the LEFT side. When converting Int→F32, the F32
constant must be the left operand.

---

## 2026-02-26 — ScatterND Fails on CUDA: "does not produce an egraph"

### What the symptom was

`test_scatter_nd` passed on native backend but failed on CUDA with "does not produce an
egraph". The CUDA compilation could not extract a valid program from the e-graph.

### What the actual root cause was

`scatter_nd` in `movement.rs` does `indices * 1` (line 353) to materialize the tensor for
reshaping. The `* 1` dispatches to `Mul<S: Into<Expression>>`, which creates a `constant(1)`
→ `Iota(1,1)` → `DType::Int`. But the ONNX parser creates all tensors as `DType::F32`
(via `named_tensor()` in `compiled_graph.rs:70`), so indices arrive as F32. This produces
`Mul(F32, Int)` — mixed dtypes.

The HLIR Mul dtype rule (`hlir.rs:886-888`) uses `(= ?dty (dtype ?lhs))` and
`(= ?dty (dtype ?rhs))` with the same `?dty` variable, requiring both inputs to have
matching dtypes. `F32 != Int` → the rule never fires → the Mul node gets **no dtype**.

Every downstream op checks `(= ?dty (dtype ?upstream))`. Without dtype on the Mul, no
CUDA kernel rewrite rules fire for any downstream op (KernelMul, KernelAdd, KernelLessThan,
etc.). When `cleanup_hlir` runs (enabled for CUDA, disabled for native), it deletes all
unrewritten HLIR ops, leaving empty e-classes → egraph extraction fails.

### Why it was hard to find

1. **Works on native**: `cleanup_hlir = false` for NativeRuntime, so unrewritten HLIR ops
   are never deleted. NativeOp dispatches on actual runtime data, not egglog dtype.
2. **Cascading failure**: The root cause (missing dtype on one Mul) silently propagated
   through every downstream op, making it look like a systemic CUDA issue rather than a
   single dtype mismatch.
3. **`scatter_elements` works fine**: The sibling op already cast indices via
   `(idx_f32 + (is_neg * adj)).cast(DType::Int)`, so only `scatter_nd` had this bug.

### The fix

Added `let indices = indices.cast(DType::Int);` at the top of `scatter_nd` in
`movement.rs`, before any arithmetic on indices. `GraphTensor::cast()` short-circuits
when `self.dtype == dtype`, so this is safe for callers already passing Int indices.
Also added the same cast in `parse_scatter_nd_node` for explicitness.

### General principle

**Always cast index tensors to `DType::Int` before arithmetic in graph-building code.**
ONNX tensors arrive as F32 from the Python bridge. Any `indices * stride` or
`indices * 1` will produce `Mul(F32, Int)` which breaks HLIR dtype propagation on CUDA.
The pattern `let indices = indices.cast(DType::Int);` at the top of any index-consuming
function is defensive and free (no-op when already Int).

---

## 2026-03-04 — Dynamic Shapes: Empty Buffer for BOOL Scalar Initializer

### What the symptom was

`test_hf_llama_decode_loop_dynamic` panicked at `bin_fn: a index 0 out of bounds (a.len=0), shape=[1, 1, 4, 4], strides=[0, 0, 0, 0]`. An Input node labeled `"new_ones"` had an empty buffer at runtime.

### What the actual root cause was

Two issues combined:

1. **`load_tensor_floats` didn't handle ONNX data_type=9 (BOOL)**. The `new_ones` initializer was a BOOL scalar (1 byte in `raw_data`). `load_tensor_floats` fell through to the fallback case, which tried `chunks_exact(4)` on 1 byte → produced 0 chunks → returned empty vec `[]`. The buffer was set with empty data.

2. **Scalar initializers with empty `dims` created 0-dimensional tensors**. ONNX represents scalars with `dims=[]`. The initializer loop computed `shape = init.dims.iter().map(|&d| d as usize).collect()` → empty vec `[]`, then called `named_tensor(name, [])` which created a tensor with 0 dimensions instead of the intended scalar `[1]`.

### Why it was hard to find

1. **Misdiagnosed as ConstantOfShape issue**: The original plan targeted `ConstantOfShape` with dynamic shapes. The shape `[1,1,4,4]` with strides `[0,0,0,0]` looked like a broadcast from a constant fill. But `parse_constant_of_shape` was never called — the `new_ones` tensor came from an ONNX initializer, not a computation node.

2. **The BOOL data type is unusual**: Most ONNX tensors are FLOAT, INT32, or INT64. BOOL initializers only appear in specific patterns (like `torch.ones()` in attention mask computation). `load_initializer_as_f32` already handled BOOL, but its sibling `load_tensor_floats` didn't.

3. **Empty vec is valid data**: `set_data(node_id, [])` doesn't panic — it silently sets an empty buffer. The error only manifests later when a downstream op tries to read index 0.

### The fix

1. Added `data_type=9` (BOOL) handling to `load_tensor_floats` in `util.rs` — same logic as `load_initializer_as_f32`: 1 byte per element, non-zero → 1.0, zero → 0.0.

2. In `compiled_graph.rs`, initializer tensor creation: if `shape.is_empty()`, set `shape = vec![1]` (scalar representation in luminal).

### General principle

**Keep data loading functions in sync.** `load_tensor_floats` and `load_initializer_as_f32` serve the same purpose (loading ONNX TensorProto data as f32) but had different data type coverage. When adding a new data type to one, check and update the other. Better yet, refactor them into a single function.

**ONNX scalars have `dims=[]`, luminal scalars have shape `[1]`.** Always convert empty dims to `[1]` when creating luminal tensors from ONNX data.

---

## 2026-03-04 — Where Node Missing Broadcast: KernelMul flatten_strides Panic on CUDA

### What the symptom was

`test_hf_llama3_1b_decode_loop_dynamic` panicked at `flatten_strides` with `left: 4, right: 1` during
CUDA `KernelMul::compile`. The `KernelMul` had `out_shape=[1, 1, a, a]` but `b_stride=[z]` (1D).

### What the actual root cause was

`parse_where_node` called `x.cond(condition, y)` without broadcasting the inputs to matching ranks.
The ONNX Where op for the attention mask had condition=[1,1,a,a] (4D), x=[1] (scalar), y=[1] (scalar).
Luminal's `cond` doesn't auto-broadcast — it passes the shape trackers directly to the HLIR node.
The resulting Mul had input A with 4D strides and input B with 1D strides.

### Why it was hard to find

1. **Only triggered by 1B model**: The tiny model's Where inputs all had matching ranks (no scalars).
2. **CUDA-only**: The native runtime's `bin_fn` uses `StridedIterator` which handles mismatched
   strides more gracefully. CUDA's `KernelMul::compile` calls `flatten_strides` which asserts
   `range.len() == strides.len()`.
3. **Delayed crash**: The mismatch was created during ONNX parsing but only manifested during
   CUDA kernel compilation (graph search phase).

### The fix

Added numpy-style broadcasting to `parse_where_node`: compute the broadcast shape across all 3
inputs, then `broadcast_to_expr` each to the common shape before calling `cond`.

### General principle

**ONNX binary/ternary ops all use numpy broadcasting.** When parsing ONNX ops that take multiple
tensor inputs (Where, Add, Mul, etc.), always broadcast all inputs to a common shape BEFORE
calling the luminal graph operation. Luminal graph ops do NOT auto-broadcast — they expect inputs
with matching shape tracker dimensions.

---

## 2026-03-05 — TopK Values Wrong on CUDA (gather_elements with sliced non-contiguous indices)

1. **Symptom**: `test_topk_values` failed on CUDA — rows 0-1 were correct but rows 2+ returned
   the value at column 0 of each row (all three top-k positions got the same value).
   Native backend was fine.

2. **Root cause**: `gather_elements` was called with a non-contiguous index tensor produced by
   `argsort(axis=1) → slice_along(..k, axis=1)`. The slice creates a ShapeTracker view of the
   [4,8] argsort buffer with dims [4,3] and strides [8,1]. When this flowed through the
   gather_elements Int arithmetic chain (cast, multiply, add) and into the final Gather CUDA
   kernel, the non-contiguous strides caused incorrect index reads for later rows.

3. **Why it was hard to find**: `test_topk_indices` passed (it only tests argsort+slice, not
   the downstream gather_elements). A standalone `test_gather_elements` with constant indices
   also passed because constant indices are contiguous. The bug only manifested when runtime-
   computed non-contiguous indices were used with data of a different size along the gather axis.

4. **Fix**: In `parse_topk_node`, compute `gather_elements(x, full_argsort, axis)` with the
   full [4,8] argsort result (same size as data), then slice the gathered values to [4,3].
   This ensures gather_elements always operates on same-sized contiguous tensors.

5. **General principle**: When building graph operations that chain shape-tracker views
   (slice, transpose, etc.) into downstream HLIR ops on CUDA, prefer operating on full
   contiguous tensors first and slicing the result afterward. Non-contiguous views flowing
   through multiple CUDA kernels can trigger stride-related bugs in the egglog-compiled code.


---

## 2026-03-07 — Non-deterministic CUDA_ERROR_ILLEGAL_ADDRESS: Multiple Missing Rank Constraints

### What the symptom was

`test_hf_llama_tiny` on CUDA failed ~70% of runs with `CUDA_ERROR_ILLEGAL_ADDRESS`. Failures
were non-deterministic due to egglog's `FxHashMap` iteration order in `random_initial_choice()`.

### What the actual root cause was

**Multiple** matmul egglog rules lacked `(= (len ?out_shape) 2)` constraints:

1. `TileMatmulSplitK` in `block/ops.rs` (disabled via comment but rule still registered)
2. `TileMatmulFullSplit` in `block/ops.rs`
3. All 4 `sgemm_v2_*.egg` rules in `host/cublas/`

The `cublaslt_*.egg` rules already had the constraint. When egglog picked TileMatmul or sgemm
for a 3D+ batched matmul, the generated CUDA kernels accessed out-of-bounds memory.

Additionally, `KernelEmbed` in `kernel/hlir.rs` had an output indexing bug:
`out[out_offset * embed_dim + embed_idx]` should be `out[out_offset + embed_idx]` because
`out_offset` already includes the embed_dim factor from `flatten_strides`.

**Most critically**, the KernelEmbed and RowEmbed "with cast" egglog rules passed the
**pre-cast** float token_ids (`?token_ids`) to the embed kernel instead of the **post-cast**
int token_ids (`?token_ids_cast`). The CUDA kernel reads token_ids as `const int*`, so float
data gets reinterpreted as enormous garbage integers, causing out-of-bounds embed table access.

### Why it was hard to find

1. **Multiple independent bug sources**: The ~70% failure rate was caused by three separate bugs
   (matmul rank, embed output indexing, embed pre-cast input). Each fix only reduced the rate
   partially, making it seem like each fix was insufficient.
2. **CudaGraph wrapping**: The crash occurred inside `CudaGraphOp::execute_internal` which
   batches multiple kernels via CUDA graphs. The error just said "CudaGraph" — it
   didn't identify which kernel crashed. Adding per-kernel debug launches was essential.
3. **Cascading failures**: When the Megakernel (containing RowEmbed with the pre-cast bug)
   corrupted the embed output, the NEXT CudaGraph group's kernels crashed reading the garbage.
   This made the Megakernel appear to be the victim, not the source.
4. **The pre-cast bug only crashes SOMETIMES**: Egglog's random choice determines whether
   KernelEmbed/RowEmbed is selected (crash) or the generic Gather path is used (works).
   Float token_id 1.0 (= 0x3F800000 = 1065353216 as int) produces an astronomically large
   embed table index, causing ILLEGAL_ADDRESS.

### The fix

- Added `(= (len ?out_shape) 2)` to TileMatmulSplitK, TileMatmulFullSplit, and all 4 sgemm_v2 rules
- Fixed KernelEmbed output indexing: `out[out_offset + embed_idx]`
- **Fixed KernelEmbed/RowEmbed "with cast" rules**: Changed input from `?token_ids` to
  `?token_ids_cast` — using the post-Cast int tensor instead of the pre-Cast float tensor

### Results

Failure rate: ~70% → 0% (20/20 passing). All three bugs needed to be fixed together.

### General principle

**When an egglog rule matches a sub-expression chain (like Cast→Mul→Add), be precise about
which intermediate result becomes each input.** The "with cast" embed rules matched
`Cast(?token_ids, ...)` to verify the Cast existed, but then passed `?token_ids` (the Cast
INPUT) instead of `?token_ids_cast` (the Cast OUTPUT) to the embed kernel. The kernel expects
int data, so the pre-cast float data was reinterpreted as garbage ints.

**Always search for sibling implementations**: KernelEmbed (in `kernel/hlir.rs`) and RowEmbed
(in `block/ops.rs`) had the SAME bug in their "with cast" rules. Fixing one without the other
only reduces the failure rate — both must be fixed.

---

## 2026-03-09 — TileMatmulFullSplit Matches Element-wise Square+Sum from LayerNorm

### What the symptom was

`test_qwen_image_transformer_tiny` on CUDA produced NaN in specific output rows. The failure
was non-deterministic (~85% failure rate) due to egglog's random e-class extraction picking
TileMatmulFullSplit for some operations.

### What the actual root cause was

The `TileMatmulFullSplit` rewrite rule in `block/ops.rs` matched any `Mul + Sum` pattern with
a 2D output, contiguous K-strides, and F32 inputs. This correctly matched real matmuls, but
ALSO matched the element-wise `x * x + Sum(last_dim)` pattern from LayerNorm/RMSNorm
(Pow(x, 2) → ReduceMean).

For a [1, 4, 64] activation tensor `x`:
- `Mul(x, x)` shape: [1, 4, 64], strides: [256z, 64z, z] for both inputs
- `Sum(dim=2)` output: [1, 4], len=2 ✓

TileMatmulFullSplit interpreted this as a [1, 64] × [64, 4] → [1, 4] matmul with:
- A = row 0 of x (64 elements), B = same buffer at column offsets

The kernel computed `C[j] = sum_k x[k] * x[j*64+k]` (cross-products) instead of the correct
`C[j] = sum_k x[j*64+k]^2` (squared sums). This produced subtly wrong values for j > 0
(correct for j=0 since cross-product with self = squared sum). These wrong values propagated
through LayerNorm → downstream operations → softmax → NaN.

Key diagnostic: adding `printf` to the kernel showed `a_ptr == b_ptr` (same buffer for both
inputs), confirming the kernel was operating on `x * x` not a real matmul.

### Why it was hard to find

1. **Individual op tests passed**: Simple Gemm tests, attention tests, and all other bisection
   tests passed because they didn't have the specific `x*x → Sum` pattern.
2. **Non-deterministic**: The bug only manifested when egglog selected TileMatmulFullSplit
   over the kernel fallback for the square+sum operation.
3. **No NaN from TileMatmulFullSplit itself**: The kernel produced wrong-but-finite values.
   NaN only appeared downstream through softmax (exp(large) → ∞ → ∞/∞ = NaN).
4. **Systematic elimination needed**: Had to disable all block ops, then enable one at a time,
   to narrow down TileMatmulFullSplit as the culprit.

### The fix

Added matmul broadcast constraints to both `TileMatmulFullSplit` and `TileMatmulSplitK` rules:

```egglog
; Assert proper matmul broadcast pattern:
; A is broadcast over N (a_n_stride = 0), B is broadcast over M (b_m_stride = 0)
(= ?a_n_stride (MNum 0))
(= ?b_m_stride (MNum 0))
```

In a real matmul `[M, K] × [K, N]`, the Mul is created by expanding dims:
- A is broadcast over N → a_n_stride = 0
- B is broadcast over M → b_m_stride = 0

In element-wise `x * x`, both strides are identical (non-zero for all dims), so the
constraints correctly reject it. The cuBLAS `.egg` rules already had these constraints.

### General principle

**Matmul Mul+Sum patterns have specific broadcast structure: one input is broadcast over M
and the other over N.** When writing egglog rules that match `Mul + Sum` patterns for matmul
optimization, always verify the broadcast pattern (`a_n_stride = 0` and `b_m_stride = 0`).
This prevents matching element-wise operations like `x*x → sum` that happen to have a 2D
output and contiguous strides.

---

## 2026-03-09 — Conv3D Permute Axis Mismatch in ONNX Conv Parser

### Symptom

`test_qwen_image_vae_decoder_tiny` panicked with:
> Permute axes (5) doesn't match shape axes (6)

at `src/shape/tracker.rs:153`, during `parse_conv_node`.

### Root cause

The Conv parser's unfold → matmul algorithm used two consecutive permutes with incorrect
index calculations. After unfold produces a 2N-dimensional tensor
`[win_0..win_{N-1}, k_0..k_{N-1}]`, the first permute swapped kernel dims to the front.
But the second permute's index math still assumed the original (pre-first-permute) ordering,
confusing kernel dimensions with window dimensions. Additionally:

1. `output_spatial_dims` was captured from wrong indices (kernel dims instead of window
   spatial dims)
2. The `split_dims` loop iterated `spatial` times instead of `spatial-1`, creating a
   spurious size-1 dimension
3. The final permute array had `1+spatial` elements for a tensor with `2+spatial` dims

For Conv2D (spatial=2) this was never caught because the xfail'd VAE decoder test was the
only test exercising the Conv parser — the transformer tests don't use Conv ONNX nodes.

### Why it was hard to find

The Conv parser was written and the VAE test immediately xfail'd due to a *different* bug
(`merge_dims` being `todo!()`). Once `merge_dims` was implemented, the Conv parser's own
bugs surfaced for the first time.

### Fix

Rewrote the unfold → matmul section with a single correct permute:

1. **One permute** to `[N, win_spatial..., C_in, k_batch, k_chan, k_spatial...]`
   — groups batch | output spatial | channel+kernel
2. **Capture** `output_spatial_dims` from correct indices `[1..1+spatial]`
3. **Merge** all channel+kernel dims from the end into one
4. **Merge** spatial dims into one → `[N, spatial_product, C_in*kernel_product]`
5. **Matmul** → `[N, spatial_product, C_out]`
6. **Split** spatial back with `spatial-1` splits (not `spatial`)
7. **Permute** C_out to position 1 with correct `2+spatial` element array

### General principle

**When chaining permutes on high-dimensional tensors, prefer a single combined permute.**
Multiple permutes with hand-computed index arrays are error-prone because each permute
redefines what indices mean. A single permute from the original layout to the target layout
is easier to verify and less likely to confuse source/destination ordering. Also, ensure
`split_dims` loop counts match: splitting N dims out of a product requires N-1 splits
(the outermost dim is the quotient, not split out separately).

---

## 2026-03-18 — CUDA Search Rejects All Candidates: Zero Dummy Data Causes NaN for Div/Pow/Mod/Erf

### What the symptom was

6 CUDA tests (`test_pow`, `test_pow_broadcast`, `test_div`, `test_mod`, `test_mod_broadcast`,
`test_erf`) consistently failed with `Failed to find a viable initial genome for group 0 after
100 attempts`. All 6 passed on native backend.

### What the actual root cause was

The CUDA two-phase initialization in `build_cuda_backend` set ALL input tensor buffers to
`0.0f32` as dummy data for profiling. When `torch.compile` decomposes a model, it passes
model weights as additional ONNX graph inputs (not initializers). Since there were no ONNX
initializers to overwrite the zeros, weight buffers stayed all-zero during search.

Operations with zero inputs produced NaN:
- `fmod(0, 0) = NaN` (Mod test)
- `weight * recip(0) = weight * inf` → with any zero weight → `0 * inf = NaN` (Div test)
- `abs(0).log() = log(0) = -inf` → downstream NaN (Pow test)
- `sign(0)` chain → operations on zero inputs (Erf test)

The `has_nan_outputs` check rejected every candidate genome, exhausting all 100 attempts.

### Why it was hard to find

1. **No panic, no crash — silent NaN rejection**: The error message said "Failed to find a
   viable initial genome" which suggested an egglog rewrite issue, not a data issue.
2. **Works on native**: `NativeRuntime::has_nan_outputs()` returns `false` by default (no NaN
   check), so zero inputs never caused problems on native.
3. **torch.compile vs direct export difference**: Directly exporting a model via
   `torch.onnx.export(model, ...)` produces initializers. But `torch.compile`'s backend
   receives a `GraphModule` where weights are graph inputs, not initializers. The ONNX file
   from `torch.compile` has 0 initializers.
4. **CudaRuntime's own `allocate_dummy_input` already uses 1.0**: The runtime knew zeros
   were problematic (comment: "Zero inputs often hide numerical issues"), but the
   `compiled_graph.rs` code used `0.0f32` independently.

### The fix

Changed dummy data from `vec![0.0f32; n_elements]` to `vec![1.0f32; n_elements]` in
`build_cuda_backend`. Using 1.0 is numerically safe: `fmod(1,1)=0`, `recip(1)=1`,
`log(1)=0`, `exp(1)≈2.7` — no NaN or inf. Profiling timing is unaffected (same number
of FLOPs and memory accesses).

### General principle

**Use small non-zero values (1.0) for dummy profiling data, never zeros.** Zero is a
singularity for many floating-point operations (division, log, fmod with zero divisor).
The CUDA runtime's `allocate_dummy_input` already followed this principle — the ONNX
pipeline's `build_cuda_backend` was inconsistent. When creating dummy data for GPU
profiling, always match the runtime's safer default.

---

## 2026-03-18 — Dynamic Decode Loop Fails: HLIR Weight Buffers Consumed After First Execute

### What the symptom was

`test_hf_llama3_1b_decode_loop_dynamic` passed step 0 (seq_len=6) but panicked on step 1
(seq_len=7) with `no entry found for key` at `cublaslt/mod.rs:294` — the CuBlasLt op couldn't
find its weight input buffer.

### What the actual root cause was

**Two bugs:**

1. **Missing `)` in egglog rule** (`luminal_cuda_lite/src/kernel/hlir.rs:3042`): The fourth
   KernelEmbed rule ("kernel embed with mul reversed") had 3 closing parens after `INil` instead
   of 4. The missing `)` failed to close the `(= ?mul_result ...)` form. This caused an egglog
   parse error during search, caught by `catch_unwind`. The rule was dead code — it never fired,
   but the parse error consumed a search iteration.

2. **HLIR buffer consumption killed weight buffers** (`luminal_cuda_lite/src/runtime.rs:1010-1057`):
   After each `execute()`, the runtime removed all HLIR buffers (weights, constants) except those
   directly connected to Output nodes. This was intended to free one-shot input data, but it also
   deleted all 168 weight buffers. On the next `graph.run()`, CuBlasLt couldn't find any of its
   weight inputs — `hlir_buffers` had 1 entry (the just-set `input_ids`) instead of 169.

### Why it was hard to find

1. **Misdirection by the egglog syntax error**: The plan identified the missing `)` as THE cause.
   Fixing it allowed the rule to parse correctly, but the real runtime failure was independent.
2. **Step 0 always succeeds**: The weight consumption happens AFTER a successful execution. So
   the first `graph.run()` works perfectly — all 169 HLIR buffers exist. The panic only occurs
   on the second call, after consumption has cleared 168 of them.
3. **The consumption code was deliberately designed**: Comments said "weight tensors must have
   `.persist()` to survive." The ONNX pipeline didn't call `.persist()` on weights, but this
   had never been a problem before because single-shot inference only calls `execute()` once.
4. **Search phase panics masked by `catch_unwind`**: The same "no entry found for key" error
   occurred during profiling of search candidates, but was silently caught. This made it look
   like only certain LLIR variants had the issue, not all of them.
5. **Debug output needed 4 iterations to find**: The first debug showed which NodeIndex was
   missing, the second showed it was an Input node, the third showed the HLIR mapping, and
   the fourth revealed `hlir_buffers_count` dropping from 169 to 1 between steps.

### The fix

1. Added missing `)` to the KernelEmbed egglog rule at `hlir.rs:3042`.
2. In `compiled_graph.rs`, added `.persist()` calls on all weight/constant tensors (anything
   not in `input_names`) after `process_onnx_nodes` completes. `.persist()` creates an Output
   node connected to the Input, which the consumption code recognizes as "do not consume."
   User inputs (like `input_ids`) are intentionally NOT persisted — they are consumed after
   each `execute()` and re-set via `set_input()` before the next call.

### General principle

**Mark weight/constant tensors as persistent in the graph-building pipeline.** The runtime's
`execute()` consumes all HLIR buffers not connected to Output nodes. This is correct behavior
for one-shot user inputs, but weights must survive across calls. Always call `.persist()` on
tensors that should outlive a single execution. In the ONNX pipeline, the distinction is clear:
`input_names` (user-provided data per step) vs everything else (weights/constants loaded once).

---

## 2026-03-20 — PT2 CUDA Search Rejects All Candidates: Integer Buffers Misinterpreted as Float NaN

### What the symptom was

`test_hf_llama_tiny` on CUDA via PT2 failed with:
`pyo3_runtime.PanicException: Failed to find a viable initial genome for group 0 after 100 attempts`

The search tried 100 different egglog rewrites and ALL were rejected by the `has_nan_outputs` check.

### What the actual root cause was

**Two issues, both required to fix:**

1. **Integer buffers misinterpreted as float in NaN check.** `has_nan_outputs` in
   `luminal_cuda_lite/src/runtime.rs` checks ALL `self.buffers` by reinterpreting raw bytes
   as `f32` and calling `is_nan()`. The PT2-translated graph has integer intermediate
   buffers (from `arange`, `cast(Int)`, integer arithmetic for embedding index computation).
   Certain valid `i32` bit patterns (e.g., large integers from `token_id * hidden_dim`)
   have exponent=0xFF and non-zero mantissa when reinterpreted as f32 — matching the
   IEEE 754 NaN pattern. This caused false NaN rejections for EVERY candidate genome.

2. **Real weights/constants loaded before search contain -inf.** The PT2 path loaded real
   safetensors weights and model constants (including the causal attention mask with `-inf`
   values) BEFORE the search. While the ONNX path also loads real initializer data before
   search, the PT2 graph's different structure (more explicit integer operations) made the
   integer NaN false-positive the blocking issue.

### Why it was hard to find

- The original plan diagnosed this as the same zero-dummy-data bug fixed on 2026-03-18.
  Changing `0.0` to `1.0` was insufficient because the root cause was different.
- `has_nan_outputs` checking ALL intermediate buffers (not just outputs) masked the real
  issue — the NaN was in integer index-computation buffers, not in the model's float outputs.
- The ONNX-translated graph didn't have this problem because it doesn't produce as many
  integer intermediate buffers (ONNX embedding uses different ops).
- The NaN pattern was identical across all 100 search attempts, which was the key clue:
  it was deterministic and independent of egglog rewrite choices, pointing to input data
  or buffer interpretation rather than graph optimization issues.

### The fix

Four changes:

1. **`luminal_cuda_lite/src/kernel/mod.rs`** (`KernelOp` trait): Added `output_dtype()`
   method with default `DType::F32`. Each kernel now reports its actual output dtype.

2. **`luminal_cuda_lite/src/kernel/hlir.rs`** and **`other_ops.rs`**: Overrode
   `output_dtype()` in all kernels with a `dtype` field (returns `self.dtype`), plus
   special cases: `KernelIota` → `DType::Int`, `KernelLessThan` → `DType::Bool`,
   `KernelCast` → `self.out_dtype`.

3. **`luminal_cuda_lite/src/runtime.rs`** (`has_nan_outputs`): Replaced fragile
   `format!("{:?}").contains("dtype: Int")` string matching with proper
   `op.to_dialect::<dyn KernelOp>().output_dtype()` check. Only F32 buffers are
   checked for NaN; integer and bool buffers are skipped.

4. **`rust/src/pt2_compiled_model.rs`** (`init_cuda_runtime`): Set ALL input nodes
   (weights, constants, user inputs) to `vec![1.0f32; n_elements]` before search via
   new `set_all_inputs_dummy_cuda` function, then reload real data after search.
   This prevents any -inf values from the causal mask from polluting intermediate
   float computations during profiling.

### General principle

**Never reinterpret integer buffer bytes as float for NaN checking.** When a graph has
mixed-dtype operations (float model computation + integer index computation), raw byte
buffers from integer kernels contain valid i32 values that look like NaN when cast to f32.
The search's `has_nan_outputs` must be dtype-aware — use the kernel's `output_dtype()`
method rather than string-matching on Debug output. Additionally, when diagnosing "all
candidates rejected" during search, check whether the rejection is from actual float NaN
or from dtype misinterpretation — the key diagnostic is whether the NaN pattern is
identical across all attempts (dtype issue) vs varying (actual numerical issue).

## 2026-03-25 — KernelExp/KernelSigmoid: Fused CUDA Kernels for Precision

1. **Symptom**: `test_hf_llama3_full` (16-layer Llama-3.2-1B) had ~1e-4 max diff vs PyTorch.
2. **Root cause**: `exp(x)` was computed as `exp2(x * 1.442695)` — the constant truncated by `{:.6}` format + extra multiply adds rounding. Sigmoid was 5 separate kernels. SumReduce had naive accumulation.
3. **Why hard**: Per-operation error was ~1e-7 but compounded over 16 layers × ~25 extra materializations. The egglog `Exp` rewrite depends on exact constant format matching.
4. **Fix**: Added `KernelExp` (uses `expf()`), `KernelSigmoid` (uses `1/(1+expf(-x))`), and Kahan summation in SumReduce. Each uses both `kernel_rewrite` and a direct egglog pattern match with range checks (e.g., `(> ?val 1.44) (< ?val 1.45)`) to bypass constant format dependency.
5. **Principle**: When decomposed CUDA kernel chains cause precision loss, add fused kernels via `kernel_rewrite`. For robustness, add BOTH the logical-op rewrite path AND a direct HLIR pattern match — the constant format in egglog can be fragile.
