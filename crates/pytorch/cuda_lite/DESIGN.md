# `luminal_cuda_lite` × PyTorch integration design

Status: **core implemented and GPU-verified**; the remaining items are listed in
"Not yet done".

## What is implemented today

- **Forked cudarc.** `luminal_cuda_lite` depends on the Luminal cudarc fork
  (`rev a42f389`) for `CudaContext::wrap_borrowed_stream` (non-owning stream).
- **Borrowed stream.** `CudaRuntime::use_borrowed_stream(raw)` /
  `use_owned_stream()`. The Python layer runs on a dedicated
  `torch.cuda.Stream` (the legacy default stream cannot host CUDA graph
  capture) and orders it against the caller's stream with
  `side.wait_stream(caller)` / `caller.wait_stream(side)`.
- **Torch-owned per-execution arena.** `CudaRuntime::arena_bytes()` returns the
  plan set's scratch requirement; the Python layer allocates exactly that with
  `torch.cuda.caching_allocator_alloc(bytes, device, side_stream)`, binds it via
  `set_arena(ptr, bytes)`, and calls `caching_allocator_delete(ptr)` immediately
  after `execute`. The runtime never frees it. A new base invalidates the
  compiled plan (it rebuilds) and forces arena-resident bytes to re-upload.
- **Zero-copy inputs.** `set_input_ptr(name, ptr, bytes, shape)` binds a graph
  input to the caller's device tensor; `range()` resolves it absolutely and the
  H2D upload step is skipped. Non-contiguous inputs fall back to host staging.
- **Zero-copy final outputs.** Every returned output is a PyTorch tensor bound
  with `set_output_ptr(index, ptr, bytes)`; the producing op writes it in place,
  the D2H readback is skipped, and the tensor is returned without a copy. A
  mutation sink is bound to the caller's input tensor (eager aliasing).
- **Parameter/weight binding.** At compile time, `ep.state_dict` tensors are
  moved to the input device once and bound zero-copy, so weights are not
  re-uploaded per execution.
- **Planner support for caller-owned buffers.** `ArenaPlan::externals` and the
  extra `external_buffers` parameter on `luminal::arena::plan_resident_over`
  let a planner exclude caller-owned buffers from packing entirely. The flag is
  threaded end-to-end: `CudaRuntime::set_external_outputs(true)` (set by the
  backend before `search`) → `CompileOptions::external_outputs` for the
  finalist budget → `storage::plan`/`plan_resident` → `resident::allocate`,
  which marks every output-slot buffer (that is not a resident mutation sink)
  external. Such buffers reserve no slab range, get no Download step, and the
  executor must resolve them through its external-pointer map. Metal passes
  `false`.

Verified on a GH200 (CUDA 13.2, torch 2.10) with `torch.compile`: a `Linear`
matches eager exactly across repeated calls with fresh inputs and per-call
arena alloc/free, and dynamic batch sizes (1/2/3/7) match eager. A unit test
(`storage::tests::external_outputs_leave_the_slab_and_require_caller_pointers`)
asserts an external output is absent from `slices`, present in `externals`, has
no Download step, and that excluding outputs strictly shrinks the slab.

## Not yet done

- **Async execution.** Every launch still `stream.synchronize()`s. On a
  borrowed side stream that is correct but serializing; removing it needs an
  explicit sync/host-readback discipline.

## Op coverage and dtypes (added after the initial integration)

- **Native `Select`** (`crates/luminal_cuda_lite/src/ops/select/`): the ternary
  `condition ? if_true : if_false` kernel. This closed the ReLU/GELU epilogue
  gap — `dynamic_cublas_bias_epilogue_rebinds_geometry`, a pre-existing failure
  on the clean baseline, now passes. The kernel copies a value verbatim, so it
  is dtype-generic.
- **F16/BF16/F64**: `dtype_bytes`, `cuda_type`, and readback now cover
  half/bfloat/double. Generated kernels prepend `cuda_fp16.h`/`cuda_bf16.h`
  when the dtypes appear. NVRTC does not ship those headers, so
  `crates/luminal_cuda_lite/build.rs` locates the build machine's toolkit
  header closure and **embeds it into the binary**; at runtime it is
  materialized to a temp directory and appended to NVRTC's include path after
  any real runtime toolkit. A machine with only the CUDA driver can therefore
  run half/bfloat kernels. `LUMINAL_CUDA_HEADERS_DIR` overrides the source
  tree (for a pinned/vendored copy). `recip` divides half inputs in float to
  avoid the `float / __half` overload ambiguity. Verified end-to-end for
  add/mul/div/recip/modulo/floor/ceil/round/trunc/sqrt/exp/sin/where/relu/
  reduce on F16, BF16, and F64.

The rest of this document records the reconnaissance and the reasoning behind
the seams, which remain accurate.

---

## 1. What we are integrating with

PyTorch already owns, per process:

- **A CUDA primary context per device**, retained on first use
  (`cuDevicePrimaryCtxRetain`). `torch.cuda.current_stream()` is a real
  `CUstream` on that context; `torch.cuda.Stream` exposes `.cuda_stream` (an
  `int`) and `torch.cuda.current_stream().cuda_stream` is the handle to borrow.
- **The caching allocator**, exposed to extensions as
  `torch.cuda.caching_allocator_alloc(size, device=None, stream=None)` →
  device pointer (int), and `torch.cuda.caching_allocator_delete(ptr)`.
  Allocating through it means PyTorch accounts for the bytes, so it can make
  room (evict/reuse) instead of the process OOMing behind PyTorch's back.
- **CUDA graph capture state** (`torch.cuda.graphs.CUDAGraph`,
  `torch.cuda.graph`), which we must not capture through accidentally.

The Luminal runtime must stop being a self-contained island inside that process.

## 2. Current runtime facts (the starting point)

From `crates/luminal_cuda_lite`:

- `CudaRuntime` (`src/runtime.rs:47`) owns its device in a **private**
  `device: Option<CudaDevice>` (`runtime.rs:109`), created lazily as
  `CudaDevice::new(0)` on the first device-profiled `search`
  (`runtime.rs:573`) or the first `execute` (`runtime.rs:788`). There is no
  injection seam.
- `CudaDevice::new` (`src/device.rs:83`) does
  `CudaContext::new(ordinal)` + `ctx.new_stream()`, and allocates the arena as
  `self.stream.alloc_zeros::<u8>(bytes)` (`device.rs:170`), recording
  `stats.arena_base` (`device.rs:174`).
- **Good news:** cudarc 0.19.8's `CudaContext::new` *is*
  `cuDevicePrimaryCtxRetain`, i.e. the **same primary context PyTorch uses**.
  Context sharing is therefore mostly free — we must not call
  `from_raw_context` (its `Drop` destroys the context).
- **Blocker:** cudarc 0.19.8 has **no raw/borrowed-stream constructor**. Only
  `default_stream`, `per_thread_stream`, `new_stream`, `fork`. A
  `CudaStream` it creates is owned and destroyed on drop. The Luminal cudarc
  fork (rev `a42f389`, v0.19.9) adds
  `CudaContext::wrap_borrowed_stream(CUstream) -> Arc<CudaStream>` with an
  `owned: bool` so the borrowed stream is not destroyed — this is the hook we
  need.
- **All I/O is host-staged.** `set_data` takes `impl Into<HostBuffer>`
  (`runtime.rs:769`) and `HostBuffer` is `{ dtype, bytes: Vec<u8> }`
  (`src/host_buffer.rs:26`). Execution copies H2D through one pinned staging
  allocation, and D2H's every output back (`device.rs:1045-1080`). Outputs are
  read from a host `HostBuffer` (`get_f32`/`get_i32`/`get_i64`/`get_bool8`,
  `runtime.rs:870-891`). Readable dtypes are only F32, Int(i32), Int64, Bool8.
- **Everything synchronizes.** `Executable::launch` is followed by
  `stream.synchronize()` (`device.rs:1064-1068`); every public launch is
  synchronous, including error paths.
- Stable keys exist for wiring external buffers: inputs are identified by
  `BufferLit` id and `NodeIndex` (`input_buffer`, `runtime.rs:735`;
  `CompiledPlan.inputs: Vec<Input{lit, ...}>`, `device.rs:399`), outputs by slot
  index (`output_slot_index`, `runtime.rs:741`). The single place an arena
  offset becomes a device address is `range(...)` (`device.rs:436-453`), which
  already takes a raw `base: u64` — externalizing the arena is tractable.

## 3. Design

### 3.1 CUDA context: share the primary context, don't create one

- Keep `CudaContext::new(ordinal)` semantics (primary retain). It is already the
  same context PyTorch uses. Do **not** use `from_raw_context` on PyTorch's
  context.
- The pyo3 wrapper reads the device ordinal from the caller's tensors
  (`tensor.device.index`), and the backend must call `torch.cuda.set_device` /
  bind the context on the calling thread before any runtime call
  (`CudaDevice::execute` already does `ctx.bind_to_thread()`).
- Multi-device: one `CudaRuntime` is single-device; the Python `CompiledModel`
  should refuse inputs whose device differs from the compiled device rather than
  silently copying. (Today's scaffold already refuses mixed devices.)

### 3.2 Stream: borrow PyTorch's current stream

- Add a borrowed-stream constructor on `CudaDevice` (requires the cudarc fork):
  ```rust
  #[cfg(feature = "device")]
  pub unsafe fn with_borrowed_stream(
      ctx: Arc<CudaContext>,
      raw_stream: sys::CUstream,
      arena: Option<ExternalArena>,
  ) -> Result<Self>;
  ```
  `wrap_borrowed_stream` must produce `owned = false`, so Luminal never calls
  `cuStreamDestroy` on PyTorch's stream.
- The Python side passes `torch.cuda.current_stream().cuda_stream` at compile /
  first-execute time. Two consequences to design for, not paper over:
  1. **Stream can change between calls.** PyTorch's "current stream" is
     thread-local and can be swapped. A long-lived `CompiledGraph` holding one
     borrowed stream is only correct while the caller stays on that stream.
     Options: (a) rebind the stream each call via a cheap
     `set_stream(raw)` on the runtime (preferred — no recompile), or (b) hold
     the stream and document that callers must not switch, which breaks under
     `torch.cuda.stream(...)` contexts. **Choose (a).**
  2. **Synchronization discipline.** On a borrowed stream, absolute
     `stream.synchronize()` after every launch serializes the whole device and
     defeats interleaving. The runtime should get an async mode: launch and
     *record* work on the borrowed stream, and synchronize only when the caller
     asks (host readback, timing, or explicit `synchronize()`). Because
     torch ops on the same stream are ordered, an output tensor whose data was
     written by Luminal is safe for subsequent PyTorch ops without a host sync.

### 3.3 Arena: allocate through PyTorch's caching allocator

The arena slab is the big, persistent allocation (`device.rs:166-178`). It must
not be a private `cuMemAlloc`/`cuMemAllocAsync` allocation that PyTorch cannot
see.

- Allocate it host-side as
  `torch.cuda.caching_allocator_alloc(bytes, device, stream)`, and hand the raw
  pointer + size to the runtime:
  ```rust
  pub struct ExternalArena { pub ptr: u64, pub bytes: usize }

  pub fn install_resident_with_arena(
      &mut self,
      plans: Vec<(CudaPlan, Bounds)>,
      bindings: ResidentBindings,
      budget: Option<usize>,
      arena: ExternalArena,
  ) -> Result<()>;
  ```
  `range()` (`device.rs:436`) already consumes `base: u64`, so kernel arguments
  and copies need no change. Only three sites currently touch the slab:
  allocation (`device.rs:170`), `slab_bytes` (`device.rs:101`), and
  `release_slab` (`device.rs:228`).
- **Ownership must be explicit.** Today `slab: CudaSlice<u8>` frees on drop.
  With an external arena it must be `Option<ExternalArena>` with
  `owned = false`; `release_slab`/`Drop` must *not* free caller memory. The
  Python owner frees with `caching_allocator_delete` when the
  `CompiledModel` is collected.
- **Sizing.** `CompileOptions::device_budget_bytes` (`search.rs:126`) already
  bounds the slab the search will install; the Python layer should set it from
  the allocator (e.g. a fraction of `torch.cuda.mem_get_info()` free memory) so
  the search refuses plans that would overcommit instead of OOMing later. Note
  the runtime grows its slab monotonically and only replaces it when a larger
  plan set is installed (`device.rs:166`), so the budget should be computed
  once per install and the arena treated as `max` across buckets.
- **Pinned staging stays.** `cuMemHostAlloc` staging (`cuda_graph.rs:162`) is
  orthogonal to torch's allocator and can remain; it is host memory. If host
  memory accounting matters, it can later move to
  `torch.empty(..., pin_memory=True)`-style storage, but that is not required
  for correctness.

### 3.4 Zero-copy I/O: bind caller device pointers

This is the payoff; without it every call does H2D + D2H and the CUDA graph is
pointless for small ops.

**Inputs.** A caller's tensor is already on the device. Instead of
`set_input(bytes)`, pass `tensor.data_ptr()` and let the plan's H2D node become
either a no-op (direct operand) or an on-device copy if the plan requires a
different layout:

```rust
pub unsafe fn set_input_device_ptr(
    &mut self, tensor: NodeIndex, device_ptr: u64, n_bytes: usize,
) -> Result<()>;
pub fn clear_input_device_ptr(&mut self, tensor: NodeIndex) -> Result<()>;
```

Requirements to enforce, not assume:

- **Exact byte size and dtype match** against the plan's binding. Reject
  otherwise; a wrong-width pointer is silent corruption.
- **Contiguity/layout.** Luminal's elected layouts assume a logical→physical
  map. A non-contiguous (e.g. transposed / sliced) torch tensor's
  `data_ptr()` is not the dense buffer the plan assumes. The Python layer must
  either require contiguous inputs or pass strides so the plan can select a
  view layout; the safe first cut is `tensor.is_contiguous()` check →
  `.contiguous()` fallback (with a copy).
- **Lifetime.** The pointer is only valid while the caller's tensor is alive and
  not freed/resized. Brief: the binding is valid for one `execute` and cleared
  after; do not retain a raw pointer across Python calls.
- **Async ordering.** If the caller produced the tensor on the borrowed stream,
  no sync is needed (same stream). If it came from another stream, PyTorch's
  own stream semantics apply; the backend must not insert a global sync.

**Outputs.** Two modes:

1. **Allocator-backed outputs.** The Python layer allocates the output tensor
   with `torch.empty(shape, dtype, device)` (so the caching allocator owns it),
   passes its `data_ptr()` in, and the runtime writes directly into it. This is
   the natural fit for the pyo3 backend because Python already builds tensors
   from returned bytes today (`backend.py::_output_tensor`).
2. **Escape-and-disclose outputs.** For view-elected outputs (non-dense), the
   runtime currently returns the backing bytes + binding (`fetch`,
   `runtime.rs:896`). Zero-copy here would hand back a torch tensor viewing the
   arena with the elected layout; this is only safe if the arena outlives the
   returned tensor, so it needs an ownership story (D2D copy out is the safe
   default).

```rust
pub unsafe fn set_output_device_ptr(
    &mut self, tensor: NodeIndex, device_ptr: u64, n_bytes: usize,
) -> Result<()>;
pub fn clear_output_device_ptr(&mut self, tensor: NodeIndex);
pub fn output_is_zero_copy(&self, tensor: NodeIndex) -> bool;
```

**Mutations / aliasing.** `.output_into()` results alias an input's storage.
Under zero-copy the aliasing must be expressed in torch's terms (the returned
tensor *is* the mutated input, as `backend.py` already does for the reference
backend via `copy_`), and the runtime's resident-binding path
(`retain_input`, `runtime.rs:753`) is the natural place for weights that should
live in the arena across calls.

### 3.5 CUDA graph capture and PyTorch

The runtime always launches captured CUDA graphs. If PyTorch is itself
capturing (e.g. `torch.cuda.graph`), nested capture is illegal. The backend
must:

- never capture on a stream that PyTorch has marked capturing, and
- prefer to replay a graph it captured itself on the borrowed stream, which is
  legal as long as no capture is active.

This is a Python-level check (`torch.cuda.is_current_stream_capturing()`) plus a
runtime-level refusal, and it should be tested.

### 3.6 Dtype coverage

The runtime can stage/read F32, Int, Int64, Bool8 only. A CUDA backend for real
models needs F16/BF16 (and likely F8) at minimum. This is a runtime limitation,
not a Python one; until it lands, `plan_dtype`/`output_bytes` in `src/lib.rs`
refuse loudly rather than silently reinterpreting bytes. The Python layer should
surface that as a `torch.compile` graph break / fallback rather than a crash.

## 4. Proposed Rust seams (ordered by dependency)

| # | Seam | Purpose |
|---|------|---------|
| 1 | `CudaDevice::with_borrowed_stream(ctx, CUstream, arena)` | borrow torch's stream; requires cudarc fork |
| 2 | `CudaDevice::from_parts(ctx, stream, arena)` | inject primary context + stream |
| 3 | `ExternalArena { ptr, bytes }` + `install_resident_with_arena` | torch-allocator arena; `release_slab`/`Drop` must not free |
| 4 | `CudaRuntime::set_device` / `load_with_device` | device field is private |
| 5 | `set_input_device_ptr` / `clear_input_device_ptr` | zero-copy inputs |
| 6 | `set_output_device_ptr` / `copy_output_to_device_ptr` | zero-copy outputs |
| 7 | `set_stream(raw)` + async `execute` + `synchronize()` | stream rebinding; stop per-launch sync |

Items 1–3 are the minimum for "integrate correctly with the allocator and
context"; 5–7 are the performance work. Precedent for the whole surface exists
in the parked HLIR backend
(`crates/luminal_cuda_lite_hlir/src/dyn_backend.rs`,
`.../runtime.rs`: `set_device_ptr`, `set_output_device_ptr`,
`output_is_zero_copy`, `copy_output_to_device_ptr`, `use_borrowed_stream`,
`use_owned_stream`).

## 5. Phasing

1. **Context + stream + arena (this design's core).** Borrow torch's stream,
   allocate the arena via `caching_allocator_alloc`, keep H2D/D2H staging.
   Correctness: no double-free, no context destruction, work ordered on torch's
   stream. *No zero-copy yet.*
2. **Zero-copy inputs**, with contiguity + size/dtype guards.
3. **Allocator-backed outputs**, then async execute (remove per-launch sync).
4. **Resident weights** for repeated calls (LLM decode), via `retain_input`.
5. **Dtype coverage** (F16/BF16) on the runtime side.

## 6. Risks / open questions

- **cudarc fork dependency.** Borrowed streams require the Luminal cudarc fork.
  Switching `luminal_cuda_lite` to it is a dependency change that should be
  validated against the current usage; alternatively, vendor a minimal
  `owned:false` stream wrapper. This is the single biggest blocker.
- **Failure/OOM behavior.** With the arena in torch's allocator, an OOM during
  `install` should raise a torch-visible `OutOfMemoryError` (or a Luminal error
  the Python layer maps to graph-break/fallback), not abort.
- **Stream-switching correctness.** Rebinding the stream per call is necessary
  but must be cheap; a cached executable captured on stream A and replayed on
  stream B is only valid if the capture used no cross-stream dependencies.
- **Arena lifetime vs returned tensors.** Zero-copy outputs that view the arena
  cannot outlive the `CompiledModel`; D2D copy-out is the safe default until a
  refcounted arena exists.
- **Dynamic shapes.** Bucketed plans size the arena to the largest bucket;
  the allocator allocation must use that same max, and `device_budget_bytes`
  must be set before `search` (it is a search-time constraint, `search.rs:126`).

## 7. Shared export pipeline

The `torch.export` preparation in `luminal_reference/export_utils.py`
(dynamic-shape capture, scalar-output boxing, SDPA-preserving decomposition
table, `sym_sum` serde workaround, guard/dead-op cleanup) is backend-neutral.
The `luminal_cuda_lite` backend imports it from the reference package rather
than duplicating 340 lines. This is an acknowledged coupling; the clean
follow-up is to lift that module into a backend-neutral Python package consumed
by both backends. It is called out here so it is a conscious choice, not an
accident.
