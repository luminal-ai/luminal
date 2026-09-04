//! CL-2: the device executor — the reference runtime's four execution
//! phases reimplemented over cudarc, consuming the identical
//! `BufferIrGraph`.
//!
//! PHASE 3 OF THE #420/#422 REJOIN (2026-09-03) rebuilt two things
//! here:
//!
//! * THE DEVICE IS PERSISTENT. [`CudaDevice`] holds the context, the
//!   one stream, the NVRTC module cache, and the arena slab, and the
//!   runtime owns it across calls. Before, every `execute` built a
//!   fresh context and a fresh kernel cache and recompiled every
//!   kernel.
//! * MEMORY COMES FROM AN ARENA. Phase 1 used to `alloc_zeros` one
//!   slice per plan buffer, all live for the whole call — the sum of
//!   every buffer the plan names. Now [`crate::arena`] reads the
//!   plan's own `BufferAlloc`/`BufferFree` lifetimes, picks an issue
//!   order that keeps few buffers live at once, and assigns each
//!   INTERIOR buffer a range of ONE runtime-owned slab, sized to the
//!   high-water mark and grown (never shrunk) across calls. Boundary,
//!   escaping and donated storage keep their own allocations — see the
//!   ownership-row table on [`crate::arena`].
//!
//! The phases themselves are unchanged in kind. Phase 1 materializes
//! the standalone rows and stages inputs (loud on missing
//! geometry/dtype, exactly like the reference). Phase 2 is the arena's
//! issue order — a topological order over Data AND Anti edges, so WAR
//! ordering is enforced by construction, chosen for a small high-water
//! mark. Phase 3 dispatches: `BufferAlloc` binds its buffer to its slab
//! range, `BufferFree` drops the binding, D2D for copies,
//! NVRTC-compiled launches for compute (the destination is the range
//! the planner assigned — no longer a fresh zeroed slice; every CL
//! kernel writes every element it owns, see the KERNEL INVARIANT note
//! below). Phase 4 copies each output SLOT's backing buffer back to a
//! host `HostBuffer`, keyed by slot index and paired with the slot's
//! [`OutputBinding`] — the escape-and-disclose contract (ruling
//! 2026-08-27): the caller gets the backing bytes (possibly
//! parent-sized, for an escaped view election) plus the layout to
//! interpret them under.
//!
//! # THE KERNEL INVARIANT (why an unzeroed destination is safe)
//!
//! A recycled slab range arrives holding the previous occupant's
//! bytes. That is sound iff no kernel READS its destination before
//! writing it, and every kernel writes every element it owns. Audited
//! 2026-09-03, one family at a time:
//!
//! * ELEMENTWISE (`kernels::binary`/`unary` and every op that lowers
//!   through them), CAST, CONSTANT, IOTA, GATHER,
//!   INDEXMAPAPPLYMATERIALIZE, COPY: one thread per destination
//!   element, `out[i] = <expr>` over `n = numel(dest_dims)`. The
//!   destination is never an operand of the launch (the executor
//!   passes `reads[..reads.len()-writes.len()]`, dropping the DPS dest
//!   operand), so it cannot even be read.
//! * REDUCE: `out[i] = acc` over `n = outer*inner` — the whole
//!   destination, `acc` seeded from `init` and folded over the input.
//! * SCATTER: TWO launches. The first is `out[i] = init[...]` over the
//!   FULL destination numel (`init_dims != dest_dims` is a bail), so
//!   the destination is completely written before the second launch
//!   scatters into it. Same stream, so the phases are ordered.
//! * CUBLASLT D: `beta = 0` on the non-fold forms, which is the BLAS
//!   skip — C (aliased to D) is not read, D is fully written. The
//!   C-fold forms read their C operand at `beta = 1`; C is a DEFINED
//!   resident — a distinct buffer, or D's own range when the bufferizer
//!   seeded D onto C's ReadWrite caller buffer through the May permit —
//!   never an undefined recycled range.
//!
//! So no memset is emitted anywhere. The one standing assumption is
//! that a destination's `numel(dest_dims)` covers its buffer's SPAN:
//! true for every codegen'd kernel because the elected destination
//! layout must be right-major contiguous (the egglog write-capability
//! guard, 2026-09-01) and for cuBLASLt because `bind_destination`
//! refuses anything but the two dense orders. A future non-dense
//! destination would leave the span's tail holding the previous
//! occupant's bytes, and would need the memset this note says we do
//! not do.

use crate::arena::{buffer_bytes, plan_arena, ArenaPlan};
use crate::host_buffer::HostBuffer;
use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    result as cu, CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind, OutputBinding};
use luminal::dtype::PlanDtype;
use luminal::prelude::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

use crate::kernels::{codegen_for, CodegenCtx};

/// STAGING AND READBACK are memcpys now (ruling D4, 2026-09-03): a
/// [`HostBuffer`] IS bytes plus a dtype tag, which is exactly what an
/// H2D/D2H copy wants. The eleven-variant match these two used to be
/// went with `TypedBuffer` to the reference runtime, where kernels
/// really do need typed Rust slices.
fn typed_to_bytes(data: &HostBuffer) -> &[u8] {
    &data.bytes
}

/// D2H: the device's bytes under the plan's dtype. Boolean readback
/// still passes the VALIDATED door — a device that wrote a byte other
/// than 0x00/0x01 into a Bool8 buffer has broken the two-legal-codes
/// contract, and this is where that becomes visible.
fn bytes_to_typed(bytes: &[u8], dtype: PlanDtype) -> Result<HostBuffer> {
    match dtype {
        PlanDtype::Bool | PlanDtype::Bool8 => HostBuffer::bool8(bytes.to_vec()),
        other => HostBuffer::new(other, bytes.to_vec()),
    }
}

struct KernelCache {
    ctx: Arc<CudaContext>,
    modules: HashMap<u64, (Arc<CudaModule>, CudaFunction)>,
}

impl KernelCache {
    fn function(&mut self, source: &str) -> Result<CudaFunction> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((_, func)) = self.modules.get(&key) {
            return Ok(func.clone());
        }
        let ptx =
            compile_ptx(source).map_err(|e| anyhow!("NVRTC failed: {e:?}\nsource:\n{source}"))?;
        let module = self.ctx.load_module(ptx).context("module load")?;
        let func = module.load_function("k").context("entry `k` missing")?;
        self.modules.insert(key, (module, func.clone()));
        Ok(func)
    }
}

/// THE PERSISTENT DEVICE: everything an execution needs that should
/// outlive one call — the context, the one stream every kernel and
/// copy is issued on, the compiled-module cache, and the arena slab.
/// The runtime owns exactly one of these and hands it to
/// [`execute_plan`] by `&mut`.
///
/// The slab is GROW-ONLY and never parked (#401 as amended by #422):
/// one runtime-owned allocation, resized upward when an installed plan
/// needs more than it holds. SERVING never releases it between
/// [`CudaRuntime::execute`](crate::CudaRuntime::execute) calls; the
/// SEARCH is the one exception — it releases the slab after every
/// profiled candidate through [`Self::release_slab`] (#422's search-time
/// policy), and the next [`execute_plan`] re-allocates through
/// `ensure_slab`. Nothing has to be invalidated on a grow — CL captures
/// no CUDA graphs and holds no device pointers across calls.
pub struct CudaDevice {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cache: KernelCache,
    slab: Option<CudaSlice<u8>>,
}

impl CudaDevice {
    /// Bind device `ordinal` and take its default stream.
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).with_context(|| format!("no CUDA device {ordinal}"))?;
        let stream = ctx.default_stream();
        Ok(CudaDevice {
            cache: KernelCache {
                ctx: ctx.clone(),
                modules: HashMap::new(),
            },
            ctx,
            stream,
            slab: None,
        })
    }

    /// Grow the slab to at least `bytes`. The old slab is released
    /// BEFORE the new one is taken, so a grow never needs both at once.
    fn ensure_slab(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 || self.slab.as_ref().map(|slab| slab.len()).unwrap_or(0) >= bytes {
            return Ok(());
        }
        self.slab = None;
        self.slab = Some(
            self.stream
                .alloc_zeros::<u8>(bytes)
                .with_context(|| format!("arena slab alloc, {bytes} bytes"))?,
        );
        Ok(())
    }

    /// The slab's current size in bytes (0 before the first plan needs
    /// one) — the runtime's resident device footprint.
    pub fn slab_bytes(&self) -> usize {
        self.slab.as_ref().map(|slab| slab.len()).unwrap_or(0)
    }

    /// RELEASE THE SLAB — the SEARCH-TIME hygiene (#422 policy, Phase
    /// 4, reversing #401's retention for this one caller):
    /// [`crate::search`] calls this after every profiled candidate, so a
    /// candidate whose arena high-water mark is outsized cannot hold
    /// that memory for the rest of the search and starve its successors.
    /// The next [`execute_plan`] re-allocates through `ensure_slab`.
    ///
    /// SERVING NEVER CALLS IT. `CudaRuntime::execute` keeps the slab
    /// exactly as Phase 3 landed it: one grow-only allocation for the
    /// runtime's life, which is the point of the persistent device.
    /// Nothing else is released here — the context, the stream and the
    /// NVRTC module cache all survive, which is what keeps kernel
    /// compilation a once-per-source cost across a whole search.
    pub fn release_slab(&mut self) {
        self.slab = None;
    }
}

/// A bound buffer: where its bytes are and how many there are. Derived
/// either from an owned [`CudaSlice`] (the standalone and donated rows)
/// or from the slab base plus an [`crate::arena::ArenaSlice`] offset.
///
/// The executor works in raw device pointers rather than cudarc's
/// slices/views because a slab range is a BORROW of one allocation:
/// with many ranges live at once — several read by a launch that also
/// writes one — no set of `CudaView`/`CudaViewMut` handles can coexist
/// under the borrow checker. Pushing the pointer as a kernel argument
/// is ABI-identical to pushing a `&CudaSlice` (cudarc pushes exactly
/// this `CUdeviceptr`), and the stream-event bookkeeping those handles
/// carry is inert here: CL issues everything on one stream, so
/// `is_managing_stream_synchronization()` is false and no event is ever
/// recorded. A multi-stream executor would owe those events.
#[derive(Debug, Clone, Copy)]
struct Bound {
    ptr: u64,
    bytes: usize,
}

fn bound_of(bindings: &FxHashMap<BufferId, Bound>, id: &BufferId, who: &str) -> Result<Bound> {
    bindings.get(id).copied().ok_or_else(|| {
        anyhow!(
            "{who}: buffer {id:?} has no live binding — it was never allocated, \
             or its BufferFree already ran"
        )
    })
}

/// Execute a bufferized plan on `device`. Returns, per output slot
/// index, a host copy of the slot's BACKING buffer plus its
/// [`OutputBinding`] (the elected layout) — the escape-and-disclose
/// fetch, universal over dense and view elections.
///
/// `staged` is a map of BORROWED payloads by BufferLit id (Phase 4). It
/// used to hold the payloads themselves, which was fine while the only
/// caller was the serving ladder — the runtime already owns them. The
/// search now stages too, and it stages the CALLER's map, which for a
/// full-size model is gigabytes of weights: a map of references costs
/// one pointer per input and no copy at all.
pub fn execute_plan(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
) -> Result<FxHashMap<usize, (HostBuffer, OutputBinding<luminal::layouts::DecodedLayout>)>> {
    // ESCAPE GUARD (ruling 2026-08-27): an output slot's backing storage
    // must SURVIVE the call — FreedBy::Caller, whatever the owner.
    // FreedBy::Program backing an output hands the caller bytes the
    // program destroys: minted non-escaping storage (Owner::System) and
    // DONATED boundary storage (Owner::Caller — validate()'s donated arm
    // forbids exactly this plan shape) alike. The pre-lowering
    // certificate enforces this for planner-built plans; hand-built /
    // externally loaded plans never met it — re-check here, loudly,
    // before any bytes move.
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let buffer = plan
                    .buffers
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("output slot {} names unknown buffer", slot.index))?;
                if buffer.freed_by != luminal::layout_ir::FreedBy::Caller {
                    bail!(
                        "output slot {} is backed by NON-ESCAPING buffer {} \
                         (FreedBy::Program, {:?}-owned) — escaped output storage \
                         must be FreedBy::Caller; refusing to hand the caller bytes \
                         the program destroys",
                        slot.index,
                        buffer.label,
                        buffer.owner,
                    );
                }
            }
        }
    }

    // Phase 2, hoisted: the arena decides the issue order AND the slab
    // layout in one device-free pass (Anti edges ride it, so WAR
    // ordering is enforced by construction). Everything below walks
    // `arena.order`.
    let arena = plan_arena(plan, buffer_bytes).context("arena planning")?;
    debug_assert!(plan
        .dag
        .edge_weights()
        .all(|e| matches!(e.kind, EdgeKind::Data | EdgeKind::Anti)));
    device.ensure_slab(arena.slab_bytes)?;
    let stream = device.stream.clone();
    let slab_base = match &device.slab {
        Some(slab) => {
            let (base, _record) = slab.device_ptr(&stream);
            base
        }
        None => 0,
    };

    // Phase 1: materialize the buffers that do NOT come from the slab —
    // the BOUNDARY and ESCAPING rows (their bytes are the caller's after
    // the call) and the DONATED row (the caller's bytes, which in CL
    // means the staged payload's device copy). Slab members stay unbound
    // until their `BufferAlloc` is issued.
    let mut owned: FxHashMap<BufferId, CudaSlice<u8>> = FxHashMap::default();
    let mut bindings: FxHashMap<BufferId, Bound> = FxHashMap::default();
    let mut geometry: FxHashMap<BufferId, PlanDtype> = FxHashMap::default();
    for (id, buffer) in &plan.buffers {
        let dtype = buffer.layout.dtype.ok_or_else(|| {
            anyhow!(
                "buffer {:?} (backing {}) carries no dtype fact",
                buffer.label,
                buffer.backs
            )
        })?;
        geometry.insert(id.clone(), dtype);
    }
    for id in arena.standalone.iter().chain(arena.donated.iter()) {
        let buffer = &plan.buffers[id];
        let bytes = buffer_bytes(buffer)?;
        let mut slice = stream
            .alloc_zeros::<u8>(bytes.max(1))
            .with_context(|| format!("device alloc {} bytes for {:?}", bytes, buffer.label))?;
        if let Some(lit) = buffer.lit {
            if let Some(data) = staged.get(&lit) {
                let host = typed_to_bytes(data);
                if host.len() != bytes {
                    bail!(
                        "staged buffer {lit} is {} bytes, plan expects {bytes} for {:?}",
                        host.len(),
                        buffer.label
                    );
                }
                stream.memcpy_htod(host, &mut slice).context("H2D")?;
            }
        }
        let ptr = {
            let (ptr, _record) = slice.device_ptr(&stream);
            ptr
        };
        bindings.insert(id.clone(), Bound { ptr, bytes });
        owned.insert(id.clone(), slice);
    }

    // CONTRACT-1 (bind-time), NARROWED. Distinct BufferIds must be
    // backed by disjoint device ranges — folded-view reads and WAR
    // ordering are both keyed on BufferId identity. This assert covers
    // the allocations the EXECUTOR makes (the standalone and donated
    // rows), which is the surface it was written for: "when raw caller
    // pointers arrive at this binding surface". It can no longer be a
    // whole-plan check, because slab members are sub-ranges of ONE
    // allocation by design — for them the question is whether two
    // SIMULTANEOUSLY LIVE ranges overlap, which is decidable at
    // planning time and is checked there (see the CONTRACT-1 live-range
    // note in `crate::arena`).
    {
        let bound: Vec<crate::binding_check::BoundRange> = owned
            .iter()
            .map(|(id, slice)| crate::binding_check::BoundRange {
                buffer: format!("{id:?}"),
                base: bindings[id].ptr,
                bytes: slice.len() as u64,
            })
            .collect();
        crate::binding_check::assert_disjoint(&bound).context("CONTRACT-1 bind-time check")?;
    }

    if std::env::var_os("LUMINAL_CL_ARENA").is_some() {
        arena_report(plan, &arena, device.slab_bytes());
    }

    // Phase 3: dispatch, in the arena's issue order.
    for node in arena.order.iter().copied() {
        match &plan.dag[node] {
            BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => {}
            BufferNode::BufferCopy { src, dst } => {
                // THE BUFFERCOPY CONTRACT, executor side (Austin, ruled
                // 2026-08-31 — see `bufferize::BufferNode::BufferCopy`):
                //
                // * The node carries ONLY {src, dst}.
                // * Semantics: a DUMB EXACT-SIZE WHOLE-BUFFER copy — one
                //   `memcpy_dtod` of the whole slice, no layout awareness,
                //   no element walk. "If a runtime chooses to do resource
                //   reuse and do unequal sized buffer that is an entirely
                //   runtime owned choice"; CL sizes both ends from the same
                //   span-of-layout rule and makes no such choice, so
                //   unequal lengths are a bug HERE and bail loudly (this is
                //   the executor's own discipline over bufferizer-authored
                //   nodes, NOT a type fence re-checking an e-graph premise).
                // * ORDERING IS THIS RUNTIME'S OBLIGATION. The plan supplied
                //   dependency structure only (data + WAR anti-edges); we
                //   discharge it by issuing in the arena's topological order
                //   onto ONE stream, which serializes the copy against every
                //   op that depends on it and every prior reader of `dst`. A
                //   multi-stream executor would owe events/barriers here.
                // * The three causes (conflict repair, boundary placement,
                //   lifetime repair) are the bufferizer's business; all
                //   three execute identically.
                let from = bound_of(&bindings, src, "copy src")?;
                let to = bound_of(&bindings, dst, "copy dst")?;
                if from.bytes != to.bytes {
                    bail!("copy length mismatch: {} -> {} bytes", from.bytes, to.bytes);
                }
                unsafe { cu::memcpy_dtod_async(to.ptr, from.ptr, from.bytes, stream.cu_stream()) }
                    .context("D2D copy")?;
            }
            BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                let label = op.label();
                // THE ARENA'S TWO EVENTS. An alloc binds its buffer to
                // the range the planner assigned it (nothing is zeroed —
                // see the KERNEL INVARIANT note at the top of this
                // module); a free drops the binding, and the range is
                // already back on the planner's free list, waiting for
                // the next alloc that fits. A buffer the planner did not
                // put in the slab (standalone, donated, or an interior
                // buffer demoted for want of a lifetime pair) is bound
                // from Phase 1 and its alloc is a no-op.
                if label == "BufferAlloc" {
                    if let Some(buffer) = writes.first() {
                        if let Some(slice) = arena.slices.get(buffer) {
                            bindings.insert(
                                buffer.clone(),
                                Bound {
                                    ptr: slab_base + slice.offset as u64,
                                    bytes: slice.bytes,
                                },
                            );
                        }
                    }
                    continue;
                }
                if label == "BufferFree" {
                    if let Some(buffer) = reads.first() {
                        bindings.remove(buffer);
                    }
                    continue;
                }
                if writes.len() != 1 {
                    bail!(
                        "{label}: CL handles single-destination ops, got {}",
                        writes.len()
                    );
                }
                let dest = bound_of(&bindings, &writes[0], label)?;
                let input_count = reads.len().saturating_sub(writes.len());
                let inputs: Vec<Bound> = reads[..input_count]
                    .iter()
                    .map(|id| bound_of(&bindings, id, label))
                    .collect::<Result<_>>()?;

                // Train 3: the HOST-CALL arm — cuBLASLt contracts
                // dispatch as one `cublasLtMatmul` library call on the
                // SAME stream as the surrounding kernels, never an
                // NVRTC kernel. The destination is the arena range the
                // planner assigned (it was a fresh zeroed slice before
                // Phase 3 of the rejoin); the C-fold forms read their C
                // operand buffer and write D at beta = 1.0f. C is USUALLY
                // a distinct live range, but when the program binds D's
                // output slot onto the same ReadWrite caller buffer that
                // holds C, the seed is admitted through CublasLtDps's May
                // permit on operand 2 and C == D — legal, because
                // `bind_destination` emits identical C and D descriptors,
                // which is the API's C == D precondition. (Recorder-
                // produced programs never bind an output onto an input
                // buffer, so this arises only for hand-authored
                // boundaries.) The non-fold forms run beta = 0, which is
                // the BLAS skip, so D's prior bytes are never read.
                if let Some(dps) = op
                    .as_any()
                    .downcast_ref::<crate::ops::cublaslt::CublasLtDps>()
                {
                    let mut call = crate::ops::cublaslt::exec::plan_call(&dps.op)
                        .with_context(|| format!("cuBLASLt call planning for {label}"))?;
                    // THE PLAN/CALL-FRAME COHERENCE FENCE — restored,
                    // strengthened, and CLASSIFIED (2026-08-31; the full
                    // taxonomy lives on `exec::bind_destination`).
                    //
                    // Correction 4 of the Option-B landing deleted the
                    // `[m, n]` frame check here as "runtime type-checking
                    // ... rule premises, not our business". THAT
                    // CLASSIFICATION WAS WRONG, and this is the note that
                    // keeps the next cleanup from repeating it:
                    //
                    //  * An E-GRAPH RE-CHECK restates a fact a rule
                    //    premise guarantees (F32 scope; C's dims/fold
                    //    matching D's by rule guard). Those are gone and
                    //    stay gone.
                    //  * A VENDOR CHECK verifies the library where its own
                    //    guarantees are vacuous (TF32 detector, ld bounds).
                    //    Those stay.
                    //  * A COHERENCE FENCE reconciles the PLAN's vocabulary
                    //    (elected layouts) with a CALL FRAME THE EXECUTOR
                    //    INVENTS (m/n/k, descriptors, orders, lds). No
                    //    e-graph rule has ever seen an `LtCall`, so nothing
                    //    upstream can guarantee the two agree. This is that
                    //    fence. It is NOT disposable.
                    //
                    // It also does what the deleted check could not: the
                    // old one compared EXTENTS only, and the regression it
                    // was supposed to catch had matching extents and a
                    // diverging ORDER (the transpose-sandwich sibling's
                    // elected destination layout is LEFT-major). So the
                    // fence RESOLVES the C/D order from the elected layout
                    // instead of asserting a convention.
                    let dest_slot = result_info.first().ok_or_else(|| {
                        anyhow!("{label}: host-call node carries no result descriptor")
                    })?;
                    crate::ops::cublaslt::exec::bind_destination(
                        &mut call,
                        &dest_slot.layout,
                        label,
                    )
                    .with_context(|| format!("cuBLASLt destination frame binding for {label}"))?;
                    // E-GRAPH RE-CHECKS STAY DEAD (corrected contract,
                    // 2026-08-31, correction 4): the F32 end-to-end
                    // re-check and the C-operand dims/fold re-checks
                    // that stood here restated facts the e-graph
                    // guarantees by rule premise (the marker's contracts
                    // match F32 dense frames; Cdesc == Ddesc by rule
                    // guard). They are REMOVED and stay removed. The
                    // frame check that stood alongside them was NOT one
                    // of them — see the coherence fence above.
                    let operand_spans: Vec<crate::ops::cublaslt::device_call::DeviceRange> = inputs
                        .iter()
                        .map(|b| crate::ops::cublaslt::device_call::DeviceRange {
                            ptr: b.ptr,
                            bytes: b.bytes,
                        })
                        .collect();
                    crate::ops::cublaslt::device_call::dispatch(
                        &call,
                        &operand_spans,
                        crate::ops::cublaslt::device_call::DeviceRange {
                            ptr: dest.ptr,
                            bytes: dest.bytes,
                        },
                        &stream,
                    )
                    .with_context(|| format!("cuBLASLt dispatch for {label}"))?;
                    continue;
                }
                let Some(kernel) = codegen_for(op.as_ref()) else {
                    bail!("no cuda codegen for {label}");
                };
                // Phase 3: codegen geometry comes from the node's OWN slot
                // descriptors, never the shared buffer table — the buffer
                // table sizes ALLOCATIONS and nothing else. A compute node
                // arriving without its descriptors is malformed: bail
                // loudly (mirror of the None-dims bail).
                if operand_info.len() != reads.len() || result_info.len() != writes.len() {
                    bail!(
                        "{label}: compute node lacks slot descriptors \
                         (operand_info {}/{}, result_info {}/{})",
                        operand_info.len(),
                        reads.len(),
                        result_info.len(),
                        writes.len()
                    );
                }
                let ctxinfo = CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                let launches = (kernel.codegen)(op.as_ref(), &ctxinfo)
                    .with_context(|| format!("codegen for {label}"))?;

                // Kernel inputs are the non-destination operands; the
                // destination is the buffer the planner assigned, in
                // place. Launches in one sequence share the stream, so
                // phase ordering (e.g. scatter's init-copy then writes)
                // is free.
                let input_ptrs: Vec<u64> = inputs.iter().map(|b| b.ptr).collect();
                let dest_ptr = dest.ptr;
                for generated in &launches {
                    let func = device.cache.function(&generated.source)?;
                    let n = generated.n as u64;
                    let cfg = LaunchConfig {
                        grid_dim: ((generated.n as u32).max(1).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = stream.launch_builder(&func);
                    for ptr in &input_ptrs {
                        builder.arg(ptr);
                    }
                    builder.arg(&dest_ptr);
                    builder.arg(&n);
                    unsafe { builder.launch(cfg) }.with_context(|| format!("launch {label}"))?;
                }
            }
        }
    }
    stream.synchronize().context("stream sync")?;

    // Phase 4: D2H each output SLOT's backing buffer — the escaped
    // buffer for a view election, the boundary buffer for a dense one —
    // keyed by slot index and paired with the binding's layout. (The
    // declared-but-unused Boundary buffer of an escaped slot never
    // reaches this plan: buffer DCE dropped it, so Phase 1 never
    // allocated it; and no free node exists for an escaping buffer, so
    // every output slot is still bound here.)
    let mut outputs = FxHashMap::default();
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let bound = bound_of(&bindings, &slot.buffer, "output slot")?;
                let mut host = vec![0u8; bound.bytes];
                unsafe { cu::memcpy_dtoh_async(&mut host, bound.ptr, stream.cu_stream()) }
                    .context("D2H")?;
                let dtype = geometry[&slot.buffer];
                outputs.insert(slot.index, (bytes_to_typed(&host, dtype)?, slot.clone()));
            }
        }
    }
    Ok(outputs)
}

/// The arena's cost, on demand (`LUMINAL_CL_ARENA=1`): the high-water
/// mark this plan needs against the sum CL-2 used to pay — one
/// allocation per plan buffer, all live for the whole call.
fn arena_report(
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    arena: &ArenaPlan,
    slab_now: usize,
) {
    let sum_all: usize = plan
        .buffers
        .values()
        .filter_map(|b| buffer_bytes(b).ok())
        .sum();
    let slab_sum: usize = arena.slices.values().map(|s| s.bytes).sum();
    let standalone: usize = arena
        .standalone
        .iter()
        .chain(arena.donated.iter())
        .filter_map(|id| plan.buffers.get(id))
        .filter_map(|b| buffer_bytes(b).ok())
        .sum();
    eprintln!(
        "[cl-arena] slab high-water {} B (peak live {} B, resident {} B) for {} \
         interior buffers summing {} B; standalone+donated {} B over {} buffers; \
         whole-plan sum (the CL-2 cost) {} B; total now {} B",
        arena.slab_bytes,
        arena.peak_live_bytes,
        slab_now,
        arena.slices.len(),
        slab_sum,
        standalone,
        arena.standalone.len() + arena.donated.len(),
        sum_all,
        arena.slab_bytes + standalone,
    );
}
