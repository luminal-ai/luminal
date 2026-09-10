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
//! THE SERVING LANDING (2026-09-10) made the BOUNDARY storage persistent
//! too, in three pieces:
//!
//! * RESIDENT INPUTS. A staged payload is copied to the device ONCE and
//!   kept, keyed by its `BufferLit` id. Later executes re-upload a lit
//!   only when the runtime marked it DIRTY (a new `set_data`) or the
//!   staged bytes are a different host allocation than the resident copy
//!   was taken from. A 60 GB model's weights therefore cross the link
//!   once — at the first profiled candidate of the search — and every
//!   later candidate and every serving tick reuses the same device
//!   bytes. (Before: every `execute_plan` allocated, zeroed and re-copied
//!   every input, which priced a serving tick at the model size.)
//! * POOLED OUTPUTS. Each output SLOT keeps one device allocation across
//!   calls, resized only when a plan sizes the slot differently. Slot
//!   index is the key, so the bucket plans of one program share their
//!   output homes.
//! * LAZY READBACK. `execute_plan` leaves outputs on the device and
//!   returns only their bindings; [`CudaDevice::read_output`] does the
//!   D2H when a caller actually asks for the bytes. A serving tick that
//!   reads one `i32` per row no longer pays for a D2H of every KV cache
//!   output — and [`CudaDevice::copy_output_to_input`] feeds such an
//!   output back into its input's resident copy with one D2D memcpy.
//!
//! The phases themselves are unchanged in kind. Phase 1 binds the
//! standalone rows (resident, pooled, or scratch) and stages what is
//! dirty (loud on missing geometry/dtype, exactly like the reference).
//! Phase 2 is the arena's issue order — a topological order over Data
//! AND Anti edges, so WAR ordering is enforced by construction, chosen
//! for a small high-water mark. Phase 3 dispatches: `BufferAlloc` binds
//! its buffer to its slab range, `BufferFree` drops the binding, D2D for
//! copies, NVRTC-compiled launches for compute (the destination is the
//! range the planner assigned — no longer a fresh zeroed slice; every CL
//! kernel writes every element it owns, see the KERNEL INVARIANT note
//! below). Phase 4 records each output SLOT's device home and its
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
//! * THE FUSED HOST OPS (paged attention, the MXFP4 MoE halves): each
//!   writes every element of its destination in one pass and reads no
//!   destination byte.
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

use crate::arena::{ArenaPlan, buffer_bytes, plan_arena};
use crate::host_buffer::HostBuffer;
use anyhow::{Context, Result, anyhow, bail};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg, result as cu,
};
use cudarc::nvrtc::compile_ptx;
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind, OutputBinding};
use luminal::dtype::PlanDtype;
use luminal::prelude::{FxHashMap, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::host::{DeviceRange as Bound, HostOpContext};
use crate::kernels::CodegenCtx;
use crate::{as_host_op, as_kernel_op};

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

/// A staged input's device copy, plus the identity of the host bytes it
/// was taken from (allocation address + length). A staged payload whose
/// bytes live at a different address is a different payload and is
/// re-uploaded; one at the same address is trusted unless the runtime
/// marked its lit dirty.
struct Resident {
    slice: CudaSlice<u8>,
    host_ptr: usize,
    len: usize,
    dtype: PlanDtype,
}

/// Where an output slot's bytes live after an execute: a pooled output
/// home (the ordinary case — a fresh escaping/boundary buffer the plan
/// wrote), or a RESIDENT INPUT (an escaped view election whose backing
/// buffer is a staged input: the slot discloses a layout over the
/// input's own bytes).
#[derive(Debug, Clone, Copy)]
enum Home {
    Output(usize),
    Input(i64),
}

/// One output buffer's persistent device home: the allocation and the
/// dtype the plan gave it. Keyed by the FIRST slot index that names the
/// buffer (slots may legally share one escaping buffer); each slot's own
/// disclosed binding rides `CudaDevice::slot_view`.
struct OutputHome {
    slice: CudaSlice<u8>,
    bytes: usize,
    dtype: PlanDtype,
}

/// THE PERSISTENT DEVICE: everything an execution needs that should
/// outlive one call — the context, the one stream every kernel and
/// copy is issued on, the compiled-module cache, the arena slab, and
/// (since the serving landing) the resident inputs and pooled outputs.
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
    /// Device-resident staged inputs, by `BufferLit` id.
    resident: HashMap<i64, Resident>,
    /// Lits the runtime re-staged since their resident copy was taken.
    dirty: HashSet<i64>,
    /// Output buffers' persistent homes, by the home slot index.
    outputs: HashMap<usize, OutputHome>,
    /// What the last execute disclosed per output slot: where its bytes
    /// live, and the slot's elected layout.
    slot_view: HashMap<usize, (Home, OutputBinding<luminal::layouts::DecodedLayout>)>,
    /// Standalone buffers that are neither inputs nor output slots (an
    /// interior buffer demoted for want of a lifetime pair), keyed by the
    /// plan's own buffer id text and size.
    scratch: HashMap<(String, usize), CudaSlice<u8>>,
    /// PREPARED DISPATCH (serving landing, 2026-09-10): a plan's arena
    /// walk and compiled launches, keyed by the caller's plan key. A
    /// serving tick re-derives none of it — codegen, NVRTC lookup and the
    /// topological walk are paid once per installed plan.
    prepared: HashMap<u64, PreparedPlan>,
}

/// One compiled launch of a codegen'd kernel.
struct PreparedLaunch {
    func: CudaFunction,
    n: u64,
    grid: u32,
    block: u32,
    skip_when_dest_is_operand: Option<usize>,
}

/// One dispatch step of a prepared plan, in arena issue order.
enum PreparedStep {
    Copy {
        src: BufferId,
        dst: BufferId,
    },
    /// Bind a slab member to its range (a standalone buffer's alloc is a no-op).
    Alloc(Option<BufferId>),
    Free(Option<BufferId>),
    Host {
        node: NodeIndex,
        reads: Vec<BufferId>,
        dest: BufferId,
        input_count: usize,
    },
    Kernel {
        reads: Vec<BufferId>,
        dest: BufferId,
        input_count: usize,
        launches: Vec<PreparedLaunch>,
    },
}

/// A plan's dispatch, derived once from the plan: everything that does
/// not depend on where the buffers happen to live this call.
pub struct PreparedPlan {
    arena: ArenaPlan,
    slot_of_buffer: FxHashMap<BufferId, usize>,
    slot_bindings: Vec<OutputBinding<luminal::layouts::DecodedLayout>>,
    geometry: FxHashMap<BufferId, PlanDtype>,
    steps: Vec<PreparedStep>,
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
            resident: HashMap::new(),
            dirty: HashSet::new(),
            outputs: HashMap::new(),
            slot_view: HashMap::new(),
            scratch: HashMap::new(),
            prepared: HashMap::new(),
        })
    }

    /// The stream every launch and copy of this device is issued on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The CUDA context this device runs in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
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
    /// one) — the runtime's resident arena footprint.
    pub fn slab_bytes(&self) -> usize {
        self.slab.as_ref().map(|slab| slab.len()).unwrap_or(0)
    }

    /// Bytes held by the resident inputs — the model's device footprint.
    pub fn resident_bytes(&self) -> usize {
        self.resident.values().map(|r| r.slice.len()).sum()
    }

    /// Bytes held by the pooled output homes.
    pub fn output_bytes(&self) -> usize {
        self.outputs.values().map(|o| o.slice.len()).sum()
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
    /// Nothing else is released here — the context, the stream, the
    /// NVRTC module cache, the resident inputs and the output homes all
    /// survive, which is what keeps kernel compilation a once-per-source
    /// cost and weight staging a once-per-search cost across a whole
    /// search.
    pub fn release_slab(&mut self) {
        self.slab = None;
    }

    /// Drop every prepared plan (a new search installs new plans).
    pub fn forget_prepared(&mut self) {
        self.prepared.clear();
    }

    /// Mark a staged lit DIRTY: its next execute re-uploads it whatever
    /// the resident copy's identity says. The runtime calls this from
    /// `set_data`.
    pub fn mark_dirty(&mut self, lit: i64) {
        self.dirty.insert(lit);
    }

    /// Drop a lit's resident copy (and any dirty mark).
    pub fn evict_input(&mut self, lit: i64) {
        self.resident.remove(&lit);
        self.dirty.remove(&lit);
    }

    /// Drop EVERY resident input. The search's staged map is borrowed
    /// from the caller; a runtime whose staged payloads are replaced
    /// wholesale (a new search over new data) evicts first so no stale
    /// address identity can be trusted.
    pub fn evict_all_inputs(&mut self) {
        self.resident.clear();
        self.dirty.clear();
    }

    /// The device range a resident input currently occupies, if any.
    pub fn resident_input(&self, lit: i64) -> Option<Bound> {
        self.resident.get(&lit).map(|r| Bound {
            ptr: {
                let (ptr, _record) = r.slice.device_ptr(&self.stream);
                ptr
            },
            bytes: r.slice.len(),
        })
    }

    /// The device range backing an output slot after the last execute,
    /// with the slot's disclosed binding.
    pub fn output_slot(
        &self,
        slot: usize,
    ) -> Option<(Bound, &OutputBinding<luminal::layouts::DecodedLayout>)> {
        let (home, binding) = self.slot_view.get(&slot)?;
        let (bound, _) = self.home_range(*home).ok()?;
        Some((bound, binding))
    }

    /// The device range and dtype of a slot home.
    fn home_range(&self, home: Home) -> Result<(Bound, PlanDtype)> {
        match home {
            Home::Output(home_slot) => {
                let home = self
                    .outputs
                    .get(&home_slot)
                    .ok_or_else(|| anyhow!("output home {home_slot} was never bound"))?;
                let (ptr, _record) = home.slice.device_ptr(&self.stream);
                Ok((
                    Bound {
                        ptr,
                        bytes: home.bytes,
                    },
                    home.dtype,
                ))
            }
            Home::Input(lit) => {
                let resident = self.resident.get(&lit).ok_or_else(|| {
                    anyhow!("input lit {lit} backs an output but is not resident")
                })?;
                let (ptr, _record) = resident.slice.device_ptr(&self.stream);
                Ok((
                    Bound {
                        ptr,
                        bytes: resident.slice.len(),
                    },
                    resident.dtype,
                ))
            }
        }
    }

    fn slot_home(&self, slot: usize) -> Result<(Bound, PlanDtype)> {
        let (home, _) = self
            .slot_view
            .get(&slot)
            .ok_or_else(|| anyhow!("output slot {slot} has not been executed"))?;
        self.home_range(*home)
    }

    /// D2H one output slot's backing bytes (synchronous). The escape-
    /// and-disclose fetch's byte half; the binding rides
    /// [`Self::output_slot`].
    pub fn read_output(&self, slot: usize) -> Result<HostBuffer> {
        let (bound, dtype) = self.slot_home(slot)?;
        let mut host = vec![0u8; bound.bytes];
        if bound.bytes > 0 {
            unsafe { cu::memcpy_dtoh_async(&mut host, bound.ptr, self.stream.cu_stream()) }
                .context("D2H")?;
            self.stream.synchronize().context("D2H sync")?;
        }
        bytes_to_typed(&host, dtype)
    }

    /// FEED AN OUTPUT BACK INTO AN INPUT: one D2D memcpy from the slot's
    /// backing bytes into the lit's resident copy. Sizes must agree. The
    /// resident copy becomes DEVICE-AUTHORITATIVE — the runtime's staged
    /// host bytes for that lit are stale from here on, and are never
    /// re-uploaded unless the lit is marked dirty again.
    pub fn copy_output_to_input(&mut self, slot: usize, lit: i64) -> Result<()> {
        let (src, bytes) = {
            let (bound, _) = self.slot_home(slot)?;
            (bound.ptr, bound.bytes)
        };
        let dst = self
            .resident
            .get(&lit)
            .ok_or_else(|| anyhow!("input lit {lit} has no resident copy to feed back into"))?;
        if dst.slice.len() != bytes {
            bail!(
                "copy_output_to_input: output slot {slot} is {bytes} bytes, input lit {lit} \
                 is {} bytes",
                dst.slice.len()
            );
        }
        let (dst_ptr, _record) = dst.slice.device_ptr(&self.stream);
        unsafe { cu::memcpy_dtod_async(dst_ptr, src, bytes, self.stream.cu_stream()) }
            .context("D2D feedback copy")?;
        self.dirty.remove(&lit);
        Ok(())
    }

    /// Stage one lit: reuse the resident copy when it is clean and was
    /// taken from these very bytes, else (re)upload — into the existing
    /// allocation when the size still fits exactly, else a fresh one.
    fn stage_input(&mut self, lit: i64, data: &HostBuffer, bytes: usize) -> Result<Bound> {
        let host = &data.bytes;
        if host.len() != bytes {
            bail!(
                "staged buffer {lit} is {} bytes, plan expects {bytes}",
                host.len()
            );
        }
        let identity = (host.as_ptr() as usize, host.len());
        let dirty = self.dirty.remove(&lit);
        let reuse_alloc = match self.resident.get(&lit) {
            Some(resident) if !dirty && (resident.host_ptr, resident.len) == identity => {
                let (ptr, _record) = resident.slice.device_ptr(&self.stream);
                return Ok(Bound { ptr, bytes });
            }
            Some(resident) => resident.slice.len() == bytes,
            None => false,
        };
        let mut slice = if reuse_alloc {
            self.resident.remove(&lit).expect("checked above").slice
        } else {
            self.resident.remove(&lit);
            unsafe { self.stream.alloc::<u8>(bytes.max(1)) }
                .with_context(|| format!("device alloc {bytes} bytes for input lit {lit}"))?
        };
        if bytes > 0 {
            self.stream.memcpy_htod(host, &mut slice).context("H2D")?;
        }
        let ptr = {
            let (ptr, _record) = slice.device_ptr(&self.stream);
            ptr
        };
        self.resident.insert(
            lit,
            Resident {
                slice,
                host_ptr: identity.0,
                len: identity.1,
                dtype: data.dtype,
            },
        );
        Ok(Bound { ptr, bytes })
    }

    /// Bind an output slot's home, resizing when the plan sizes it
    /// differently than the last one did.
    fn output_home(&mut self, slot: usize, bytes: usize, dtype: PlanDtype) -> Result<Bound> {
        let fits = self
            .outputs
            .get(&slot)
            .is_some_and(|home| home.slice.len() == bytes.max(1));
        if !fits {
            self.outputs.remove(&slot);
            let slice = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }
                .with_context(|| format!("device alloc {bytes} bytes for output slot {slot}"))?;
            self.outputs.insert(
                slot,
                OutputHome {
                    slice,
                    bytes,
                    dtype,
                },
            );
        }
        let home = self.outputs.get_mut(&slot).expect("inserted above");
        home.bytes = bytes;
        home.dtype = dtype;
        let (ptr, _record) = home.slice.device_ptr(&self.stream);
        Ok(Bound { ptr, bytes })
    }

    fn scratch_home(&mut self, key: String, bytes: usize) -> Result<Bound> {
        let entry = self.scratch.entry((key.clone(), bytes));
        let slice = match entry {
            std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
            std::collections::hash_map::Entry::Vacant(vacant) => {
                let slice = unsafe { self.stream.alloc::<u8>(bytes.max(1)) }
                    .with_context(|| format!("device alloc {bytes} bytes for {key}"))?;
                vacant.insert(slice)
            }
        };
        let (ptr, _record) = slice.device_ptr(&self.stream);
        Ok(Bound { ptr, bytes })
    }
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
/// index, the slot's [`OutputBinding`] (the elected layout); the bytes
/// stay on the device until [`CudaDevice::read_output`] asks for them —
/// the escape-and-disclose fetch, universal over dense and view
/// elections, now lazy.
///
/// `staged` is a map of BORROWED payloads by BufferLit id (Phase 4). It
/// used to hold the payloads themselves, which was fine while the only
/// caller was the serving ladder — the runtime already owns them. The
/// search now stages too, and it stages the CALLER's map, which for a
/// full-size model is gigabytes of weights: a map of references costs
/// one pointer per input and no copy at all — and since the serving
/// landing, no H2D either once a lit is resident and clean.
///
/// This entry prepares the plan's dispatch afresh (the search's transient
/// candidates); serving goes through [`execute_plan_keyed`], which keeps
/// the prepared dispatch across calls.
pub fn execute_plan(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
) -> Result<FxHashMap<usize, OutputBinding<luminal::layouts::DecodedLayout>>> {
    let prepared = prepare_plan(device, plan)?;
    run_prepared(device, &prepared, plan, staged)
}

/// [`execute_plan`] with the plan's dispatch PREPARED ONCE under `key`
/// and reused by every later call with the same key: the arena walk, the
/// codegen and the NVRTC module lookups are paid on the first call only.
/// The caller owns the key's meaning (the runtime keys by search epoch and
/// bucket) and must not reuse a key for a different plan.
pub fn execute_plan_keyed(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
    key: u64,
) -> Result<FxHashMap<usize, OutputBinding<luminal::layouts::DecodedLayout>>> {
    if !device.prepared.contains_key(&key) {
        let prepared = prepare_plan(device, plan)?;
        device.prepared.insert(key, prepared);
    }
    // The prepared plan is borrowed out of the device for the run: take
    // it, run, put it back (a run touches the device's other fields).
    let prepared = device.prepared.remove(&key).expect("inserted above");
    let result = run_prepared(device, &prepared, plan, staged);
    device.prepared.insert(key, prepared);
    result
}

/// Derive a plan's dispatch: validate the output boundary, plan the
/// arena, and compile every codegen'd kernel through the module cache.
fn prepare_plan(
    device: &mut CudaDevice,
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
) -> Result<PreparedPlan> {
    // ESCAPE GUARD (ruling 2026-08-27): an output slot's backing storage
    // must SURVIVE the call — FreedBy::Caller, whatever the owner.
    // FreedBy::Program backing an output hands the caller bytes the
    // program destroys: minted non-escaping storage (Owner::System) and
    // DONATED boundary storage (Owner::Caller — validate()'s donated arm
    // forbids exactly this plan shape) alike. The pre-lowering
    // certificate enforces this for planner-built plans; hand-built /
    // externally loaded plans never met it — re-check here, loudly,
    // before any bytes move.
    //
    // The same walk records which buffer each output slot names, so
    // Phase 1 can give it the slot's pooled home.
    let mut slot_of_buffer: FxHashMap<BufferId, usize> = FxHashMap::default();
    let mut slot_bindings: Vec<OutputBinding<luminal::layouts::DecodedLayout>> = Vec::new();
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
                slot_of_buffer
                    .entry(slot.buffer.clone())
                    .or_insert(slot.index);
                slot_bindings.push(slot.clone());
            }
        }
    }

    // Phase 2, hoisted: the arena decides the issue order AND the slab
    // layout in one device-free pass (Anti edges ride it, so WAR
    // ordering is enforced by construction). Everything below walks
    // `arena.order`.
    let arena = plan_arena(plan, buffer_bytes).context("arena planning")?;
    debug_assert!(
        plan.dag
            .edge_weights()
            .all(|e| matches!(e.kind, EdgeKind::Data | EdgeKind::Anti))
    );
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

    // The dispatch steps: codegen and compile every kernel now.
    let mut steps = Vec::with_capacity(arena.order.len());
    for node in arena.order.iter().copied() {
        let step = match &plan.dag[node] {
            BufferNode::BufferInput { .. } | BufferNode::BufferOutput { .. } => continue,
            BufferNode::BufferCopy { src, dst } => PreparedStep::Copy {
                src: src.clone(),
                dst: dst.clone(),
            },
            BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                let label = op.label();
                if label == "BufferAlloc" {
                    PreparedStep::Alloc(
                        writes
                            .first()
                            .filter(|buffer| arena.slices.contains_key(*buffer))
                            .cloned(),
                    )
                } else if label == "BufferFree" {
                    PreparedStep::Free(
                        reads
                            .first()
                            .filter(|buffer| arena.slices.contains_key(*buffer))
                            .cloned(),
                    )
                } else {
                    if writes.len() != 1 {
                        bail!(
                            "{label}: CL handles single-destination ops, got {}",
                            writes.len()
                        );
                    }
                    // Codegen geometry comes from the node's OWN slot
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
                    let input_count = reads.len().saturating_sub(writes.len());
                    if as_host_op(op.as_ref()).is_some() {
                        PreparedStep::Host {
                            node,
                            reads: reads.clone(),
                            dest: writes[0].clone(),
                            input_count,
                        }
                    } else {
                        let Some(kernel) = as_kernel_op(op.as_ref()) else {
                            bail!("no CUDA execution interface for {label}");
                        };
                        let ctxinfo =
                            CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                        let generated = kernel
                            .codegen(&ctxinfo)
                            .with_context(|| format!("codegen for {label}"))?;
                        let mut launches = Vec::with_capacity(generated.len());
                        for source in generated {
                            let func = device.cache.function(&source.source)?;
                            let (grid, block) = match source.launch {
                                Some(geometry) => (geometry.grid.max(1), geometry.block.max(1)),
                                None => ((source.n as u32).max(1).div_ceil(256), 256),
                            };
                            launches.push(PreparedLaunch {
                                func,
                                n: source.n as u64,
                                grid,
                                block,
                                skip_when_dest_is_operand: source.skip_when_dest_is_operand,
                            });
                        }
                        PreparedStep::Kernel {
                            reads: reads.clone(),
                            dest: writes[0].clone(),
                            input_count,
                            launches,
                        }
                    }
                }
            }
        };
        steps.push(step);
    }
    Ok(PreparedPlan {
        arena,
        slot_of_buffer,
        slot_bindings,
        geometry,
        steps,
    })
}

/// Run a prepared plan: bind this call's storage, issue the steps, record
/// the output slots.
fn run_prepared(
    device: &mut CudaDevice,
    prepared: &PreparedPlan,
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    staged: &FxHashMap<i64, &HostBuffer>,
) -> Result<FxHashMap<usize, OutputBinding<luminal::layouts::DecodedLayout>>> {
    let arena = &prepared.arena;
    device.ensure_slab(arena.slab_bytes)?;
    let stream = device.stream.clone();
    let slab_base = match &device.slab {
        Some(slab) => {
            let (base, _record) = slab.device_ptr(&stream);
            base
        }
        None => 0,
    };

    // Phase 1: bind the buffers that do NOT come from the slab — the
    // BOUNDARY and ESCAPING rows (their bytes are the caller's after
    // the call) and the DONATED row (the caller's bytes, which in CL
    // means the staged payload's resident copy). Slab members stay
    // unbound until their `BufferAlloc` is issued.
    let mut bindings: FxHashMap<BufferId, Bound> = FxHashMap::default();
    // Per HOME slot (the first slot naming a buffer): where its bytes
    // live — a pooled output home or a resident input.
    let mut home_of: FxHashMap<usize, Home> = FxHashMap::default();
    for id in arena.standalone.iter().chain(arena.donated.iter()) {
        let buffer = &plan.buffers[id];
        let bytes = buffer_bytes(buffer)?;
        let staged_lit = buffer.lit.filter(|lit| staged.contains_key(lit));
        let bound = if let Some(lit) = staged_lit {
            // A STAGED input: resident on the device. It may ALSO back an
            // output slot (an escaped view election over an input's own
            // bytes, or an in-place output delivered into it); the slot
            // then reads the resident copy.
            if let Some(&slot) = prepared.slot_of_buffer.get(id) {
                home_of.insert(slot, Home::Input(lit));
            }
            device
                .stage_input(lit, staged[&lit], bytes)
                .with_context(|| format!("staging {:?}", buffer.label))?
        } else if let Some(&slot) = prepared.slot_of_buffer.get(id) {
            home_of.insert(slot, Home::Output(slot));
            device.output_home(slot, bytes, prepared.geometry[id])?
        } else if let Some(lit) = buffer.lit {
            // A lit-bearing standalone buffer that backs no output slot
            // is an INPUT (boundary or donated): it must be staged. It
            // used to be zero-filled silently, which is how a missing
            // `set_data` became a wrong answer instead of an error.
            bail!(
                "input {:?} (lit {lit}) has no staged payload — set_data it before execute",
                buffer.label
            );
        } else {
            device.scratch_home(format!("{id:?}"), bytes)?
        };
        bindings.insert(id.clone(), bound);
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
    //
    // TWO BufferIds MAY share a range on purpose: an in-place output
    // delivered into its input (`output_into`) binds both to one lit —
    // the same `Bound` — and is exactly the sharing CONTRACT 1 licenses
    // (same storage, same BufferLit). Deduplicate by range before the
    // check.
    {
        let mut by_range: FxHashMap<(u64, usize), BufferId> = FxHashMap::default();
        for (id, b) in bindings.iter().filter(|(_, b)| b.bytes > 0) {
            by_range
                .entry((b.ptr, b.bytes))
                .or_insert_with(|| id.clone());
        }
        let bound: Vec<crate::binding_check::BoundRange> = by_range
            .iter()
            .map(|((ptr, bytes), id)| crate::binding_check::BoundRange {
                buffer: format!("{id:?}"),
                base: *ptr,
                bytes: *bytes as u64,
            })
            .collect();
        crate::binding_check::assert_disjoint(&bound).context("CONTRACT-1 bind-time check")?;
    }

    if std::env::var_os("LUMINAL_CL_ARENA").is_some() {
        arena_report(plan, arena, device.slab_bytes());
    }

    // `LUMINAL_CL_PROFILE_OPS=1`: synchronize after every node and
    // attribute wall time per op label — a development probe for finding
    // the hot ops of a plan, never on in serving (the syncs serialize
    // the stream).
    // `=trace` additionally lists every step in issue order — the
    // launch-count view a decode tick lives or dies by.
    let profile_mode = std::env::var("LUMINAL_CL_PROFILE_OPS").ok();
    let profile_ops = profile_mode.is_some();
    let profile_trace = profile_mode.as_deref() == Some("trace");
    let mut op_times: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut op_clock = std::time::Instant::now();
    let mut step_index = 0usize;

    // Phase 3: dispatch, in the arena's issue order.
    for step in &prepared.steps {
        if profile_ops {
            op_clock = std::time::Instant::now();
        }
        let profile_label: Option<String> = match step {
            PreparedStep::Alloc(Some(buffer)) => {
                let slice = &arena.slices[buffer];
                bindings.insert(
                    buffer.clone(),
                    Bound {
                        ptr: slab_base + slice.offset as u64,
                        bytes: slice.bytes,
                    },
                );
                None
            }
            PreparedStep::Alloc(None) => None,
            PreparedStep::Free(Some(buffer)) => {
                bindings.remove(buffer);
                None
            }
            PreparedStep::Free(None) => None,
            PreparedStep::Copy { src, dst } => {
                // THE BUFFERCOPY CONTRACT, executor side (Austin, ruled
                // 2026-08-31 — see `bufferize::BufferNode::BufferCopy`):
                // a DUMB EXACT-SIZE WHOLE-BUFFER copy, one `memcpy_dtod`,
                // ordered by issue on the one stream.
                let from = bound_of(&bindings, src, "copy src")?;
                let to = bound_of(&bindings, dst, "copy dst")?;
                if from.bytes != to.bytes {
                    bail!("copy length mismatch: {} -> {} bytes", from.bytes, to.bytes);
                }
                unsafe { cu::memcpy_dtod_async(to.ptr, from.ptr, from.bytes, stream.cu_stream()) }
                    .context("D2D copy")?;
                profile_ops.then(|| "BufferCopy".to_string())
            }
            PreparedStep::Host {
                node,
                reads,
                dest,
                input_count,
            } => {
                let BufferNode::Compute {
                    op,
                    operand_info,
                    result_info,
                    ..
                } = &plan.dag[*node]
                else {
                    unreachable!("a prepared host step names a compute node")
                };
                let label = op.label();
                let dest = bound_of(&bindings, dest, label)?;
                let inputs: Vec<Bound> = reads[..*input_count]
                    .iter()
                    .map(|id| bound_of(&bindings, id, label))
                    .collect::<Result<_>>()?;
                let host = as_host_op(op.as_ref()).expect("prepared as a host op");
                let ctx = HostOpContext {
                    stream: &stream,
                    inputs: &inputs,
                    dest,
                    operand_info,
                    result_info,
                };
                // SAFETY: the plan's arena bindings are live through the
                // stream synchronization below; bufferization enforces the
                // op's alias and memory-effect contract.
                unsafe { host.execute(&ctx) }
                    .with_context(|| format!("host execution for {label}"))?;
                profile_ops.then(|| profile_name(label, result_info))
            }
            PreparedStep::Kernel {
                reads,
                dest,
                input_count,
                launches,
            } => {
                let dest_bound = bound_of(&bindings, dest, "kernel dest")?;
                let inputs: Vec<Bound> = reads[..*input_count]
                    .iter()
                    .map(|id| bound_of(&bindings, id, "kernel operand"))
                    .collect::<Result<_>>()?;
                // Kernel inputs are the non-destination operands; the
                // destination is the buffer the planner assigned, in
                // place. Launches in one sequence share the stream, so
                // phase ordering (e.g. scatter's init-copy then writes)
                // is free.
                let input_ptrs: Vec<u64> = inputs.iter().map(|b| b.ptr).collect();
                let dest_ptr = dest_bound.ptr;
                for launch in launches {
                    if let Some(k) = launch.skip_when_dest_is_operand
                        && inputs.get(k).is_some_and(|b| b.ptr == dest_ptr)
                    {
                        continue;
                    }
                    let cfg = LaunchConfig {
                        grid_dim: (launch.grid, 1, 1),
                        block_dim: (launch.block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = stream.launch_builder(&launch.func);
                    for ptr in &input_ptrs {
                        builder.arg(ptr);
                    }
                    builder.arg(&dest_ptr);
                    builder.arg(&launch.n);
                    unsafe { builder.launch(cfg) }.context("kernel launch")?;
                }
                profile_ops.then(|| {
                    prepared_label(plan, reads, dest).unwrap_or_else(|| "kernel".to_string())
                })
            }
        };
        if let Some(label) = profile_label {
            stream.synchronize().context("profile sync")?;
            let ms = op_clock.elapsed().as_secs_f64() * 1e3;
            if profile_trace {
                eprintln!("[cl-trace] {step_index:>5} {ms:>8.3} ms  {label}");
            }
            step_index += 1;
            let entry = op_times.entry(label).or_default();
            entry.0 += 1;
            entry.1 += ms;
        }
    }
    stream.synchronize().context("stream sync")?;
    if profile_ops {
        let mut rows: Vec<_> = op_times.into_iter().collect();
        rows.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap());
        let total: f64 = rows.iter().map(|r| r.1.1).sum();
        eprintln!("[cl-profile] {total:.2} ms over {} node kinds", rows.len());
        for (label, (count, ms)) in rows.iter().take(25) {
            eprintln!("[cl-profile] {ms:>9.3} ms  {count:>5} x  {label}");
        }
    }

    // Phase 4: record each output SLOT's disclosed binding against the
    // home whose bytes back it — the escaped buffer for a view election,
    // the boundary buffer for a dense one — keyed by slot index. The
    // bytes stay on the device until `read_output` asks for them.
    device.slot_view.clear();
    let mut outputs = FxHashMap::default();
    for slot in &prepared.slot_bindings {
        let home_slot = prepared.slot_of_buffer[&slot.buffer];
        let home = *home_of.get(&home_slot).ok_or_else(|| {
            anyhow!(
                "output slot {} names buffer {:?}, which Phase 1 never bound",
                slot.index,
                slot.buffer
            )
        })?;
        device.slot_view.insert(slot.index, (home, slot.clone()));
        outputs.insert(slot.index, slot.clone());
    }
    Ok(outputs)
}

/// The profiler's row label for a compute node: the op label, with the
/// result size when it is large.
fn profile_name(
    label: &str,
    result_info: &[luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
) -> String {
    let bytes: usize = result_info
        .iter()
        .filter_map(|slot| {
            let numel = slot.layout.literal_span_elements()?;
            let width = slot
                .layout
                .dtype
                .and_then(|d| crate::host_buffer::dtype_bytes(d).ok())?;
            Some(numel * width)
        })
        .sum();
    if bytes >= 64 << 20 {
        format!("{label} [{} MiB out]", bytes >> 20)
    } else {
        label.to_string()
    }
}

/// The profiler's label for a prepared kernel step: found by its
/// destination buffer (each compute node writes exactly one).
fn prepared_label(
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    _reads: &[BufferId],
    dest: &BufferId,
) -> Option<String> {
    plan.dag.node_weights().find_map(|node| match node {
        BufferNode::Compute {
            op,
            writes,
            result_info,
            ..
        } if writes.first() == Some(dest)
            && op.label() != "BufferAlloc"
            && op.label() != "BufferFree" =>
        {
            Some(profile_name(op.label(), result_info))
        }
        _ => None,
    })
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
