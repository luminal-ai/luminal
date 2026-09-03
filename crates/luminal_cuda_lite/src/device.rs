//! CL-2: the device executor — the reference runtime's four execution
//! phases reimplemented over cudarc, consuming the identical
//! `BufferIrGraph`.
//!
//! Phase 1 materializes every plan buffer on the device up front
//! (loud on missing geometry/dtype, exactly like the reference; alloc
//! and free plan ops are no-ops in this first cut). Phase 2 toposorts
//! the dag — `Anti` (WAR) edges ride petgraph, so the order is
//! load-bearing for free. Phase 3 dispatches: D2D for copies,
//! NVRTC-compiled launches for compute (out-of-place: inputs are the
//! operand buffers, the destination is a fresh zeroed slice swapped in
//! after the launch — mirroring the reference's alias-safety
//! convention; `ties` are ordering-only in CL-2). Phase 4 copies each
//! output SLOT's backing buffer back to a host `TypedBuffer`, keyed by
//! slot index and paired with the slot's [`OutputBinding`] — the
//! escape-and-disclose contract (ruling 2026-08-27): the caller gets
//! the backing bytes (possibly parent-sized, for an escaped view
//! election) plus the layout to interpret them under.

use anyhow::{anyhow, bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode, EdgeKind, OutputBinding};
use luminal::dtype::PlanDtype;
use luminal::prelude::FxHashMap;
use std::collections::HashMap;
use std::sync::Arc;

use crate::kernels::{codegen_for, CodegenCtx};

fn dtype_bytes(dtype: PlanDtype) -> Result<usize> {
    Ok(match dtype {
        PlanDtype::F32 => 4,
        PlanDtype::Int => 4,
        PlanDtype::Int64 => 8,
        PlanDtype::Bool | PlanDtype::Bool8 => 1,
        other => bail!("cuda-lite CL-2 has no device representation for {other:?}"),
    })
}

fn typed_to_bytes(data: &TypedBuffer) -> &[u8] {
    match data {
        TypedBuffer::F32(v) => bytemuck_cast(v),
        TypedBuffer::I32(v) => bytemuck_cast(v),
        TypedBuffer::I64(v) => bytemuck_cast(v),
        TypedBuffer::Bool8(v) => v.as_slice(),
        TypedBuffer::F8E4M3(_) => unreachable!("dtype_bytes refuses F8 first"),
        // F64 is executable on the reference runtime (ruling
        // 2026-09-02) but has no CL kernel and no device
        // representation, so `dtype_bytes` refuses it by name before a
        // buffer ever reaches this bridge.
        TypedBuffer::F64(_) => unreachable!("dtype_bytes refuses F64 first"),
    }
}

fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    // Plain-old-data reinterpretation for f32/i32/i64 payloads.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn bytes_to_typed(bytes: &[u8], dtype: PlanDtype) -> Result<TypedBuffer> {
    Ok(match dtype {
        PlanDtype::F32 => TypedBuffer::F32(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        PlanDtype::Int => TypedBuffer::I32(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        PlanDtype::Int64 => TypedBuffer::I64(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_ne_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        PlanDtype::Bool | PlanDtype::Bool8 => TypedBuffer::bool8(bytes.to_vec())?,
        other => bail!("cuda-lite CL-2 cannot read back {other:?}"),
    })
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

/// Execute a bufferized plan on device 0. Returns, per output slot
/// index, a host copy of the slot's BACKING buffer plus its
/// [`OutputBinding`] (the elected layout) — the escape-and-disclose
/// fetch, universal over dense and view elections.
pub fn execute_plan(
    plan: &BufferIrGraph<crate::layouts::CudaLayout>,
    staged: &FxHashMap<i64, TypedBuffer>,
) -> Result<FxHashMap<usize, (TypedBuffer, OutputBinding<crate::layouts::CudaLayout>)>> {
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

    let ctx = CudaContext::new(0).context("no CUDA device 0")?;
    let stream = ctx.default_stream();
    let mut cache = KernelCache {
        ctx: ctx.clone(),
        modules: HashMap::new(),
    };

    // Phase 1: materialize every buffer on device — ALLOCATION BY
    // ASSIGNMENT LOOKUP (corrected contract, 2026-08-31): the buffer
    // backs one tensor, its carried layout gives the span in elements
    // and the dtype fact gives the byte width. No walk, no voting.
    let mut storage: FxHashMap<BufferId, CudaSlice<u8>> = FxHashMap::default();
    let mut geometry: FxHashMap<BufferId, (Vec<usize>, PlanDtype)> = FxHashMap::default();
    for (id, buffer) in &plan.buffers {
        let dims = buffer.layout.mirror.literal_extents().ok_or_else(|| {
            anyhow!(
                "buffer {:?} (backing {}) has symbolic layout extents — not executable",
                buffer.label,
                buffer.backs
            )
        })?;
        let numel = buffer
            .layout
            .mirror
            .literal_span_elements()
            .ok_or_else(|| {
                anyhow!(
                    "buffer {:?} (backing {}) has no literal span — symbolic or \
                 undisclosed-reach layouts are not executable",
                    buffer.label,
                    buffer.backs
                )
            })?;
        let dtype = buffer.layout.dtype.ok_or_else(|| {
            anyhow!(
                "buffer {:?} (backing {}) carries no dtype fact",
                buffer.label,
                buffer.backs
            )
        })?;
        let bytes = numel * dtype_bytes(dtype)?;
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
        storage.insert(id.clone(), slice);
        geometry.insert(id.clone(), (dims, dtype));
    }

    // CONTRACT-1 (bind-time): distinct BufferIds must be backed by
    // disjoint device ranges — folded-view reads and WAR ordering are
    // both keyed on BufferId identity. Fresh `alloc_zeros` per buffer
    // makes this hold by construction today; the assert is the
    // contract's enforcement face for when raw caller pointers arrive
    // at this binding surface. Loud refusal, never mistranslation.
    {
        let bound: Vec<crate::binding_check::BoundRange> = storage
            .iter()
            .map(|(id, slice)| {
                let (base, _sync) = slice.device_ptr(&stream);
                crate::binding_check::BoundRange {
                    buffer: format!("{id:?}"),
                    base: base as u64,
                    bytes: slice.len() as u64,
                }
            })
            .collect();
        crate::binding_check::assert_disjoint(&bound).context("CONTRACT-1 bind-time check")?;
    }

    // Phase 2: toposort — Anti edges are ordinary edges here, so WAR
    // ordering is enforced by construction.
    let order = luminal::prelude::petgraph::algo::toposort(&plan.dag, None)
        .map_err(|_| anyhow!("plan dag has a cycle"))?;
    debug_assert!(plan
        .dag
        .edge_weights()
        .all(|e| matches!(e.kind, EdgeKind::Data | EdgeKind::Anti)));

    // Phase 3: dispatch.
    for node in order {
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
                //   runtime owned choice"; CL-2 pre-materializes exactly
                //   sized slices and makes no such choice, so unequal
                //   lengths are a bug HERE and bail loudly (this is the
                //   executor's own discipline over bufferizer-authored
                //   nodes, NOT a type fence re-checking an e-graph premise).
                // * ORDERING IS THIS RUNTIME'S OBLIGATION. The plan supplied
                //   dependency structure only (data + WAR anti-edges); we
                //   discharge it by issuing in toposort order onto ONE
                //   stream, which serializes the copy against every op that
                //   depends on it and every prior reader of `dst`. A
                //   multi-stream executor would owe events/barriers here.
                // * The three causes (conflict repair, boundary placement,
                //   lifetime repair) are the bufferizer's business; all
                //   three execute identically.
                let src_slice = storage
                    .get(src)
                    .ok_or_else(|| anyhow!("copy src unknown"))?
                    .clone();
                let dst_slice = storage
                    .get_mut(dst)
                    .ok_or_else(|| anyhow!("copy dst unknown"))?;
                if src_slice.len() != dst_slice.len() {
                    bail!(
                        "copy length mismatch: {} -> {} bytes",
                        src_slice.len(),
                        dst_slice.len()
                    );
                }
                stream
                    .memcpy_dtod(&src_slice, dst_slice)
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
                if label == "BufferAlloc" || label == "BufferFree" {
                    continue; // storage is pre-materialized in CL-2
                }
                // Train 3: the HOST-CALL arm — cuBLASLt contracts
                // dispatch as one `cublasLtMatmul` library call on the
                // SAME stream as the surrounding kernels, never an
                // NVRTC kernel. The destination follows the executor's
                // out-of-place convention (fresh zeroed slice, swapped
                // into storage after the call), so the C-fold forms
                // read their C operand buffer and write fresh D
                // (C != D pointers, beta = 1.0f — legal, identical
                // layouts by the marker's rule guard).
                if let Some(dps) = op
                    .as_any()
                    .downcast_ref::<crate::ops::cublaslt::CublasLtDps>()
                {
                    let mut call = crate::ops::cublaslt::exec::plan_call(&dps.op)
                        .with_context(|| format!("cuBLASLt call planning for {label}"))?;
                    if writes.len() != 1 {
                        bail!("{label}: single-destination contract, got {}", writes.len());
                    }
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
                    let input_count = reads.len().saturating_sub(writes.len());
                    let inputs: Vec<CudaSlice<u8>> = reads[..input_count]
                        .iter()
                        .map(|id| storage.get(id).unwrap().clone())
                        .collect();
                    let operand_refs: Vec<&CudaSlice<u8>> = inputs.iter().collect();
                    // E-GRAPH RE-CHECKS STAY DEAD (corrected contract,
                    // 2026-08-31, correction 4): the F32 end-to-end
                    // re-check and the C-operand dims/fold re-checks
                    // that stood here restated facts the e-graph
                    // guarantees by rule premise (the marker's contracts
                    // match F32 dense frames; Cdesc == Ddesc by rule
                    // guard). They are REMOVED and stay removed. The
                    // frame check that stood alongside them was NOT one
                    // of them — see the coherence fence above.
                    let (dest_dims, dest_dtype) = geometry.get(&writes[0]).unwrap().clone();
                    let dest_bytes = dest_dims.iter().product::<usize>() * dtype_bytes(dest_dtype)?;
                    let mut dest = stream
                        .alloc_zeros::<u8>(dest_bytes.max(1))
                        .context("dest alloc")?;
                    crate::ops::cublaslt::device_call::dispatch(
                        &call,
                        &operand_refs,
                        &mut dest,
                        &stream,
                    )
                    .with_context(|| format!("cuBLASLt dispatch for {label}"))?;
                    storage.insert(writes[0].clone(), dest);
                    continue;
                }
                let Some(kernel) = codegen_for(op.as_ref()) else {
                    bail!("no cuda codegen for {label}");
                };
                // Phase 3: codegen geometry comes from the node's OWN slot
                // descriptors, never the shared buffer table — `geometry`
                // stays for allocation sizing and the copy check only. A
                // compute node arriving without its descriptors is
                // malformed: bail loudly (mirror of the None-dims bail).
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
                // destination is a fresh zeroed slice (out-of-place),
                // swapped into storage after the sequence. Launches in
                // one sequence share the stream, so phase ordering
                // (e.g. scatter's init-copy then writes) is free.
                if writes.len() != 1 {
                    bail!(
                        "{label}: CL-2 handles single-destination ops, got {}",
                        writes.len()
                    );
                }
                let input_count = reads.len().saturating_sub(writes.len());
                let inputs: Vec<CudaSlice<u8>> = reads[..input_count]
                    .iter()
                    .map(|id| storage.get(id).unwrap().clone())
                    .collect();
                let (dest_dims, dest_dtype) = geometry.get(&writes[0]).unwrap().clone();
                let dest_bytes = dest_dims.iter().product::<usize>() * dtype_bytes(dest_dtype)?;
                let mut dest = stream
                    .alloc_zeros::<u8>(dest_bytes.max(1))
                    .context("dest alloc")?;

                for generated in &launches {
                    let func = cache.function(&generated.source)?;
                    let n = generated.n as u64;
                    let cfg = LaunchConfig {
                        grid_dim: (((generated.n as u32).max(1) + 255) / 256, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut builder = stream.launch_builder(&func);
                    for input in &inputs {
                        builder.arg(input);
                    }
                    builder.arg(&mut dest);
                    builder.arg(&n);
                    unsafe { builder.launch(cfg) }.with_context(|| format!("launch {label}"))?;
                }
                storage.insert(writes[0].clone(), dest);
            }
        }
    }
    stream.synchronize().context("stream sync")?;

    // Phase 4: D2H each output SLOT's backing buffer — the escaped
    // buffer for a view election, the boundary buffer for a dense one —
    // keyed by slot index and paired with the binding's layout. (The
    // declared-but-unused Boundary buffer of an escaped slot never
    // reaches this plan: buffer DCE dropped it, so Phase 1 never
    // allocated it; and no free node exists for an escaping buffer.)
    let mut outputs = FxHashMap::default();
    for node in plan.dag.node_weights() {
        if let BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let slice = storage
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("output slot {} names unknown buffer", slot.index))?;
                let mut host = vec![0u8; slice.len()];
                stream.memcpy_dtoh(slice, &mut host).context("D2H")?;
                let (_, dtype) = geometry.get(&slot.buffer).unwrap();
                outputs.insert(slot.index, (bytes_to_typed(&host, *dtype)?, slot.clone()));
            }
        }
    }
    Ok(outputs)
}
