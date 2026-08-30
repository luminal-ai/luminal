//! `SinkAttention` — host op wrapping the FA3/Hopper AttentionSink paged
//! batch-prefill kernel.
//!
//! Runtime inputs (7): `q` (s, nq*hd) bf16, `k_pool`/`v_pool`
//! (num_slots, nkv*hd) bf16 (post-scatter pool states), `kv_indices` (c,)
//! Int (compact page table, page_size 1), `qo_indptr`/`kv_indptr` (r,) Int
//! on DEVICE (read back per execute), `sinks` (nq,) F32.
//!
//! Output: either the reference chain's heads-major `(nq, s, hd)` layout or
//! the kernel-native token-major `(s, nq * hd)` layout, in F32 or BF16. The
//! token-major spelling is exposed only when the compiler proves the exact
//! transpose/materialize/BF16 consumer boundary, allowing FA3 to write
//! directly into a tensor-core projection input.
//! Decode is the same kernel at qo_len=1 (no SM90 decode-with-sink kernel).
//!
//! The rewrite rule (sink_attention.egg) matches the paged gpt-oss sink
//! attention chain and unions this op in; which host-mask Input feeds the
//! chain ("mask_sliding" vs "mask_full") selects window_left.

use std::sync::Arc;

use luminal::{
    egglog_utils::api::{Rule, SortDef, sort},
    egglog_utils::base::{DTYPE, EXPRESSION, F64, OP_KIND},
    egglog_utils::{SerializedEGraph, extract_dtype, extract_expr},
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::{Expression, Symbol},
};

use crate::cudarc::driver::{CudaStream, DevicePtr, result};

use super::super::{DeviceBuffer, HostOp};
use super::jit;
use super::{
    INT_WORKSPACE_SIZE, bytes_to_i32_vec, page_locked_workspace, sink_attention_workspaces,
};

/// Grow-only device scratch (q transpose in, kernel output out), reused
/// across calls instead of a per-call alloc + trailing sync. Stream-ordered
/// reuse on the same stream is safe (each execute's indptr-readback sync
/// drains prior consumers); on a stream change (tests) the old buffers are
/// leaked rather than dropped, since their context may be gone. Bounded by
/// the largest tick, so leaking on replace/grow costs nothing real.
static SCRATCH: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>, u64)>> =
    std::sync::Mutex::new(None);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PlanKey {
    execution_id: u64,
    stream: usize,
    qo_indptr_ptr: u64,
    qo_indptr_bytes: usize,
    kv_indptr_ptr: u64,
    kv_indptr_bytes: usize,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct CachedPlan {
    key: PlanKey,
    info: [i64; 16],
    info_len: i32,
    nnz_qo: usize,
    total_pages: usize,
}

/// The FA3 planner depends on batch metadata and head geometry, not layer
/// weights or the runtime sliding-window value. All same-shaped attention
/// layers in one model tick can therefore share its output. The execution id
/// deliberately bounds reuse to one runtime call so changed indptr contents
/// at the same device addresses are always observed on the next tick.
static PLAN_CACHE: std::sync::Mutex<Option<CachedPlan>> = std::sync::Mutex::new(None);
static DECODE_PLAN_CACHE: std::sync::Mutex<Option<CachedPlan>> = std::sync::Mutex::new(None);

#[derive(Debug)]
pub struct SinkAttention {
    pub num_qo_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The 's' (batch tokens) dimension expression.
    pub batch_dim: Expression,
    /// Softmax scale; 0.0 = default `1/sqrt(head_dim)`.
    pub sm_scale: f64,
    /// FlashInfer window_left convention (visible previous positions);
    /// -1 = full attention. Selects the swa .so variant at compile time.
    pub window_left: i64,
    /// Graph-visible storage dtype. FA3 writes BF16 natively; F32 retains the
    /// reference chain's dtype through a fused layout transpose + upcast.
    pub output_dtype: DType,
    /// Whether the graph-visible result is FA3's native token-major
    /// `(s, heads * dim)` layout. False retains the reference heads-major
    /// `(heads, s, dim)` layout.
    pub token_major_output: bool,
}

impl Default for SinkAttention {
    fn default() -> Self {
        Self {
            num_qo_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            batch_dim: Expression::default(),
            sm_scale: 0.0,
            window_left: -1,
            output_dtype: DType::F32,
            token_major_output: false,
        }
    }
}

impl EgglogOp for SinkAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "SinkAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("batch_dim", EXPRESSION),
                ("sm_scale", F64),
                ("window_left", F64),
                ("output_dtype", DTYPE),
                ("token_major_output", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        // q, k_pool, v_pool, kv_indices, qo_indptr, kv_indptr, sinks
        7
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The FA3 kernels are Hopper-only (sm_90a WGMMA/TMA): emit no rules
        // on other architectures so the search never selects the op there.
        if crate::device_compute_major() != 9 {
            return vec![];
        }
        vec![Rule::raw(include_str!("sink_attention.egg"))]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let num_qo_heads = extract_expr(egraph, kind_children[0], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let num_kv_heads = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let head_dim = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .exec(&FxHashMap::default())
            .unwrap();
        let batch_dim = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let sm_scale: f64 = egraph.enodes[kind_children[4]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let window_left = egraph.enodes[kind_children[5]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            .round() as i64;
        let output_dtype = extract_dtype(egraph, kind_children[6]);
        let token_major_output = egraph.enodes[kind_children[7]]
            .0
            .replace('"', "")
            .parse::<f64>()
            .unwrap()
            != 0.0;

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            batch_dim,
            sm_scale,
            window_left,
            output_dtype,
            token_major_output,
        };

        // JIT at extract time so the ~45s nvcc cost never lands inside a
        // GA profiling trial (same rationale as FlashInferAttention).
        let _ = jit::ensure_compiled_fa3(head_dim, window_left >= 0);
        let _ = jit::ensure_compiled(head_dim, window_left >= 0);

        // The rule passes the FLAT gather index (proof anchor); recover the
        // compact per-token page table the kernel consumes.
        let flat_idx_node = input_enodes[3];
        let gather_idx = super::find_indptrs::try_find_compact_gather_idx(egraph, flat_idx_node)
            .expect("SinkAttention matched a gather without recoverable compact gather_idx");
        let final_inputs = vec![
            input_enodes[0], // q (bf16)
            input_enodes[1], // k_pool
            input_enodes[2], // v_pool
            gather_idx,      // compact kv_indices
            input_enodes[4], // qo_indptr
            input_enodes[5], // kv_indptr
            input_enodes[6], // sinks (f32)
        ];

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        (op, final_inputs)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl SinkAttention {
    fn external_fa3_library(&self) -> Option<String> {
        (self.head_dim == 64 && self.output_dtype == DType::Bf16 && self.token_major_output)
            .then(|| std::env::var("LUMINAL_EXTERNAL_FA3_LIBRARY").ok())
            .flatten()
    }

    fn initialize_external_fa3(&self) -> anyhow::Result<Option<&'static jit::Fa3Lib>> {
        let Some(path) = self.external_fa3_library() else {
            return Ok(None);
        };
        let path = std::ffi::CString::new(path)
            .map_err(|_| anyhow::anyhow!("LUMINAL_EXTERNAL_FA3_LIBRARY contains a NUL byte"))?;
        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let ret = unsafe { (lib.external_init)(path.as_ptr()) };
        anyhow::ensure!(ret == 0, "SinkAttention: external FA3 init failed ({ret})");
        Ok(Some(lib))
    }

    /// Return `(request_count, total_pages)` when the descriptor is a uniform
    /// one-query-row decode batch. Serving buffers carry authoritative host
    /// mirrors, so this check adds no device synchronization. Direct callers
    /// without mirrors simply retain the planned FA3 prefill path.
    fn uniform_decode_geometry(
        &self,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> Option<(usize, usize)> {
        let supported_output = (self.output_dtype == DType::Bf16)
            || (self.output_dtype == DType::F32 && !self.token_major_output);
        if !supported_output
            || std::env::var("LUMINAL_CUDA_SINK_DECODE").ok().as_deref() == Some("0")
        {
            return None;
        }
        let logical_rows = self.batch_dim.exec(dyn_map)?;
        // FA3's persistent scheduler has lower launch overhead for very small
        // batches; the parallel FA2 decoder wins once there are enough query
        // heads to fill the device. Keep the crossover configurable for other
        // GPUs while using the measured general serving default here.
        let min_batch = std::env::var("LUMINAL_CUDA_SINK_DECODE_MIN_BATCH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(if self.external_fa3_library().is_some() {
                1
            } else {
                3
            });
        if logical_rows < min_batch {
            return None;
        }
        // Captured child graphs intentionally keep only device buffers. The
        // serving ABI's explicit request-row and context symbols are the
        // authoritative geometry there; this is also the recapture boundary.
        if let Some(request_rows) = dyn_map.get(&Symbol::from('r')).copied()
            && request_rows >= 1
            && logical_rows + 1 == request_rows
        {
            let total_pages = dyn_map
                .get(&Symbol::from('c'))
                .copied()
                .or_else(|| {
                    buffers
                        .get(inputs.get(5)?)?
                        .host_bytes()
                        .map(|bytes| bytes_to_i32_vec(bytes.to_vec()))
                        .and_then(|values| usize::try_from(*values.last()?).ok())
                })
                .unwrap_or_else(|| {
                    buffers
                        .get(inputs.get(3).unwrap())
                        .map_or(0, |buffer| buffer.len() / std::mem::size_of::<i32>())
                });
            return Some((logical_rows, total_pages));
        }

        // The final/search child graph may not retain `r` or host mirrors, but
        // DeviceBuffer lengths remain the exact logical tensor lengths even
        // when storage comes from a larger arena slot. Page-size-one decode
        // has matching i32 qo/kv indptr tensors of batch + 1 entries, so their
        // shape is sufficient to recover the graph's fixed request count.
        let qo_indptr = buffers.get(inputs.get(4)?)?;
        let kv_indptr = buffers.get(inputs.get(5)?)?;
        let i32_bytes = std::mem::size_of::<i32>();
        if qo_indptr.len() == kv_indptr.len()
            && qo_indptr.len() >= 2 * i32_bytes
            && qo_indptr.len() % i32_bytes == 0
        {
            let batch_size = qo_indptr.len() / i32_bytes - 1;
            if batch_size == logical_rows {
                let total_pages = dyn_map.get(&Symbol::from('c')).copied().unwrap_or_else(|| {
                    buffers
                        .get(inputs.get(3).unwrap())
                        .map_or(0, |buffer| buffer.len() / i32_bytes)
                });
                return Some((batch_size, total_pages));
            }
        }

        let qo = bytes_to_i32_vec(buffers.get(inputs.get(4)?)?.host_bytes()?.to_vec());
        let kv = bytes_to_i32_vec(buffers.get(inputs.get(5)?)?.host_bytes()?.to_vec());
        if qo.len() != kv.len()
            || qo.len() < 2
            || qo[0] != 0
            || kv[0] != 0
            || !qo.windows(2).all(|w| w[1] == w[0] + 1)
            || !kv.windows(2).all(|w| w[1] >= w[0])
        {
            return None;
        }
        let batch_size = qo.len() - 1;
        if logical_rows != batch_size {
            return None;
        }
        let total_pages = usize::try_from(*kv.last()?).ok()?;
        Some((batch_size, total_pages))
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_decode(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        execution_id: u64,
        enable_cuda_graph: bool,
        batch_size: usize,
        total_pages: usize,
    ) -> anyhow::Result<CachedPlan> {
        let qo_indptr_buf = buffers
            .get(&inputs[4])
            .copied()
            .ok_or_else(|| anyhow::anyhow!("SinkAttention decode: missing qo_indptr"))?;
        let kv_indptr_buf = buffers
            .get(&inputs[5])
            .copied()
            .ok_or_else(|| anyhow::anyhow!("SinkAttention decode: missing kv_indptr"))?;
        let key = PlanKey {
            execution_id,
            stream: stream.cu_stream() as usize,
            qo_indptr_ptr: qo_indptr_buf.ptr(),
            qo_indptr_bytes: qo_indptr_buf.len(),
            kv_indptr_ptr: kv_indptr_buf.ptr(),
            kv_indptr_bytes: kv_indptr_buf.len(),
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        };
        if let Some(plan) = DECODE_PLAN_CACHE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .filter(|plan| plan.key == key)
        {
            return Ok(plan);
        }

        // The specialized decoder's schedule is identity/non-split and does
        // not depend on context lengths. Supply a shape-correct host array to
        // the stable C ABI; the live device indptr remains the kernel input.
        let mut kv_indptr = vec![0i32; batch_size + 1];
        let lib = jit::ensure_compiled(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = sink_attention_workspaces(stream);
        let page_locked = page_locked_workspace();
        let mut info = [0i64; 16];
        let mut info_len = 0i32;
        let ret = unsafe {
            (lib.sink_decode_plan)(
                float_ws_ptr as *mut std::ffi::c_void,
                super::FLOAT_WORKSPACE_SIZE,
                int_ws_ptr as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                page_locked.0 as *mut std::ffi::c_void,
                kv_indptr.as_mut_ptr(),
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                enable_cuda_graph,
                stream.cu_stream() as *mut std::ffi::c_void,
                info.as_mut_ptr(),
                &mut info_len,
            )
        };
        anyhow::ensure!(ret == 0, "SinkAttention: decode plan failed ({ret})");
        anyhow::ensure!(
            (0..=info.len() as i32).contains(&info_len),
            "SinkAttention: invalid decode plan length {info_len}"
        );
        let plan = CachedPlan {
            key,
            info,
            info_len,
            nnz_qo: batch_size,
            total_pages,
        };
        *DECODE_PLAN_CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(plan);
        Ok(plan)
    }

    fn plan(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
        enable_cuda_graph: bool,
    ) -> anyhow::Result<CachedPlan> {
        anyhow::ensure!(
            inputs.len() == 7,
            "SinkAttention expects 7 inputs, got {}",
            inputs.len()
        );
        let qo_indptr_buf = buffers
            .get(&inputs[4])
            .copied()
            .ok_or_else(|| anyhow::anyhow!("SinkAttention: missing buffer for qo_indptr"))?;
        let kv_indptr_buf = buffers
            .get(&inputs[5])
            .copied()
            .ok_or_else(|| anyhow::anyhow!("SinkAttention: missing buffer for kv_indptr"))?;
        let key = PlanKey {
            execution_id,
            stream: stream.cu_stream() as usize,
            qo_indptr_ptr: qo_indptr_buf.ptr(),
            qo_indptr_bytes: qo_indptr_buf.len(),
            kv_indptr_ptr: kv_indptr_buf.ptr(),
            kv_indptr_bytes: kv_indptr_buf.len(),
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        };
        if let Some(plan) = PLAN_CACHE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .filter(|plan| plan.key == key)
        {
            return Ok(plan);
        }

        // Serving runtimes retain authoritative host mirrors for these tiny
        // dynamic inputs, avoiding a GPU round trip and the mid-graph
        // synchronization it would require. Direct HostOp callers retain the
        // device-readback fallback.
        let (qo_bytes, kv_bytes) = match (qo_indptr_buf.host_bytes(), kv_indptr_buf.host_bytes()) {
            (Some(qo), Some(kv)) => (qo.to_vec(), kv.to_vec()),
            _ => {
                let mut qo = vec![0u8; qo_indptr_buf.len()];
                let mut kv = vec![0u8; kv_indptr_buf.len()];
                unsafe {
                    result::memcpy_dtoh_async(&mut qo, qo_indptr_buf.ptr(), stream.cu_stream())?;
                    result::memcpy_dtoh_async(&mut kv, kv_indptr_buf.ptr(), stream.cu_stream())?;
                }
                stream.synchronize()?;
                (qo, kv)
            }
        };
        let mut qo_indptr = bytes_to_i32_vec(qo_bytes);
        let mut kv_indptr = bytes_to_i32_vec(kv_bytes);
        anyhow::ensure!(
            qo_indptr.len() == kv_indptr.len() && qo_indptr.len() >= 2,
            "SinkAttention: malformed indptrs (qo len {}, kv len {})",
            qo_indptr.len(),
            kv_indptr.len()
        );
        let batch_size = qo_indptr.len() - 1;
        let logical_nnz_qo = self
            .batch_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("SinkAttention: unresolved batch dimension"))?;
        let mirrored_nnz_qo = usize::try_from(*qo_indptr.last().unwrap())
            .map_err(|_| anyhow::anyhow!("SinkAttention: negative qo total"))?;
        if mirrored_nnz_qo != logical_nnz_qo {
            // Search compiles several dynamic buckets after the serving graph
            // has uploaded one high-water dummy descriptor. The sole end
            // pointer of a one-request profiling descriptor is recoverable
            // from the bucket shape; real multi-request descriptors must
            // agree exactly because their interior boundaries are semantic.
            anyhow::ensure!(
                batch_size == 1,
                "SinkAttention: qo_indptr total {mirrored_nnz_qo} does not match dynamic batch {logical_nnz_qo} for {batch_size} requests"
            );
            anyhow::ensure!(
                logical_nnz_qo <= i32::MAX as usize,
                "SinkAttention: dynamic batch {logical_nnz_qo} exceeds i32"
            );
            qo_indptr[1] = logical_nnz_qo as i32;
        }
        let nnz_qo = usize::try_from(*qo_indptr.last().unwrap())
            .map_err(|_| anyhow::anyhow!("SinkAttention: negative qo total"))?;
        let total_pages = usize::try_from(*kv_indptr.last().unwrap())
            .map_err(|_| anyhow::anyhow!("SinkAttention: negative kv total"))?;
        let mut kv_len_arr: Vec<i32> = kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();

        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = sink_attention_workspaces(stream);
        let page_locked = page_locked_workspace();
        let mut info = [0i64; 16];
        let mut info_len = 0i32;
        let plan_ret = unsafe {
            (lib.prefill_plan)(
                float_ws_ptr as *mut std::ffi::c_void,
                super::FLOAT_WORKSPACE_SIZE,
                int_ws_ptr as *mut std::ffi::c_void,
                page_locked.0 as *mut std::ffi::c_void,
                INT_WORKSPACE_SIZE,
                qo_indptr.as_mut_ptr(),
                kv_indptr.as_mut_ptr(),
                kv_len_arr.as_mut_ptr(),
                nnz_qo as i32,
                batch_size as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                /*page_size=*/ 1,
                enable_cuda_graph,
                stream.cu_stream() as *mut std::ffi::c_void,
                info.as_mut_ptr(),
                &mut info_len,
            )
        };
        anyhow::ensure!(plan_ret == 0, "SinkAttention: fa3 plan failed ({plan_ret})");
        anyhow::ensure!(
            (0..=info.len() as i32).contains(&info_len),
            "SinkAttention: invalid plan length {info_len}"
        );
        let plan = CachedPlan {
            key,
            info,
            info_len,
            nnz_qo,
            total_pages,
        };
        *PLAN_CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(plan);
        Ok(plan)
    }

    fn scratch_base(stream: &Arc<CudaStream>, scratch_bytes: usize) -> anyhow::Result<u64> {
        let stream_key = stream.cu_stream() as usize;
        let mut scratch = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let needs_new = !matches!(&*scratch,
            Some((key, buf, _)) if *key == stream_key && buf.len() >= scratch_bytes);
        if needs_new {
            let buf = unsafe { stream.alloc::<u8>(scratch_bytes.next_power_of_two())? };
            if let Some((_, old, _)) = scratch.take() {
                // Captured child graphs can retain the old address after the
                // process switches from its capture stream to the run stream.
                std::mem::forget(old);
            }
            let ptr = buf.device_ptr(stream).0;
            *scratch = Some((stream_key, buf, ptr));
        }
        Ok(scratch.as_ref().unwrap().2)
    }

    /// Allocate fixed-address scratch and upload a graph-capacity schedule
    /// before standalone capture begins. `execute` then records only GPU work.
    pub(crate) fn prepare_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        if let Some((batch_size, total_pages)) =
            self.uniform_decode_geometry(inputs, buffers, dyn_map)
        {
            if self.initialize_external_fa3()?.is_some() {
                let _ = sink_attention_workspaces(stream);
                return Ok(());
            }
            self.plan_decode(stream, inputs, buffers, 0, true, batch_size, total_pages)?;
            if self.output_dtype != DType::Bf16 || !self.token_major_output {
                let temp_bytes = (batch_size * self.num_qo_heads * self.head_dim * 2).max(1);
                Self::scratch_base(stream, temp_bytes)?;
            }
            return Ok(());
        }
        let plan = self.plan(stream, inputs, buffers, dyn_map, 0, true)?;
        let temp_bytes = (plan.nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        let q_layout_is_token_major = plan.nnz_qo == 1;
        let output_layout_is_graph_native =
            self.output_dtype == DType::Bf16 && (self.token_major_output || plan.nnz_qo == 1);
        let scratch_slots =
            usize::from(!q_layout_is_token_major) + usize::from(!output_layout_is_graph_native);
        Self::scratch_base(stream, temp_bytes * scratch_slots.max(1))?;
        Ok(())
    }

    /// Refresh the graph-stable FA3 schedule once immediately before the
    /// surrounding model graph is launched. All captured sink-attention ops
    /// in that graph share the descriptor and integer workspace.
    pub(crate) fn prepare_graph_execution(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        if let Some((batch_size, total_pages)) =
            self.uniform_decode_geometry(inputs, buffers, dyn_map)
        {
            if self.initialize_external_fa3()?.is_some() {
                return Ok(());
            }
            self.plan_decode(
                stream,
                inputs,
                buffers,
                execution_id,
                true,
                batch_size,
                total_pages,
            )?;
            return Ok(());
        }
        self.plan(stream, inputs, buffers, dyn_map, execution_id, true)?;
        Ok(())
    }
}

impl HostOp for SinkAttention {
    fn prepare_cuda_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        SinkAttention::prepare_graph_capture(self, stream, inputs, buffers, dyn_map)
    }

    fn prepare_cuda_graph_execution(
        &self,
        stream: &Arc<CudaStream>,
        _self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        SinkAttention::prepare_graph_execution(self, stream, inputs, buffers, dyn_map, execution_id)
    }

    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
    ) -> anyhow::Result<()> {
        self.execute_with_id(stream, self_node, inputs, buffers, dyn_map, 0)
    }

    fn execute_with_id(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &DynMap,
        execution_id: u64,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            inputs.len() == 7,
            "SinkAttention expects 7 inputs, got {}",
            inputs.len()
        );
        let buf = |n: NodeIndex, what: &str| -> anyhow::Result<DeviceBuffer> {
            buffers
                .get(&n)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("SinkAttention: missing buffer for {what}"))
        };
        let q = buf(inputs[0], "q")?;
        let k_pool = buf(inputs[1], "k_pool")?;
        let v_pool = buf(inputs[2], "v_pool")?;
        let kv_indices = buf(inputs[3], "kv_indices")?;
        let qo_indptr_buf = buf(inputs[4], "qo_indptr")?;
        let kv_indptr_buf = buf(inputs[5], "kv_indptr")?;
        let sinks = buf(inputs[6], "sinks")?;
        let out = buf(self_node, "output")?;

        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        let decode_geometry = self.uniform_decode_geometry(inputs, buffers, dyn_map);
        if std::env::var_os("LUMINAL_CUDA_DEBUG_SINK_DECODE").is_some() {
            eprintln!(
                "SinkAttention decode candidate: geometry={decode_geometry:?} dtype={:?} token_major={} rows={:?} qo_bytes={} kv_bytes={} q_host={} kv_host={} s={:?} r={:?} c={:?}",
                self.output_dtype,
                self.token_major_output,
                self.batch_dim.exec(dyn_map),
                qo_indptr_buf.len(),
                kv_indptr_buf.len(),
                qo_indptr_buf.host_bytes().is_some(),
                kv_indptr_buf.host_bytes().is_some(),
                dyn_map.get(&Symbol::from('s')),
                dyn_map.get(&Symbol::from('r')),
                dyn_map.get(&Symbol::from('c')),
            );
        }
        if let Some((batch_size, total_pages)) = decode_geometry {
            let native_bytes = batch_size * self.num_qo_heads * self.head_dim * 2;
            let output_bytes =
                batch_size * self.num_qo_heads * self.head_dim * (self.output_dtype.bits() / 8);
            anyhow::ensure!(
                q.capacity() >= native_bytes && out.capacity() >= output_bytes,
                "SinkAttention decode q/output is too small for batch {batch_size}"
            );
            anyhow::ensure!(
                kv_indices.len() >= total_pages * std::mem::size_of::<i32>(),
                "SinkAttention decode kv_indices buffer smaller than kv_indptr total"
            );
            let sm_scale = if self.sm_scale == 0.0 {
                1.0 / (self.head_dim as f32).sqrt()
            } else {
                self.sm_scale as f32
            };
            if let Some(external) = self.initialize_external_fa3()? {
                let max_context_len = std::env::var("LUMINAL_EXTERNAL_FA3_MAX_CONTEXT")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(4096);
                let page_size = std::env::var("LUMINAL_KV_PAGE_SIZE")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&value| value > 0 && value.is_power_of_two())
                    .unwrap_or(1);
                let page_bytes = self.num_kv_heads * self.head_dim * 2 * page_size;
                anyhow::ensure!(
                    page_bytes > 0 && k_pool.len().is_multiple_of(page_bytes),
                    "SinkAttention: malformed BF16 KV cache size"
                );
                let num_pages = k_pool.len() / page_bytes;
                let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) =
                    sink_attention_workspaces(stream);
                let run_ret = unsafe {
                    (external.external_sink_decode_run)(
                        float_ws_ptr as *mut std::ffi::c_void,
                        super::FLOAT_WORKSPACE_SIZE,
                        int_ws_ptr as *mut std::ffi::c_void,
                        INT_WORKSPACE_SIZE,
                        q.ptr() as *const std::ffi::c_void,
                        k_pool.ptr() as *const std::ffi::c_void,
                        v_pool.ptr() as *const std::ffi::c_void,
                        kv_indices.ptr() as *const i32,
                        qo_indptr_buf.ptr() as *const i32,
                        kv_indptr_buf.ptr() as *const i32,
                        sinks.ptr() as *const f32,
                        out.ptr() as *mut std::ffi::c_void,
                        batch_size as i32,
                        self.num_qo_heads as i32,
                        self.num_kv_heads as i32,
                        max_context_len as i32,
                        num_pages as i32,
                        page_size as i32,
                        sm_scale,
                        self.window_left as i32,
                        cu_stream,
                    )
                };
                anyhow::ensure!(
                    run_ret == 0,
                    "SinkAttention: external FA3 decode failed ({run_ret})"
                );
                return Ok(());
            }
            let mut plan = self.plan_decode(
                stream,
                inputs,
                buffers,
                execution_id,
                false,
                batch_size,
                total_pages,
            )?;
            let decode_lib = jit::ensure_compiled(self.head_dim, self.window_left >= 0);
            let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = sink_attention_workspaces(stream);
            let direct_native_output = self.output_dtype == DType::Bf16 && self.token_major_output;
            let native_output_ptr = if direct_native_output {
                out.ptr()
            } else {
                Self::scratch_base(stream, native_bytes.max(1))?
            };
            let run_ret = unsafe {
                (decode_lib.sink_decode_run)(
                    float_ws_ptr as *mut std::ffi::c_void,
                    int_ws_ptr as *mut std::ffi::c_void,
                    plan.info.as_mut_ptr(),
                    plan.info_len,
                    q.ptr() as *const std::ffi::c_void,
                    k_pool.ptr() as *const std::ffi::c_void,
                    v_pool.ptr() as *const std::ffi::c_void,
                    kv_indptr_buf.ptr() as *mut i32,
                    kv_indices.ptr() as *mut i32,
                    sinks.ptr() as *const f32,
                    native_output_ptr as *mut std::ffi::c_void,
                    batch_size as i32,
                    self.num_qo_heads as i32,
                    self.num_kv_heads as i32,
                    sm_scale,
                    self.window_left as i32,
                    cu_stream,
                )
            };
            anyhow::ensure!(
                run_ret == 0,
                "SinkAttention: specialized decode failed ({run_ret})"
            );
            if !direct_native_output {
                let fa3_lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
                let transpose = match self.output_dtype {
                    DType::F32 => fa3_lib.transpose_output_f32,
                    DType::Bf16 => fa3_lib.transpose_output_bf16,
                    dtype => anyhow::bail!(
                        "SinkAttention: unsupported specialized decode output {dtype:?}"
                    ),
                };
                let ret = unsafe {
                    transpose(
                        native_output_ptr as *const std::ffi::c_void,
                        out.ptr() as *mut std::ffi::c_void,
                        batch_size as i32,
                        self.num_qo_heads as i32,
                        self.head_dim as i32,
                        cu_stream,
                    )
                };
                anyhow::ensure!(ret == 0, "SinkAttention: decode output transpose failed");
            }
            return Ok(());
        }
        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = sink_attention_workspaces(stream);

        let key = PlanKey {
            execution_id,
            stream: stream.cu_stream() as usize,
            qo_indptr_ptr: qo_indptr_buf.ptr(),
            qo_indptr_bytes: qo_indptr_buf.len(),
            kv_indptr_ptr: kv_indptr_buf.ptr(),
            kv_indptr_bytes: kv_indptr_buf.len(),
            num_qo_heads: self.num_qo_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        };
        let cached = PLAN_CACHE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .filter(|plan| plan.key == key);

        let mut plan = if let Some(plan) = cached {
            plan
        } else {
            // Serving runtimes can retain authoritative host mirrors for
            // these tiny dynamic inputs, avoiding a GPU round trip and the
            // mid-graph synchronization it requires. Direct HostOp callers
            // and other integrations retain the device-readback fallback.
            let (qo_bytes, kv_bytes) =
                match (qo_indptr_buf.host_bytes(), kv_indptr_buf.host_bytes()) {
                    (Some(qo), Some(kv)) => (qo.to_vec(), kv.to_vec()),
                    _ => {
                        // Both copies are independent: queue them together and
                        // synchronize once rather than once per indptr.
                        let mut qo = vec![0u8; qo_indptr_buf.len()];
                        let mut kv = vec![0u8; kv_indptr_buf.len()];
                        unsafe {
                            result::memcpy_dtoh_async(
                                &mut qo,
                                qo_indptr_buf.ptr(),
                                stream.cu_stream(),
                            )?;
                            result::memcpy_dtoh_async(
                                &mut kv,
                                kv_indptr_buf.ptr(),
                                stream.cu_stream(),
                            )?;
                        }
                        stream.synchronize()?;
                        (qo, kv)
                    }
                };
            let mut qo_indptr = bytes_to_i32_vec(qo_bytes);
            let mut kv_indptr = bytes_to_i32_vec(kv_bytes);
            anyhow::ensure!(
                qo_indptr.len() == kv_indptr.len() && qo_indptr.len() >= 2,
                "SinkAttention: malformed indptrs (qo len {}, kv len {})",
                qo_indptr.len(),
                kv_indptr.len()
            );
            let batch_size = qo_indptr.len() - 1;
            let logical_nnz_qo = self
                .batch_dim
                .exec(dyn_map)
                .ok_or_else(|| anyhow::anyhow!("SinkAttention: unresolved batch dimension"))?;
            let mirrored_nnz_qo = usize::try_from(*qo_indptr.last().unwrap())
                .map_err(|_| anyhow::anyhow!("SinkAttention: negative qo total"))?;
            if mirrored_nnz_qo != logical_nnz_qo {
                // Search compiles several dynamic buckets after the serving
                // graph has uploaded one high-water dummy descriptor. For the
                // one-request profiling descriptor, specialize its sole end
                // pointer to the bucket's actual dynamic row count. Real
                // multi-request descriptors must already agree exactly; their
                // interior boundaries cannot be reconstructed from a shape.
                anyhow::ensure!(
                    batch_size == 1,
                    "SinkAttention: qo_indptr total {mirrored_nnz_qo} does not match dynamic batch {logical_nnz_qo} for {batch_size} requests"
                );
                anyhow::ensure!(
                    logical_nnz_qo <= i32::MAX as usize,
                    "SinkAttention: dynamic batch {logical_nnz_qo} exceeds i32"
                );
                qo_indptr[1] = logical_nnz_qo as i32;
            }
            let nnz_qo = usize::try_from(*qo_indptr.last().unwrap())
                .map_err(|_| anyhow::anyhow!("SinkAttention: negative qo total"))?;
            let total_pages = usize::try_from(*kv_indptr.last().unwrap())
                .map_err(|_| anyhow::anyhow!("SinkAttention: negative kv total"))?;
            let mut kv_len_arr: Vec<i32> = kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();

            let page_locked = page_locked_workspace();
            let mut info = [0i64; 16];
            let mut info_len = 0i32;
            let plan_ret = unsafe {
                (lib.prefill_plan)(
                    float_ws_ptr as *mut std::ffi::c_void,
                    super::FLOAT_WORKSPACE_SIZE,
                    int_ws_ptr as *mut std::ffi::c_void,
                    page_locked.0 as *mut std::ffi::c_void,
                    INT_WORKSPACE_SIZE,
                    qo_indptr.as_mut_ptr(),
                    kv_indptr.as_mut_ptr(),
                    kv_len_arr.as_mut_ptr(),
                    nnz_qo as i32,
                    batch_size as i32,
                    self.num_qo_heads as i32,
                    self.num_kv_heads as i32,
                    /*page_size=*/ 1,
                    false,
                    cu_stream,
                    info.as_mut_ptr(),
                    &mut info_len,
                )
            };
            anyhow::ensure!(plan_ret == 0, "SinkAttention: fa3 plan failed ({plan_ret})");
            anyhow::ensure!(
                (0..=info.len() as i32).contains(&info_len),
                "SinkAttention: invalid plan length {info_len}"
            );
            let plan = CachedPlan {
                key,
                info,
                info_len,
                nnz_qo,
                total_pages,
            };
            *PLAN_CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(plan);
            plan
        };

        let nnz_qo = plan.nnz_qo;
        let total_pages = plan.total_pages;
        anyhow::ensure!(
            kv_indices.len() >= total_pages * std::mem::size_of::<i32>(),
            "SinkAttention: kv_indices buffer smaller than kv_indptr total"
        );

        // Kernel-native (s, heads, dim) BF16 scratch, from the grow-only pool.
        // Reuse is stream-ordered; the refresh-path sync above covers growth.
        let temp_bytes = (nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        anyhow::ensure!(
            q.capacity() >= temp_bytes && out.capacity() >= temp_bytes,
            "SinkAttention q/output capacity is too small: q={}/{} out={}/{} required={temp_bytes} bytes for nnz_qo={nnz_qo}",
            q.len(),
            q.capacity(),
            out.len(),
            out.capacity(),
        );
        // With one query token, (heads, 1, dim) and (1, heads, dim) are the
        // same contiguous byte order. Reuse q directly and reserve scratch
        // only for the kernel output. Batched queries retain the explicit
        // heads-major -> token-major transpose below.
        let q_layout_is_token_major = nnz_qo == 1;
        // A singleton BF16 result is already byte-identical in the graph and
        // kernel layouts, so FA3 can write the graph output directly. Other
        // BF16 batches only need the output-layout transpose; F32 additionally
        // needs the native temporary before transpose+upcast.
        let output_layout_is_graph_native =
            self.output_dtype == DType::Bf16 && (self.token_major_output || nnz_qo == 1);
        let scratch_slots =
            usize::from(!q_layout_is_token_major) + usize::from(!output_layout_is_graph_native);
        let scratch_bytes = temp_bytes * scratch_slots.max(1);
        let mut scratch_guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let stream_key = key.stream;
        let needs_new = !matches!(&*scratch_guard,
            Some((key, buf, _)) if *key == stream_key && buf.len() >= scratch_bytes);
        if needs_new {
            // Allocation and old-buffer retirement are stream ordered. On a
            // cache hit an earlier same-execution layer already established a
            // sufficiently large allocation for this batch shape.
            let buf = unsafe { stream.alloc::<u8>(scratch_bytes.next_power_of_two())? };
            if let Some((_, old, _)) = scratch_guard.take() {
                std::mem::forget(old); // context may be gone; never free
            }
            let ptr = buf.device_ptr(stream).0;
            *scratch_guard = Some((stream_key, buf, ptr));
        }
        // Cache the raw pointer at allocation time. Calling `device_ptr` while
        // a parent stream capture is active can make cudarc insert allocator
        // event dependencies and invalidate otherwise legal capture.
        let base_ptr = scratch_guard.as_ref().unwrap().2;
        let q_temp_ptr = if q_layout_is_token_major {
            q.ptr()
        } else {
            base_ptr
        };
        let native_output_ptr = if output_layout_is_graph_native {
            out.ptr()
        } else {
            base_ptr + usize::from(!q_layout_is_token_major) as u64 * temp_bytes as u64
        };

        // A HostOp output is not in-place. Catch an allocator/extractor
        // regression before handing an overlapping range to FA3, where the
        // eventual stream synchronization would otherwise report only an
        // opaque illegal address after the responsible launch has returned.
        if output_layout_is_graph_native {
            let out_start = out.ptr();
            let out_end = out_start + temp_bytes as u64;
            for (name, input) in [
                ("q", q),
                ("k_pool", k_pool),
                ("v_pool", v_pool),
                ("kv_indices", kv_indices),
                ("qo_indptr", qo_indptr_buf),
                ("kv_indptr", kv_indptr_buf),
                ("sinks", sinks),
            ] {
                let input_start = input.ptr();
                let input_end = input_start + input.len() as u64;
                anyhow::ensure!(
                    out_end <= input_start || input_end <= out_start,
                    "SinkAttention token-major output [{out_start:#x}, {out_end:#x}) overlaps {name} input [{input_start:#x}, {input_end:#x})"
                );
            }
        }

        // The graph's q is (heads, s, dim) — the same heads-major layout
        // world as the output point — but the kernel reads token-major
        // (s, heads, dim) q. The layouts are byte-identical at s == 1
        // (decode), which is how this survived every single-token path;
        // prefill (s > 1) needs the transpose.
        if !q_layout_is_token_major {
            let qtr_ret = unsafe {
                (lib.transpose_q_bf16)(
                    q.ptr() as *const std::ffi::c_void,
                    q_temp_ptr as *mut std::ffi::c_void,
                    nnz_qo as i32,
                    self.num_qo_heads as i32,
                    self.head_dim as i32,
                    cu_stream,
                )
            };
            anyhow::ensure!(
                qtr_ret == 0,
                "SinkAttention: q transpose failed ({qtr_ret})"
            );
        }

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let run_ret = unsafe {
            (lib.prefill_run)(
                int_ws_ptr as *mut std::ffi::c_void,
                plan.info.as_mut_ptr(),
                plan.info_len,
                q_temp_ptr as *mut std::ffi::c_void,
                k_pool.ptr() as *mut std::ffi::c_void,
                v_pool.ptr() as *mut std::ffi::c_void,
                kv_indices.ptr() as *mut i32,
                sinks.ptr() as *mut f32,
                native_output_ptr as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.num_kv_heads as i32,
                /*page_size=*/ 1,
                sm_scale,
                self.window_left as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(run_ret == 0, "SinkAttention: fa3 run failed ({run_ret})");

        if !output_layout_is_graph_native {
            let transpose = match self.output_dtype {
                DType::F32 => lib.transpose_output_f32,
                DType::Bf16 => lib.transpose_output_bf16,
                dtype => anyhow::bail!("SinkAttention: unsupported output dtype {dtype:?}"),
            };
            let tr_ret = unsafe {
                transpose(
                    native_output_ptr as *const std::ffi::c_void,
                    out.ptr() as *mut std::ffi::c_void,
                    nnz_qo as i32,
                    self.num_qo_heads as i32,
                    self.head_dim as i32,
                    cu_stream,
                )
            };
            anyhow::ensure!(tr_ret == 0, "SinkAttention: output transpose failed");
        }

        // No trailing sync: scratch is pooled (never freed), so the enqueued
        // kernels own it under stream ordering; the next tick's plan-refresh
        // path syncs before touching shared host buffers or growing scratch.
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.output_dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.output_dtype
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("SinkAttention")
    }
}
