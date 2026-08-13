//! `SinkAttention` — host op wrapping the FA3/Hopper AttentionSink paged
//! batch-prefill kernel.
//!
//! Runtime inputs (7): `q` (s, nq*hd) bf16, `k_pool`/`v_pool`
//! (num_slots, nkv*hd) bf16 (post-scatter pool states), `kv_indices` (c,)
//! Int (compact page table, page_size 1), `qo_indptr`/`kv_indptr` (r,) Int
//! on DEVICE (read back per execute), `sinks` (nq,) F32.
//!
//! Output: (nq, s, hd) F32 — the layout+dtype of the reference chain's
//! attention output point, produced by a fused transpose+upcast kernel.
//! Decode is the same kernel at qo_len=1 (no SM90 decode-with-sink kernel).
//!
//! The rewrite rule (sink_attention.egg) matches the paged gpt-oss sink
//! attention chain and unions this op in; which host-mask Input feeds the
//! chain ("mask_sliding" vs "mask_full") selects window_left.

use std::sync::Arc;

use luminal::{
    egglog_utils::api::{Rule, SortDef, sort},
    egglog_utils::base::{EXPRESSION, F64, OP_KIND},
    egglog_utils::{SerializedEGraph, extract_expr},
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::cudarc::driver::{CudaStream, DevicePtr, result};

use super::super::{DeviceBuffer, HostOp};
use super::jit;
use super::{INT_WORKSPACE_SIZE, bytes_to_i32_vec, flashinfer_workspaces, page_locked_workspace};

/// Grow-only device scratch (q transpose in, kernel output out), reused
/// across calls instead of a per-call alloc + trailing sync. Stream-ordered
/// reuse on the same stream is safe (each execute's indptr-readback sync
/// drains prior consumers); on a stream change (tests) the old buffers are
/// leaked rather than dropped, since their context may be gone. Bounded by
/// the largest tick, so leaking on replace/grow costs nothing real.
static SCRATCH: std::sync::Mutex<Option<(usize, crate::cudarc::driver::CudaSlice<u8>)>> =
    std::sync::Mutex::new(None);

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

        let extracted = Self {
            num_qo_heads,
            num_kv_heads,
            head_dim,
            batch_dim,
            sm_scale,
            window_left,
        };

        // JIT at extract time so the ~45s nvcc cost never lands inside a
        // GA profiling trial (same rationale as FlashInferAttention).
        let _ = jit::ensure_compiled_fa3(head_dim, window_left >= 0);

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

impl HostOp for SinkAttention {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _dyn_map: &DynMap,
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

        let lib = jit::ensure_compiled_fa3(self.head_dim, self.window_left >= 0);
        let (_float_ws, float_ws_ptr, _int_ws, int_ws_ptr) = flashinfer_workspaces(stream);

        // Read the indptrs back to the host: the FA3 plan is host-side and
        // runs once per execute. The first read's synchronize also drains
        // the previous execute's async traffic out of the shared pinned plan
        // buffer and the scratch pools before this call rewrites them.
        let read_device_i32s = |b: DeviceBuffer| -> anyhow::Result<Vec<i32>> {
            let mut host_bytes = vec![0u8; b.len()];
            unsafe {
                result::memcpy_dtoh_async(&mut host_bytes, b.ptr(), stream.cu_stream())?;
            }
            stream.synchronize()?;
            Ok(bytes_to_i32_vec(host_bytes))
        };
        let mut qo_indptr = read_device_i32s(qo_indptr_buf)?;
        let mut kv_indptr = read_device_i32s(kv_indptr_buf)?;
        anyhow::ensure!(
            qo_indptr.len() == kv_indptr.len() && qo_indptr.len() >= 2,
            "SinkAttention: malformed indptrs (qo len {}, kv len {})",
            qo_indptr.len(),
            kv_indptr.len()
        );
        let batch_size = qo_indptr.len() - 1;
        let nnz_qo = *qo_indptr.last().unwrap() as usize;
        let total_pages = *kv_indptr.last().unwrap() as usize;
        anyhow::ensure!(
            kv_indices.len() >= total_pages * std::mem::size_of::<i32>(),
            "SinkAttention: kv_indices buffer smaller than kv_indptr total"
        );
        // page_size = 1: per-sequence kv length in tokens == pages.
        let mut kv_len_arr: Vec<i32> = kv_indptr.windows(2).map(|w| w[1] - w[0]).collect();

        let page_locked = page_locked_workspace();

        let mut plan_info = [0i64; 16];
        let mut plan_info_len: i32 = 0;
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
                cu_stream,
                plan_info.as_mut_ptr(),
                &mut plan_info_len,
            )
        };
        anyhow::ensure!(plan_ret == 0, "SinkAttention: fa3 plan failed ({plan_ret})");

        // Kernel-native (s, heads, dim) bf16 scratch (front half: transposed
        // q in, back half: kernel out), from the grow-only pool. Reuse is
        // stream-ordered; the refresh-path sync above covers growth.
        let temp_bytes = (nnz_qo * self.num_qo_heads * self.head_dim * 2).max(1);
        let mut scratch_guard = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        let stream_key = stream.cu_stream() as usize;
        let needs_new = !matches!(&*scratch_guard,
            Some((key, buf)) if *key == stream_key && buf.len() >= 2 * temp_bytes);
        if needs_new {
            stream.synchronize()?; // in-flight users of the old scratch
            let buf = unsafe { stream.alloc::<u8>((2 * temp_bytes).next_power_of_two())? };
            if let Some((_, old)) = scratch_guard.take() {
                std::mem::forget(old); // context may be gone; never free
            }
            *scratch_guard = Some((stream_key, buf));
        }
        let base_ptr = scratch_guard.as_ref().unwrap().1.device_ptr(stream).0;
        let (q_temp_ptr, temp_ptr) = (base_ptr, base_ptr + temp_bytes as u64);

        // The graph's q is (heads, s, dim) — the same heads-major layout
        // world as the output point — but the kernel reads token-major
        // (s, heads, dim) q. The layouts are byte-identical at s == 1
        // (decode), which is how this survived every single-token path;
        // prefill (s > 1) needs the transpose.
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

        let sm_scale = if self.sm_scale == 0.0 {
            1.0 / (self.head_dim as f32).sqrt()
        } else {
            self.sm_scale as f32
        };
        let run_ret = unsafe {
            (lib.prefill_run)(
                int_ws_ptr as *mut std::ffi::c_void,
                plan_info.as_mut_ptr(),
                plan_info_len,
                q_temp_ptr as *mut std::ffi::c_void,
                k_pool.ptr() as *mut std::ffi::c_void,
                v_pool.ptr() as *mut std::ffi::c_void,
                kv_indices.ptr() as *mut i32,
                sinks.ptr() as *mut f32,
                temp_ptr as *mut std::ffi::c_void,
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

        let tr_ret = unsafe {
            (lib.transpose_output_f32)(
                temp_ptr as *const std::ffi::c_void,
                out.ptr() as *mut std::ffi::c_void,
                nnz_qo as i32,
                self.num_qo_heads as i32,
                self.head_dim as i32,
                cu_stream,
            )
        };
        anyhow::ensure!(tr_ret == 0, "SinkAttention: output transpose failed");

        // No trailing sync: scratch is pooled (never freed), so the enqueued
        // kernels own it under stream ordering; the next tick's plan-refresh
        // path syncs before touching shared host buffers or growing scratch.
        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.batch_dim * self.num_qo_heads * self.head_dim
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4 // F32 output
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("SinkAttention")
    }
}
