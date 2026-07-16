//! FusedMoE: the gpt-oss MXFP4 MoE as one host op — a stream-ordered pair
//! of GEMV launches (see `decode.cu`) at every batch size: phase 1 gate_up
//! dequant-GEMV + clamped SwiGLU, phase 2 down projection + top-k weighted
//! mix. (An expert-grouped tensor-core chain for large prefill batches was
//! built and benched during bring-up but removed before upstreaming:
//! post-row-blocking, it only beat the GEMV by ~10% at 512 pairs.)
//!
//! Installed by the union-only rewrite in `fused_moe_rewrite.egg`, whose LHS
//! is the model's dense reference spelling (`moe_naive`). Enforcement of the
//! fused choice is the candidate memory filter: the raw naive spelling
//! materializes expert-dense tensors no genome can afford.
//!
//! Inputs (graph edges, in order):
//!   0: x            [s, hidden]               F32
//!   1: topk_idx     [s, >=top_k] row-major    Int  (argsort tensor; row
//!      stride derived from the buffer length, GLUMoE-style)
//!   2: topk_weights [s, top_k]                F32
//!   3: gu_blocks    [E, 2*inter, hidden/2]    U8 (packed fp4, lo nibble = even k)
//!   4: gu_scales    [E, 2*inter, hidden/32]   U8 (e8m0)
//!   5: gu_bias      [E, 2*inter]              BF16
//!   6: dn_blocks    [E, hidden, inter/2]      U8
//!   7: dn_scales    [E, hidden, inter/32]     U8
//!   8: dn_bias      [E, hidden]               BF16
//!
//! Output: [s, hidden] F32.
//!
//! `num_experts` is derived from the weight buffer lengths (not op metadata).
//! No host readback, no mid-execute synchronize: every launch is
//! stream-ordered, and the kernels live in a process-wide cache
//! (decode.rs's static), so cloning this op during GA profiling costs
//! nothing.

use std::sync::Arc;

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    cudarc::driver::CudaStream,
    host::{DeviceBuffer, HostOp},
};

use super::decode;

const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;

/// Force the process-wide NVRTC compile of the decode module, once.
/// Extraction has no runtime stream, so this binds the device's primary
/// context itself; if no device is available the compile stays lazy (first
/// execute will pay it, as before).
fn ensure_kernels_compiled() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        if let Ok(ctx) = crate::cudarc::driver::CudaContext::new(0) {
            let stream = ctx.default_stream();
            decode::warm(&stream);
        }
    });
}

#[derive(Debug, Clone, Default)]
pub struct FusedMoE {
    hidden: Expression,
    intermediate: Expression,
    top_k: Expression,
}

impl EgglogOp for FusedMoE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusedMoE",
            &[
                ("hidden", EXPRESSION),
                ("intermediate", EXPRESSION),
                ("top_k", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (FusedMoE ?hidden ?intermediate ?top_k) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["fused_moe_rewrite.egg"]),
        ]
    }

    fn n_inputs(&self) -> usize {
        9
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let hidden = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let intermediate = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let top_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let extracted = FusedMoE {
            hidden,
            intermediate,
            top_k,
        };
        // Compile the decode kernels now (flashinfer precedent: JIT at
        // extract, not first execute) so the NVRTC cost never lands inside a
        // timed profiling trial.
        ensure_kernels_compiled();
        (
            LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>),
            input_enodes,
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for FusedMoE {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 9 {
            anyhow::bail!("FusedMoE expected 9 inputs, got {}", inputs.len());
        }

        let hidden = self
            .hidden
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE hidden dim is unresolved"))?;
        let intermediate = self
            .intermediate
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE intermediate dim is unresolved"))?;
        let top_k = self
            .top_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE top_k is unresolved"))?;
        if top_k == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            hidden % 32 == 0 && intermediate % 32 == 0,
            "FusedMoE dims must be multiples of 32 (e8m0 group width): hidden={hidden}, intermediate={intermediate}"
        );
        let gate_up_n = 2 * intermediate;

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("FusedMoE output byte size is unresolved"))?;
        anyhow::ensure!(
            output_bytes % (hidden * 4) == 0,
            "FusedMoE output bytes {output_bytes} not divisible by row bytes {}",
            hidden * 4
        );
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }
        let num_pairs = seq * top_k;

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("FusedMoE missing {name} buffer for LLIR node {node:?}")
            })
        };
        let x_buf = get_buffer("x", inputs[0])?;
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?;
        let topk_vals_buf = get_buffer("topk weights", inputs[2])?;
        let gu_blocks_buf = get_buffer("gate_up blocks", inputs[3])?;
        let gu_scales_buf = get_buffer("gate_up scales", inputs[4])?;
        let gu_bias_buf = get_buffer("gate_up bias", inputs[5])?;
        let dn_blocks_buf = get_buffer("down blocks", inputs[6])?;
        let dn_scales_buf = get_buffer("down scales", inputs[7])?;
        let dn_bias_buf = get_buffer("down bias", inputs[8])?;
        let output_buf = get_buffer("output", self_node)?;

        // num_experts is derived from the resident weight buffers.
        let gu_stride = gate_up_n * hidden / 2;
        anyhow::ensure!(
            gu_stride > 0 && gu_blocks_buf.len() % gu_stride == 0,
            "FusedMoE gate_up blocks len {} not a multiple of per-expert stride {gu_stride}",
            gu_blocks_buf.len()
        );
        let num_experts = gu_blocks_buf.len() / gu_stride;
        anyhow::ensure!(num_experts > 0, "FusedMoE derived zero experts");
        let checks: [(&str, usize, usize); 5] = [
            (
                "gate_up scales",
                gu_scales_buf.len(),
                num_experts * gate_up_n * hidden / 32,
            ),
            (
                "gate_up bias",
                gu_bias_buf.len(),
                num_experts * gate_up_n * 2,
            ),
            (
                "down blocks",
                dn_blocks_buf.len(),
                num_experts * hidden * intermediate / 2,
            ),
            (
                "down scales",
                dn_scales_buf.len(),
                num_experts * hidden * intermediate / 32,
            ),
            ("down bias", dn_bias_buf.len(), num_experts * hidden * 2),
        ];
        for (name, have, want) in checks {
            anyhow::ensure!(
                have >= want,
                "FusedMoE {name} buffer too small: {have} < {want}"
            );
        }
        anyhow::ensure!(
            x_buf.len() >= seq * hidden * 4,
            "FusedMoE x buffer too small: {} < {}",
            x_buf.len(),
            seq * hidden * 4
        );
        anyhow::ensure!(
            output_buf.len() >= output_bytes,
            "FusedMoE output buffer too small: {} < {output_bytes}",
            output_buf.len()
        );

        // Row strides derived from buffer lengths (the rewrite binds the full
        // [s, E] argsort tensor as topk_idx; weights come from the softmax as
        // [s, top_k], but derive anyway).
        let derive_stride = |name: &str, len_bytes: usize| -> anyhow::Result<usize> {
            anyhow::ensure!(
                len_bytes.is_multiple_of(4) && (len_bytes / 4).is_multiple_of(seq),
                "FusedMoE {name} buffer len {len_bytes} not divisible into {seq} rows"
            );
            let stride = len_bytes / 4 / seq;
            anyhow::ensure!(
                stride >= top_k,
                "FusedMoE {name} row stride {stride} < top_k {top_k}"
            );
            Ok(stride)
        };
        let idx_row_stride = derive_stride("topk index", topk_idx_buf.len())?;
        let vals_row_stride = derive_stride("topk weights", topk_vals_buf.len())?;
        anyhow::ensure!(
            vals_row_stride == top_k,
            "FusedMoE topk weights must be contiguous [s, top_k]; got row stride {vals_row_stride}"
        );

        let span = tracing::span!(
            tracing::Level::TRACE,
            "FusedMoE",
            seq,
            num_pairs,
            num_experts,
            hidden,
            intermediate
        );
        let _guard = span.enter();

        let hidden_scratch = unsafe { stream.alloc::<u8>(num_pairs * intermediate * 4)? };
        let hs_ptr = {
            use crate::cudarc::driver::DevicePtr;
            hidden_scratch.device_ptr(stream).0
        };
        decode::fused_moe_decode(
            stream,
            x_buf.ptr(),
            gu_blocks_buf.ptr(),
            gu_scales_buf.ptr(),
            gu_bias_buf.ptr(),
            dn_blocks_buf.ptr(),
            dn_scales_buf.ptr(),
            dn_bias_buf.ptr(),
            topk_idx_buf.ptr(),
            topk_vals_buf.ptr(),
            hs_ptr,
            output_buf.ptr(),
            hidden,
            intermediate,
            top_k,
            seq,
            idx_row_stride,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
        )?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        // [seq, hidden] F32; 's' is the seq dim by model convention (GLUMoE
        // does the same).
        Expression::from('s') * self.hidden
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4 // F32
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("FusedMoE")
    }
}
