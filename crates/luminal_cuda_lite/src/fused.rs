//! THE FRONTEND SIDE OF THE FUSED SERVING OPS: how a model SPELLS the
//! extern logical ops this runtime implements. Each helper records one
//! [`luminal::graph::LogicalOp::Extern`] term whose constructor,
//! propagation rules and implementation ride the corresponding registry
//! row ([`crate::ops::paged_attention`], [`crate::ops::moe_mxfp4`]).
//!
//! These are runtime-specific by construction — a graph that names them
//! is a CUDA-lite graph — which is why they live here and not in
//! `luminal_nn`. A model crate that wants to stay backend-neutral spells
//! the decomposed attention/MoE from `luminal_nn` instead.

use luminal::dtype::DType;
use luminal::graph::ExternParam;
use luminal::prelude::GraphTensor;
use luminal::shape::IntExpr;

pub use crate::ops::moe_mxfp4::{DownSpec, GateUpSpec};
pub use crate::ops::paged_attention::PagedAttentionSpec;

/// The operands of one paged attention step (see
/// [`crate::ops::paged_attention`] for each tensor's shape and dtype).
#[derive(Clone, Copy)]
pub struct PagedAttentionInputs {
    /// `[s, heads * head_dim]` F32 — the rotated queries.
    pub q: GraphTensor,
    /// `[slots, kv_heads * head_dim]` F32 — the cache AFTER this step's
    /// keys were written.
    pub k_cache: GraphTensor,
    /// `[slots, kv_heads * head_dim]` F32 — likewise for values.
    pub v_cache: GraphTensor,
    /// `[c]` Int — every request's context slots, concatenated in
    /// request order and position order within a request.
    pub slot_table: GraphTensor,
    /// `[r]` Int — CSR query-row boundaries per request (`r` = requests + 1).
    pub qo_indptr: GraphTensor,
    /// `[r]` Int — CSR context-row boundaries per request.
    pub kv_indptr: GraphTensor,
    /// `[s]` Int — absolute position of each query token in its sequence.
    pub q_pos: GraphTensor,
    /// `[heads]` F32 — per-head attention sink logits (use a very
    /// negative vector for a model without sinks).
    pub sinks: GraphTensor,
}

/// Record a fused paged-attention step. Returns `[s, heads * head_dim]`.
pub fn paged_attention(inputs: PagedAttentionInputs, spec: PagedAttentionSpec) -> GraphTensor {
    let q = inputs.q;
    assert_eq!(q.dtype, DType::F32, "paged_attention: q must be F32");
    assert_eq!(
        q.rank(),
        2,
        "paged_attention: q must be [s, heads*head_dim]"
    );
    for (name, t) in [
        ("slot_table", inputs.slot_table),
        ("qo_indptr", inputs.qo_indptr),
        ("kv_indptr", inputs.kv_indptr),
        ("q_pos", inputs.q_pos),
    ] {
        assert_eq!(t.dtype, DType::Int, "paged_attention: {name} must be Int");
        assert_eq!(t.rank(), 1, "paged_attention: {name} must be rank 1");
    }
    assert!(
        spec.kv_heads > 0 && spec.heads.is_multiple_of(spec.kv_heads),
        "paged_attention: heads must be a multiple of kv_heads"
    );
    assert!(
        spec.head_dim.is_multiple_of(32),
        "paged_attention: head_dim must be a multiple of 32"
    );
    let out_dims = q.dims();
    q.graph().extern_op(
        crate::ops::paged_attention::LOGICAL_CONSTRUCTOR,
        &[
            q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.slot_table,
            inputs.qo_indptr,
            inputs.kv_indptr,
            inputs.q_pos,
            inputs.sinks,
        ],
        vec![
            ExternParam::I64(spec.heads as i64),
            ExternParam::I64(spec.kv_heads as i64),
            ExternParam::I64(spec.head_dim as i64),
            ExternParam::I64(spec.window as i64),
            ExternParam::F64(spec.scale),
        ],
        out_dims,
        DType::F32,
    )
}

/// One MXFP4 expert projection bank: `blocks` `[E, n, k/2]` U8,
/// `scales` `[E, n, k/32]` F8UE8M0, `bias` `[E, n]` Bf16.
#[derive(Clone, Copy)]
pub struct Mxfp4Experts {
    pub blocks: GraphTensor,
    pub scales: GraphTensor,
    pub bias: GraphTensor,
}

impl Mxfp4Experts {
    fn check(&self, who: &str) {
        assert_eq!(self.blocks.dtype, DType::U8, "{who}: blocks must be U8");
        assert_eq!(
            self.scales.dtype,
            DType::F8UE8M0,
            "{who}: scales must be F8UE8M0"
        );
        assert_eq!(self.bias.dtype, DType::Bf16, "{who}: bias must be Bf16");
        assert_eq!(self.blocks.rank(), 3, "{who}: blocks must be [E, n, k/2]");
        assert_eq!(self.scales.rank(), 3, "{who}: scales must be [E, n, k/32]");
        assert_eq!(self.bias.rank(), 2, "{who}: bias must be [E, n]");
    }
}

/// Record the gate/up half of an MXFP4 MoE block. `x` is `[s, hidden]`
/// F32, `expert_ids` `[s, top_k]` Int; returns `[s, top_k, inter]` F32
/// (the clamped-SwiGLU hidden activations per route).
pub fn moe_gate_up_mxfp4(
    x: GraphTensor,
    expert_ids: GraphTensor,
    experts: Mxfp4Experts,
    spec: GateUpSpec,
) -> GraphTensor {
    assert_eq!(x.dtype, DType::F32, "moe_gate_up_mxfp4: x must be F32");
    assert_eq!(x.rank(), 2, "moe_gate_up_mxfp4: x must be [s, hidden]");
    assert_eq!(
        expert_ids.dtype,
        DType::Int,
        "moe_gate_up_mxfp4: expert_ids must be Int"
    );
    assert_eq!(
        expert_ids.rank(),
        2,
        "moe_gate_up_mxfp4: expert_ids must be [s, top_k]"
    );
    experts.check("moe_gate_up_mxfp4");
    let s = x.dims()[0];
    let out_dims: Vec<IntExpr> = vec![s, IntExpr::from(spec.top_k), IntExpr::from(spec.inter)];
    x.graph().extern_op(
        crate::ops::moe_mxfp4::GATE_UP_LOGICAL_CONSTRUCTOR,
        &[x, expert_ids, experts.blocks, experts.scales, experts.bias],
        vec![
            ExternParam::I64(spec.inter as i64),
            ExternParam::I64(spec.top_k as i64),
            ExternParam::F64(spec.alpha),
            ExternParam::F64(spec.limit),
        ],
        out_dims,
        DType::F32,
    )
}

/// Record the down half of an MXFP4 MoE block. `hidden` is
/// `[s, top_k, inter]` F32 (the gate/up result), `expert_ids` and
/// `weights` `[s, top_k]`; returns `[s, hidden]` F32, the route-weighted
/// sum over the selected experts.
pub fn moe_down_mxfp4(
    hidden: GraphTensor,
    expert_ids: GraphTensor,
    weights: GraphTensor,
    experts: Mxfp4Experts,
    spec: DownSpec,
) -> GraphTensor {
    assert_eq!(
        hidden.dtype,
        DType::F32,
        "moe_down_mxfp4: hidden must be F32"
    );
    assert_eq!(
        hidden.rank(),
        3,
        "moe_down_mxfp4: hidden must be [s, top_k, inter]"
    );
    assert_eq!(
        expert_ids.dtype,
        DType::Int,
        "moe_down_mxfp4: expert_ids must be Int"
    );
    assert_eq!(
        weights.dtype,
        DType::F32,
        "moe_down_mxfp4: weights must be F32"
    );
    experts.check("moe_down_mxfp4");
    let s = hidden.dims()[0];
    let out_dims: Vec<IntExpr> = vec![s, IntExpr::from(spec.hidden)];
    hidden.graph().extern_op(
        crate::ops::moe_mxfp4::DOWN_LOGICAL_CONSTRUCTOR,
        &[
            hidden,
            expert_ids,
            weights,
            experts.blocks,
            experts.scales,
            experts.bias,
        ],
        vec![
            ExternParam::I64(spec.hidden as i64),
            ExternParam::I64(spec.top_k as i64),
        ],
        out_dims,
        DType::F32,
    )
}

/// Shrink a tensor's leading axis to `rows` (a zero-start slice recorded
/// as a plain shrink, so the result's dim is exactly `rows` — not
/// `min(capacity, rows)`). This is how a serving graph turns a
/// fixed-capacity per-tick input into the bucket's `s` rows.
pub fn take_rows(tensor: GraphTensor, rows: impl Into<IntExpr>) -> GraphTensor {
    let mut new_dims = tensor.dims();
    assert!(
        !new_dims.is_empty(),
        "take_rows: tensor must have a leading axis"
    );
    new_dims[0] = rows.into();
    let operand = (tensor.id, tensor.dims());
    let dtype = tensor.dtype;
    let value = tensor.graph().logical.apply_movement(
        &operand,
        luminal::graph::Movement::Shrink {
            new_dims: new_dims.clone(),
        },
    );
    match value {
        Some(id) => GraphTensor::from_id(id, new_dims, tensor.graph_ref, dtype),
        None => GraphTensor::from_id(
            luminal::graph::unrecorded_value_pub(),
            new_dims,
            tensor.graph_ref,
            dtype,
        ),
    }
}
