//! Scaled-dot-product attention lowerings (port batch 6).
//!
//! All `scaled_dot_product_attention` ATen variants (unified, efficient,
//! flash, flash_for_cpu, cudnn) share one body that lowers to
//! `softmax((Q@K^T)*scale + mask) @ V`. Args are resolved by name so one
//! body serves every variant; only the first output slot is bound.
//!
//! Torch-kernel parity notes: Q/K/V arrive at the op's compute dtype (F32
//! for bf16/f16, the translator's opmath pattern) and the probabilities are
//! rounded to the operands' dtype before P@V, as the fused kernels do;
//! `is_causal` is a top-left iota mask; bool masks are keep-masks and
//! fully-masked query rows output zeros; grouped K/V heads are
//! repeat-interleaved (GQA).

use anyhow::{Context, Result};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, same_dim};
use crate::pt2_schema::{Node, NodeInput};

/// Broadcast-add an additive offset (causal mask, attention bias) onto the
/// scores. Handles dtype promotion and rank extension: a `(S_q, S_k)` offset
/// broadcasts across any batch/head prefix dims of `scores`.
fn add_offset(scores: GraphTensor, offset: GraphTensor) -> GraphTensor {
    let (s, o) = util::ensure_same_dtype(scores, offset);
    let (s, o) = util::broadcast_binary(s, o);
    s + o
}

fn named_input<'a>(node: &'a Node, name: &str) -> Option<&'a NodeInput> {
    node.inputs.iter().find(|input| input.name == name)
}

impl Translator<'_> {
    /// Translate all `scaled_dot_product_attention` ATen variants (unified,
    /// efficient, flash, flash_for_cpu, cudnn) into
    /// `softmax((Q@K^T)*scale + causal_mask + attn_bias) @ V`. Args resolve
    /// by name so one body serves every variant; only tuple slot 0 (the
    /// attention output) is bound — logsumexp and later slots are
    /// inference-time dead ends, left unbound by design.
    pub(super) fn translate_sdpa(&mut self, node: &Node) -> Result<()> {
        let query = self.operand(&node.inputs[0])?;
        let mut key = self.operand(&node.inputs[1])?;
        let mut value = self.operand(&node.inputs[2])?;

        let q_ndim = query.rank();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );

        // Q/K share head_dim and K/V share seq by op contract, but dynamo
        // gives each placeholder its own SymInt — unify so matmul
        // dim-equality checks hold. The recorder has no mutable ShapeTracker,
        // so re-view the tensor when the dims differ but have equal extents;
        // when they already compare equal this is a no-op.
        if key.rank() == q_ndim && value.rank() == q_ndim {
            let q_head_dim = query.dims()[q_ndim - 1];
            let k_head_dim = key.dims()[q_ndim - 1];
            if !same_dim(q_head_dim, k_head_dim)
                && let (Some(q), Some(k)) = (q_head_dim.to_usize(), k_head_dim.to_usize())
            {
                anyhow::ensure!(q == k, "SDPA: query/key head_dim mismatch ({q} vs {k})");
                let mut target = key.dims();
                target[q_ndim - 1] = q_head_dim;
                key = util::reshape_tensor(key, &target);
            }
            let k_seq = key.dims()[q_ndim - 2];
            let v_seq = value.dims()[q_ndim - 2];
            if !same_dim(k_seq, v_seq)
                && let (Some(k), Some(v)) = (k_seq.to_usize(), v_seq.to_usize())
            {
                anyhow::ensure!(k == v, "SDPA: key/value seq mismatch ({k} vs {v})");
                let mut target = value.dims();
                target[q_ndim - 2] = k_seq;
                value = util::reshape_tensor(value, &target);
            }
        }

        // attn_bias (Efficient/Cudnn/Unified) or attn_mask (FlashForCpu/Unified),
        // read as recorded: a Bool mask is a predicate, not an operand of
        // the score arithmetic.
        let mut additive: Option<GraphTensor> = None;
        for name in ["attn_bias", "attn_mask"] {
            if let Some(input) = named_input(node, name)
                && let Some(tensor) = self.optional_raw_operand(input)?
            {
                additive = Some(tensor);
                break;
            }
        }
        let is_causal = self.named_bool_arg(node, "is_causal").unwrap_or(false);

        let dropout_p = self.named_float_arg(node, "dropout_p").unwrap_or(0.0) as f32;
        anyhow::ensure!(
            dropout_p == 0.0,
            "SDPA: dropout_p={dropout_p} unsupported (inference only)"
        );

        // Default scale 1/sqrt(head_dim) needs a concrete head_dim;
        // symbolic-shape HF graphs always pass `scale` explicitly.
        let scale = match self.named_float_arg(node, "scale") {
            Some(v) => v,
            None => {
                let head_dim = query.dims().last().and_then(|d| d.to_usize()).context(
                    "SDPA: query head_dim must be concrete to derive the default \
                         scale (pass `scale` explicitly for symbolic head dims)",
                )?;
                1.0_f64 / (head_dim as f64).sqrt()
            }
        };

        // GQA: repeat-interleave K/V heads up to Q's count, gated on
        // `enable_gqa` — only the unified op carries the flag, and the
        // pipeline emits only the unified op (sdpa decomps are stripped).
        // Flagless graphs with mismatched heads fail the ensure below.
        let enable_gqa = self.named_bool_arg(node, "enable_gqa").unwrap_or(false);
        if enable_gqa && q_ndim >= 3 && !same_dim(query.dims()[q_ndim - 3], key.dims()[q_ndim - 3])
        {
            let h_axis = q_ndim - 3;
            let h_q = query.dims()[h_axis]
                .to_usize()
                .context("SDPA GQA: query head count must be concrete")?;
            let h_kv = key.dims()[h_axis]
                .to_usize()
                .context("SDPA GQA: kv head count must be concrete")?;
            anyhow::ensure!(
                h_kv > 0 && h_q % h_kv == 0,
                "SDPA GQA: query heads ({h_q}) must be a positive multiple of kv heads ({h_kv})"
            );
            let group = h_q / h_kv;
            key = key
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
            value = value
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
        } else if q_ndim >= 3 {
            anyhow::ensure!(
                same_dim(query.dims()[q_ndim - 3], key.dims()[q_ndim - 3]),
                "SDPA: query/key head counts differ ({:?} vs {:?}) without enable_gqa",
                query.dims()[q_ndim - 3],
                key.dims()[q_ndim - 3]
            );
        }

        // scores = (Q @ K^T) * scale.
        // Q/K arrive at the compute dtype, so the scores, the scale, the
        // masks and the softmax are all F32 for half-precision inputs.
        let (q_for_mm, k_for_mm) =
            util::ensure_same_dtype(query, key.transpose(q_ndim - 2, q_ndim - 1));
        let scores = q_for_mm.matmul(k_for_mm);
        let scale_const = self.constant_like(scores, scale);
        let mut scores = scores * scale_const;

        if is_causal {
            let s_q = scores.dims()[q_ndim - 2];
            let s_k = scores.dims()[q_ndim - 1];
            let row = self.cx.arange(s_q).cast(DType::F32).expand_dim(1, s_k);
            let col = self.cx.arange(s_k).cast(DType::F32).expand_dim(0, s_q);
            // 1.0 strictly above the diagonal (j > i = masked); -1e9 ~= -inf.
            let masked = col.gt(row).cast(DType::F32);
            let neg_inf = self.constant_like(masked, -1e9);
            scores = add_offset(scores, masked * neg_inf);
        }

        // Bool masks: track per-row any-keep to zero fully-masked rows
        // after softmax.
        let mut row_any_keep: Option<GraphTensor> = None;
        if let Some(mask) = additive {
            let offset = if mask.dtype == DType::Bool {
                let keep = mask.cast(DType::F32);
                let key_axis = keep.rank() - 1;
                row_any_keep = Some(
                    keep.max(key_axis)
                        .expand_to_shape_on_axes(keep.dims(), key_axis),
                );
                let one = self.cx.constant_f32(1.0).expand_rhs(keep.dims());
                (one - keep) * self.constant_like(keep, -1e9)
            } else {
                self.widen(mask)
            };
            scores = add_offset(scores, offset);
        }

        let mut attn = scores.softmax(q_ndim - 1);
        if let Some(indicator) = row_any_keep {
            let (a, i) = util::ensure_same_dtype(attn, indicator);
            let (a, i) = util::broadcast_binary(a, i);
            attn = a * i;
        }
        // torch parity, part two: the fused kernels round the probabilities
        // to the operands' dtype before the P@V GEMM. No-op on fp32.
        let operand_dtype = self.opmath.map_or(value.dtype, |opmath| opmath.common);
        let probabilities = super::convert(super::convert(attn, operand_dtype), value.dtype);
        let out = probabilities.matmul(value);

        // Tuple outputs serialize as one `as_tensors` list or one entry per
        // element — flatten to slot order; slot 0 is the attention output.
        let name = Self::tensor_output_names(node)
            .into_iter()
            .find(|name| !name.is_empty())
            .with_context(|| {
                format!("SDPA: no output tensor name found on node {}", node.target)
            })?;
        self.bind_value(name, out);
        Ok(())
    }
}
