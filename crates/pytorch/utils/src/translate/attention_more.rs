//! Additional ATen lowerings ported from the parked translator (port batch 6+).
#![allow(dead_code)]

use anyhow::{Context, Result};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::Node;

/// Broadcast-add an additive offset (causal mask, attention bias) onto the
/// scores. Handles dtype promotion and rank extension: a `(S_q, S_k)` offset
/// broadcasts across any batch/head prefix dims of `scores`.
fn add_offset(scores: GraphTensor, offset: GraphTensor) -> GraphTensor {
    let (s, o) = util::ensure_same_dtype(scores, offset);
    let (s, o) = util::broadcast_binary(s, o);
    s + o
}

impl Translator<'_> {
    /// A tensor input resolved by name (`None` when absent or non-tensor).
    fn named_tensor_input(&mut self, node: &Node, name: &str) -> Result<Option<GraphTensor>> {
        let Some(index) = node.inputs.iter().position(|input| input.name == name) else {
            return Ok(None);
        };
        self.optional_tensor_operand(&node.inputs[index])
    }

    /// Bind every tensor output an SDPA node declares. Slot 0 is the
    /// attention output; slot 1 is the log-sum-exp of the masked/scaled
    /// scores (a real value, cast to the declared dtype); later tuple slots
    /// (rng state, seed/offset, cumulative-sequence cache, debug masks) are
    /// inference-time dead ends, bound as correctly-shaped zeros of the
    /// declared metadata so consumers of `getitem_k` still resolve. Tuple
    /// outputs serialize either as one `as_tensors` entry or as one entry per
    /// element, so bind by flattened name rather than through `bind_outputs`
    /// (which cannot name `as_tensors` groups).
    fn bind_sdpa_outputs(
        &mut self,
        node: &Node,
        out: GraphTensor,
        scores: GraphTensor,
    ) -> Result<()> {
        let mut names = Self::tensor_output_names(node).into_iter();
        let first = names
            .next()
            .with_context(|| format!("{}: no output tensor name found", node.target))?;
        self.values.insert(first, out);

        if let Some(lse_name) = names.next() {
            // logsumexp(x) = x - log_softmax(x); stable and real, then take
            // one lane along the reduced axis so the shape matches torch's
            // `[B, H, S_q]` log-sum-exp.
            let axis = scores.rank() - 1;
            let lse = (scores - scores.log_softmax(axis))
                .slice_along(0..1, axis)
                .squeeze(axis);
            let meta = self.tensor_meta(&lse_name)?.clone();
            let dtype = super::dtype_of(meta.dtype)?;
            self.values.insert(lse_name, lse.cast(dtype));
        }

        for name in names {
            let meta = self.tensor_meta(&name)?.clone();
            let shape = self.tensor_meta_to_shape(&meta)?;
            let dtype = super::dtype_of(meta.dtype)?;
            let placeholder = self.full_tensor(shape, dtype, 0.0);
            self.values.insert(name, placeholder);
        }
        Ok(())
    }

    /// Translate all `scaled_dot_product_attention` ATen variants (unified,
    /// efficient, flash, flash_for_cpu, cudnn) into
    /// `softmax((Q@K^T)*scale + causal_mask + attn_bias) @ V`. Args resolve
    /// by name so one body serves every variant; slot 0 (the attention
    /// output) is bound plus the log-sum-exp and all remaining declared
    /// tensor slots (see `bind_sdpa_outputs`).
    ///
    /// Torch-kernel parity notes: bf16/f16 run the score chain in F32, probs
    /// return to the value dtype for PV — mirroring torch's kernels, which
    /// compute in `opmath_type<scalar_t>` (= f32 for bf16/f16; see pytorch
    /// `aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`, `accum_t`, and
    /// the FlashAttention-2 paper, arXiv:2307.08691 §3); `is_causal` is a
    /// top-left iota mask (symbolic-seq safe); bool masks are keep-masks and
    /// fully-masked query rows output zeros; grouped K/V heads are
    /// repeat-interleaved (GQA); Q/K head_dim and K/V seq SymInts are unified
    /// per the op contract.
    pub(super) fn translate_sdpa(&mut self, node: &Node) -> Result<()> {
        let query = self.get_input_tensor(node, 0)?;
        let mut key = self.get_input_tensor(node, 1)?;
        let mut value = self.get_input_tensor(node, 2)?;

        let q_ndim = query.rank();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );

        // The new recorder resolves symbolic dims to concrete hints at input
        // time, so Q/K head_dim and K/V seq already compare equal by value;
        // no ShapeTracker dim unification is needed here.

        let additive = self
            .named_tensor_input(node, "attn_bias")?
            .or(self.named_tensor_input(node, "attn_mask")?);
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
        if q_ndim >= 3 && key.rank() == q_ndim {
            let h_axis = q_ndim - 3;
            let q_heads = query.dims()[h_axis];
            let k_heads = key.dims()[h_axis];
            if enable_gqa && !util::same_dim(q_heads, k_heads) {
                let h_q = q_heads
                    .to_usize()
                    .context("SDPA GQA: query head count must be concrete")?;
                let h_kv = k_heads
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
            } else {
                anyhow::ensure!(
                    util::same_dim(q_heads, k_heads),
                    "SDPA: query/key head counts differ ({q_heads:?} vs {k_heads:?}) without enable_gqa"
                );
            }
        }

        // scores = (Q @ K^T) * scale.
        let mut perm: Vec<usize> = (0..q_ndim).collect();
        perm.swap(q_ndim - 2, q_ndim - 1);
        let (q_for_mm, k_for_mm) = util::ensure_same_dtype(query, key.permute(perm));
        // torch parity: fused kernels accumulate QK^T in fp32 and never
        // materialize low-precision scores (CPU: opmath_type = f32, see
        // pytorch's FlashAttentionKernel.cpp; CUDA likewise via tensor-core
        // fp32 accumulators — FA2 paper, arXiv:2307.08691 §3). Cast Q/K
        // before the matmul; scale, masks, and softmax inherit F32 from here.
        let low_precision = matches!(q_for_mm.dtype, DType::Bf16 | DType::F16);
        let (q_for_mm, k_for_mm) = if low_precision {
            (q_for_mm.cast(DType::F32), k_for_mm.cast(DType::F32))
        } else {
            (q_for_mm, k_for_mm)
        };
        let mut scores = q_for_mm.matmul(k_for_mm);
        let scale_tensor = self.constant_like(scores, scale);
        scores *= scale_tensor;

        if is_causal {
            let s_q = scores.dims()[q_ndim - 2];
            let s_k = scores.dims()[q_ndim - 1];
            let row = self.cx.arange(s_q).cast(DType::F32).expand_dim(1, s_k);
            let col = self.cx.arange(s_k).cast(DType::F32).expand_dim(0, s_q);
            // 1.0 strictly above the diagonal (j > i = masked); -1e9 ≈ -inf.
            let masked = col.gt(row).cast(DType::F32);
            scores = add_offset(scores, masked * (-1e9_f32));
        }

        // Bool masks: track per-row any-keep to zero fully-masked rows
        // after softmax (see doc comment).
        let mut row_any_keep: Option<GraphTensor> = None;
        if let Some(mask) = additive {
            let offset = if mask.dtype == DType::Bool {
                let keep = mask.cast(DType::F32);
                let key_axis = keep.rank() - 1;
                row_any_keep = Some(
                    keep.max(key_axis)
                        .expand_to_shape_on_axes(keep.dims(), key_axis),
                );
                let one = self.constant_like(keep, 1.0);
                (one - keep) * -1e9_f32
            } else {
                mask
            };
            scores = add_offset(scores, offset);
        }

        let mut attn = scores.softmax(scores.rank() - 1);
        if let Some(indicator) = row_any_keep {
            let (a, i) = util::ensure_same_dtype(attn, indicator);
            let (a, i) = util::broadcast_binary(a, i);
            attn = a * i;
        }
        // torch parity, part two: probs round back to the input dtype for
        // the P@V GEMM (keeps V out of an fp32 matmul). No-op on fp32.
        let out = attn.cast(value.dtype).matmul(value);

        self.bind_sdpa_outputs(node, out, scores)
    }
}
