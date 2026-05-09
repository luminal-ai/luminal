use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Which SDPA variant we're translating. Governs argument positions and
/// which output slots are consumed downstream.
#[derive(Clone, Copy, Debug)]
pub enum SdpaVariant {
    /// `aten._scaled_dot_product_efficient_attention.default(q, k, v, attn_bias,
    ///     compute_log_sumexp, dropout_p=0., is_causal=False, *, scale=None)
    ///     -> (output, log_sumexp, philox_seed, philox_offset)`
    Efficient,
    /// `aten._scaled_dot_product_flash_attention.default(q, k, v, dropout_p=0.,
    ///     is_causal=False, return_debug_mask=False, *, scale=None)
    ///     -> (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
    ///         rng_state, unused, debug_attn_mask)`
    Flash,
    /// `aten._scaled_dot_product_flash_attention_for_cpu.default(q, k, v,
    ///     dropout_p=0., is_causal=False, *, attn_mask=None, scale=None)
    ///     -> (output, logsumexp)`
    FlashForCpu,
    /// `aten._scaled_dot_product_cudnn_attention.default(q, k, v, attn_bias,
    ///     compute_log_sumexp, dropout_p=0., is_causal=False,
    ///     return_debug_mask=False, *, scale=None)
    ///     -> (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
    ///         philox_seed, philox_offset, debug_attn_mask)`
    Cudnn,
    /// `aten.scaled_dot_product_attention.default(q, k, v, attn_mask=None,
    ///     dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False)
    ///     -> Tensor` (single output, no tuple).
    Unified,
}

impl<'a> Translator<'a> {
    /// Translate any SDPA op variant into `softmax((Q@K^T)*scale + causal_mask +
    /// attn_bias) @ V`. Stores the primary `output` by the node's first output
    /// name. Other tuple outputs (logsumexp, philox_seed, etc.) are unused in
    /// inference — left unbound; the downstream `getitem(node, 0)` resolves
    /// to `output` via the tuple-output name list.
    pub(crate) fn translate_sdpa(&mut self, node: &Node, variant: SdpaVariant) -> Result<()> {
        let query = self.get_input_tensor(node, 0)?;
        let key = self.get_input_tensor(node, 1)?;
        let value = self.get_input_tensor(node, 2)?;

        // Resolve args by NAME rather than positional index. PT2 serializes
        // kwargs inline in `node.inputs` with `kind=2`, so any arg that wasn't
        // passed positionally by the caller shifts the indices of subsequent
        // positional args. Name-based lookup is unambiguous across variants
        // and across caller argument-passing styles.
        let arg_by_name =
            |name: &str| -> Option<&NodeInput> { node.inputs.iter().find(|i| i.name == name) };
        let tensor_arg = |name: &str| -> Option<GraphTensor> {
            arg_by_name(name)
                .and_then(|i| i.arg.as_tensor_name())
                .and_then(|n| self.get_tensor(n).ok())
        };
        let float_arg =
            |name: &str| -> Option<f64> { arg_by_name(name).and_then(|i| i.arg.as_float()) };
        let bool_arg =
            |name: &str| -> Option<bool> { arg_by_name(name).and_then(|i| i.arg.as_bool()) };

        // attn_bias (Efficient/Cudnn/Unified) or attn_mask (FlashForCpu/Unified).
        let additive = tensor_arg("attn_bias").or_else(|| tensor_arg("attn_mask"));

        let dropout_p = float_arg("dropout_p").unwrap_or(0.0) as f32;
        anyhow::ensure!(
            dropout_p == 0.0,
            "SDPA: dropout_p={dropout_p} unsupported (inference only)"
        );
        let is_causal = bool_arg("is_causal").unwrap_or(false);
        // Silence compiler warnings — variant arg remains for branch-specific
        // logic (output tuple-name resolution below) and for future divergence.
        let _ = variant;

        // `scale` kwarg, default 1/sqrt(head_dim).
        let head_dim = query
            .shape
            .dims
            .last()
            .and_then(|d| d.to_usize())
            .context("SDPA: query head_dim must be concrete")?;
        let default_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let scale = float_arg("scale")
            .map(|v| v as f32)
            .unwrap_or(default_scale);

        // Math form: scores = (Q @ K^T) * scale; + causal_mask; + attn_bias;
        // attn = softmax(scores, dim=-1); out = attn @ V.
        let q_ndim = query.shape.len();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );
        // Transpose last two dims of key.
        let mut perm: Vec<usize> = (0..q_ndim).collect();
        perm.swap(q_ndim - 2, q_ndim - 1);
        let key_t = key.permute(perm);
        let (q_for_mm, k_for_mm) = ensure_same_dtype(query, key_t);
        let scores = q_for_mm.matmul(k_for_mm);
        let scale_t = self
            .graph
            .constant_float(scale)
            .cast(scores.dtype)
            .expand_rhs(scores.shape);
        let mut scores = scores * scale_t;

        if is_causal {
            let s_q = scores
                .shape
                .dims
                .get(q_ndim - 2)
                .and_then(|d| d.to_usize())
                .context("SDPA is_causal: S_q must be concrete")?;
            let s_k = scores
                .shape
                .dims
                .get(q_ndim - 1)
                .and_then(|d| d.to_usize())
                .context("SDPA is_causal: S_k must be concrete")?;
            let size = s_q.max(s_k);
            // triu with diagonal=1 = 1 strictly above diagonal, 0 elsewhere.
            let mut mask = self.graph.triu(size, 1).cast(DType::F32);
            if s_q != size || s_k != size {
                mask = mask.slice_along(0..s_q, 0).slice_along(0..s_k, 1);
            }
            // -1e9 * mask ≈ -inf where masked, 0 otherwise. Broadcast across
            // batch/head prefix dims of `scores`.
            let neg_large = mask * (-1e9_f32);
            let mut neg_large = neg_large.cast(scores.dtype);
            for _ in 0..(q_ndim - 2) {
                neg_large = neg_large.expand_dim(0, Expression::from(1usize));
            }
            let (scores_b, mask_b) = broadcast_binary(scores, neg_large);
            scores = scores_b + mask_b;
        }
        if let Some(bias) = additive {
            let (scores_b, bias_b) = ensure_same_dtype(scores, bias);
            let (scores_b, bias_b) = broadcast_binary(scores_b, bias_b);
            scores = scores_b + bias_b;
        }

        let attn = scores.softmax(q_ndim - 1);
        let (attn, value) = ensure_same_dtype(attn, value);
        let out = attn.matmul(value);

        // Store the primary output by name. The other tuple outputs are
        // inference-time dead ends — downstream getitem(node, 0) resolves to
        // the same tensor name we bind here, because pt2 serializes the
        // multi-output name list with output[0] as the primary slot.
        let out_name = if let Some(ts) = node.outputs.first().and_then(|o| o.as_tensors.as_ref()) {
            ts.first().map(|t| t.name.clone())
        } else if variant == SdpaVariant::Unified {
            node.outputs
                .first()
                .and_then(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
        } else {
            node.outputs
                .first()
                .and_then(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
                .or_else(|| {
                    node.outputs
                        .first()
                        .and_then(|o| o.as_tensors.as_ref())
                        .and_then(|ts| ts.first().map(|t| t.name.clone()))
                })
        };

        if let Some(name) = out_name
            && !name.is_empty()
        {
            self.tensors.insert(name, out);
        } else {
            anyhow::bail!("SDPA: no output tensor name found on node {}", node.target);
        }

        Ok(())
    }
}

impl PartialEq for SdpaVariant {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (SdpaVariant::Efficient, SdpaVariant::Efficient)
                | (SdpaVariant::Flash, SdpaVariant::Flash)
                | (SdpaVariant::FlashForCpu, SdpaVariant::FlashForCpu)
                | (SdpaVariant::Cudnn, SdpaVariant::Cudnn)
                | (SdpaVariant::Unified, SdpaVariant::Unified)
        )
    }
}
