use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Broadcast-add an additive offset (causal mask, attention bias) onto the
/// scores. Handles dtype promotion and rank extension: a `(S_q, S_k)` offset
/// broadcasts across any batch/head prefix dims of `scores`.
fn add_offset(scores: GraphTensor, offset: GraphTensor) -> GraphTensor {
    let (s, o) = ensure_same_dtype(scores, offset);
    let (s, o) = broadcast_binary(s, o);
    s + o
}

impl<'a> Translator<'a> {
    /// Translate all `scaled_dot_product_attention` ATen variants (unified,
    /// efficient, flash, flash_for_cpu, cudnn) into
    /// `softmax((Q@K^T)*scale + causal_mask + attn_bias) @ V`. Args resolve
    /// by name so one body serves every variant; only tuple slot 0 (the
    /// attention output) is bound — logsumexp and later slots are
    /// inference-time dead ends, left unbound by design.
    ///
    /// Torch-kernel parity notes: bf16/f16 run the score chain in F32, probs
    /// return to the value dtype for PV — mirroring torch's kernels, which
    /// compute in `opmath_type<scalar_t>` (= f32 for bf16/f16; see pytorch
    /// `aten/src/ATen/native/cpu/FlashAttentionKernel.cpp`, `accum_t`, and
    /// the FlashAttention-2 paper, arXiv:2307.08691 §3); `is_causal` is a
    /// top-left iota mask
    /// (symbolic-seq safe); bool masks are keep-masks and fully-masked query
    /// rows output zeros; grouped K/V heads are repeat-interleaved (GQA);
    /// Q/K head_dim and K/V seq SymInts are unified per the op contract.
    pub(crate) fn translate_sdpa(&mut self, node: &Node) -> Result<()> {
        let query = self.get_input_tensor(node, 0)?;
        let mut key = self.get_input_tensor(node, 1)?;
        let mut value = self.get_input_tensor(node, 2)?;

        let q_ndim = query.shape.len();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );

        // Q/K share head_dim and K/V share seq by op contract, but dynamo
        // gives each placeholder its own SymInt — unify so matmul
        // dim-equality checks hold.
        if key.shape.len() == q_ndim && value.shape.len() == q_ndim {
            key.shape.dims[q_ndim - 1] = query.shape.dims[q_ndim - 1];
            value.shape.dims[q_ndim - 2] = key.shape.dims[q_ndim - 2];
        }

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
        let is_causal = bool_arg("is_causal").unwrap_or(false);

        let dropout_p = float_arg("dropout_p").unwrap_or(0.0) as f32;
        anyhow::ensure!(
            dropout_p == 0.0,
            "SDPA: dropout_p={dropout_p} unsupported (inference only)"
        );

        // Default scale 1/sqrt(head_dim) needs a concrete head_dim;
        // symbolic-shape HF graphs always pass `scale` explicitly.
        let scale = match float_arg("scale") {
            Some(v) => v as f32,
            None => {
                let head_dim = query.shape.dims.last().and_then(|d| d.to_usize()).context(
                    "SDPA: query head_dim must be concrete to derive the default \
                         scale (pass `scale` explicitly for symbolic head dims)",
                )?;
                1.0_f32 / (head_dim as f32).sqrt()
            }
        };

        // GQA: repeat-interleave K/V heads up to Q's count, gated on
        // `enable_gqa` — only the unified op carries the flag, and the
        // pipeline emits only the unified op (sdpa decomps are stripped).
        // Flagless graphs with mismatched heads fail the ensure below.
        let enable_gqa = bool_arg("enable_gqa").unwrap_or(false);
        if enable_gqa && q_ndim >= 3 && query.shape.dims[q_ndim - 3] != key.shape.dims[q_ndim - 3] {
            let h_axis = q_ndim - 3;
            let h_q = query.shape.dims[h_axis]
                .to_usize()
                .context("SDPA GQA: query head count must be concrete")?;
            let h_kv = key.shape.dims[h_axis]
                .to_usize()
                .context("SDPA GQA: kv head count must be concrete")?;
            anyhow::ensure!(
                h_kv > 0 && h_q % h_kv == 0,
                "SDPA GQA: query heads ({h_q}) must be a positive multiple of kv heads ({h_kv})"
            );
            let group = Expression::from(h_q / h_kv);
            key = key
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
            value = value
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
        } else if q_ndim >= 3 {
            anyhow::ensure!(
                query.shape.dims[q_ndim - 3] == key.shape.dims[q_ndim - 3],
                "SDPA: query/key head counts differ ({:?} vs {:?}) without enable_gqa",
                query.shape.dims[q_ndim - 3],
                key.shape.dims[q_ndim - 3]
            );
        }

        // scores = (Q @ K^T) * scale.
        let mut perm: Vec<usize> = (0..q_ndim).collect();
        perm.swap(q_ndim - 2, q_ndim - 1);
        let (q_for_mm, k_for_mm) = ensure_same_dtype(query, key.permute(perm));
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
        let mut scores = self.apply_scalar_op(q_for_mm.matmul(k_for_mm), scale, BinaryOp::Mul);

        if is_causal {
            let s_q = scores.shape.dims[q_ndim - 2];
            let s_k = scores.shape.dims[q_ndim - 1];
            let row = self.graph.arange(s_q).cast(DType::F32).expand_dim(1, s_k);
            let col = self.graph.arange(s_k).cast(DType::F32).expand_dim(0, s_q);
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
                let key_axis = keep.shape.len() - 1;
                row_any_keep = Some(
                    keep.max(key_axis)
                        .expand_to_shape_on_axes(keep.shape, key_axis),
                );
                let one = keep.graph().constant_float(1.0).expand_rhs(keep.shape);
                (one - keep) * -1e9_f32
            } else {
                mask
            };
            scores = add_offset(scores, offset);
        }

        // Tuple outputs serialize as one `as_tensors` list or one entry per
        // element — flatten to slot order; slot 0 is the attention output.
        let output_names: Vec<String> = node
            .outputs
            .iter()
            .flat_map(|o| {
                if let Some(t) = o.as_tensor.as_ref() {
                    vec![t.name.clone()]
                } else if let Some(ts) = o.as_tensors.as_ref() {
                    ts.iter().map(|t| t.name.clone()).collect()
                } else {
                    vec![]
                }
            })
            .collect();

        let mut attn = scores.softmax(q_ndim - 1);
        if let Some(indicator) = row_any_keep {
            let (a, i) = ensure_same_dtype(attn, indicator);
            let (a, i) = broadcast_binary(a, i);
            attn = a * i;
        }
        // torch parity, part two: probs round back to the input dtype for
        // the P@V GEMM (keeps V out of an fp32 matmul). No-op on fp32.
        let out = attn.cast(value.dtype).matmul(value);

        match output_names.first().filter(|n| !n.is_empty()) {
            Some(name) => {
                self.tensors.insert(name.clone(), out);
                Ok(())
            }
            None => anyhow::bail!("SDPA: no output tensor name found on node {}", node.target),
        }
    }
}
