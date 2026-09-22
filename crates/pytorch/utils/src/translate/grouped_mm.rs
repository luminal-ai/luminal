//! Grouped GEMM / MoE lowerings (port batch 6).
//!
//! `aten._grouped_mm.default(input, weight, offs)` and the HuggingFace
//! `transformers.grouped_mm_fallback.default` alias: per-token expert
//! gather + matmul, so the expert routing stays a runtime tensor.

use anyhow::{Context, Result};
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::Node;

impl Translator<'_> {
    /// Translate `aten._grouped_mm.default(input, weight, offs)` -> `Tensor[S, N]`.
    ///
    /// Grouped matmul: `input` is `[S, K]` (tokens sorted by expert), `weight`
    /// is `[G, K, N]` (per-expert weights), `offs` is `[G]` cumulative token
    /// counts. Output `[S, N]` where token m (in group g s.t.
    /// `offs[g-1] <= m < offs[g]`) is multiplied by `weight[g]`.
    ///
    /// Implementation: for each token m we (a) compute its expert id from
    /// offs, (b) gather only that expert's `[K, N]` slice from weight, and (c)
    /// do a single per-token matmul.
    ///
    /// Why not the straightforward `[G, S, K] @ [G, K, N] -> [G, S, N]` +
    /// mask: it forces a full F32 cast of the entire `[G, K, N]` weight tensor
    /// as a search-time intermediate, which OOMs on real MoE checkpoints.
    /// Gathering first keeps the F32 cast on `[S, K, N]` instead. We do NOT
    /// introduce an F32 cast on the whole weight bank.
    pub(super) fn translate_grouped_mm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        let weight = self.operand(&node.inputs[1])?;
        let offs = self.operand(&node.inputs[2])?;
        let out_dtype = self.output_meta_dtype(node)?;
        // Exact-class op (`offs` is an int index tensor), so the arm states
        // torch's opmath itself: the GEMM runs wide and rounds once.
        let compute_dtype = super::opmath_compute(out_dtype);

        anyhow::ensure!(
            input.rank() == 2,
            "_grouped_mm: input must be 2D, got {}D",
            input.rank()
        );
        anyhow::ensure!(
            weight.rank() == 3,
            "_grouped_mm: weight must be 3D, got {}D",
            weight.rank()
        );
        anyhow::ensure!(
            offs.rank() == 1,
            "_grouped_mm: offs must be 1D, got {}D",
            offs.rank()
        );

        let s = input.dims()[0];
        let g = weight.dims()[0];
        let k = weight.dims()[1];
        let n = weight.dims()[2];

        // expert_id[m] = number of g s.t. m >= offs[g], clamped to [0, G-1].
        // Same value as HF MoE's `expert_ids.clamp(0, num_experts-1)` for
        // invalid expert IDs from EP, AND protects search-time profiling:
        // dummy-1 input bytes give offs=[1,...,1], which pushes the raw count
        // to G for any token with index >= 1 and would OOB the weight gather.
        //
        // Stay in Int throughout — arange / offs are already Int, ge -> Bool
        // -> cast(Int), sum stays Int, and the binary `minimum` handles the
        // clamp without an F32 round-trip.
        let g_usize = g
            .to_usize()
            .context("_grouped_mm: G (num_experts) must be concrete")?;
        let s_arange = self.cx.arange(s); // Int [S]
        let ge_int = s_arange
            .expand_dim(0, g)
            .ge(offs.expand_dim(1, s)) // Bool [G, S]
            .cast(DType::Int); // Int [G, S]
        let raw = ge_int.sum(vec![0]); // Int [S], values in [0, G]
        let cap = self
            .cx
            .constant_i64((g_usize - 1) as i64)
            .cast(DType::Int)
            .expand_dim(0, s); // Int [S], all G-1
        let expert_id = raw.minimum(cap); // Int [S]

        // Flat gather index into weight (treated as a length-G*K*N 1D buffer):
        //   flat[m, k_, n_] = expert_id[m] * (K*N) + k_ * N + n_
        let io = k * n;
        let base = expert_id * io;
        let within = self.cx.iota((k, n), |c| c[0] * n + c[1]);
        let exp_base = base.expand_dim(1, k).expand_dim(2, n);
        let exp_within = within.expand_dim(0, s);
        let flat_idx = exp_base + exp_within;

        // Gather -> [S, K, N], then meet both operands at the compute dtype.
        // Gathering first keeps a half-precision widening on [S, K, N] and
        // off the whole expert weight bank.
        let weight_gathered = weight.gather1d(flat_idx).cast(compute_dtype);
        let input = input.cast(compute_dtype);

        // Per-token matmul: [S, 1, K] @ [S, K, N] -> [S, 1, N] -> [S, N].
        let result = input.unsqueeze(1).matmul(weight_gathered).squeeze(1);

        Ok(super::convert(result, out_dtype))
    }
}
