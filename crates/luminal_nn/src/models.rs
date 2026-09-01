//! FULL-MODEL tests, smallest first (Austin's directive 2026-08-06):
//! whole models composed from the nn modules, run end-to-end through the
//! native ladder (load → search → execute) against scalar references
//! computed in plain loops. Rungs: (1) MLP; (2) a decoder block with a
//! KV cache — Embedding → paged attention → FFN → tied logits; (3) the
//! same block with the MoE FFN; (4) a two-layer decoder with LayerNorm
//! driven for TWO decode steps, the caches flowing runtime-out →
//! runtime-in between steps.

use crate::{Embedding, Linear, MoE, paged_attention};
use luminal::prelude::*;
use luminal::shape::IntExpr;

/// The smallest true model: Linear → relu → Linear → relu → Linear.
pub struct Mlp {
    pub layers: Vec<Linear>,
}

impl Mlp {
    /// `dims` = [in, hidden.., out]; a relu follows every layer but the
    /// last.
    pub fn new(dims: &[usize], ns: &Ns, cx: &mut Graph) -> Self {
        assert!(dims.len() >= 2, "an MLP needs at least in and out dims");
        let layers = dims
            .windows(2)
            .enumerate()
            .map(|(i, pair)| Linear::new(pair[0], pair[1], true, &ns.child("layers").index(i), cx))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        let last = self.layers.len() - 1;
        for (index, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if index != last {
                x = x.relu();
            }
        }
        x
    }
}

/// The FFN flavor a decoder block carries.
pub enum FeedForward {
    /// up → relu → down.
    Dense { up: Box<Linear>, down: Box<Linear> },
    /// Mixture of experts (top-k routing).
    Moe(Box<MoE>),
}

impl FeedForward {
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        match self {
            FeedForward::Dense { up, down } => down.forward(up.forward(x).relu()),
            FeedForward::Moe(moe) => moe.forward(x),
        }
    }
}

/// MODEL 2/3: one decoder block over a paged KV cache. Token ids embed,
/// attention reads/writes the cache, the FFN applies, and logits come
/// from the tied embedding (`Embedding::reverse`). Residuals around both
/// sublayers; optional pre-norms arrive with [`TinyDecoder`].
pub struct DecoderBlock {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ff: FeedForward,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl DecoderBlock {
    /// x (s, d_model) + cache slots → (x', k_cache', v_cache').
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let (attn, k_cache, v_cache) = paged_attention(
            self.wq.forward(x),
            self.wk.forward(x),
            self.wv.forward(x),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            prev_seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let x = x + self.wo.forward(attn);
        let x = x + self.ff.forward(x);
        (x, k_cache, v_cache)
    }
}

/// MODEL 4: a two-layer decoder — per-layer pre-LayerNorm, per-layer KV
/// caches, tied logits through a final norm. One forward = one decode
/// step; multi-step decode re-runs the (shape-specialized) program with
/// the cache OUTPUTS flowing back in as the next step's cache INPUTS.
pub struct TinyDecoder {
    pub embed: Embedding,
    pub norms: Vec<crate::LayerNorm>,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: crate::LayerNorm,
}

impl TinyDecoder {
    /// ids (s,) Int + one (k, v) cache pair per layer → (logits, caches').
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = self.embed.forward(ids);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            x = self.norms[layer].forward(x);
            let (next, k_cache, v_cache) = block.forward(
                x,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                prev_seq,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}

/// RUNG 5 (2026-08-07): real llama anatomy minus rope (ruling: rope/cos
/// deferred for the initial reference runtime) — pre-RMSNorms, GQA
/// attention over the paged KV cache (n_kv_heads < n_heads), SwiGLU FFN,
/// residuals around both sublayers.
/// Gated-FFN activation family: llama/qwen use SwiGLU (silu gate),
/// gemma uses GeGLU (tanh-approximated gelu gate).
#[derive(Clone, Copy)]
pub enum GatedFfn {
    SwiGlu,
    GeGlu,
}

pub struct LlamaBlock {
    pub ffn_kind: GatedFfn,
    pub attn_norm: crate::LayerNorm, // RMS: mean_norm = false
    pub wq: Linear,                  // d → n_heads·head_dim
    pub wk: Linear,                  // d → n_kv_heads·head_dim
    pub wv: Linear,
    pub wo: Linear,
    /// Qwen3-style per-head QK RMSNorm weights (q, k), each (head_dim,),
    /// applied after projection and before the cache/attention path.
    pub qk_norm: Option<(GraphTensor, GraphTensor)>,
    pub ffn_norm: crate::LayerNorm,
    pub gate: Linear, // d → ff
    pub up: Linear,   // d → ff
    pub down: Linear, // ff → d
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl LlamaBlock {
    pub fn new(
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ns: &Ns,
        cx: &mut Graph,
    ) -> Self {
        Self::new_with_ffn(d, ff, n_heads, n_kv_heads, GatedFfn::SwiGlu, ns, cx)
    }

    pub fn new_with_ffn(
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ffn_kind: GatedFfn,
        ns: &Ns,
        cx: &mut Graph,
    ) -> Self {
        let head_dim = d / n_heads;
        let kv_dim = n_kv_heads * head_dim;
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        Self {
            ffn_kind,
            attn_norm: crate::LayerNorm::new(
                d,
                false,
                false,
                false,
                1e-5,
                &ns.child("input_layernorm"),
                cx,
            ),
            wq: Linear::new(d, d, false, &attn.child("q_proj"), cx),
            wk: Linear::new(d, kv_dim, false, &attn.child("k_proj"), cx),
            wv: Linear::new(d, kv_dim, false, &attn.child("v_proj"), cx),
            wo: Linear::new(d, d, false, &attn.child("o_proj"), cx),
            qk_norm: None,
            ffn_norm: crate::LayerNorm::new(
                d,
                false,
                false,
                false,
                1e-5,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
            gate: Linear::new(d, ff, false, &mlp.child("gate_proj"), cx),
            up: Linear::new(d, ff, false, &mlp.child("up_proj"), cx),
            down: Linear::new(ff, d, false, &mlp.child("down_proj"), cx),
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// Mint the QK-norm weights (Qwen3 anatomy) — builder form so the
    /// existing constructors stay unchanged.
    pub fn with_qk_norm(mut self, ns: &Ns, cx: &mut Graph) -> Self {
        let attn = ns.child("self_attn");
        self.qk_norm = Some((
            cx.named_tensor(attn.child("q_norm").leaf("weight"), self.head_dim),
            cx.named_tensor(attn.child("k_norm").leaf("weight"), self.head_dim),
        ));
        self
    }

    /// x (s, d) + cache slots (slots, kv_dim) → (x', k_cache', v_cache').
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let mut q = self.wq.forward(normed);
        let mut k = self.wk.forward(normed);
        if let Some((q_weight, k_weight)) = self.qk_norm {
            q = crate::rms_norm_heads(q, self.head_dim, q_weight, 1e-6);
            k = crate::rms_norm_heads(k, self.head_dim, k_weight, 1e-6);
        }
        let (attn, k_cache, v_cache) = paged_attention(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            prev_seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        );
        let x = x + self.wo.forward(attn);
        let ff = self.ffn(x);
        (x + ff, k_cache, v_cache)
    }

    /// The llama/qwen full-anatomy forward: RoPE threaded between the
    /// QK-norm and the cache path (the [`GemmaBlock::forward`]
    /// ordering), and the causal mask driven by a `q_pos` (s,) Int DATA
    /// input via [`crate::paged_attention_positional`] — so a decode
    /// loop searches the graph once and executes per token. Rope tables
    /// (s, head_dim) and the pairing matrix (head_dim, head_dim) are
    /// host-built ([`crate::rope_tables_split_half`] /
    /// [`crate::rope_pairing_matrix`]) — the concat-free rope spelling.
    /// Scale stays on the scores (the llama/qwen convention).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rope(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let mut q = self.wq.forward(normed);
        let mut k = self.wk.forward(normed);
        if let Some((q_weight, k_weight)) = self.qk_norm {
            q = crate::rms_norm_heads(q, self.head_dim, q_weight, 1e-6);
            k = crate::rms_norm_heads(k, self.head_dim, k_weight, 1e-6);
        }
        q = crate::rotary_apply(q, self.head_dim, rope_cos, rope_sin, rope_rot);
        k = crate::rotary_apply(k, self.head_dim, rope_cos, rope_sin, rope_rot);
        let (attn, k_cache, v_cache) = crate::paged_attention_positional(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            None,
            1.0 / (self.head_dim as f32).sqrt(),
        );
        let x = x + self.wo.forward(attn);
        let ff = self.ffn(x);
        (x + ff, k_cache, v_cache)
    }

    /// [`Self::forward_rope`] with the causal/isolation mask arriving
    /// as DATA — the page-table batch form ([`crate::paged_attention_masked`]);
    /// rope tables carry each batch row's own position.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rope_masked(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        mask: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let mut q = self.wq.forward(normed);
        let mut k = self.wk.forward(normed);
        if let Some((q_weight, k_weight)) = self.qk_norm {
            q = crate::rms_norm_heads(q, self.head_dim, q_weight, 1e-6);
            k = crate::rms_norm_heads(k, self.head_dim, k_weight, 1e-6);
        }
        q = crate::rotary_apply(q, self.head_dim, rope_cos, rope_sin, rope_rot);
        k = crate::rotary_apply(k, self.head_dim, rope_cos, rope_sin, rope_rot);
        let (attn, k_cache, v_cache) = crate::paged_attention_masked(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            mask,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            1.0 / (self.head_dim as f32).sqrt(),
        );
        let x = x + self.wo.forward(attn);
        let ff = self.ffn(x);
        (x + ff, k_cache, v_cache)
    }

    fn ffn(&self, x: GraphTensor) -> GraphTensor {
        let ff_in = self.ffn_norm.forward(x);
        let gated = match self.ffn_kind {
            GatedFfn::SwiGlu => self.gate.forward(ff_in).silu(),
            GatedFfn::GeGlu => self.gate.forward(ff_in).gelu_fast_tanh_approximation(),
        };
        self.down.forward(gated * self.up.forward(ff_in))
    }
}

/// GemmaBlock — the FULL gemma-3 layer anatomy (ruling 2026-08-10: minis
/// exercise every architectural construct of the big model, shrinking
/// only shapes):
///  * SANDWICH NORMS: four learned RMSNorms — pre-attention, post-
///    attention applied to the sublayer output INSIDE the residual add,
///    pre-feedforward, post-feedforward likewise inside the residual;
///  * head_dim DECOUPLED from d / n_heads (q_dim = n_heads·head_dim may
///    differ from d — gemma-3-4B: 2048 vs 2560);
///  * per-head QK RMSNorm (learned (head_dim,) weights, before RoPE);
///  * attention scale FOLDED INTO Q (the QK matmul itself is scale-free);
///  * split-half RoPE computed in-graph from the position input, with
///    the LAYER-ROLE theta: local layers θ=10k / no position scaling,
///    global layers θ=1M with linear position scaling (pos · scale);
///  * SLIDING-WINDOW attention on local layers (5-of-6 pattern in the
///    real model; the alternation is the construct, the ratio a shape);
///  * GeGLU feed-forward (tanh-approximated gelu gate).
pub struct GemmaBlock {
    pub input_norm: crate::LayerNorm,
    pub post_attn_norm: crate::LayerNorm,
    pub pre_ff_norm: crate::LayerNorm,
    pub post_ff_norm: crate::LayerNorm,
    pub wq: Linear, // d → n_heads·head_dim (decoupled)
    pub wk: Linear, // d → n_kv_heads·head_dim
    pub wv: Linear,
    pub wo: Linear, // n_heads·head_dim → d
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    /// Local layers use the sliding window + local theta; globals use
    /// full context + global theta + position scaling.
    pub local: bool,
    pub window: usize,
    pub rope_theta: f32,
    pub pos_scale: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl GemmaBlock {
    /// `local` layers follow gemma's alternation rule (the caller picks
    /// the pattern); theta and position scaling derive from the role.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        local: bool,
        window: usize,
        ns: &Ns,
        cx: &mut Graph,
    ) -> Self {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |segment: &str, cx: &mut Graph| {
            crate::LayerNorm::new(d, true, false, false, 1e-6, &ns.child(segment), cx)
                .with_unit_offset()
        };
        Self {
            input_norm: rms("input_layernorm", cx),
            post_attn_norm: rms("post_attention_layernorm", cx),
            pre_ff_norm: rms("pre_feedforward_layernorm", cx),
            post_ff_norm: rms("post_feedforward_layernorm", cx),
            wq: Linear::new(d, q_dim, false, &attn.child("q_proj"), cx),
            wk: Linear::new(d, kv_dim, false, &attn.child("k_proj"), cx),
            wv: Linear::new(d, kv_dim, false, &attn.child("v_proj"), cx),
            wo: Linear::new(q_dim, d, false, &attn.child("o_proj"), cx),
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), head_dim),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), head_dim),
            gate: Linear::new(d, ff, false, &mlp.child("gate_proj"), cx),
            up: Linear::new(d, ff, false, &mlp.child("up_proj"), cx),
            down: Linear::new(ff, d, false, &mlp.child("down_proj"), cx),
            local,
            window,
            rope_theta: if local { 10_000.0 } else { 1_000_000.0 },
            pos_scale: if local { 1.0 } else { 1.0 / 8.0 },
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// x (s, d) + cache slots (slots, kv_dim) + this layer role's rope
    /// tables (s, head_dim) and the pairing matrix (head_dim, head_dim)
    /// → (x', k_cache', v_cache'). Tables are host-built from the
    /// block's own `rope_theta`/`pos_scale` (see
    /// [`crate::rope_tables_split_half`]) — the concat-free rope
    /// spelling (rejoin-divergence workaround, 2026-08-10; in-graph
    /// angle synthesis returns with the divergence ruling).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.input_norm.forward(x);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // QK-norm before RoPE; attention scale folded into q.
        let q = crate::rotary_apply(
            crate::rms_norm_heads(
                self.wq.forward(normed),
                self.head_dim,
                self.q_norm + 1.0,
                1e-6,
            ) * scale,
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = crate::rotary_apply(
            crate::rms_norm_heads(
                self.wk.forward(normed),
                self.head_dim,
                self.k_norm + 1.0,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let (attn, k_cache, v_cache) = crate::paged_attention_windowed(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            prev_seq,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.local.then_some(self.window),
            1.0, // scale lives in Q (folded above) — the matmul is scale-free
        );
        // Sandwich residuals: post-norms wrap the sublayer OUTPUT.
        let x = x + self.post_attn_norm.forward(self.wo.forward(attn));
        let ff_in = self.pre_ff_norm.forward(x);
        let gated = self.gate.forward(ff_in).gelu_fast_tanh_approximation();
        let ff = self.down.forward(gated * self.up.forward(ff_in));
        (x + self.post_ff_norm.forward(ff), k_cache, v_cache)
    }

    /// [`Self::forward`] with the query position as DATA (q_pos Int
    /// input) — the step-invariant decode form; window, dual-theta rope
    /// tables, folded scale, and sandwich norms all preserved.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_positional(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.input_norm.forward(x);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q = crate::rotary_apply(
            crate::rms_norm_heads(
                self.wq.forward(normed),
                self.head_dim,
                self.q_norm + 1.0,
                1e-6,
            ) * scale,
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = crate::rotary_apply(
            crate::rms_norm_heads(
                self.wk.forward(normed),
                self.head_dim,
                self.k_norm + 1.0,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let (attn, k_cache, v_cache) = crate::paged_attention_positional(
            q,
            k,
            self.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.local.then_some(self.window),
            1.0, // scale folded into q above
        );
        let x = x + self.post_attn_norm.forward(self.wo.forward(attn));
        let ff_in = self.pre_ff_norm.forward(x);
        let gated = self.gate.forward(ff_in).gelu_fast_tanh_approximation();
        let ff = self.down.forward(gated * self.up.forward(ff_in));
        (x + self.post_ff_norm.forward(ff), k_cache, v_cache)
    }
}

#[cfg(test)]
mod tests {
    use super::{DecoderBlock, FeedForward, LlamaBlock, Mlp, TinyDecoder};
    use crate::{Embedding, Linear, MoE};
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::reference::ReferenceRuntime;
    use luminal::shape::IntExpr;
    use rustc_hash::FxHashMap;
    use scalar_refs::*;

    /// MODEL 1: a full 4→8→6→3 MLP, batch 2, through the search ladder —
    /// every layer's weights and biases bound as named tensors, the whole
    /// forward against a scalar reference.
    #[test]
    fn mlp_forward_matches_scalar_reference() {
        const DIMS: [usize; 4] = [4, 8, 6, 3];
        const BATCH: usize = 2;

        let mut cx = Graph::new();
        let model = Mlp::new(&DIMS, &Ns::root(), &mut cx);
        let x = cx.tensor((BATCH, DIMS[0]));
        let out = model.forward(x).output();

        let x_data = weights(BATCH * DIMS[0], 7);
        let mut layer_data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for (index, pair) in DIMS.windows(2).enumerate() {
            layer_data.push((
                weights(pair[0] * pair[1], index),
                weights(pair[1], index + 50),
            ));
        }

        // Scalar reference.
        let mut activation = x_data.clone();
        let mut width = DIMS[0];
        for (index, pair) in DIMS.windows(2).enumerate() {
            let (w, b) = &layer_data[index];
            let (in_w, out_w) = (pair[0], pair[1]);
            let mut next = vec![0f32; BATCH * out_w];
            for r in 0..BATCH {
                for c in 0..out_w {
                    let mut acc = b[c];
                    for k in 0..in_w {
                        acc += activation[r * width + k] * w[k * out_w + c];
                    }
                    next[r * out_w + c] = if index != DIMS.len() - 2 {
                        acc.max(0.0)
                    } else {
                        acc
                    };
                }
            }
            activation = next;
            width = out_w;
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone().into());
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            data.insert(layer.weight.id, w.clone().into());
            data.insert(layer.bias.unwrap().id, b.clone().into());
        }
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let outcome = rt
            .search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        eprintln!(
            "[mlp] search attribution: {} (plans profiled {}, cache hits {})",
            outcome.timings.summary(),
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );
        rt.set_data(x.id, x_data);
        for (layer, (w, b)) in model.layers.iter().zip(&layer_data) {
            rt.set_data(layer.weight.id, w.clone());
            rt.set_data(layer.bias.unwrap().id, b.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &activation);
    }

    /// RUNG 5: the llama-anatomy block (RMSNorm + GQA kv_groups=2 +
    /// SwiGLU, no rope) — one decode step through the DEFAULT ladder vs
    /// a scalar reference; prints attribution + refusal breakdown.
    #[test]
    fn llama_block_matches_scalar_reference() {
        const D5: usize = 8; // n_heads 4 × head_dim 2
        const N_H: usize = 4;
        const N_KV: usize = 2; // kv_groups = 2 — the untested GQA path
        const HD: usize = 2;
        const KV_DIM: usize = N_KV * HD;
        const FF5: usize = 12;
        const SLOTS5: usize = 4;
        const CTX5: usize = 2;
        const EPS: f32 = 1e-5;

        let mut cx = Graph::new();
        let block = LlamaBlock::new(D5, FF5, N_H, N_KV, &Ns::root().child("blk"), &mut cx);
        let x = cx.tensor((1, D5));
        let k_cache = cx.tensor((SLOTS5, KV_DIM));
        let v_cache = cx.tensor((SLOTS5, KV_DIM));
        let gather_idx = cx.tensor_dtyped(CTX5, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let (out, kc, vc) = block.forward(
            x,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(1usize),
        );
        let out = out.output();
        let kc = kc.output();
        let vc = vc.output();

        let x_vals = weights(D5, 70);
        let pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (x.id, x_vals.clone().into()),
            (block.wq.weight.id, weights(D5 * D5, 71).into()),
            (block.wk.weight.id, weights(D5 * KV_DIM, 72).into()),
            (block.wv.weight.id, weights(D5 * KV_DIM, 73).into()),
            (block.wo.weight.id, weights(D5 * D5, 74).into()),
            (block.gate.weight.id, weights(D5 * FF5, 75).into()),
            (block.up.weight.id, weights(D5 * FF5, 76).into()),
            (block.down.weight.id, weights(FF5 * D5, 77).into()),
            (k_cache.id, weights(SLOTS5 * KV_DIM, 78).into()),
            (v_cache.id, weights(SLOTS5 * KV_DIM, 79).into()),
            (gather_idx.id, vec![0i32, 1].into()),
            (scatter_idx.id, vec![1i32].into()),
        ];

        // Scalar reference.
        let get = |seed: usize, n: usize| weights(n, seed);
        let normed = ref_rms_norm(&x_vals, EPS);
        let q = ref_matmul(&normed, &get(71, D5 * D5), D5, D5);
        let k_new = ref_matmul(&normed, &get(72, D5 * KV_DIM), D5, KV_DIM);
        let v_new = ref_matmul(&normed, &get(73, D5 * KV_DIM), D5, KV_DIM);
        let mut ref_kc = get(78, SLOTS5 * KV_DIM);
        let mut ref_vc = get(79, SLOTS5 * KV_DIM);
        let attn = ref_paged_step_gqa(
            &q,
            &k_new,
            &v_new,
            &mut ref_kc,
            &mut ref_vc,
            &[0, 1],
            1,
            N_H,
            N_KV,
            HD,
            1,
            None,
            1.0 / (HD as f32).sqrt(),
        );
        let attn_proj = ref_matmul(&attn, &get(74, D5 * D5), D5, D5);
        let x1: Vec<f32> = x_vals.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
        let ff_in = ref_rms_norm(&x1, EPS);
        let gate = ref_silu(&ref_matmul(&ff_in, &get(75, D5 * FF5), D5, FF5));
        let up = ref_matmul(&ff_in, &get(76, D5 * FF5), D5, FF5);
        let hidden: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| a * b).collect();
        let ff = ref_matmul(&hidden, &get(77, FF5 * D5), FF5, D5);
        let expected: Vec<f32> = x1.iter().zip(&ff).map(|(a, b)| a + b).collect();

        let data: FxHashMap<_, _> = pairs.iter().cloned().collect();
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        let outcome = rt
            .search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        eprintln!(
            "[llama-block] attribution: {} | refusals: {}",
            outcome.timings.summary(),
            outcome.refusal_breakdown.summary()
        );
        for exemplar in &outcome.refusal_breakdown.exemplars {
            eprintln!("[llama-block] refusal exemplar: {exemplar}");
        }
        for (id, values) in &pairs {
            rt.set_data(*id, values.clone());
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("out"), &expected);
        assert_close(rt.get_f32(kc.id).expect("k cache"), &ref_kc);
        assert_close(rt.get_f32(vc.id).expect("v cache"), &ref_vc);
    }

    /// RUNG 6: scaling measurement — the llama block at 1/2/4/8 layers
    /// (d=8) and widths d=8/16/32 (1 layer), one decode step each,
    /// fixed 8-genome budget for comparability (+ the rung-5 default-
    /// budget point as reference). Prints one row per config. Run
    /// explicitly by name: `cargo test --release measure_scaling -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement harness — run explicitly by name (release)"]
    fn measure_scaling_curves() {
        let budget = ImplementationSearchOptions {
            generations: 2,
            generation_size: 4,
            mutations: 2,
            trials: 1,
            seed: 0,
        };
        eprintln!(
            "config | wall | saturation | extract | exec(best) | genomes refused (cycles/dead-ends)"
        );
        // (layers, d, use_default_budget): the fixed 8-genome budget gives
        // comparable refusal RATES; depth ≥ 2 needs the default budget to
        // complete at all (the choice-cycle cliff — see the report).
        for (layers, d, default_budget) in [
            (1usize, 8usize, false),
            (1, 16, false),
            (1, 32, false),
            (2, 8, true),
            (4, 8, true),
            (8, 8, true),
        ] {
            let n_heads = 4;
            let n_kv = 2;
            let ff = d + d / 2;
            let kv_dim = n_kv * (d / n_heads);
            const SLOTS6: usize = 4;
            const CTX6: usize = 2;

            let start = std::time::Instant::now();
            let mut cx = Graph::new();
            let blocks: Vec<LlamaBlock> = (0..layers)
                .map(|l| {
                    LlamaBlock::new(
                        d,
                        ff,
                        n_heads,
                        n_kv,
                        &Ns::root().child("layers").index(l),
                        &mut cx,
                    )
                })
                .collect();
            let x = cx.tensor((1, d));
            let caches: Vec<_> = (0..layers)
                .map(|_| (cx.tensor((SLOTS6, kv_dim)), cx.tensor((SLOTS6, kv_dim))))
                .collect();
            let gather_idx = cx.tensor_dtyped(CTX6, DType::Int);
            let scatter_idx = cx.tensor_dtyped(1, DType::Int);
            let mut h = x;
            let mut outs = Vec::new();
            for (layer, block) in blocks.iter().enumerate() {
                let (next, kc, vc) = block.forward(
                    h,
                    caches[layer].0,
                    caches[layer].1,
                    gather_idx,
                    scatter_idx,
                    IntExpr::from(1usize),
                );
                h = next;
                outs.push((kc.output(), vc.output()));
            }
            let _ = h.output();

            let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
                (x.id, weights(d, 90).into()),
                (gather_idx.id, vec![0i32, 1].into()),
                (scatter_idx.id, vec![1i32].into()),
            ];
            for (layer, block) in blocks.iter().enumerate() {
                pairs.push((block.wq.weight.id, weights(d * d, 91 + layer).into()));
                pairs.push((block.wk.weight.id, weights(d * kv_dim, 92 + layer).into()));
                pairs.push((block.wv.weight.id, weights(d * kv_dim, 93 + layer).into()));
                pairs.push((block.wo.weight.id, weights(d * d, 94 + layer).into()));
                pairs.push((block.gate.weight.id, weights(d * ff, 95 + layer).into()));
                pairs.push((block.up.weight.id, weights(d * ff, 96 + layer).into()));
                pairs.push((block.down.weight.id, weights(ff * d, 97 + layer).into()));
                pairs.push((
                    caches[layer].0.id,
                    weights(SLOTS6 * kv_dim, 98 + layer).into(),
                ));
                pairs.push((
                    caches[layer].1.id,
                    weights(SLOTS6 * kv_dim, 99 + layer).into(),
                ));
            }
            let data: FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt = ReferenceRuntime::load(&cx).expect("native load");
            let chosen = if default_budget {
                ImplementationSearchOptions::default()
            } else {
                budget.clone()
            };
            match rt.search(&data, &chosen) {
                Ok(outcome) => {
                    eprintln!(
                        "L{layers} d{d} | {:.1}s | {:.1}s | {:.1}s | {:.3}ms | {}",
                        start.elapsed().as_secs_f64(),
                        outcome.timings.saturation_nanos as f64 / 1e9,
                        outcome.timings.extract_nanos as f64 / 1e9,
                        outcome.best_nanos as f64 / 1e6,
                        outcome.refusal_breakdown.summary(),
                    );
                }
                Err(err) => {
                    eprintln!(
                        "L{layers} d{d} | {:.1}s | SEARCH REFUSED: {err:#}",
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        }
    }

    /// DIAGNOSTIC (run by name): draw random genomes on the 2-layer
    /// llama graph until one refuses, then dump the cyclic pairs'
    /// (logical, layout) anatomy — the "what ARE the Copy⟷Copy welds"
    /// question, answered from a live specimen.
    #[test]
    #[ignore = "diagnostic — run explicitly by name (release)"]
    fn probe_deadlock_anatomy() {
        let mut cx = Graph::new();
        let blocks: Vec<LlamaBlock> = (0..2)
            .map(|l| LlamaBlock::new(8, 12, 4, 2, &Ns::root().child("layers").index(l), &mut cx))
            .collect();
        let x = cx.tensor((1, 8));
        let caches: Vec<_> = (0..2)
            .map(|_| (cx.tensor((4, 4)), cx.tensor((4, 4))))
            .collect();
        let gather_idx = cx.tensor_dtyped(2, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let mut h = x;
        for (layer, block) in blocks.iter().enumerate() {
            let (next, kc, vc) = block.forward(
                h,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                IntExpr::from(1usize),
            );
            h = next;
            kc.output();
            vc.output();
        }
        h.output();

        // Assemble + saturate once, then draw single-genome searches
        // until a refusal is recorded (tiny budgets, varying seed).
        let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (x.id, weights(8, 90).into()),
            (gather_idx.id, vec![0i32, 1].into()),
            (scatter_idx.id, vec![1i32].into()),
        ];
        for (layer, block) in blocks.iter().enumerate() {
            pairs.push((block.wq.weight.id, weights(64, 91 + layer).into()));
            pairs.push((block.wk.weight.id, weights(32, 92 + layer).into()));
            pairs.push((block.wv.weight.id, weights(32, 93 + layer).into()));
            pairs.push((block.wo.weight.id, weights(64, 94 + layer).into()));
            pairs.push((block.gate.weight.id, weights(96, 95 + layer).into()));
            pairs.push((block.up.weight.id, weights(96, 96 + layer).into()));
            pairs.push((block.down.weight.id, weights(96, 97 + layer).into()));
            pairs.push((caches[layer].0.id, weights(16, 98 + layer).into()));
            pairs.push((caches[layer].1.id, weights(16, 99 + layer).into()));
        }
        let data: rustc_hash::FxHashMap<_, _> = pairs.iter().cloned().collect();
        for seed in 0..16 {
            let mut rt = luminal::reference::ReferenceRuntime::load(&cx).expect("native load");
            let outcome = rt.search(
                &data,
                &luminal::implementation_search::ImplementationSearchOptions {
                    generations: 1,
                    generation_size: 1,
                    mutations: 0,
                    trials: 1,
                    seed,
                },
            );
            match outcome {
                Err(_) => {
                    eprintln!(
                        "[anatomy] seed {seed} refused; last failure dissected by the search's own session (see exemplars above); re-deriving:"
                    );
                    // Re-run the same single genome through a session we
                    // control so the blockage record is inspectable.
                    // (search consumed the runtime; rebuild the pipeline)
                    let program = cx.logical.native_program().expect("clean");
                    let text = format!(
                        "{}\n\n{}",
                        luminal::egglog_snippet::assembled_program(),
                        program.text
                    );
                    let mut egraph = luminal::egglog_snippet::new_egraph();
                    egraph.parse_and_run_program(None, &text).expect("runs");
                    let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
                    let allow = luminal::reference::reference_allow_list();
                    let mut session =
                        luminal::extractor::ExtractionSession::new(&serialized, Some(&allow));
                    let index =
                        luminal::extractor::producer_index_with_ops(&serialized, Some(&allow));
                    use rand::{Rng, SeedableRng};
                    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                    let mut genome = luminal::extractor::Genome::default();
                    for (class, candidates) in &index {
                        let pick = &candidates[rng.random_range(0..candidates.len())];
                        genome.choices.insert(class.clone(), pick.1.clone());
                    }
                    if session.extract_with_genome(&genome).is_err() {
                        eprintln!("{}", session.blockage_anatomy());
                        return;
                    }
                    eprintln!(
                        "[anatomy] re-derived genome extracted (serialization nondeterminism) — trying next seed"
                    );
                }
                Ok(_) => continue,
            }
        }
        eprintln!("[anatomy] no refusal in 16 single-genome draws");
    }

    // ── shared scalar-reference pieces (single query row, s = 1) ──

    struct BlockFixture {
        cx: Graph,
        block: DecoderBlock,
        embed: Embedding,
        ids: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    }

    const D: usize = 4; // d_model = n_heads · head_dim
    const N_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const VOCAB: usize = 5;
    const SLOTS: usize = 4;
    const CTX: usize = 2;
    const PREV_SEQ: usize = 1;
    const FF_HIDDEN: usize = 6;
    const EXPERTS: usize = 2;

    fn block_fixture(
        ff: fn(&mut Graph) -> FeedForward,
    ) -> (BlockFixture, GraphTensor, GraphTensor, GraphTensor) {
        let mut cx = Graph::new();
        let embed = Embedding::new(VOCAB, D, &Ns::root().child("embed"), &mut cx);
        let a = Ns::root().child("attn");
        let block = DecoderBlock {
            wq: Linear::new(D, D, false, &a.child("q"), &mut cx),
            wk: Linear::new(D, D, false, &a.child("k"), &mut cx),
            wv: Linear::new(D, D, false, &a.child("v"), &mut cx),
            wo: Linear::new(D, D, false, &a.child("o"), &mut cx),
            ff: ff(&mut cx),
            n_heads: N_HEADS,
            n_kv_heads: N_HEADS,
            head_dim: HEAD_DIM,
        };
        let ids = cx.tensor_dtyped(1, DType::Int);
        let k_cache = cx.tensor((SLOTS, D));
        let v_cache = cx.tensor((SLOTS, D));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let x = embed.forward(ids);
        let (x, kc, vc) = block.forward(
            x,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(PREV_SEQ),
        );
        let logits = embed.reverse(x).output();
        let kc = kc.output();
        let vc = vc.output();
        (
            BlockFixture {
                cx,
                block,
                embed,
                ids,
                k_cache,
                v_cache,
                gather_idx,
                scatter_idx,
            },
            logits,
            kc,
            vc,
        )
    }

    /// Everything the block binds, with deterministic weights; returns
    /// (tensor-keyed data map, scalar-side copies).
    #[allow(clippy::type_complexity)]
    fn block_data(
        fx: &BlockFixture,
    ) -> (
        FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
        Vec<(petgraph::graph::NodeIndex, TypedBuffer)>,
    ) {
        let token = 3usize;
        let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (fx.ids.id, vec![token as i32].into()),
            (fx.embed.weight.id, weights(VOCAB * D, 1).into()),
            (fx.block.wq.weight.id, weights(D * D, 2).into()),
            (fx.block.wk.weight.id, weights(D * D, 3).into()),
            (fx.block.wv.weight.id, weights(D * D, 4).into()),
            (fx.block.wo.weight.id, weights(D * D, 5).into()),
            (fx.k_cache.id, weights(SLOTS * D, 6).into()),
            (fx.v_cache.id, weights(SLOTS * D, 8).into()),
            (fx.gather_idx.id, vec![0i32, 1].into()),
            (fx.scatter_idx.id, vec![1i32].into()),
        ];
        match &fx.block.ff {
            FeedForward::Dense { up, down } => {
                pairs.push((up.weight.id, weights(D * FF_HIDDEN, 9).into()));
                pairs.push((down.weight.id, weights(FF_HIDDEN * D, 10).into()));
            }
            FeedForward::Moe(moe) => {
                pairs.push((moe.router.id, weights(D * EXPERTS, 9).into()));
                pairs.push((moe.expert_weights.id, weights(EXPERTS * D * D, 10).into()));
            }
        }
        (pairs.iter().cloned().collect(), pairs)
    }

    /// Scalar reference for the whole block graph: embed → block → tied
    /// logits, plus the updated caches.
    fn block_reference(fx: &BlockFixture) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let embed_w = weights(VOCAB * D, 1);
        let token = 3usize;
        let x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
        let mut k_cache = weights(SLOTS * D, 6);
        let mut v_cache = weights(SLOTS * D, 8);
        let (wq, wk, wv, wo) = (
            weights(D * D, 2),
            weights(D * D, 3),
            weights(D * D, 4),
            weights(D * D, 5),
        );
        type FfnReference = dyn Fn(&[f32]) -> Vec<f32>;
        let ff: Box<FfnReference> = match &fx.block.ff {
            FeedForward::Dense { .. } => {
                let (up, down) = (weights(D * FF_HIDDEN, 9), weights(FF_HIDDEN * D, 10));
                Box::new(move |x: &[f32]| {
                    let hidden: Vec<f32> = ref_matmul(x, &up, D, FF_HIDDEN)
                        .iter()
                        .map(|v| v.max(0.0))
                        .collect();
                    ref_matmul(&hidden, &down, FF_HIDDEN, D)
                })
            }
            FeedForward::Moe(_) => {
                let (router, experts) = (weights(D * EXPERTS, 9), weights(EXPERTS * D * D, 10));
                Box::new(move |x: &[f32]| ref_moe_k1(x, &router, &experts, D, EXPERTS))
            }
        };
        let x2 = ref_block_step(
            &x,
            &wq,
            &wk,
            &wv,
            &wo,
            &*ff,
            &mut k_cache,
            &mut v_cache,
            &[0, 1],
            1,
            N_HEADS,
            HEAD_DIM,
            D,
        );
        // Tied logits: x2 · Eᵀ.
        let logits: Vec<f32> = (0..VOCAB)
            .map(|v| (0..D).map(|i| x2[i] * embed_w[v * D + i]).sum())
            .collect();
        (logits, k_cache, v_cache)
    }

    /// MODEL 2: the full decoder block (dense FFN) through the DEFAULT
    /// search ladder, with the stage attribution printed.
    #[test]
    fn decoder_block_matches_scalar_reference() {
        let (fx, logits, kc, vc) = block_fixture(|cx| FeedForward::Dense {
            up: Box::new(Linear::new(
                D,
                FF_HIDDEN,
                false,
                &Ns::root().child("up"),
                cx,
            )),
            down: Box::new(Linear::new(
                FF_HIDDEN,
                D,
                false,
                &Ns::root().child("down"),
                cx,
            )),
        });
        let (data, pairs) = block_data(&fx);
        let (ref_logits, ref_kc, ref_vc) = block_reference(&fx);

        let mut rt = ReferenceRuntime::load(&fx.cx).expect("native load");
        let outcome = rt
            .search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        eprintln!(
            "[decoder-block] search attribution: {} (plans profiled {}, cache hits {})",
            outcome.timings.summary(),
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );
        for (id, values) in pairs {
            rt.set_data(id, values);
        }
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        assert_close(rt.get_f32(kc.id).expect("k cache"), &ref_kc);
        assert_close(rt.get_f32(vc.id).expect("v cache"), &ref_vc);
    }

    /// MODEL 3: the same block with the MoE FFN (harness search budget —
    /// the flagship default-budget run is model 2).
    #[test]
    fn moe_decoder_block_matches_scalar_reference() {
        let (fx, logits, kc, vc) = block_fixture(|cx| {
            FeedForward::Moe(Box::new(MoE {
                expert_weights: cx.named_tensor("Experts", (EXPERTS, D, D)),
                router: cx.named_tensor("Router", (D, EXPERTS)),
                k: 1,
            }))
        });
        let (_, pairs) = block_data(&fx);
        let (ref_logits, ref_kc, ref_vc) = block_reference(&fx);

        let rt = luminal::test_support::run_reference(&fx.cx, &pairs);
        assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
        assert_close(rt.get_f32(kc.id).expect("k cache"), &ref_kc);
        assert_close(rt.get_f32(vc.id).expect("v cache"), &ref_vc);
    }

    /// MODEL 4: a TWO-LAYER decoder with pre-LayerNorms, decoded for TWO
    /// steps — the step-1 cache OUTPUTS feed the step-2 cache INPUTS
    /// through the binding surface (each step is its own shape-specialized
    /// program; harness search budget per step).
    #[test]
    fn tiny_decoder_two_steps_match_scalar_reference() {
        const LAYERS: usize = 2;
        const EPS: f32 = 1e-5;
        let tokens = [3usize, 1];

        // Scalar caches persist across BOTH steps, per layer.
        let mut ref_k: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut ref_v: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut runtime_k: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];
        let mut runtime_v: Vec<Vec<f32>> = vec![vec![0.0; SLOTS * D]; LAYERS];

        for (step, token) in tokens.iter().enumerate() {
            let prev_seq = step; // tokens already in the cache
            let ctx = step + 1; // slots visible this step
            let mut cx = Graph::new();
            let embed = Embedding::new(VOCAB, D, &Ns::root().child("embed"), &mut cx);
            let mut norms = Vec::new();
            let mut blocks = Vec::new();
            for l in 0..LAYERS {
                let lns = Ns::root().child("layers").index(l);
                norms.push(crate::LayerNorm::new(
                    D,
                    false,
                    false,
                    true,
                    EPS,
                    &lns.child("norm"),
                    &mut cx,
                ));
                blocks.push(DecoderBlock {
                    wq: Linear::new(D, D, false, &lns.child("q"), &mut cx),
                    wk: Linear::new(D, D, false, &lns.child("k"), &mut cx),
                    wv: Linear::new(D, D, false, &lns.child("v"), &mut cx),
                    wo: Linear::new(D, D, false, &lns.child("o"), &mut cx),
                    ff: FeedForward::Dense {
                        up: Box::new(Linear::new(D, FF_HIDDEN, false, &lns.child("up"), &mut cx)),
                        down: Box::new(Linear::new(
                            FF_HIDDEN,
                            D,
                            false,
                            &lns.child("down"),
                            &mut cx,
                        )),
                    },
                    n_heads: N_HEADS,
                    n_kv_heads: N_HEADS,
                    head_dim: HEAD_DIM,
                });
            }
            let model = TinyDecoder {
                embed,
                norms,
                blocks,
                final_norm: crate::LayerNorm::new(
                    D,
                    false,
                    false,
                    true,
                    EPS,
                    &Ns::root().child("final_norm"),
                    &mut cx,
                ),
            };
            let ids = cx.tensor_dtyped(1, DType::Int);
            let cache_inputs: Vec<(GraphTensor, GraphTensor)> = (0..LAYERS)
                .map(|_| (cx.tensor((SLOTS, D)), cx.tensor((SLOTS, D))))
                .collect();
            let gather_idx = cx.tensor_dtyped(ctx, DType::Int);
            let scatter_idx = cx.tensor_dtyped(1, DType::Int);
            let (logits, caches_out) = model.forward(
                ids,
                &cache_inputs,
                gather_idx,
                scatter_idx,
                IntExpr::from(prev_seq),
            );
            let logits = logits.output();
            let caches_out: Vec<_> = caches_out
                .into_iter()
                .map(|(k, v)| (k.output(), v.output()))
                .collect();

            // Per-layer weights, deterministic and step-independent.
            let layer_weights = |layer: usize| {
                let base = 20 + layer * 10;
                (
                    weights(D * D, base),
                    weights(D * D, base + 1),
                    weights(D * D, base + 2),
                    weights(D * D, base + 3),
                    weights(D * FF_HIDDEN, base + 4),
                    weights(FF_HIDDEN * D, base + 5),
                )
            };
            let embed_w = weights(VOCAB * D, 1);
            let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
                (ids.id, vec![*token as i32].into()),
                (model.embed.weight.id, embed_w.clone().into()),
                (gather_idx.id, (0..ctx as i32).collect::<Vec<i32>>().into()),
                (scatter_idx.id, vec![step as i32].into()),
            ];
            for layer in 0..LAYERS {
                let (wq, wk, wv, wo, up, down) = layer_weights(layer);
                let block = &model.blocks[layer];
                pairs.push((block.wq.weight.id, wq.into()));
                pairs.push((block.wk.weight.id, wk.into()));
                pairs.push((block.wv.weight.id, wv.into()));
                pairs.push((block.wo.weight.id, wo.into()));
                let FeedForward::Dense {
                    up: up_l,
                    down: down_l,
                } = &block.ff
                else {
                    unreachable!()
                };
                pairs.push((up_l.weight.id, up.into()));
                pairs.push((down_l.weight.id, down.into()));
                pairs.push((cache_inputs[layer].0.id, runtime_k[layer].clone().into()));
                pairs.push((cache_inputs[layer].1.id, runtime_v[layer].clone().into()));
            }

            // Scalar reference for this step.
            let mut x: Vec<f32> = embed_w[token * D..(token + 1) * D].to_vec();
            let gather: Vec<usize> = (0..ctx).collect();
            for layer in 0..LAYERS {
                let (wq, wk, wv, wo, up, down) = layer_weights(layer);
                x = ref_layer_norm(&x, EPS);
                let ff = move |x: &[f32]| {
                    let hidden: Vec<f32> = ref_matmul(x, &up, D, FF_HIDDEN)
                        .iter()
                        .map(|v| v.max(0.0))
                        .collect();
                    ref_matmul(&hidden, &down, FF_HIDDEN, D)
                };
                x = ref_block_step(
                    &x,
                    &wq,
                    &wk,
                    &wv,
                    &wo,
                    &ff,
                    &mut ref_k[layer],
                    &mut ref_v[layer],
                    &gather,
                    step,
                    N_HEADS,
                    HEAD_DIM,
                    D,
                );
            }
            let x = ref_layer_norm(&x, EPS);
            let ref_logits: Vec<f32> = (0..VOCAB)
                .map(|v| (0..D).map(|i| x[i] * embed_w[v * D + i]).sum())
                .collect();

            // Two layers double the genome decision points: the harness
            // budget's 8 genomes all hit choice-cycle discards, so this
            // model runs the DEFAULT budget (64 genomes).
            let data: FxHashMap<_, _> = pairs.iter().cloned().collect();
            let mut rt = ReferenceRuntime::load(&cx).expect("native load");
            rt.search(&data, &ImplementationSearchOptions::default())
                .expect("search finds a plan");
            for (id, values) in &pairs {
                rt.set_data(*id, values.clone());
            }
            rt.execute().expect("winner executes");
            assert_close(rt.get_f32(logits.id).expect("logits"), &ref_logits);
            for layer in 0..LAYERS {
                let k_out = rt.get_f32(caches_out[layer].0.id).expect("k cache").clone();
                let v_out = rt.get_f32(caches_out[layer].1.id).expect("v cache").clone();
                assert_close(&k_out, &ref_k[layer]);
                assert_close(&v_out, &ref_v[layer]);
                runtime_k[layer] = k_out; // runtime-out → next step's runtime-in
                runtime_v[layer] = v_out;
            }
        }
    }
}

#[cfg(test)]
mod forward_rope_tests {
    use super::LlamaBlock;
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use scalar_refs::*;

    /// forward_rope ≡ the hand-composed IntExpr path: one graph, one
    /// set of block weights, the same decode-step inputs — the new
    /// method on one side; rms_norm_heads → rotary_apply →
    /// paged_attention(IntExpr) → residual/FFN spelled out on the
    /// other. Outputs and both cache outs must agree exactly, proving
    /// the rope threading and the data-driven mask change nothing at
    /// the pinned position.
    #[test]
    fn llama_block_forward_rope_matches_expression_path() {
        const D: usize = 8;
        const FF: usize = 12;
        const N_HEADS: usize = 4;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = D / N_HEADS;
        const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
        const SLOTS: usize = 4;
        const CTX: usize = 3;
        let position = 1usize;

        let mut cx = Graph::new();
        let block = LlamaBlock::new(
            D,
            FF,
            N_HEADS,
            N_KV_HEADS,
            &Ns::root().child("blk"),
            &mut cx,
        )
        .with_qk_norm(&Ns::root().child("blk"), &mut cx);
        let x = cx.tensor((1, D));
        let k_cache = cx.tensor((SLOTS, KV_DIM));
        let v_cache = cx.tensor((SLOTS, KV_DIM));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let rope_cos = cx.tensor((1, HEAD_DIM));
        let rope_sin = cx.tensor((1, HEAD_DIM));
        let rope_rot = cx.tensor((HEAD_DIM, HEAD_DIM));

        let (out_rope, k_rope, v_rope) = block.forward_rope(
            x,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let out_rope = out_rope.output();
        let k_rope = k_rope.output();
        let v_rope = v_rope.output();

        // The same anatomy composed by hand on the IntExpr path.
        let normed = block.attn_norm.forward(x);
        let (q_weight, k_weight) = block.qk_norm.expect("qk-norm minted");
        let q = crate::rotary_apply(
            crate::rms_norm_heads(block.wq.forward(normed), HEAD_DIM, q_weight, 1e-6),
            HEAD_DIM,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = crate::rotary_apply(
            crate::rms_norm_heads(block.wk.forward(normed), HEAD_DIM, k_weight, 1e-6),
            HEAD_DIM,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let (attn, k_expr, v_expr) = crate::paged_attention(
            q,
            k,
            block.wv.forward(normed),
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(position),
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
        );
        let x_mid = x + block.wo.forward(attn);
        let ff_in = block.ffn_norm.forward(x_mid);
        let ff = block
            .down
            .forward(block.gate.forward(ff_in).silu() * block.up.forward(ff_in));
        let out_expr = (x_mid + ff).output();
        let k_expr = k_expr.output();
        let v_expr = v_expr.output();

        let mut pairs: Vec<(petgraph::graph::NodeIndex, TypedBuffer)> = vec![
            (x.id, weights(D, 1).into()),
            (k_cache.id, weights(SLOTS * KV_DIM, 2).into()),
            (v_cache.id, weights(SLOTS * KV_DIM, 3).into()),
            (gather_idx.id, vec![0i32, 1, 2].into()),
            (scatter_idx.id, vec![1i32].into()),
            (q_pos.id, vec![position as i32].into()),
        ];
        let (cos, sin) = crate::rope_tables_split_half(&[position as f32], HEAD_DIM, 10_000.0, 1.0);
        pairs.push((rope_cos.id, cos.into()));
        pairs.push((rope_sin.id, sin.into()));
        pairs.push((
            rope_rot.id,
            crate::rope_pairing_matrix(HEAD_DIM, false).into(),
        ));
        for (seed, tensor) in [
            block.wq.weight,
            block.wk.weight,
            block.wv.weight,
            block.wo.weight,
            block.gate.weight,
            block.up.weight,
            block.down.weight,
            q_weight,
            k_weight,
        ]
        .into_iter()
        .enumerate()
        {
            let n: usize = tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("static dim"))
                .product();
            pairs.push((tensor.id, weights(n, 10 + seed).into()));
        }

        let rt = luminal::test_support::run_reference(&cx, &pairs);
        let rope_out = rt.get_f32(out_rope.id).expect("rope out").clone();
        let expr_out = rt.get_f32(out_expr.id).expect("expr out").clone();
        assert_close(&rope_out, &expr_out);
        assert_close(
            rt.get_f32(k_rope.id).expect("k rope"),
            rt.get_f32(k_expr.id).expect("k expr"),
        );
        assert_close(
            rt.get_f32(v_rope.id).expect("v rope"),
            rt.get_f32(v_expr.id).expect("v expr"),
        );
    }
}
