//! MiniDit — the flux2 family mini MODEL DEFINITION (relocated out of
//! luminal_nn 2026-08-13: model definitions live in their example
//! crates; luminal_nn keeps only the building blocks). The runner in
//! src/main.rs drives one denoising velocity prediction; the scalar
//! fidelity test lives in tests/fidelity.rs.

#[path = "../../../common/model_support.rs"]
mod model_support;

use crate::model_support::{scatter_rows, LayerNorm, Linear, Namespace};
use luminal::prelude::*;

/// MiniDit — the flux2 family mini. Family-unique constructs carried
/// faithfully: sinusoidal t/guidance conditioning summed through SiLU
/// MLPs; SHARED adaLN modulation tables cut into (shift, scale, gate)
/// triples; gated residuals `x += gate ⊙ sublayer`; no-affine
/// LayerNorms; one DOUBLE-stream block (separate img/txt weights, one
/// joint attention over [txt ‖ img]); one SINGLE-stream block (fused
/// qkv+mlp in-projection, fused out-projection over [attn ‖ mlp]);
/// per-head QK RMSNorm; interleaved-pair multi-axis RoPE from host
/// tables; non-causal maskless SDPA; AdaLayerNormContinuous head with
/// the REVERSED (scale, shift) order. Patchify/VAE/scheduler are
/// host-side in the flux2 example and stay outside the family constructs.
pub struct MiniDit {
    pub x_embed: Linear,
    pub ctx_embed: Linear,
    pub t_mlp1: Linear,
    pub t_mlp2: Linear,
    pub g_mlp1: Linear,
    pub g_mlp2: Linear,
    pub mod_img: Linear,
    pub mod_txt: Linear,
    pub mod_single: Linear,
    pub norm_out: Linear,
    pub proj_out: Linear,
    pub img_q: Linear,
    pub img_k: Linear,
    pub img_v: Linear,
    pub img_out: Linear,
    pub txt_q: Linear,
    pub txt_k: Linear,
    pub txt_v: Linear,
    pub txt_out: Linear,
    pub img_qnorm: GraphTensor,
    pub img_knorm: GraphTensor,
    pub txt_qnorm: GraphTensor,
    pub txt_knorm: GraphTensor,
    pub ff_in: Linear,
    pub ff_out: Linear,
    pub ctx_ff_in: Linear,
    pub ctx_ff_out: Linear,
    pub single_proj: Linear,
    /// The single-stream out-projection, SPLIT into its attn-rows and
    /// mlp-rows halves: out = attn·W_attn + mlp·W_mlp — algebraically
    /// identical to flux2's fused `to_out @ [attn ‖ mlp]`, spelled
    /// without the concat (rejoin-divergence workaround; the fused
    /// spelling returns with the divergence ruling).
    pub single_out_attn: Linear,
    pub single_out_mlp: Linear,
    pub single_qnorm: GraphTensor,
    pub single_knorm: GraphTensor,
    ln: LayerNorm, // no-affine LayerNorm, shared (stateless)
    d: usize,
    head_dim: usize,
    mlp: usize,
    t_half: usize,
    s_txt: usize,
}

impl MiniDit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        txt_dim: usize,
        d: usize,
        n_heads: usize,
        mlp: usize,
        t_half: usize,
        s_txt: usize,
        cx: &mut Graph,
    ) -> Self {
        let head_dim = d / n_heads;
        Self {
            x_embed: Linear::new(
                in_channels,
                d,
                false,
                DType::F32,
                &Namespace::root().child("x_embed"),
                cx,
            ),
            ctx_embed: Linear::new(
                txt_dim,
                d,
                false,
                DType::F32,
                &Namespace::root().child("ctx_embed"),
                cx,
            ),
            t_mlp1: Linear::new(
                2 * t_half,
                d,
                false,
                DType::F32,
                &Namespace::root().child("t_mlp1"),
                cx,
            ),
            t_mlp2: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("t_mlp2"),
                cx,
            ),
            g_mlp1: Linear::new(
                2 * t_half,
                d,
                false,
                DType::F32,
                &Namespace::root().child("g_mlp1"),
                cx,
            ),
            g_mlp2: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("g_mlp2"),
                cx,
            ),
            mod_img: Linear::new(
                d,
                6 * d,
                false,
                DType::F32,
                &Namespace::root().child("mod_img"),
                cx,
            ),
            mod_txt: Linear::new(
                d,
                6 * d,
                false,
                DType::F32,
                &Namespace::root().child("mod_txt"),
                cx,
            ),
            mod_single: Linear::new(
                d,
                3 * d,
                false,
                DType::F32,
                &Namespace::root().child("mod_single"),
                cx,
            ),
            norm_out: Linear::new(
                d,
                2 * d,
                false,
                DType::F32,
                &Namespace::root().child("norm_out"),
                cx,
            ),
            proj_out: Linear::new(
                d,
                in_channels,
                false,
                DType::F32,
                &Namespace::root().child("proj_out"),
                cx,
            ),
            img_q: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("img_q"),
                cx,
            ),
            img_k: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("img_k"),
                cx,
            ),
            img_v: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("img_v"),
                cx,
            ),
            img_out: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("img_out"),
                cx,
            ),
            txt_q: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("txt_q"),
                cx,
            ),
            txt_k: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("txt_k"),
                cx,
            ),
            txt_v: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("txt_v"),
                cx,
            ),
            txt_out: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("txt_out"),
                cx,
            ),
            img_qnorm: cx.named_tensor("ImgQNorm", head_dim, DType::F32),
            img_knorm: cx.named_tensor("ImgKNorm", head_dim, DType::F32),
            txt_qnorm: cx.named_tensor("TxtQNorm", head_dim, DType::F32),
            txt_knorm: cx.named_tensor("TxtKNorm", head_dim, DType::F32),
            ff_in: Linear::new(
                d,
                2 * mlp,
                false,
                DType::F32,
                &Namespace::root().child("ff_in"),
                cx,
            ),
            ff_out: Linear::new(
                mlp,
                d,
                false,
                DType::F32,
                &Namespace::root().child("ff_out"),
                cx,
            ),
            ctx_ff_in: Linear::new(
                d,
                2 * mlp,
                false,
                DType::F32,
                &Namespace::root().child("ctx_ff_in"),
                cx,
            ),
            ctx_ff_out: Linear::new(
                mlp,
                d,
                false,
                DType::F32,
                &Namespace::root().child("ctx_ff_out"),
                cx,
            ),
            single_proj: Linear::new(
                d,
                3 * d + 2 * mlp,
                false,
                DType::F32,
                &Namespace::root().child("single_proj"),
                cx,
            ),
            single_out_attn: Linear::new(
                d,
                d,
                false,
                DType::F32,
                &Namespace::root().child("single_out_attn"),
                cx,
            ),
            single_out_mlp: Linear::new(
                mlp,
                d,
                false,
                DType::F32,
                &Namespace::root().child("single_out_mlp"),
                cx,
            ),
            single_qnorm: cx.named_tensor("SglQNorm", head_dim, DType::F32),
            single_knorm: cx.named_tensor("SglKNorm", head_dim, DType::F32),
            ln: LayerNorm::new(
                d,
                false,
                false,
                true,
                1e-6,
                DType::F32,
                &Namespace::root().child("ln"),
                cx,
            ),
            d,
            head_dim,
            mlp,
            t_half,
            s_txt,
        }
    }

    /// Sinusoidal embedding of a (1,) scalar: [cos(1000x·fᵢ) ‖ sin(1000x·fᵢ)],
    /// fᵢ = 10000^(-i/half) — flip_sin_to_cos ordering, as flux2.
    fn sinusoid(&self, x: GraphTensor) -> GraphTensor {
        let mut cos_parts = Vec::with_capacity(self.t_half);
        let mut sin_parts = Vec::with_capacity(self.t_half);
        for i in 0..self.t_half {
            let freq = (-(i as f32) * (10000f32).ln() / self.t_half as f32).exp();
            let arg = x * (1000.0 * freq);
            cos_parts.push(arg.cos());
            sin_parts.push(arg.sin());
        }
        let mut parts = cos_parts;
        parts.extend(sin_parts);
        let mut cat = parts[0];
        for part in &parts[1..] {
            cat = cat.concat_along(*part, 0);
        }
        cat.unsqueeze(0) // (1, 2·half)
    }

    /// latent (s_img, in_ch), text (s_txt, txt_dim), t (1,), guidance (1,),
    /// rope tables (s_txt+s_img, head_dim), the interleaved pairing
    /// matrix (head_dim, head_dim), and a zeros base (s_txt+s_img, d)
    /// for the scatter-assembled joint sequence → velocity (s_img, in_ch).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: GraphTensor,
        text: GraphTensor,
        t: GraphTensor,
        guidance: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        joint_base: GraphTensor,
    ) -> GraphTensor {
        let (d, mlp, s_txt) = (self.d, self.mlp, self.s_txt);
        let temb = self
            .t_mlp2
            .forward(self.t_mlp1.forward(self.sinusoid(t)).silu())
            + self
                .g_mlp2
                .forward(self.g_mlp1.forward(self.sinusoid(guidance)).silu()); // (1, d)
        let cond = temb.silu();
        let m_img = self.mod_img.forward(cond); // (1, 6d): 2 × (shift, scale, gate)
        let m_txt = self.mod_txt.forward(cond);
        let m_single = self.mod_single.forward(cond); // (1, 3d)
        let triple = |m: GraphTensor, set: usize| {
            let base = set * 3 * d;
            (
                m.slice_along(base..base + d, 1),             // shift
                m.slice_along(base + d..base + 2 * d, 1),     // scale
                m.slice_along(base + 2 * d..base + 3 * d, 1), // gate
            )
        };
        let ada = |x: GraphTensor, scale: GraphTensor, shift: GraphTensor| {
            let dims = x.dims();
            x * (scale + 1.0).expand(dims.clone()) + shift.expand(dims)
        };
        let gate = |x: GraphTensor, g: GraphTensor| {
            let dims = x.dims();
            x * g.expand(dims)
        };
        let heads = |x: GraphTensor| x.split_dims(1, self.head_dim).permute((1, 0, 2)); // (H,S,hd)
        let unheads = |x: GraphTensor| x.permute((1, 0, 2)).merge_dims(1, 2); // (S,d)
        let head_rms = |x: GraphTensor, weight: GraphTensor| {
            let dims = x.dims();
            let inv = ((x * x).mean(2) + 1e-6).sqrt().reciprocal(); // (H,S)
            x * inv.unsqueeze(2).expand(dims.clone())
                * weight.unsqueeze(0).unsqueeze(0).expand(dims)
        };
        let rope = |x: GraphTensor| {
            // Interleaved-pair rotation via the pairing matrix — the
            // concat-free spelling (rejoin-divergence workaround):
            // rope(x) = x ⊙ cos + (x @ R) ⊙ sin on (H, S, hd).
            let dims = x.dims();
            let rotated = x.matmul(rope_rot);
            x * rope_cos.unsqueeze(0).expand(dims.clone())
                + rotated * rope_sin.unsqueeze(0).expand(dims)
        };
        let sdpa = |q: GraphTensor, k: GraphTensor, v: GraphTensor| {
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let scores = q.matmul(k.permute((0, 2, 1))) * scale; // (H,S,S)
            scores.softmax(2).matmul(v) // (H,S,hd)
        };
        let swiglu =
            |u: GraphTensor| u.slice_along(0..mlp, 1).silu() * u.slice_along(mlp..2 * mlp, 1);

        // ---- double-stream block (txt first in every concat/split) ----
        let (shift0, scale0, gate0) = triple(m_img, 0);
        let (shift1, scale1, gate1) = triple(m_img, 1);
        let (c_shift0, c_scale0, c_gate0) = triple(m_txt, 0);
        let (c_shift1, c_scale1, c_gate1) = triple(m_txt, 1);
        let mut img = self.x_embed.forward(latent); // (s_img, d)
        let mut txt = self.ctx_embed.forward(text); // (s_txt, d)
        let img_n = ada(self.ln.forward(img), scale0, shift0);
        let txt_n = ada(self.ln.forward(txt), c_scale0, c_shift0);
        let q_img = head_rms(heads(self.img_q.forward(img_n)), self.img_qnorm);
        let k_img = head_rms(heads(self.img_k.forward(img_n)), self.img_knorm);
        let q_txt = head_rms(heads(self.txt_q.forward(txt_n)), self.txt_qnorm);
        let k_txt = head_rms(heads(self.txt_k.forward(txt_n)), self.txt_knorm);
        // V's sequence concat happens FLAT, before the head split — the
        // head reshape commutes with row concat, and pads over matmul
        // outputs (compute) never form the pure-view stack the rejoin
        // divergence needs. q/k concat after head_rms (a compute) for
        // the same reason.
        let v_all = heads(
            self.txt_v
                .forward(txt_n)
                .concat_along(self.img_v.forward(img_n), 0),
        );
        let attn = unheads(sdpa(
            rope(q_txt.concat_along(q_img, 1)),
            rope(k_txt.concat_along(k_img, 1)),
            v_all,
        )); // (s, d)
        let attn_txt = attn.slice_along(0..s_txt, 0);
        let attn_img = attn.slice_along(s_txt.., 0);
        img += gate(self.img_out.forward(attn_img), gate0);
        txt += gate(self.txt_out.forward(attn_txt), c_gate0);
        let ff = swiglu(
            self.ff_in
                .forward(ada(self.ln.forward(img), scale1, shift1)),
        );
        img += gate(self.ff_out.forward(ff), gate1);
        let c_ff = swiglu(
            self.ctx_ff_in
                .forward(ada(self.ln.forward(txt), c_scale1, c_shift1)),
        );
        txt += gate(self.ctx_ff_out.forward(c_ff), c_gate1);

        // ---- single-stream block over [txt ‖ img] ----
        // The joint sequence assembles by SCATTER writes into a zero
        // base (the paged-attention family's own row-assembly spelling)
        // instead of concat's pad+add: the head SLICES this tensor, and
        // a slice distributing down to a pad's clamp view re-creates
        // the rejoin-divergence stack (measured: stage-8 probe). Scatter
        // is a compute write — the slice stops there.
        let graph = latent.graph();
        let txt_positions = graph.arange(s_txt);
        let img_positions = graph.iota(latent.dims()[0], move |c| c[0] + s_txt);
        let mut hidden = scatter_rows(
            img,
            img_positions,
            scatter_rows(txt, txt_positions, joint_base),
        ); // (s, d)
        let (s_shift, s_scale, s_gate) = triple(m_single, 0);
        let normed = ada(self.ln.forward(hidden), s_scale, s_shift);
        let proj = self.single_proj.forward(normed); // (s, 3d + 2·mlp)
        let q = head_rms(heads(proj.slice_along(0..d, 1)), self.single_qnorm);
        let k = head_rms(heads(proj.slice_along(d..2 * d, 1)), self.single_knorm);
        let v = heads(proj.slice_along(2 * d..3 * d, 1));
        let attn = unheads(sdpa(rope(q), rope(k), v)); // (s, d)
        let mlp_out = swiglu(proj.slice_along(3 * d..3 * d + 2 * mlp, 1)); // (s, mlp)
                                                                           // Fused out-projection over [attn ‖ mlp], spelled as the
                                                                           // row-split sum (see the single_out_* field note).
        hidden += gate(
            self.single_out_attn.forward(attn) + self.single_out_mlp.forward(mlp_out),
            s_gate,
        );

        // ---- AdaLayerNormContinuous head: (scale, shift) — REVERSED ----
        let img_final = hidden.slice_along(s_txt.., 0); // (s_img, d)
        let head = self.norm_out.forward(cond); // (1, 2d)
        let scale = head.slice_along(0..d, 1);
        let shift = head.slice_along(d..2 * d, 1);
        self.proj_out
            .forward(ada(self.ln.forward(img_final), scale, shift))
    }
}

/// Host-side interleaved-pair RoPE tables for the mini DiT grid
/// (mirrors flux2's host-precomputed tables): rows are the s_txt text
/// tokens with ids (0,0,0,ℓ), then the h·w image tokens with ids
/// (0,hi,wi,0) row-major. Four axes × 2 dims each (half = 1 per axis, so
/// the frequency is θ⁰ = 1 and θ drops out); every cos/sin value is
/// written twice — the repeat_interleave that matches adjacent-pair
/// rotation. Tables are (s_txt + h·w, 8); head_dim must be 8.
pub fn mini_dit_rope_tables(s_txt: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut ids: Vec<[f32; 4]> = (0..s_txt).map(|l| [0.0, 0.0, 0.0, l as f32]).collect();
    for hi in 0..h {
        for wi in 0..w {
            ids.push([0.0, hi as f32, wi as f32, 0.0]);
        }
    }
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for id in ids {
        for coordinate in id {
            for _ in 0..2 {
                cos.push(coordinate.cos());
                sin.push(coordinate.sin());
            }
        }
    }
    (cos, sin)
}
