//! MiniDit — the flux2 family mini MODEL DEFINITION (relocated out of
//! luminal_nn 2026-08-13: model definitions live in their example
//! crates; luminal_nn keeps only the building blocks). The runner in
//! src/main.rs drives one denoising velocity prediction; the scalar
//! fidelity test lives in tests/fidelity.rs.

use crate::namespace::Namespace;
use luminal::prelude::*;
use luminal_nn::scatter_rows;

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
    pub x_embed_weight: GraphTensor,
    pub x_embed_bias: Option<GraphTensor>,
    pub ctx_embed_weight: GraphTensor,
    pub ctx_embed_bias: Option<GraphTensor>,
    pub t_mlp1_weight: GraphTensor,
    pub t_mlp1_bias: Option<GraphTensor>,
    pub t_mlp2_weight: GraphTensor,
    pub t_mlp2_bias: Option<GraphTensor>,
    pub g_mlp1_weight: GraphTensor,
    pub g_mlp1_bias: Option<GraphTensor>,
    pub g_mlp2_weight: GraphTensor,
    pub g_mlp2_bias: Option<GraphTensor>,
    pub mod_img_weight: GraphTensor,
    pub mod_img_bias: Option<GraphTensor>,
    pub mod_txt_weight: GraphTensor,
    pub mod_txt_bias: Option<GraphTensor>,
    pub mod_single_weight: GraphTensor,
    pub mod_single_bias: Option<GraphTensor>,
    pub norm_out_weight: GraphTensor,
    pub norm_out_bias: Option<GraphTensor>,
    pub proj_out_weight: GraphTensor,
    pub proj_out_bias: Option<GraphTensor>,
    pub img_q_weight: GraphTensor,
    pub img_q_bias: Option<GraphTensor>,
    pub img_k_weight: GraphTensor,
    pub img_k_bias: Option<GraphTensor>,
    pub img_v_weight: GraphTensor,
    pub img_v_bias: Option<GraphTensor>,
    pub img_out_weight: GraphTensor,
    pub img_out_bias: Option<GraphTensor>,
    pub txt_q_weight: GraphTensor,
    pub txt_q_bias: Option<GraphTensor>,
    pub txt_k_weight: GraphTensor,
    pub txt_k_bias: Option<GraphTensor>,
    pub txt_v_weight: GraphTensor,
    pub txt_v_bias: Option<GraphTensor>,
    pub txt_out_weight: GraphTensor,
    pub txt_out_bias: Option<GraphTensor>,
    pub img_qnorm: GraphTensor,
    pub img_knorm: GraphTensor,
    pub txt_qnorm: GraphTensor,
    pub txt_knorm: GraphTensor,
    pub ff_in_weight: GraphTensor,
    pub ff_in_bias: Option<GraphTensor>,
    pub ff_out_weight: GraphTensor,
    pub ff_out_bias: Option<GraphTensor>,
    pub ctx_ff_in_weight: GraphTensor,
    pub ctx_ff_in_bias: Option<GraphTensor>,
    pub ctx_ff_out_weight: GraphTensor,
    pub ctx_ff_out_bias: Option<GraphTensor>,
    pub single_proj_weight: GraphTensor,
    pub single_proj_bias: Option<GraphTensor>,
    /// The single-stream out-projection, SPLIT into its attn-rows and
    /// mlp-rows halves: out = attn·W_attn + mlp·W_mlp — algebraically
    /// identical to flux2's fused `to_out @ [attn ‖ mlp]`, spelled
    /// without the concat (rejoin-divergence workaround; the fused
    /// spelling returns with the divergence ruling).
    pub single_out_attn_weight: GraphTensor,
    pub single_out_attn_bias: Option<GraphTensor>,
    pub single_out_mlp_weight: GraphTensor,
    pub single_out_mlp_bias: Option<GraphTensor>,
    pub single_qnorm: GraphTensor,
    pub single_knorm: GraphTensor,
    ln_weight: Option<GraphTensor>,
    ln_bias: Option<GraphTensor>,
    ln_epsilon: f32, // no-affine LayerNorm, shared (stateless)
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
            x_embed_weight: cx.named_tensor(
                Namespace::root().child("x_embed").leaf("weight"),
                vec![in_channels, d],
                DType::F32,
            ),
            x_embed_bias: None,
            ctx_embed_weight: cx.named_tensor(
                Namespace::root().child("ctx_embed").leaf("weight"),
                vec![txt_dim, d],
                DType::F32,
            ),
            ctx_embed_bias: None,
            t_mlp1_weight: cx.named_tensor(
                Namespace::root().child("t_mlp1").leaf("weight"),
                vec![2 * t_half, d],
                DType::F32,
            ),
            t_mlp1_bias: None,
            t_mlp2_weight: cx.named_tensor(
                Namespace::root().child("t_mlp2").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            t_mlp2_bias: None,
            g_mlp1_weight: cx.named_tensor(
                Namespace::root().child("g_mlp1").leaf("weight"),
                vec![2 * t_half, d],
                DType::F32,
            ),
            g_mlp1_bias: None,
            g_mlp2_weight: cx.named_tensor(
                Namespace::root().child("g_mlp2").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            g_mlp2_bias: None,
            mod_img_weight: cx.named_tensor(
                Namespace::root().child("mod_img").leaf("weight"),
                vec![d, 6 * d],
                DType::F32,
            ),
            mod_img_bias: None,
            mod_txt_weight: cx.named_tensor(
                Namespace::root().child("mod_txt").leaf("weight"),
                vec![d, 6 * d],
                DType::F32,
            ),
            mod_txt_bias: None,
            mod_single_weight: cx.named_tensor(
                Namespace::root().child("mod_single").leaf("weight"),
                vec![d, 3 * d],
                DType::F32,
            ),
            mod_single_bias: None,
            norm_out_weight: cx.named_tensor(
                Namespace::root().child("norm_out").leaf("weight"),
                vec![d, 2 * d],
                DType::F32,
            ),
            norm_out_bias: None,
            proj_out_weight: cx.named_tensor(
                Namespace::root().child("proj_out").leaf("weight"),
                vec![d, in_channels],
                DType::F32,
            ),
            proj_out_bias: None,
            img_q_weight: cx.named_tensor(
                Namespace::root().child("img_q").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            img_q_bias: None,
            img_k_weight: cx.named_tensor(
                Namespace::root().child("img_k").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            img_k_bias: None,
            img_v_weight: cx.named_tensor(
                Namespace::root().child("img_v").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            img_v_bias: None,
            img_out_weight: cx.named_tensor(
                Namespace::root().child("img_out").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            img_out_bias: None,
            txt_q_weight: cx.named_tensor(
                Namespace::root().child("txt_q").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            txt_q_bias: None,
            txt_k_weight: cx.named_tensor(
                Namespace::root().child("txt_k").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            txt_k_bias: None,
            txt_v_weight: cx.named_tensor(
                Namespace::root().child("txt_v").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            txt_v_bias: None,
            txt_out_weight: cx.named_tensor(
                Namespace::root().child("txt_out").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            txt_out_bias: None,
            img_qnorm: cx.named_tensor("ImgQNorm", vec![head_dim], DType::F32),
            img_knorm: cx.named_tensor("ImgKNorm", vec![head_dim], DType::F32),
            txt_qnorm: cx.named_tensor("TxtQNorm", vec![head_dim], DType::F32),
            txt_knorm: cx.named_tensor("TxtKNorm", vec![head_dim], DType::F32),
            ff_in_weight: cx.named_tensor(
                Namespace::root().child("ff_in").leaf("weight"),
                vec![d, 2 * mlp],
                DType::F32,
            ),
            ff_in_bias: None,
            ff_out_weight: cx.named_tensor(
                Namespace::root().child("ff_out").leaf("weight"),
                vec![mlp, d],
                DType::F32,
            ),
            ff_out_bias: None,
            ctx_ff_in_weight: cx.named_tensor(
                Namespace::root().child("ctx_ff_in").leaf("weight"),
                vec![d, 2 * mlp],
                DType::F32,
            ),
            ctx_ff_in_bias: None,
            ctx_ff_out_weight: cx.named_tensor(
                Namespace::root().child("ctx_ff_out").leaf("weight"),
                vec![mlp, d],
                DType::F32,
            ),
            ctx_ff_out_bias: None,
            single_proj_weight: cx.named_tensor(
                Namespace::root().child("single_proj").leaf("weight"),
                vec![d, 3 * d + 2 * mlp],
                DType::F32,
            ),
            single_proj_bias: None,
            single_out_attn_weight: cx.named_tensor(
                Namespace::root().child("single_out_attn").leaf("weight"),
                vec![d, d],
                DType::F32,
            ),
            single_out_attn_bias: None,
            single_out_mlp_weight: cx.named_tensor(
                Namespace::root().child("single_out_mlp").leaf("weight"),
                vec![mlp, d],
                DType::F32,
            ),
            single_out_mlp_bias: None,
            single_qnorm: cx.named_tensor("SglQNorm", vec![head_dim], DType::F32),
            single_knorm: cx.named_tensor("SglKNorm", vec![head_dim], DType::F32),
            ln_weight: None,
            ln_bias: None,
            ln_epsilon: 1e-6,
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
        let temb = luminal_nn::linear(
            luminal_nn::linear(self.sinusoid(t), self.t_mlp1_weight, self.t_mlp1_bias).silu(),
            self.t_mlp2_weight,
            self.t_mlp2_bias,
        ) + luminal_nn::linear(
            luminal_nn::linear(
                self.sinusoid(guidance),
                self.g_mlp1_weight,
                self.g_mlp1_bias,
            )
            .silu(),
            self.g_mlp2_weight,
            self.g_mlp2_bias,
        ); // (1, d)
        let cond = temb.silu();
        let m_img = luminal_nn::linear(cond, self.mod_img_weight, self.mod_img_bias); // (1, 6d): 2 × (shift, scale, gate)
        let m_txt = luminal_nn::linear(cond, self.mod_txt_weight, self.mod_txt_bias);
        let m_single = luminal_nn::linear(cond, self.mod_single_weight, self.mod_single_bias); // (1, 3d)
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
        let heads = |x: GraphTensor| x.split_dims(1, self.head_dim).permute(vec![1, 0, 2]); // (H,S,hd)
        let unheads = |x: GraphTensor| x.permute(vec![1, 0, 2]).merge_dims(1, 2); // (S,d)
        let head_rms = |x: GraphTensor, weight: GraphTensor| {
            let dims = x.dims();
            let inv = ((x * x).mean(vec![2]) + 1e-6).sqrt().reciprocal(); // (H,S)
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
            let scores = q.matmul(k.permute(vec![0, 2, 1])) * scale; // (H,S,S)
            scores.softmax(vec![2]).matmul(v) // (H,S,hd)
        };
        let swiglu =
            |u: GraphTensor| u.slice_along(0..mlp, 1).silu() * u.slice_along(mlp..2 * mlp, 1);

        // ---- double-stream block (txt first in every concat/split) ----
        let (shift0, scale0, gate0) = triple(m_img, 0);
        let (shift1, scale1, gate1) = triple(m_img, 1);
        let (c_shift0, c_scale0, c_gate0) = triple(m_txt, 0);
        let (c_shift1, c_scale1, c_gate1) = triple(m_txt, 1);
        let mut img = luminal_nn::linear(latent, self.x_embed_weight, self.x_embed_bias); // (s_img, d)
        let mut txt = luminal_nn::linear(text, self.ctx_embed_weight, self.ctx_embed_bias); // (s_txt, d)
        let img_n = ada(
            luminal_nn::layer_norm(img, self.ln_weight, self.ln_bias, self.ln_epsilon),
            scale0,
            shift0,
        );
        let txt_n = ada(
            luminal_nn::layer_norm(txt, self.ln_weight, self.ln_bias, self.ln_epsilon),
            c_scale0,
            c_shift0,
        );
        let q_img = head_rms(
            heads(luminal_nn::linear(
                img_n,
                self.img_q_weight,
                self.img_q_bias,
            )),
            self.img_qnorm,
        );
        let k_img = head_rms(
            heads(luminal_nn::linear(
                img_n,
                self.img_k_weight,
                self.img_k_bias,
            )),
            self.img_knorm,
        );
        let q_txt = head_rms(
            heads(luminal_nn::linear(
                txt_n,
                self.txt_q_weight,
                self.txt_q_bias,
            )),
            self.txt_qnorm,
        );
        let k_txt = head_rms(
            heads(luminal_nn::linear(
                txt_n,
                self.txt_k_weight,
                self.txt_k_bias,
            )),
            self.txt_knorm,
        );
        // V's sequence concat happens FLAT, before the head split — the
        // head reshape commutes with row concat, and pads over matmul
        // outputs (compute) never form the pure-view stack the rejoin
        // divergence needs. q/k concat after head_rms (a compute) for
        // the same reason.
        let v_all = heads(
            luminal_nn::linear(txt_n, self.txt_v_weight, self.txt_v_bias).concat_along(
                luminal_nn::linear(img_n, self.img_v_weight, self.img_v_bias),
                0,
            ),
        );
        let attn = unheads(sdpa(
            rope(q_txt.concat_along(q_img, 1)),
            rope(k_txt.concat_along(k_img, 1)),
            v_all,
        )); // (s, d)
        let attn_txt = attn.slice_along(0..s_txt, 0);
        let attn_img = attn.slice_along(s_txt.., 0);
        img += gate(
            luminal_nn::linear(attn_img, self.img_out_weight, self.img_out_bias),
            gate0,
        );
        txt += gate(
            luminal_nn::linear(attn_txt, self.txt_out_weight, self.txt_out_bias),
            c_gate0,
        );
        let ff = swiglu(luminal_nn::linear(
            ada(
                luminal_nn::layer_norm(img, self.ln_weight, self.ln_bias, self.ln_epsilon),
                scale1,
                shift1,
            ),
            self.ff_in_weight,
            self.ff_in_bias,
        ));
        img += gate(
            luminal_nn::linear(ff, self.ff_out_weight, self.ff_out_bias),
            gate1,
        );
        let c_ff = swiglu(luminal_nn::linear(
            ada(
                luminal_nn::layer_norm(txt, self.ln_weight, self.ln_bias, self.ln_epsilon),
                c_scale1,
                c_shift1,
            ),
            self.ctx_ff_in_weight,
            self.ctx_ff_in_bias,
        ));
        txt += gate(
            luminal_nn::linear(c_ff, self.ctx_ff_out_weight, self.ctx_ff_out_bias),
            c_gate1,
        );

        // ---- single-stream block over [txt ‖ img] ----
        // The joint sequence assembles by SCATTER writes into a zero
        // base (the paged-attention family's own row-assembly spelling)
        // instead of concat's pad+add: the head SLICES this tensor, and
        // a slice distributing down to a pad's clamp view re-creates
        // the rejoin-divergence stack (measured: stage-8 probe). Scatter
        // is a compute write — the slice stops there.
        let graph = latent.graph();
        let txt_positions = graph.arange(s_txt);
        let img_positions = graph.iota(vec![latent.dims()[0]], move |c| c[0] + s_txt);
        let mut hidden = scatter_rows(
            img,
            img_positions,
            scatter_rows(txt, txt_positions, joint_base),
        ); // (s, d)
        let (s_shift, s_scale, s_gate) = triple(m_single, 0);
        let normed = ada(
            luminal_nn::layer_norm(hidden, self.ln_weight, self.ln_bias, self.ln_epsilon),
            s_scale,
            s_shift,
        );
        let proj = luminal_nn::linear(normed, self.single_proj_weight, self.single_proj_bias); // (s, 3d + 2·mlp)
        let q = head_rms(heads(proj.slice_along(0..d, 1)), self.single_qnorm);
        let k = head_rms(heads(proj.slice_along(d..2 * d, 1)), self.single_knorm);
        let v = heads(proj.slice_along(2 * d..3 * d, 1));
        let attn = unheads(sdpa(rope(q), rope(k), v)); // (s, d)
        let mlp_out = swiglu(proj.slice_along(3 * d..3 * d + 2 * mlp, 1)); // (s, mlp)
        // Fused out-projection over [attn ‖ mlp], spelled as the
        // row-split sum (see the single_out_* field note).
        hidden += gate(
            luminal_nn::linear(attn, self.single_out_attn_weight, self.single_out_attn_bias)
                + luminal_nn::linear(
                    mlp_out,
                    self.single_out_mlp_weight,
                    self.single_out_mlp_bias,
                ),
            s_gate,
        );

        // ---- AdaLayerNormContinuous head: (scale, shift) — REVERSED ----
        let img_final = hidden.slice_along(s_txt.., 0); // (s_img, d)
        let head = luminal_nn::linear(cond, self.norm_out_weight, self.norm_out_bias); // (1, 2d)
        let scale = head.slice_along(0..d, 1);
        let shift = head.slice_along(d..2 * d, 1);
        luminal_nn::linear(
            ada(
                luminal_nn::layer_norm(img_final, self.ln_weight, self.ln_bias, self.ln_epsilon),
                scale,
                shift,
            ),
            self.proj_out_weight,
            self.proj_out_bias,
        )
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
