//! MiniWhisper — the whisper-family mini (relocated from luminal_nn,
//! ruling 2026-08-13: mini model definitions live in their example
//! crates; luminal_nn keeps only the building blocks).

#[path = "../../../common/model_support.rs"]
mod model_support;

use crate::model_support::{attention, LayerNorm, Linear, Namespace};
use luminal::prelude::*;

/// The whisper-family mini: one encoder block (bidirectional
/// self-attention, GELU FFN) and one decoder cross-attention block that
/// attends over the encoder output — the construct nothing else covers.
pub struct MiniWhisper {
    pub enc_norm: LayerNorm,
    pub enc_wq: Linear,
    pub enc_wk: Linear,
    pub enc_wv: Linear,
    pub enc_wo: Linear,
    pub enc_up: Linear,
    pub enc_down: Linear,
    pub dec_norm: LayerNorm,
    pub dec_wq: Linear,
    pub dec_wk: Linear,
    pub dec_wv: Linear,
    pub dec_wo: Linear,
    pub dec_up: Linear,
    pub dec_down: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MiniWhisper {
    pub fn new(d: usize, ff: usize, n_heads: usize, cx: &mut Graph) -> Self {
        let ns = Namespace::root();
        let linear = |a, b, seg: &str, cx: &mut Graph| {
            Linear::new(a, b, false, DType::F32, &Namespace::root().child(seg), cx)
        };
        Self {
            enc_norm: LayerNorm::new(
                d,
                false,
                false,
                true,
                1e-5,
                DType::F32,
                &ns.child("enc_norm"),
                cx,
            ),
            enc_wq: linear(d, d, "enc_wq", cx),
            enc_wk: linear(d, d, "enc_wk", cx),
            enc_wv: linear(d, d, "enc_wv", cx),
            enc_wo: linear(d, d, "enc_wo", cx),
            enc_up: linear(d, ff, "enc_up", cx),
            enc_down: linear(ff, d, "enc_down", cx),
            dec_norm: LayerNorm::new(
                d,
                false,
                false,
                true,
                1e-5,
                DType::F32,
                &ns.child("dec_norm"),
                cx,
            ),
            dec_wq: linear(d, d, "dec_wq", cx),
            dec_wk: linear(d, d, "dec_wk", cx),
            dec_wv: linear(d, d, "dec_wv", cx),
            dec_wo: linear(d, d, "dec_wo", cx),
            dec_up: linear(d, ff, "dec_up", cx),
            dec_down: linear(ff, d, "dec_down", cx),
            n_heads,
            head_dim: d / n_heads,
        }
    }

    /// audio (s_enc, d) + token activations (s_dec, d) → (s_dec, d).
    pub fn forward(&self, audio: GraphTensor, tokens: GraphTensor) -> GraphTensor {
        // Encoder block: pre-LN self-attention + GELU FFN, residuals.
        let normed = self.enc_norm.forward(audio);
        let self_attn = attention(
            self.enc_wq.forward(normed),
            self.enc_wk.forward(normed),
            self.enc_wv.forward(normed),
            self.n_heads,
            self.head_dim,
        );
        let enc = audio + self.enc_wo.forward(self_attn);
        let enc = enc
            + self
                .enc_down
                .forward(self.enc_up.forward(enc).gelu_fast_tanh_approximation());

        // Decoder cross-attention block: queries from tokens, keys and
        // values from the ENCODER OUTPUT.
        let normed = self.dec_norm.forward(tokens);
        let cross = attention(
            self.dec_wq.forward(normed),
            self.dec_wk.forward(enc),
            self.dec_wv.forward(enc),
            self.n_heads,
            self.head_dim,
        );
        let x = tokens + self.dec_wo.forward(cross);
        x + self
            .dec_down
            .forward(self.dec_up.forward(x).gelu_fast_tanh_approximation())
    }
}
