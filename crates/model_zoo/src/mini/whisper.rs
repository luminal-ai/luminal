//! MiniWhisper — the whisper-family mini (relocated from luminal_nn,
//! ruling 2026-08-13: mini model definitions live in their example
//! crates; luminal_nn keeps only the building blocks).

use crate::namespace::Namespace;
use luminal::prelude::*;
use luminal_nn::attention;

/// The whisper-family mini: one encoder block (bidirectional
/// self-attention, GELU FFN) and one decoder cross-attention block that
/// attends over the encoder output — the construct nothing else covers.
pub struct MiniWhisper {
    pub enc_norm_weight: Option<GraphTensor>,
    pub enc_norm_bias: Option<GraphTensor>,
    pub enc_norm_epsilon: f32,
    pub enc_wq_weight: GraphTensor,
    pub enc_wq_bias: Option<GraphTensor>,
    pub enc_wk_weight: GraphTensor,
    pub enc_wk_bias: Option<GraphTensor>,
    pub enc_wv_weight: GraphTensor,
    pub enc_wv_bias: Option<GraphTensor>,
    pub enc_wo_weight: GraphTensor,
    pub enc_wo_bias: Option<GraphTensor>,
    pub enc_up_weight: GraphTensor,
    pub enc_up_bias: Option<GraphTensor>,
    pub enc_down_weight: GraphTensor,
    pub enc_down_bias: Option<GraphTensor>,
    pub dec_norm_weight: Option<GraphTensor>,
    pub dec_norm_bias: Option<GraphTensor>,
    pub dec_norm_epsilon: f32,
    pub dec_wq_weight: GraphTensor,
    pub dec_wq_bias: Option<GraphTensor>,
    pub dec_wk_weight: GraphTensor,
    pub dec_wk_bias: Option<GraphTensor>,
    pub dec_wv_weight: GraphTensor,
    pub dec_wv_bias: Option<GraphTensor>,
    pub dec_wo_weight: GraphTensor,
    pub dec_wo_bias: Option<GraphTensor>,
    pub dec_up_weight: GraphTensor,
    pub dec_up_bias: Option<GraphTensor>,
    pub dec_down_weight: GraphTensor,
    pub dec_down_bias: Option<GraphTensor>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MiniWhisper {
    pub fn new(d: usize, ff: usize, n_heads: usize, cx: &mut Graph) -> Self {
        let linear = |a, b, seg: &str, cx: &mut Graph| {
            (
                cx.named_tensor(
                    Namespace::root().child(seg).leaf("weight"),
                    vec![a, b],
                    DType::F32,
                ),
                None,
            )
        };
        {
            let (enc_wq_weight, enc_wq_bias) = linear(d, d, "enc_wq", cx);
            let (enc_wk_weight, enc_wk_bias) = linear(d, d, "enc_wk", cx);
            let (enc_wv_weight, enc_wv_bias) = linear(d, d, "enc_wv", cx);
            let (enc_wo_weight, enc_wo_bias) = linear(d, d, "enc_wo", cx);
            let (enc_up_weight, enc_up_bias) = linear(d, ff, "enc_up", cx);
            let (enc_down_weight, enc_down_bias) = linear(ff, d, "enc_down", cx);
            let (dec_wq_weight, dec_wq_bias) = linear(d, d, "dec_wq", cx);
            let (dec_wk_weight, dec_wk_bias) = linear(d, d, "dec_wk", cx);
            let (dec_wv_weight, dec_wv_bias) = linear(d, d, "dec_wv", cx);
            let (dec_wo_weight, dec_wo_bias) = linear(d, d, "dec_wo", cx);
            let (dec_up_weight, dec_up_bias) = linear(d, ff, "dec_up", cx);
            let (dec_down_weight, dec_down_bias) = linear(ff, d, "dec_down", cx);
            Self {
                enc_norm_weight: None,
                enc_norm_bias: None,
                enc_norm_epsilon: 1e-5,
                enc_wq_weight,
                enc_wq_bias,
                enc_wk_weight,
                enc_wk_bias,
                enc_wv_weight,
                enc_wv_bias,
                enc_wo_weight,
                enc_wo_bias,
                enc_up_weight,
                enc_up_bias,
                enc_down_weight,
                enc_down_bias,
                dec_norm_weight: None,
                dec_norm_bias: None,
                dec_norm_epsilon: 1e-5,
                dec_wq_weight,
                dec_wq_bias,
                dec_wk_weight,
                dec_wk_bias,
                dec_wv_weight,
                dec_wv_bias,
                dec_wo_weight,
                dec_wo_bias,
                dec_up_weight,
                dec_up_bias,
                dec_down_weight,
                dec_down_bias,
                n_heads,
                head_dim: d / n_heads,
            }
        }
    }

    /// audio (s_enc, d) + token activations (s_dec, d) → (s_dec, d).
    pub fn forward(&self, audio: GraphTensor, tokens: GraphTensor) -> GraphTensor {
        // Encoder block: pre-LN self-attention + GELU FFN, residuals.
        let normed = luminal_nn::layer_norm(
            audio,
            self.enc_norm_weight,
            self.enc_norm_bias,
            self.enc_norm_epsilon,
        );
        let self_attn = attention(
            luminal_nn::linear(normed, self.enc_wq_weight, self.enc_wq_bias),
            luminal_nn::linear(normed, self.enc_wk_weight, self.enc_wk_bias),
            luminal_nn::linear(normed, self.enc_wv_weight, self.enc_wv_bias),
            self.n_heads,
            self.head_dim,
        );
        let enc = audio + luminal_nn::linear(self_attn, self.enc_wo_weight, self.enc_wo_bias);
        let enc = enc
            + luminal_nn::linear(
                luminal_nn::linear(enc, self.enc_up_weight, self.enc_up_bias)
                    .gelu_fast_tanh_approximation(),
                self.enc_down_weight,
                self.enc_down_bias,
            );

        // Decoder cross-attention block: queries from tokens, keys and
        // values from the ENCODER OUTPUT.
        let normed = luminal_nn::layer_norm(
            tokens,
            self.dec_norm_weight,
            self.dec_norm_bias,
            self.dec_norm_epsilon,
        );
        let cross = attention(
            luminal_nn::linear(normed, self.dec_wq_weight, self.dec_wq_bias),
            luminal_nn::linear(enc, self.dec_wk_weight, self.dec_wk_bias),
            luminal_nn::linear(enc, self.dec_wv_weight, self.dec_wv_bias),
            self.n_heads,
            self.head_dim,
        );
        let x = tokens + luminal_nn::linear(cross, self.dec_wo_weight, self.dec_wo_bias);
        x + luminal_nn::linear(
            luminal_nn::linear(x, self.dec_up_weight, self.dec_up_bias)
                .gelu_fast_tanh_approximation(),
            self.dec_down_weight,
            self.dec_down_bias,
        )
    }
}
