//! openai/whisper-tiny.en as pure logical ops — 100% of THIS
//! checkpoint (ruling 2026-08-12): conv frontend (k3s1 + k3s2, exact
//! erf GELU), a stored sinusoidal position table on the encoder and a
//! LEARNED position table on the decoder, torch-style LayerNorm
//! (mean + std, learned weight AND bias, eps 1e-5), the HF whisper
//! projection-bias asymmetry (q/v/out biased, k unbiased), plain MHA
//! 6×64 with 1/8 on the scores, causal cached decoder self-attention
//! over the slot pool, UNCACHED cross-attention whose K/V recompute
//! from the encoder output every step (encoder + decoder are ONE
//! graph — faithful to the parked example), exact-erf GELU MLPs, and
//! the TIED output head. No rope, no RMSNorm, no GQA.

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal::shape::IntExpr;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, KvCachePool, causal_bias, paged_attention,
};

#[derive(Clone)]
pub struct WhisperDims {
    pub n_mels: usize,
    pub audio_ctx: usize,
    pub state: usize,
    pub heads: usize,
    pub audio_layers: usize,
    pub text_layers: usize,
    pub text_ctx: usize,
    pub vocab: usize,
    pub ff: usize,
    pub eps: f32,
}

impl WhisperDims {
    pub fn whisper_tiny_en() -> Self {
        Self {
            n_mels: 80,
            audio_ctx: 1500,
            state: 384,
            heads: 6,
            audio_layers: 4,
            text_layers: 4,
            text_ctx: 448,
            vocab: 51_864,
            ff: 1536,
            eps: 1e-5,
        }
    }

    pub fn tiny() -> Self {
        Self {
            n_mels: 4,
            audio_ctx: 6,
            state: 8,
            heads: 2,
            audio_layers: 2,
            text_layers: 2,
            text_ctx: 8,
            vocab: 19,
            ff: 12,
            eps: 1e-5,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.state / self.heads
    }

    /// Mel frames before the stride-2 conv halves them.
    pub fn mel_frames(&self) -> usize {
        self.audio_ctx * 2
    }
}

fn conv1d_bias(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> GraphTensor {
    let padded = x.pad(
        vec![
            (IntExpr::from(0), IntExpr::from(0)),
            (IntExpr::from(padding), IntExpr::from(padding)),
        ],
        0.0,
    );
    let unfolded = padded
        .unfold([1usize, kernel], [1usize, stride], [1usize, 1usize])
        .squeeze(2);
    let flat = unfolded.permute(vec![1, 0, 2]).merge_dims(1, 2);
    let out = flat.matmul(weight.permute(vec![1, 0]));
    let n_windows = out.dims()[0];
    (out + bias.expand_dim(0, n_windows)).transpose(0, 1)
}

/// The HF whisper bias asymmetry: q/v/out biased, k unbiased.
pub struct WhisperAttn {
    pub q_weight: GraphTensor,
    pub q_bias: Option<GraphTensor>,
    pub k_weight: GraphTensor,
    pub k_bias: Option<GraphTensor>,
    pub v_weight: GraphTensor,
    pub v_bias: Option<GraphTensor>,
    pub out_weight: GraphTensor,
    pub out_bias: Option<GraphTensor>,
    heads: usize,
    head_dim: usize,
}

impl WhisperAttn {
    fn new(ns: &Namespace, d: &WhisperDims, cx: &mut Graph) -> Self {
        Self {
            q_weight: cx.named_tensor(
                ns.child("q_proj").leaf("weight"),
                vec![d.state, d.state],
                DType::F32,
            ),
            q_bias: Some(cx.named_tensor(
                ns.child("q_proj").leaf("bias"),
                vec![d.state],
                DType::F32,
            )),
            k_weight: cx.named_tensor(
                ns.child("k_proj").leaf("weight"),
                vec![d.state, d.state],
                DType::F32,
            ),
            k_bias: None,
            v_weight: cx.named_tensor(
                ns.child("v_proj").leaf("weight"),
                vec![d.state, d.state],
                DType::F32,
            ),
            v_bias: Some(cx.named_tensor(
                ns.child("v_proj").leaf("bias"),
                vec![d.state],
                DType::F32,
            )),
            out_weight: cx.named_tensor(
                ns.child("out_proj").leaf("weight"),
                vec![d.state, d.state],
                DType::F32,
            ),
            out_bias: Some(cx.named_tensor(
                ns.child("out_proj").leaf("bias"),
                vec![d.state],
                DType::F32,
            )),
            heads: d.heads,
            head_dim: d.head_dim(),
        }
    }

    /// Bidirectional maskless attention (encoder self / cross): q from
    /// `x`, k/v from `kv_source`, scale 1/sqrt(head_dim) on scores.
    fn bidirectional(&self, x: GraphTensor, kv_source: GraphTensor) -> GraphTensor {
        let attn = luminal_nn::attention(
            luminal_nn::linear(x, self.q_weight, self.q_bias),
            luminal_nn::linear(kv_source, self.k_weight, self.k_bias),
            luminal_nn::linear(kv_source, self.v_weight, self.v_bias),
            self.heads,
            self.head_dim,
        );
        luminal_nn::linear(attn, self.out_weight, self.out_bias)
    }

    /// Causal cached decoder self-attention over the slot pool.
    #[allow(clippy::too_many_arguments)]
    fn causal_cached(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let query = luminal_nn::linear(x, self.q_weight, self.q_bias);
        let context_positions = query.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            query,
            luminal_nn::linear(x, self.k_weight, self.k_bias),
            luminal_nn::linear(x, self.v_weight, self.v_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, context_positions),
            AttentionGeometry::new(self.heads, self.heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        (
            luminal_nn::linear(attn, self.out_weight, self.out_bias),
            k_cache,
            v_cache,
        )
    }
}

fn layer_norm(
    d: &WhisperDims,
    ns: &Namespace,
    cx: &mut Graph,
) -> (Option<GraphTensor>, Option<GraphTensor>, f32) {
    // torch LayerNorm: mean-centered, learned weight AND bias.
    (
        Some(cx.named_tensor((ns).leaf("weight"), vec![d.state], DType::F32)),
        Some(cx.named_tensor((ns).leaf("bias"), vec![d.state], DType::F32)),
        d.eps,
    )
}

pub struct EncoderLayer {
    pub attn_norm_weight: Option<GraphTensor>,
    pub attn_norm_bias: Option<GraphTensor>,
    pub attn_norm_epsilon: f32,
    pub attn: WhisperAttn,
    pub ff_norm_weight: Option<GraphTensor>,
    pub ff_norm_bias: Option<GraphTensor>,
    pub ff_norm_epsilon: f32,
    pub fc1_weight: GraphTensor,
    pub fc1_bias: Option<GraphTensor>,
    pub fc2_weight: GraphTensor,
    pub fc2_bias: Option<GraphTensor>,
}

pub struct DecoderLayer {
    pub self_norm_weight: Option<GraphTensor>,
    pub self_norm_bias: Option<GraphTensor>,
    pub self_norm_epsilon: f32,
    pub self_attn: WhisperAttn,
    pub cross_norm_weight: Option<GraphTensor>,
    pub cross_norm_bias: Option<GraphTensor>,
    pub cross_norm_epsilon: f32,
    pub cross_attn: WhisperAttn,
    pub ff_norm_weight: Option<GraphTensor>,
    pub ff_norm_bias: Option<GraphTensor>,
    pub ff_norm_epsilon: f32,
    pub fc1_weight: GraphTensor,
    pub fc1_bias: Option<GraphTensor>,
    pub fc2_weight: GraphTensor,
    pub fc2_bias: Option<GraphTensor>,
}

pub struct Whisper {
    pub dims: WhisperDims,
    pub conv1_w: GraphTensor,
    pub conv1_b: GraphTensor,
    pub conv2_w: GraphTensor,
    pub conv2_b: GraphTensor,
    pub enc_pos: GraphTensor,
    pub enc_layers: Vec<EncoderLayer>,
    pub enc_final_norm_weight: Option<GraphTensor>,
    pub enc_final_norm_bias: Option<GraphTensor>,
    pub enc_final_norm_epsilon: f32,
    pub embed: GraphTensor,
    pub dec_pos: GraphTensor,
    pub dec_layers: Vec<DecoderLayer>,
    pub dec_final_norm_weight: Option<GraphTensor>,
    pub dec_final_norm_bias: Option<GraphTensor>,
    pub dec_final_norm_epsilon: f32,
}

impl Whisper {
    pub fn init(cx: &mut Graph, d: &WhisperDims) -> Self {
        let enc = Namespace::root().child("model").child("encoder");
        let dec = Namespace::root().child("model").child("decoder");
        let enc_layers = (0..d.audio_layers)
            .map(|l| {
                let ns = enc.child("layers").index(l);
                {
                    let (attn_norm_weight, attn_norm_bias, attn_norm_epsilon) =
                        layer_norm(d, &ns.child("self_attn_layer_norm"), cx);
                    let (ff_norm_weight, ff_norm_bias, ff_norm_epsilon) =
                        layer_norm(d, &ns.child("final_layer_norm"), cx);
                    EncoderLayer {
                        attn_norm_weight,
                        attn_norm_bias,
                        attn_norm_epsilon,
                        attn: WhisperAttn::new(&ns.child("self_attn"), d, cx),
                        ff_norm_weight,
                        ff_norm_bias,
                        ff_norm_epsilon,
                        fc1_weight: cx.named_tensor(
                            ns.child("fc1").leaf("weight"),
                            vec![d.state, d.ff],
                            DType::F32,
                        ),
                        fc1_bias: Some(cx.named_tensor(
                            ns.child("fc1").leaf("bias"),
                            vec![d.ff],
                            DType::F32,
                        )),
                        fc2_weight: cx.named_tensor(
                            ns.child("fc2").leaf("weight"),
                            vec![d.ff, d.state],
                            DType::F32,
                        ),
                        fc2_bias: Some(cx.named_tensor(
                            ns.child("fc2").leaf("bias"),
                            vec![d.state],
                            DType::F32,
                        )),
                    }
                }
            })
            .collect();
        let dec_layers = (0..d.text_layers)
            .map(|l| {
                let ns = dec.child("layers").index(l);
                {
                    let (self_norm_weight, self_norm_bias, self_norm_epsilon) =
                        layer_norm(d, &ns.child("self_attn_layer_norm"), cx);
                    let (cross_norm_weight, cross_norm_bias, cross_norm_epsilon) =
                        layer_norm(d, &ns.child("encoder_attn_layer_norm"), cx);
                    let (ff_norm_weight, ff_norm_bias, ff_norm_epsilon) =
                        layer_norm(d, &ns.child("final_layer_norm"), cx);
                    DecoderLayer {
                        self_norm_weight,
                        self_norm_bias,
                        self_norm_epsilon,
                        self_attn: WhisperAttn::new(&ns.child("self_attn"), d, cx),
                        cross_norm_weight,
                        cross_norm_bias,
                        cross_norm_epsilon,
                        cross_attn: WhisperAttn::new(&ns.child("encoder_attn"), d, cx),
                        ff_norm_weight,
                        ff_norm_bias,
                        ff_norm_epsilon,
                        fc1_weight: cx.named_tensor(
                            ns.child("fc1").leaf("weight"),
                            vec![d.state, d.ff],
                            DType::F32,
                        ),
                        fc1_bias: Some(cx.named_tensor(
                            ns.child("fc1").leaf("bias"),
                            vec![d.ff],
                            DType::F32,
                        )),
                        fc2_weight: cx.named_tensor(
                            ns.child("fc2").leaf("weight"),
                            vec![d.ff, d.state],
                            DType::F32,
                        ),
                        fc2_bias: Some(cx.named_tensor(
                            ns.child("fc2").leaf("bias"),
                            vec![d.state],
                            DType::F32,
                        )),
                    }
                }
            })
            .collect();
        {
            let (enc_final_norm_weight, enc_final_norm_bias, enc_final_norm_epsilon) =
                layer_norm(d, &enc.child("layer_norm"), cx);
            let (dec_final_norm_weight, dec_final_norm_bias, dec_final_norm_epsilon) =
                layer_norm(d, &dec.child("layer_norm"), cx);
            Self {
                dims: d.clone(),
                conv1_w: cx
                    .named_tensor(
                        enc.child("conv1").leaf("weight"),
                        vec![d.state, d.n_mels, 3usize],
                        DType::F32,
                    )
                    .view()
                    .merge_dims(1, 2)
                    .finish(),
                conv1_b: cx.named_tensor(
                    enc.child("conv1").leaf("bias"),
                    vec![d.state],
                    DType::F32,
                ),
                conv2_w: cx
                    .named_tensor(
                        enc.child("conv2").leaf("weight"),
                        vec![d.state, d.state, 3usize],
                        DType::F32,
                    )
                    .view()
                    .merge_dims(1, 2)
                    .finish(),
                conv2_b: cx.named_tensor(
                    enc.child("conv2").leaf("bias"),
                    vec![d.state],
                    DType::F32,
                ),
                enc_pos: cx.named_tensor(
                    enc.child("embed_positions").leaf("weight"),
                    vec![d.audio_ctx, d.state],
                    DType::F32,
                ),
                enc_layers,
                enc_final_norm_weight,
                enc_final_norm_bias,
                enc_final_norm_epsilon,
                embed: cx.named_tensor(
                    dec.child("embed_tokens").leaf("weight"),
                    vec![d.vocab, d.state],
                    DType::F32,
                ),
                dec_pos: cx.named_tensor(
                    dec.child("embed_positions").leaf("weight"),
                    vec![d.text_ctx, d.state],
                    DType::F32,
                ),
                dec_layers,
                dec_final_norm_weight,
                dec_final_norm_bias,
                dec_final_norm_epsilon,
            }
        }
    }

    /// mel (n_mels, 2·audio_ctx) → encoder output (audio_ctx, state).
    pub fn encode(&self, mel: GraphTensor) -> GraphTensor {
        let x = conv1d_bias(mel, self.conv1_w, self.conv1_b, 3, 1, 1).gelu();
        let x = conv1d_bias(x, self.conv2_w, self.conv2_b, 3, 2, 1).gelu();
        let mut x = x.transpose(0, 1) + self.enc_pos;
        for layer in &self.enc_layers {
            x = x + layer.attn.bidirectional(
                luminal_nn::layer_norm(
                    x,
                    layer.attn_norm_weight,
                    layer.attn_norm_bias,
                    layer.attn_norm_epsilon,
                ),
                luminal_nn::layer_norm(
                    x,
                    layer.attn_norm_weight,
                    layer.attn_norm_bias,
                    layer.attn_norm_epsilon,
                ),
            );
            let ff_in = luminal_nn::layer_norm(
                x,
                layer.ff_norm_weight,
                layer.ff_norm_bias,
                layer.ff_norm_epsilon,
            );
            x += luminal_nn::linear(
                luminal_nn::linear(ff_in, layer.fc1_weight, layer.fc1_bias).gelu(),
                layer.fc2_weight,
                layer.fc2_bias,
            );
        }
        luminal_nn::layer_norm(
            x,
            self.enc_final_norm_weight,
            self.enc_final_norm_bias,
            self.enc_final_norm_epsilon,
        )
    }

    /// One decode step: token (1,) Int at position q_pos, over the
    /// per-layer self-attention slot pools; cross-attention K/V
    /// recompute from `xa` (the encoder output) in the same graph.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &self,
        token: GraphTensor,
        q_pos: GraphTensor,
        xa: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(token.dtype, DType::Int);
        let pos_row = luminal_nn::gather_rows(self.dec_pos, q_pos);
        let mut x = luminal_nn::embedding(token, self.embed) + pos_row;
        let mut caches_out = Vec::with_capacity(self.dec_layers.len());
        for (layer_index, layer) in self.dec_layers.iter().enumerate() {
            let (attn, k_cache, v_cache) = layer.self_attn.causal_cached(
                luminal_nn::layer_norm(
                    x,
                    layer.self_norm_weight,
                    layer.self_norm_bias,
                    layer.self_norm_epsilon,
                ),
                pool.layers[layer_index].0,
                pool.layers[layer_index].1,
                gather_idx,
                scatter_idx,
                q_pos,
            );
            x += attn;
            caches_out.push((k_cache, v_cache));
            x = x + layer.cross_attn.bidirectional(
                luminal_nn::layer_norm(
                    x,
                    layer.cross_norm_weight,
                    layer.cross_norm_bias,
                    layer.cross_norm_epsilon,
                ),
                xa,
            );
            let ff_in = luminal_nn::layer_norm(
                x,
                layer.ff_norm_weight,
                layer.ff_norm_bias,
                layer.ff_norm_epsilon,
            );
            x += luminal_nn::linear(
                luminal_nn::linear(ff_in, layer.fc1_weight, layer.fc1_bias).gelu(),
                layer.fc2_weight,
                layer.fc2_bias,
            );
        }
        let x = luminal_nn::layer_norm(
            x,
            self.dec_final_norm_weight,
            self.dec_final_norm_bias,
            self.dec_final_norm_epsilon,
        );
        (luminal_nn::embedding_projection(x, self.embed), caches_out)
    }
}
