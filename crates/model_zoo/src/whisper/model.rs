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

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, KvCache, KvCachePool, LayerNorm, Linear, Namespace,
    causal_bias, paged_attention,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal::shape::IntExpr;

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
    let flat = unfolded.permute((1, 0, 2)).merge_dims(1, 2);
    let out = flat.matmul(weight.permute((1, 0)));
    let n_windows = out.dims()[0];
    (out + bias.expand_dim(0, n_windows)).transpose(0, 1)
}

/// The HF whisper bias asymmetry: q/v/out biased, k unbiased.
pub struct WhisperAttn {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    heads: usize,
    head_dim: usize,
}

impl WhisperAttn {
    fn new(ns: &Namespace, d: &WhisperDims, cx: &mut Graph) -> Self {
        Self {
            q: Linear::new(d.state, d.state, true, DType::F32, &ns.child("q_proj"), cx),
            k: Linear::new(d.state, d.state, false, DType::F32, &ns.child("k_proj"), cx),
            v: Linear::new(d.state, d.state, true, DType::F32, &ns.child("v_proj"), cx),
            out: Linear::new(
                d.state,
                d.state,
                true,
                DType::F32,
                &ns.child("out_proj"),
                cx,
            ),
            heads: d.heads,
            head_dim: d.head_dim(),
        }
    }

    /// Bidirectional maskless attention (encoder self / cross): q from
    /// `x`, k/v from `kv_source`, scale 1/sqrt(head_dim) on scores.
    fn bidirectional(&self, x: GraphTensor, kv_source: GraphTensor) -> GraphTensor {
        let attn = luminal_nn::attention(
            self.q.forward(x),
            self.k.forward(kv_source),
            self.v.forward(kv_source),
            self.heads,
            self.head_dim,
        );
        self.out.forward(attn)
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
        let query = self.q.forward(x);
        let context_positions = query.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            query,
            self.k.forward(x),
            self.v.forward(x),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, context_positions),
            AttentionGeometry::new(self.heads, self.heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        (self.out.forward(attn), k_cache, v_cache)
    }
}

fn layer_norm(d: &WhisperDims, ns: &Namespace, cx: &mut Graph) -> LayerNorm {
    // torch LayerNorm: mean-centered, learned weight AND bias.
    LayerNorm::new(d.state, true, true, true, d.eps, DType::F32, ns, cx)
}

pub struct EncoderLayer {
    pub attn_norm: LayerNorm,
    pub attn: WhisperAttn,
    pub ff_norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct DecoderLayer {
    pub self_norm: LayerNorm,
    pub self_attn: WhisperAttn,
    pub cross_norm: LayerNorm,
    pub cross_attn: WhisperAttn,
    pub ff_norm: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

pub struct Whisper {
    pub dims: WhisperDims,
    pub conv1_w: GraphTensor,
    pub conv1_b: GraphTensor,
    pub conv2_w: GraphTensor,
    pub conv2_b: GraphTensor,
    pub enc_pos: GraphTensor,
    pub enc_layers: Vec<EncoderLayer>,
    pub enc_final_norm: LayerNorm,
    pub embed: Embedding,
    pub dec_pos: GraphTensor,
    pub dec_layers: Vec<DecoderLayer>,
    pub dec_final_norm: LayerNorm,
}

impl Whisper {
    pub fn init(cx: &mut Graph, d: &WhisperDims) -> Self {
        let enc = Namespace::root().child("model").child("encoder");
        let dec = Namespace::root().child("model").child("decoder");
        let enc_layers = (0..d.audio_layers)
            .map(|l| {
                let ns = enc.child("layers").index(l);
                EncoderLayer {
                    attn_norm: layer_norm(d, &ns.child("self_attn_layer_norm"), cx),
                    attn: WhisperAttn::new(&ns.child("self_attn"), d, cx),
                    ff_norm: layer_norm(d, &ns.child("final_layer_norm"), cx),
                    fc1: Linear::new(d.state, d.ff, true, DType::F32, &ns.child("fc1"), cx),
                    fc2: Linear::new(d.ff, d.state, true, DType::F32, &ns.child("fc2"), cx),
                }
            })
            .collect();
        let dec_layers = (0..d.text_layers)
            .map(|l| {
                let ns = dec.child("layers").index(l);
                DecoderLayer {
                    self_norm: layer_norm(d, &ns.child("self_attn_layer_norm"), cx),
                    self_attn: WhisperAttn::new(&ns.child("self_attn"), d, cx),
                    cross_norm: layer_norm(d, &ns.child("encoder_attn_layer_norm"), cx),
                    cross_attn: WhisperAttn::new(&ns.child("encoder_attn"), d, cx),
                    ff_norm: layer_norm(d, &ns.child("final_layer_norm"), cx),
                    fc1: Linear::new(d.state, d.ff, true, DType::F32, &ns.child("fc1"), cx),
                    fc2: Linear::new(d.ff, d.state, true, DType::F32, &ns.child("fc2"), cx),
                }
            })
            .collect();
        Self {
            dims: d.clone(),
            conv1_w: cx
                .named_tensor(
                    enc.child("conv1").leaf("weight"),
                    (d.state, d.n_mels, 3usize),
                    DType::F32,
                )
                .view()
                .merge_dims(1, 2)
                .finish(),
            conv1_b: cx.named_tensor(enc.child("conv1").leaf("bias"), d.state, DType::F32),
            conv2_w: cx
                .named_tensor(
                    enc.child("conv2").leaf("weight"),
                    (d.state, d.state, 3usize),
                    DType::F32,
                )
                .view()
                .merge_dims(1, 2)
                .finish(),
            conv2_b: cx.named_tensor(enc.child("conv2").leaf("bias"), d.state, DType::F32),
            enc_pos: cx.named_tensor(
                enc.child("embed_positions").leaf("weight"),
                (d.audio_ctx, d.state),
                DType::F32,
            ),
            enc_layers,
            enc_final_norm: layer_norm(d, &enc.child("layer_norm"), cx),
            embed: Embedding::new(d.vocab, d.state, DType::F32, &dec.child("embed_tokens"), cx),
            dec_pos: cx.named_tensor(
                dec.child("embed_positions").leaf("weight"),
                (d.text_ctx, d.state),
                DType::F32,
            ),
            dec_layers,
            dec_final_norm: layer_norm(d, &dec.child("layer_norm"), cx),
        }
    }

    /// mel (n_mels, 2·audio_ctx) → encoder output (audio_ctx, state).
    pub fn encode(&self, mel: GraphTensor) -> GraphTensor {
        let x = conv1d_bias(mel, self.conv1_w, self.conv1_b, 3, 1, 1).gelu();
        let x = conv1d_bias(x, self.conv2_w, self.conv2_b, 3, 2, 1).gelu();
        let mut x = x.transpose(0, 1) + self.enc_pos;
        for layer in &self.enc_layers {
            x = x + layer
                .attn
                .bidirectional(layer.attn_norm.forward(x), layer.attn_norm.forward(x));
            let ff_in = layer.ff_norm.forward(x);
            x += layer.fc2.forward(layer.fc1.forward(ff_in).gelu());
        }
        self.enc_final_norm.forward(x)
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
        let mut x = self.embed.forward(token) + pos_row;
        let mut caches_out = Vec::with_capacity(self.dec_layers.len());
        for (layer_index, layer) in self.dec_layers.iter().enumerate() {
            let (attn, k_cache, v_cache) = layer.self_attn.causal_cached(
                layer.self_norm.forward(x),
                pool.layers[layer_index].0,
                pool.layers[layer_index].1,
                gather_idx,
                scatter_idx,
                q_pos,
            );
            x += attn;
            caches_out.push((k_cache, v_cache));
            x = x + layer
                .cross_attn
                .bidirectional(layer.cross_norm.forward(x), xa);
            let ff_in = layer.ff_norm.forward(x);
            x += layer.fc2.forward(layer.fc1.forward(ff_in).gelu());
        }
        let x = self.dec_final_norm.forward(x);
        (self.embed.reverse(x), caches_out)
    }
}
