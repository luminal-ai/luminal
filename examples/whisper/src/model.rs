use luminal::{dtype::DType, graph::Graph, prelude::GraphTensor, shape::Expression};
use luminal_nn::LayerNorm;

// whisper-tiny.en hyperparameters
pub const N_MELS: usize = 80;
pub const N_AUDIO_CTX: usize = 1500;
pub const N_AUDIO_STATE: usize = 384;
pub const N_AUDIO_HEAD: usize = 6;
pub const N_AUDIO_LAYER: usize = 4;

pub const N_TEXT_CTX: usize = 448;
pub const N_TEXT_STATE: usize = 384;
pub const N_TEXT_HEAD: usize = 6;
pub const N_TEXT_LAYER: usize = 4;

pub const HEAD_DIM: usize = N_AUDIO_STATE / N_AUDIO_HEAD; // 64
pub const FF_DIM: usize = 4 * N_AUDIO_STATE; // 1536

pub const N_VOCAB: usize = 51864;
pub const LAYER_NORM_EPS: f32 = 1e-5;

pub const TOKEN_SOT: u32 = 50257; // <|startoftranscript|>
pub const TOKEN_NO_TIMESTAMPS: u32 = 50362;
pub const TOKEN_EOT: u32 = 50256;

fn linear_with_bias(x: GraphTensor, w: GraphTensor, b: GraphTensor) -> GraphTensor {
    let out = x.matmul(w.t());
    let prefix: Vec<Expression> = out.dims()[..out.dims().len() - 1].to_vec();
    out + b.expand_lhs(prefix)
}

fn linear_no_bias(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    x.matmul(w.t())
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

/// 1D convolution with bias. Input: (ch_in, length). Weight: (ch_out, ch_in*kernel)
/// (HF stores it as (ch_out, ch_in, kernel) which flat-loads identically). Output: (ch_out, out_length).
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
            (Expression::from(0), Expression::from(0)),
            (Expression::from(padding), Expression::from(padding)),
        ],
        0.0,
    );
    let unfolded = padded.unfold([1usize, kernel], [1usize, stride], [1usize, 1usize]);
    // unfolded: (ch_in, n_windows, 1, kernel)
    let unfolded = unfolded.squeeze(2);
    // (ch_in, n_windows, kernel) -> (n_windows, ch_in, kernel) -> (n_windows, ch_in*kernel)
    let permuted = unfolded.permute((1, 0, 2));
    let flat = permuted.merge_dims(1, 2);
    // (n_windows, ch_in*kernel) @ (ch_in*kernel, ch_out) -> (n_windows, ch_out)
    let out = flat.matmul(weight.t());
    let n_windows = out.dims()[0];
    let bias_expanded = bias.expand_dim(0, n_windows);
    let out = out + bias_expanded;
    // (n_windows, ch_out) -> (ch_out, n_windows)
    out.transpose(0, 1)
}

/// Standard LayerNorm with mean-norm, std-norm, weight and bias (matches torch.nn.LayerNorm).
fn standard_layernorm(name: &str, dim: usize, cx: &mut Graph) -> LayerNorm {
    LayerNorm::new(
        dim,
        Some(&format!("{name}.weight")),
        Some(&format!("{name}.bias")),
        true,
        LAYER_NORM_EPS,
        cx,
    )
}

struct AttentionWeights {
    q_proj: GraphTensor,
    q_bias: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    v_bias: GraphTensor,
    out_proj: GraphTensor,
    out_bias: GraphTensor,
}

impl AttentionWeights {
    fn new(prefix: &str, dim: usize, cx: &mut Graph) -> Self {
        Self {
            q_proj: persist(cx, format!("{prefix}.q_proj.weight"), (dim, dim)),
            q_bias: persist(cx, format!("{prefix}.q_proj.bias"), dim),
            k_proj: persist(cx, format!("{prefix}.k_proj.weight"), (dim, dim)),
            v_proj: persist(cx, format!("{prefix}.v_proj.weight"), (dim, dim)),
            v_bias: persist(cx, format!("{prefix}.v_proj.bias"), dim),
            out_proj: persist(cx, format!("{prefix}.out_proj.weight"), (dim, dim)),
            out_bias: persist(cx, format!("{prefix}.out_proj.bias"), dim),
        }
    }
}

fn split_heads(x: GraphTensor) -> GraphTensor {
    // (seq, dim) -> (n_heads, seq, head_dim)
    x.split_dims(1, HEAD_DIM).transpose(0, 1)
}

fn merge_heads(x: GraphTensor) -> GraphTensor {
    // (n_heads, seq, head_dim) -> (seq, n_heads, head_dim) -> (seq, dim)
    x.transpose(0, 1).merge_dims(1, 2)
}

fn embedding_lookup(embedding: GraphTensor, ids: GraphTensor) -> GraphTensor {
    let seq = ids.dims1();
    embedding.gather(
        (ids * N_TEXT_STATE).expand_dim(1, N_TEXT_STATE)
            + ids.graph().arange(N_TEXT_STATE).expand_dim(0, seq),
    )
}

/// Encoder self-attention (full, non-causal). Input/output shape (seq, dim).
fn encoder_self_attention(x: GraphTensor, w: &AttentionWeights) -> GraphTensor {
    let q = linear_with_bias(x, w.q_proj, w.q_bias);
    let k = linear_no_bias(x, w.k_proj);
    let v = linear_with_bias(x, w.v_proj, w.v_bias);

    let q = split_heads(q);
    let k = split_heads(k);
    let v = split_heads(v);

    let scale = (HEAD_DIM as f32).sqrt().recip();
    let scores = q.matmul(k.transpose(1, 2)) * scale;
    let weights = scores.softmax(2);
    let attn = weights.matmul(v);
    let merged = merge_heads(attn);
    linear_with_bias(merged, w.out_proj, w.out_bias)
}

/// Decoder self-attention with KV cache. Returns (out, k_cache_out, v_cache_out).
fn decoder_self_attention(
    x: GraphTensor,
    w: &AttentionWeights,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    max_seq: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = x.graph();
    let seq = x.dims()[0];
    let prev = Expression::from('p');
    let total = prev + seq;

    let q = linear_with_bias(x, w.q_proj, w.q_bias);
    let k = linear_no_bias(x, w.k_proj);
    let v = linear_with_bias(x, w.v_proj, w.v_bias);

    let k_new = split_heads(k); // (n_heads, seq, head_dim)
    let v_new = split_heads(v);

    // Build flat scatter indices to write new K/V into the cache at positions [prev..prev+seq).
    let h_offset = cx.arange(N_TEXT_HEAD) * (max_seq * HEAD_DIM);
    let p_offset = (cx.arange(seq) + prev) * HEAD_DIM;
    let d_offset = cx.arange(HEAD_DIM);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, HEAD_DIM)
        + p_offset.expand_dim(0, N_TEXT_HEAD).expand_dim(2, HEAD_DIM)
        + d_offset.expand_dim(0, N_TEXT_HEAD).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    let mut k_full = k_cache_out.slice((.., ..total, ..));
    let mut v_full = v_cache_out.slice((.., ..total, ..));
    // LUM-545: model invariant `prev + seq <= max_seq`, but the frontend
    // cannot yet propagate expression-bound assertions, so `slice` reports
    // `min(max_seq, p+s)`. Normalize the visible cache axis to `total`.
    k_full.shape.dims[1] = total;
    v_full.shape.dims[1] = total;

    let q = split_heads(q);

    let scale = (HEAD_DIM as f32).sqrt().recip();
    let scores = q.matmul(k_full.transpose(1, 2)) * scale;

    // Causal mask
    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total));
    let mask_3d = mask.cast(DType::F32).expand_dim(0, N_TEXT_HEAD);
    let masked = scores + mask_3d * (-1e10f32);

    let weights = masked.softmax(2);
    let attn = weights.matmul(v_full);
    let merged = merge_heads(attn);
    let out = linear_with_bias(merged, w.out_proj, w.out_bias);
    (out, k_cache_out, v_cache_out)
}

/// Cross-attention: query from decoder, key/value from encoder output `xa`.
fn cross_attention(x: GraphTensor, xa: GraphTensor, w: &AttentionWeights) -> GraphTensor {
    let q = linear_with_bias(x, w.q_proj, w.q_bias);
    let k = linear_no_bias(xa, w.k_proj);
    let v = linear_with_bias(xa, w.v_proj, w.v_bias);

    let q = split_heads(q);
    let k = split_heads(k);
    let v = split_heads(v);

    let scale = (HEAD_DIM as f32).sqrt().recip();
    let scores = q.matmul(k.transpose(1, 2)) * scale;
    let weights = scores.softmax(2);
    let attn = weights.matmul(v);
    let merged = merge_heads(attn);
    linear_with_bias(merged, w.out_proj, w.out_bias)
}

struct EncoderLayer {
    self_attn: AttentionWeights,
    self_attn_ln: LayerNorm,
    fc1: GraphTensor,
    fc1_b: GraphTensor,
    fc2: GraphTensor,
    fc2_b: GraphTensor,
    final_ln: LayerNorm,
}

impl EncoderLayer {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("model.encoder.layers.{idx}");
        Self {
            self_attn: AttentionWeights::new(&format!("{prefix}.self_attn"), N_AUDIO_STATE, cx),
            self_attn_ln: standard_layernorm(
                &format!("{prefix}.self_attn_layer_norm"),
                N_AUDIO_STATE,
                cx,
            ),
            fc1: persist(cx, format!("{prefix}.fc1.weight"), (FF_DIM, N_AUDIO_STATE)),
            fc1_b: persist(cx, format!("{prefix}.fc1.bias"), FF_DIM),
            fc2: persist(cx, format!("{prefix}.fc2.weight"), (N_AUDIO_STATE, FF_DIM)),
            fc2_b: persist(cx, format!("{prefix}.fc2.bias"), N_AUDIO_STATE),
            final_ln: standard_layernorm(&format!("{prefix}.final_layer_norm"), N_AUDIO_STATE, cx),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let h = self.self_attn_ln.forward(x);
        let h = encoder_self_attention(h, &self.self_attn);
        let x = x + h;

        let h = self.final_ln.forward(x);
        let h = linear_with_bias(h, self.fc1, self.fc1_b).gelu();
        let h = linear_with_bias(h, self.fc2, self.fc2_b);
        x + h
    }
}

struct DecoderLayer {
    self_attn: AttentionWeights,
    self_attn_ln: LayerNorm,
    cross_attn: AttentionWeights,
    cross_attn_ln: LayerNorm,
    fc1: GraphTensor,
    fc1_b: GraphTensor,
    fc2: GraphTensor,
    fc2_b: GraphTensor,
    final_ln: LayerNorm,
}

impl DecoderLayer {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("model.decoder.layers.{idx}");
        Self {
            self_attn: AttentionWeights::new(&format!("{prefix}.self_attn"), N_TEXT_STATE, cx),
            self_attn_ln: standard_layernorm(
                &format!("{prefix}.self_attn_layer_norm"),
                N_TEXT_STATE,
                cx,
            ),
            cross_attn: AttentionWeights::new(&format!("{prefix}.encoder_attn"), N_TEXT_STATE, cx),
            cross_attn_ln: standard_layernorm(
                &format!("{prefix}.encoder_attn_layer_norm"),
                N_TEXT_STATE,
                cx,
            ),
            fc1: persist(cx, format!("{prefix}.fc1.weight"), (FF_DIM, N_TEXT_STATE)),
            fc1_b: persist(cx, format!("{prefix}.fc1.bias"), FF_DIM),
            fc2: persist(cx, format!("{prefix}.fc2.weight"), (N_TEXT_STATE, FF_DIM)),
            fc2_b: persist(cx, format!("{prefix}.fc2.bias"), N_TEXT_STATE),
            final_ln: standard_layernorm(&format!("{prefix}.final_layer_norm"), N_TEXT_STATE, cx),
        }
    }

    fn forward(
        &self,
        x: GraphTensor,
        xa: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
        max_seq: usize,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let h = self.self_attn_ln.forward(x);
        let (h, k_out, v_out) =
            decoder_self_attention(h, &self.self_attn, k_cache_in, v_cache_in, max_seq);
        let x = x + h;

        let h = self.cross_attn_ln.forward(x);
        let h = cross_attention(h, xa, &self.cross_attn);
        let x = x + h;

        let h = self.final_ln.forward(x);
        let h = linear_with_bias(h, self.fc1, self.fc1_b).gelu();
        let h = linear_with_bias(h, self.fc2, self.fc2_b);
        (x + h, k_out, v_out)
    }
}

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
    pub max_seq: usize,
}

impl KVCache {
    pub fn new(cx: &mut Graph, max_seq: usize) -> Self {
        let mut k_caches = Vec::with_capacity(N_TEXT_LAYER);
        let mut v_caches = Vec::with_capacity(N_TEXT_LAYER);
        for l in 0..N_TEXT_LAYER {
            k_caches.push(persist(
                cx,
                format!("kv_cache.{l}.k"),
                (N_TEXT_HEAD, max_seq, HEAD_DIM),
            ));
            v_caches.push(persist(
                cx,
                format!("kv_cache.{l}.v"),
                (N_TEXT_HEAD, max_seq, HEAD_DIM),
            ));
        }
        Self {
            k_caches,
            v_caches,
            max_seq,
        }
    }
}

pub struct WhisperEncoder {
    conv1_w: GraphTensor,
    conv1_b: GraphTensor,
    conv2_w: GraphTensor,
    conv2_b: GraphTensor,
    positional_embedding: GraphTensor,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
}

impl WhisperEncoder {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            conv1_w: persist(
                cx,
                "model.encoder.conv1.weight",
                (N_AUDIO_STATE, N_MELS * 3),
            ),
            conv1_b: persist(cx, "model.encoder.conv1.bias", N_AUDIO_STATE),
            conv2_w: persist(
                cx,
                "model.encoder.conv2.weight",
                (N_AUDIO_STATE, N_AUDIO_STATE * 3),
            ),
            conv2_b: persist(cx, "model.encoder.conv2.bias", N_AUDIO_STATE),
            positional_embedding: persist(
                cx,
                "model.encoder.embed_positions.weight",
                (N_AUDIO_CTX, N_AUDIO_STATE),
            ),
            layers: (0..N_AUDIO_LAYER)
                .map(|i| EncoderLayer::new(i, cx))
                .collect(),
            layer_norm: standard_layernorm("model.encoder.layer_norm", N_AUDIO_STATE, cx),
        }
    }

    /// Input mel spectrogram: (N_MELS, 3000). Output: (N_AUDIO_CTX=1500, N_AUDIO_STATE).
    pub fn forward(&self, mel: GraphTensor) -> GraphTensor {
        let h = conv1d_bias(mel, self.conv1_w, self.conv1_b, 3, 1, 1).gelu();
        let h = conv1d_bias(h, self.conv2_w, self.conv2_b, 3, 2, 1).gelu();
        // h: (N_AUDIO_STATE, N_AUDIO_CTX) -> (N_AUDIO_CTX, N_AUDIO_STATE)
        let mut x = h.transpose(0, 1) + self.positional_embedding;
        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.layer_norm.forward(x)
    }
}

pub struct WhisperDecoder {
    embed_tokens: GraphTensor,
    embed_positions: GraphTensor,
    layers: Vec<DecoderLayer>,
    layer_norm: LayerNorm,
}

impl WhisperDecoder {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            embed_tokens: persist(
                cx,
                "model.decoder.embed_tokens.weight",
                (N_VOCAB, N_TEXT_STATE),
            ),
            embed_positions: persist(
                cx,
                "model.decoder.embed_positions.weight",
                (N_TEXT_CTX, N_TEXT_STATE),
            ),
            layers: (0..N_TEXT_LAYER)
                .map(|i| DecoderLayer::new(i, cx))
                .collect(),
            layer_norm: standard_layernorm("model.decoder.layer_norm", N_TEXT_STATE, cx),
        }
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        xa: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = embedding_lookup(self.embed_tokens, token_ids);
        x += embedding_lookup(self.embed_positions, pos_ids);

        let mut cache_outputs = Vec::with_capacity(N_TEXT_LAYER);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                xa,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
                kv_cache.max_seq,
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        let x = self.layer_norm.forward(x);
        // Tied embeddings: projection to vocab
        let logits = x.matmul(self.embed_tokens.t());
        (logits, cache_outputs)
    }
}

pub struct Whisper {
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
}

impl Whisper {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            encoder: WhisperEncoder::init(cx),
            decoder: WhisperDecoder::init(cx),
        }
    }
}
