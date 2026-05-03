//! Mistral3 text encoder (text branch only) for the Flux 2 pipeline.
//!
//! ## What we need to produce
//!
//! Flux 2's text-conditioning is `joint_attention_dim = 15360 = 3 × 5120`,
//! constructed by stacking the **post-residual hidden states** at layer
//! indices 10, 20, 30 of the Mistral 3 Small text branch:
//!
//! ```text
//! out = stack([hidden_states[10], hidden_states[20], hidden_states[30]], dim=1)
//!     # shape (B, 3, S, 5120)
//! out = out.permute(0, 2, 1, 3).reshape(B, S, 15360)
//! ```
//!
//! `hidden_states[k]` follows the HuggingFace convention: index 0 is the
//! token embeddings (pre-layer-0), index k is the post-residual output of
//! layer k-1. So `[10, 20, 30]` taps after layers 9, 19, 29 — meaning we
//! only need to run **layers 0..30** (= 30 layers), not the full 40.
//!
//! ## Architecture (`text_encoder/config.json`)
//!
//! Standard Mistral / Llama-shape decoder-only LM:
//! * 5120 hidden, 32 heads, 8 kv heads (GQA, kv_groups=4), head_dim 128
//! * 32768 intermediate (SwiGLU MLP), RMSNorm eps 1e-5
//! * RoPE theta 1e9, no sliding window
//! * vocab 131072
//! * BF16 storage, ten safetensors shards
//! * On-disk weight prefix is `language_model.model.{embed_tokens, layers.i, norm}.*`
//!   because the parent class is multimodal — there's also a `vision_tower`
//!   and a `multi_modal_projector` we ignore.
//!
//! ## Memory
//!
//! 30 layers × ≈530 M params each + 671 M embedding ≈ 16.6 B params at BF16
//! ≈ **33 GB** GPU memory for weights. Activations for 1 token-sequence of
//! 512 tokens are negligible (a few hundred MB). So this fits comfortably on
//! the 96 GB GH200, leaving 60 GB free — enough headroom to keep the VAE
//! resident alongside (336 MB) and load the transformer separately afterwards.

use luminal::{dtype::DType, graph::Graph, prelude::*, shape::Expression};

// ── Mistral 3 Small architecture constants for FLUX.2-dev ────────────────────
pub const HIDDEN: usize = 5120;
pub const NUM_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = NUM_HEADS / NUM_KV_HEADS; // 4
pub const HEAD_DIM: usize = 128;
pub const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 4096 — wait, 32*128 = 4096 != 5120
pub const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
pub const INTERMEDIATE: usize = 32768;
pub const RMS_EPS: f32 = 1e-5;
pub const ROPE_THETA: f32 = 1.0e9;
pub const VOCAB_SIZE: usize = 131072;
/// We only need layers 0..29 to capture hidden_states[30] at the post-29
/// residual. Layers 30..39 of the full Mistral 3 model are not loaded.
pub const NUM_LAYERS_USED: usize = 30;
/// Indices into `hidden_states` (HF convention, first item is the embedding)
/// we tap and concatenate to get the 15360-dim Flux 2 text features.
pub const TAP_LAYERS: [usize; 3] = [10, 20, 30];
/// Concatenated channel dimension after stacking the 3 taps.
pub const OUTPUT_DIM: usize = 3 * HIDDEN; // 15360 = joint_attention_dim

/// Storage dtype — Mistral 3 ships in BF16.
pub const WEIGHT_DTYPE: DType = DType::Bf16;

// =============================================================================
// Helpers (mirror the patterns in the existing `examples/qwen` & `gemma4_moe`)
// =============================================================================

fn linear_no_bias(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    x.matmul(w.cast(x.dtype).t())
}

fn rmsnorm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let last = x.shape.last_axis();
    let normed = x.std_norm(last, eps);
    let dims = x.dims();
    let prefix: Vec<Expression> = dims[..dims.len() - 1].to_vec();
    normed * weight.expand_lhs(prefix).cast(x.dtype)
}

/// Rotary position embedding — half-rotation convention (`[x0, x1] →
/// [x0*cos - x1*sin, x1*cos + x0*sin]` where `x0`, `x1` are the first and
/// second halves of the head dim). Matches Llama / Mistral.
///
/// Inputs:
/// * `x`: `(seq, n_heads, head_dim)`
/// * `pos_ids`: `(seq,)` Int
/// * `theta`: RoPE base
fn apply_rope(x: GraphTensor, pos_ids: GraphTensor, n_heads: usize, theta: f32) -> GraphTensor {
    let cx = x.graph();
    let seq = x.dims()[0];
    let half = HEAD_DIM / 2;

    // Frequencies: theta^(-2i/D) for i in 0..D/2 — represented as 1 / theta^(2i/D)
    let exponents = cx.arange_options(0, HEAD_DIM, 2).cast(DType::F32) / HEAD_DIM as f32;
    use luminal::prelude::F32Pow;
    let inv_freqs = theta.pow(exponents).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1)); // (seq, half)

    let cos = emb.cos().expand_dim(1, n_heads); // (seq, n_heads, half)
    let sin = emb.sin().expand_dim(1, n_heads);

    let x0 = x.slice((.., .., ..half));
    let x1 = x.slice((.., .., half..));
    let r0 = x0.cast(DType::F32) * cos - x1.cast(DType::F32) * sin;
    let r1 = x1.cast(DType::F32) * cos + x0.cast(DType::F32) * sin;
    r0.concat_along(r1, 2)
}

/// Standard scaled dot-product attention over `(n_heads, seq_q, head_dim)`,
/// `(n_heads, seq_k, head_dim)`, `(n_heads, seq_k, head_dim)` with a causal
/// mask. Returns `(seq_q, n_heads * head_dim)`.
fn causal_sdpa(q: GraphTensor, k: GraphTensor, v: GraphTensor) -> GraphTensor {
    let cx = q.graph();
    let n_heads = q.dims()[0];
    let seq = q.dims()[1];
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let scores = q.matmul(k.transpose(1, 2)) * scale;
    // Causal mask: positions where k_pos > q_pos are masked.
    let q_pos = cx.arange(seq).cast(DType::F32);
    let k_pos = cx.arange(seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_pos.expand_dim(1, seq));
    let mask = mask.cast(DType::F32).expand_dim(0, n_heads);
    let masked = scores + mask * (-1e10_f32);
    let weights = masked.softmax(2);
    let attn = weights.matmul(v); // (n_heads, seq_q, head_dim)
    attn.transpose(0, 1).merge_dims(1, 2) // (seq_q, n_heads*head_dim)
}

// =============================================================================
// One Mistral 3 layer (RMSNorm → GQA self-attn + residual → RMSNorm → SwiGLU
// MLP + residual). Identical in shape to the existing `examples/qwen`'s
// `QwenLayer`.
// =============================================================================

struct MistralLayer {
    attn_rms: GraphTensor,  // (HIDDEN,)
    q_proj: GraphTensor,    // (Q_DIM, HIDDEN) — Q dim = 32*128 = 4096
    k_proj: GraphTensor,    // (KV_DIM, HIDDEN)
    v_proj: GraphTensor,    // (KV_DIM, HIDDEN)
    o_proj: GraphTensor,    // (HIDDEN, Q_DIM)
    mlp_rms: GraphTensor,   // (HIDDEN,)
    gate_proj: GraphTensor, // (INTERMEDIATE, HIDDEN)
    up_proj: GraphTensor,   // (INTERMEDIATE, HIDDEN)
    down_proj: GraphTensor, // (HIDDEN, INTERMEDIATE)
}

impl MistralLayer {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("language_model.model.layers.{idx}");
        let mk = |name: &str, shape: (usize, usize), cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(format!("{prefix}.{name}"), shape)
                .as_dtype(WEIGHT_DTYPE)
                .persist()
        };
        let mk1 = |name: &str, n: usize, cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(format!("{prefix}.{name}"), n)
                .as_dtype(WEIGHT_DTYPE)
                .persist()
        };
        Self {
            attn_rms: mk1("input_layernorm.weight", HIDDEN, cx),
            q_proj: mk("self_attn.q_proj.weight", (Q_DIM, HIDDEN), cx),
            k_proj: mk("self_attn.k_proj.weight", (KV_DIM, HIDDEN), cx),
            v_proj: mk("self_attn.v_proj.weight", (KV_DIM, HIDDEN), cx),
            o_proj: mk("self_attn.o_proj.weight", (HIDDEN, Q_DIM), cx),
            mlp_rms: mk1("post_attention_layernorm.weight", HIDDEN, cx),
            gate_proj: mk("mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN), cx),
            up_proj: mk("mlp.up_proj.weight", (INTERMEDIATE, HIDDEN), cx),
            down_proj: mk("mlp.down_proj.weight", (HIDDEN, INTERMEDIATE), cx),
        }
    }

    fn forward(&self, x: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
        let h = rmsnorm(x, self.attn_rms, RMS_EPS);
        let q = linear_no_bias(h, self.q_proj);
        let k = linear_no_bias(h, self.k_proj);
        let v = linear_no_bias(h, self.v_proj);

        // (seq, dim) → (seq, n_heads, head_dim) → ... → (n_heads, seq, head_dim)
        let q = q.split_dims(1, HEAD_DIM); // (seq, NUM_HEADS, HEAD_DIM)
        let k = k.split_dims(1, HEAD_DIM); // (seq, NUM_KV_HEADS, HEAD_DIM)
        let v = v.split_dims(1, HEAD_DIM);

        let q = apply_rope(q, pos_ids, NUM_HEADS, ROPE_THETA);
        let k = apply_rope(k, pos_ids, NUM_KV_HEADS, ROPE_THETA);

        // GQA expand: tile k, v along the kv_groups axis to match num_heads.
        let k = k
            .transpose(0, 1) // (NUM_KV_HEADS, seq, HEAD_DIM)
            .expand_dim(1, KV_GROUPS) // (NUM_KV_HEADS, KV_GROUPS, seq, HEAD_DIM)
            .merge_dims(0, 1); // (NUM_HEADS, seq, HEAD_DIM)
        let v = v.transpose(0, 1).expand_dim(1, KV_GROUPS).merge_dims(0, 1);
        let q = q.transpose(0, 1); // (NUM_HEADS, seq, HEAD_DIM)

        let attn = causal_sdpa(q, k, v); // (seq, Q_DIM)
        let attn_out = linear_no_bias(attn, self.o_proj); // (seq, HIDDEN)
        let x = x + attn_out;

        let h = rmsnorm(x, self.mlp_rms, RMS_EPS);
        let gate = linear_no_bias(h, self.gate_proj).silu();
        let up = linear_no_bias(h, self.up_proj);
        let mlp = linear_no_bias(gate * up, self.down_proj);
        x + mlp
    }
}

// =============================================================================
// Top-level text encoder
// =============================================================================

pub struct Mistral3TextEncoder {
    pub embed_tokens: GraphTensor, // (VOCAB_SIZE, HIDDEN) — used as a gather table
    layers: Vec<MistralLayer>,
}

impl Mistral3TextEncoder {
    pub fn init(cx: &mut Graph) -> Self {
        let embed_tokens = cx
            .named_tensor(
                "language_model.model.embed_tokens.weight",
                (VOCAB_SIZE, HIDDEN),
            )
            .as_dtype(WEIGHT_DTYPE)
            .persist();
        let layers = (0..NUM_LAYERS_USED)
            .map(|i| MistralLayer::new(i, cx))
            .collect();
        Self {
            embed_tokens,
            layers,
        }
    }

    /// Run the prompt through the (truncated) text encoder and return the
    /// **stacked-and-flattened** `(seq, OUTPUT_DIM=15360)` text features the
    /// Flux 2 transformer's `context_embedder` consumes.
    ///
    /// Steps mirror diffusers' `_get_mistral_3_small_prompt_embeds`:
    ///   1. Gather `embed_tokens[input_ids]` → `(seq, HIDDEN)`.
    ///   2. Run layers; capture `hidden_states[10/20/30]` (in HF convention,
    ///      = post-residual at layers 9, 19, 29).
    ///   3. Stack along a new "tap" axis: `(seq, 3, HIDDEN)`.
    ///   4. Flatten the tap axis into the channel axis: `(seq, 3*HIDDEN)`.
    pub fn forward(&self, input_ids: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
        let seq = input_ids.dims1();
        // Token embedding lookup via gather. Mirror the qwen / llama pattern:
        // build a flat index table (id * HIDDEN + col) that picks the right
        // row from the embed_tokens (VOCAB_SIZE × HIDDEN) buffer. The source
        // is BF16 so the gathered slice is BF16 too — cast to F32 immediately
        // so the rest of the network runs in F32 with BF16 weights upcast at
        // each matmul (see `linear_no_bias`).
        let mut x = self.embed_tokens.gather(
            (input_ids * HIDDEN).expand_dim(1, HIDDEN)
                + input_ids.graph().arange(HIDDEN).expand_dim(0, seq),
        );
        x = x.cast(DType::F32);

        // Run layers, taking snapshots at the right HF-convention layer indices.
        // hidden_states[10] = post-residual after layer 9, so we capture AFTER
        // running layer 9. Same for 19 and 29.
        let mut taps: Vec<GraphTensor> = Vec::with_capacity(TAP_LAYERS.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, pos_ids);
            // Map: TAP_LAYERS = [10, 20, 30] meaning "post-layer 9/19/29".
            if TAP_LAYERS.iter().any(|&k| idx + 1 == k) {
                taps.push(x);
            }
        }

        // Stack as (seq, n_taps, HIDDEN) then flatten last two dims.
        let mut stacked = taps[0].expand_dim(1, 1_usize); // (seq, 1, HIDDEN)
        for t in &taps[1..] {
            stacked = stacked.concat_along(t.expand_dim(1, 1_usize), 1);
        }
        // (seq, 3, HIDDEN) → (seq, 3*HIDDEN)
        stacked.merge_dims(1, 2)
    }
}

// =============================================================================
// Chat-template formatting (text-only path) — produces the byte string that
// then gets fed to a tokenizer. Matches the Mistral 3 chat template applied
// by diffusers' `_get_mistral_3_small_prompt_embeds`.
// =============================================================================

/// The system message Flux 2's pipeline uses by default for txt2img.
/// Verbatim from `diffusers.pipelines.flux2.system_messages.SYSTEM_MESSAGE`.
pub const SYSTEM_MESSAGE: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

/// Format `(system, user)` into the wire-format string the tokenizer expects
/// after `apply_chat_template(..., add_generation_prompt=False)`. The
/// template inserts `<s>` (BOS) on its own, so we don't add it here — the
/// tokenizer will emit it via `add_bos_token = true`.
pub fn format_chat(system_message: &str, user_prompt: &str) -> String {
    // The Mistral 3 jinja template renders to:
    //   <bos>[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT][INST]{user}[/INST]
    // The bracketed tags are individual added-tokens in tokenizer.json, so
    // they'll round-trip through the tokenizer as single ids.
    format!("[SYSTEM_PROMPT]{system_message}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_matches_jinja_output() {
        // Sanity check: the result is the deterministic concatenation we
        // expect for a text-only prompt.
        let s = format_chat("hello world", "make a cat");
        assert_eq!(
            s,
            "[SYSTEM_PROMPT]hello world[/SYSTEM_PROMPT][INST]make a cat[/INST]"
        );
    }

    #[test]
    fn architecture_constants_consistent() {
        assert_eq!(NUM_HEADS * HEAD_DIM, Q_DIM);
        assert_eq!(NUM_KV_HEADS * HEAD_DIM, KV_DIM);
        assert!(NUM_HEADS.is_multiple_of(NUM_KV_HEADS));
        assert_eq!(KV_GROUPS, NUM_HEADS / NUM_KV_HEADS);
        assert_eq!(OUTPUT_DIM, TAP_LAYERS.len() * HIDDEN);
        // hidden_states[30] requires running 30 layers (0..29 inclusive).
        assert_eq!(NUM_LAYERS_USED, *TAP_LAYERS.iter().max().unwrap());
    }
}
