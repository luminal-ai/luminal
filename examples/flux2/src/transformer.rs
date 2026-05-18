//! Flux2Transformer2DModel — the diffusion transformer / DiT — in pure HLIR.
//!
//! Mirrors the diffusers reference (`diffusers.models.transformers.transformer_flux2`)
//! op-for-op. Architecture summary:
//!
//! ## Top-level forward (per denoising step)
//!
//! ```text
//! latent (S_img, 128) ─┐
//!                      ├─ x_embedder ──────► img (S_img, 6144)
//! text (S_txt, 15360) ─┴─ context_embedder ► txt (S_txt, 6144)
//!
//! timestep, guidance ─► time_guidance_embed ► temb (6144)
//!                       ├─ double_mod_img(temb)  ─► (4096*9 = 36864) modulation
//!                       ├─ double_mod_txt(temb)  ─► (36864) modulation
//!                       └─ single_mod(temb)      ─► (18432) modulation
//!
//! img_ids (S_img, 4), txt_ids (S_txt, 4) ─► pos_embed ─► (cos, sin) of shape (S, 128)
//!                                                       (concatenated txt then img)
//!
//! 8x DoubleStream: (img, txt) -> (img, txt)        ◄── temb_mod_{img,txt}, rope
//! concat: hidden = [txt, img]      (length S_txt + S_img)
//! 48x SingleStream: hidden -> hidden               ◄── temb_mod, rope
//! drop txt prefix: hidden = hidden[S_txt:]
//!
//! norm_out(hidden, temb) ─► proj_out ─► (S_img, 128)
//! ```
//!
//! ## Per-block (DoubleStream)
//!
//! ```text
//! mod_img split → (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp)
//! mod_txt split → (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp)
//!
//! img' = LN(img) * (1+scale_msa) + shift_msa
//! txt' = LN(txt) * (1+c_scale_msa) + c_shift_msa
//! q_img, k_img, v_img = to_q(img'), to_k(img'), to_v(img')
//! q_txt, k_txt, v_txt = add_q_proj(txt'), add_k_proj(txt'), add_v_proj(txt')
//! q,k = RMSNorm_{qk}(reshape to (heads, head_dim))
//! q = [norm_added_q(q_txt) ; norm_q(q_img)]   along sequence axis
//! k = [norm_added_k(k_txt) ; norm_k(k_img)]
//! v = [v_txt ; v_img]
//! q,k = apply_rotary(q,k, rope)
//! attn = scaled_dot_product(q, k, v)            // standard
//! attn = flatten(heads, head_dim)
//! attn_txt, attn_img = split(attn, [S_txt, S_img])
//! img += gate_msa * to_out.0(attn_img)
//! img += gate_mlp * FF(LN(img) * (1+scale_mlp) + shift_mlp)
//! txt += c_gate_msa * to_add_out(attn_txt)
//! txt += c_gate_mlp * FF_context(LN(txt) * (1+c_scale_mlp) + c_shift_mlp)
//! ```
//!
//! ## Per-block (SingleStream — parallel attention + MLP)
//!
//! ```text
//! mod split → (shift, scale, gate)
//! h = LN(hidden) * (1+scale) + shift
//! qkv_mlp = to_qkv_mlp_proj(h)   // → 3*6144 + 2*mlp_hidden=2*18432
//! qkv, mlp_in = split([3*6144, 2*18432])
//! q,k,v = chunk(qkv, 3)
//! q,k = RMSNorm + RoPE
//! attn = sdpa(q,k,v); attn = flatten heads
//! mlp = SwiGLU(mlp_in)            // mlp_in has 2*mlp_hidden, halved
//! out = to_out([attn; mlp])
//! hidden += gate * out
//! ```
//!
//! ## Status
//!
//! - **Architecture: complete.** Every weight in `flux2-dev`'s 7 BF16 shards
//!   has a place in this graph (see [`Flux2Transformer::init`]).
//! - **Numerical validation: not yet done.** The transformer hasn't been run
//!   end-to-end against the diffusers reference — that requires downloading
//!   60+ GB of weights and is the next step.
//! - **Test coverage:** the FFN, modulation split, and 4D RoPE construction
//!   are unit-tested against a Rust scalar reference in the test module at
//!   the bottom of this file.

use luminal::{dtype::DType, graph::Graph, prelude::*};

// ── architecture constants for `black-forest-labs/FLUX.2-dev` ───────────────
//
// `FLUX2_NUM_LAYERS` / `FLUX2_NUM_SINGLE_LAYERS` env vars override the
// counts at runtime. Reducing them is useful for end-to-end pipeline
// validation with a much smaller compile-time cost — at the full
// 8 + 48 layer count the egglog egraph for the transformer can blow
// past 200 GB of CPU RAM.
pub fn num_layers() -> usize {
    std::env::var("FLUX2_NUM_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}
pub fn num_single_layers() -> usize {
    std::env::var("FLUX2_NUM_SINGLE_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48)
}
pub const NUM_HEADS: usize = 48;
pub const HEAD_DIM: usize = 128;
pub const HIDDEN: usize = NUM_HEADS * HEAD_DIM; // 6144
pub const MLP_HIDDEN: usize = 18432;
pub const JOINT_ATTENTION_DIM: usize = 15360;
pub const TIMESTEP_GUIDANCE_CHANNELS: usize = 256;
pub const IN_CHANNELS: usize = 128;
pub const PATCH_SIZE: usize = 1;
pub const RMS_EPS: f32 = 1e-6;
pub const RMS_NORM_HEAD_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 2000.0;
pub const ROPE_AXES: [usize; 4] = [32, 32, 32, 32];

/// Storage dtype for transformer weights. The Flux 2 checkpoint ships in
/// BF16; we keep it that way and cast to F32 only at the points where the
/// numerics matter (matmul accumulation, normalization).
pub const WEIGHT_DTYPE: DType = DType::Bf16;

// =============================================================================
// Small helpers
// =============================================================================

fn linear_no_bias(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    // For 2D inputs we go through `kernel::linear_no_bias_bf16_w`, which
    // is a direct mixed-precision SGEMM (F32 A × BF16 B^T → F32) that
    // converts BF16 → F32 on each load instead of materializing a
    // separate F32 cast tensor. Two reasons we don't use the egglog
    // matmul lowering for these:
    //   1. The cublaslt 2D rule fails to fire reliably for some matmul
    //      shapes (see kernel::matmul2d's docs); even one bad genome
    //      pick on the broadcast Mul + SumReduce fallback creates an
    //      `(M, N, K)` intermediate that OOMs the GPU.
    //   2. Explicitly casting all BF16 weights to F32 first would more
    //      than double the transformer's working set (~120 GB) and
    //      wouldn't fit. The kernel keeps weights as BF16 in memory.
    //
    // Higher-rank cases (3D batched matmul inside attention) fall
    // through to the standard matmul lowering — those go through the
    // separate `matmul_3d` / `matmul_3d_t` helpers in `sdpa` below.
    if x.shape.len() == 2 && w.shape.len() == 2 {
        luminal_cuda_lite::kernel::linear_no_bias_bf16_w(x, w)
    } else {
        x.matmul(w.cast(x.dtype).t())
    }
}

/// Pre-norm RMSNorm over the trailing axis with weight (`scale`); no shift.
fn rmsnorm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let w = if weight.dtype == DType::F32 {
        weight
    } else {
        weight.cast(DType::F32)
    };
    let x_rank = x.dims().len();
    let w_rank = w.dims().len();
    x.std_norm(x_rank - 1, eps) * w.expand_lhs(&x.dims()[..x_rank - w_rank])
}

/// LayerNorm with no affine parameters (mean-norm + std-norm only).
/// Matches `nn.LayerNorm(dim, elementwise_affine=False)` in PyTorch.
fn layernorm_noaffine(x: GraphTensor, eps: f32) -> GraphTensor {
    let last = x.shape.last_axis();
    x.layer_norm(last, eps)
}

/// Apply rotary embedding. `x` is `(S, H, D)` and `(cos, sin)` are `(S, D)`.
///
/// Diffusers Flux 2 uses `repeat_interleave_real=True`. The rotation pairs
/// adjacent dims: `[x0, x1, x2, x3, ...]` → rotated `[-x1, x0, -x3, x2, ...]`.
/// This matches `apply_rotary_emb(use_real_unbind_dim=-1)` with
/// `freqs_repeat_interleave_real=True`.
fn apply_rope(x: GraphTensor, cos: GraphTensor, sin: GraphTensor) -> GraphTensor {
    // x: (S, H, D); cos/sin: (S, D) -> explicitly broadcast to (S, H, D).
    let (_s, h, d_expr) = x.dims3();
    let d = d_expr.to_usize().expect("head_dim must be static");
    assert!(d % 2 == 0, "RoPE head_dim must be even");

    let pairs = x.split_dims(2, 2_usize);
    let x_a = pairs.slice((.., .., .., ..1)).squeeze(3);
    let x_b = pairs.slice((.., .., .., 1..)).squeeze(3);

    let neg_b = x_b * (-1.0_f32);
    let rotated_pairs = neg_b
        .expand_dim(3, 1_usize)
        .concat_along(x_a.expand_dim(3, 1_usize), 3);
    let x_rot = rotated_pairs.merge_dims(2, 3);

    let cos_b = cos.expand_dim(1, h);
    let sin_b = sin.expand_dim(1, h);
    x.cast(DType::F32) * cos_b.cast(DType::F32) + x_rot.cast(DType::F32) * sin_b.cast(DType::F32)
}

/// Scaled dot-product attention with NO mask, no causal: standard SDPA.
/// q, k, v: `(H, S, D)`. Returns `(S, H, D)`.
///
/// Routes through the direct batched matmul kernels for the same reason
/// the text encoder does — see `text_encoder::causal_sdpa` for context.
fn sdpa(q: GraphTensor, k: GraphTensor, v: GraphTensor) -> GraphTensor {
    let head_dim = q.dims()[2].to_usize().expect("head_dim must be static");
    let scale = (head_dim as f32).sqrt().recip();
    // The kernel needs contiguous batches; materialize the strided views
    // produced upstream (transpose / split_dims chains).
    let q = q * 1.0_f32;
    let k = k * 1.0_f32;
    let v = v * 1.0_f32;
    let scores = luminal_cuda_lite::kernel::matmul_3d_t(q, k) * scale; // (H, S, S)
    let attn_w = scores.softmax(2);
    let attn = luminal_cuda_lite::kernel::matmul_3d(attn_w, v); // (H, S, D)
    attn.transpose(0, 1) // (S, H, D)
}

/// SwiGLU: split `x` along last axis into `(x1, x2)`, return `silu(x1) * x2`.
/// `x` shape `(..., 2 * mlp_hidden)`. Handles 2D and 3D inputs.
fn swiglu(x: GraphTensor) -> GraphTensor {
    let dims = x.dims();
    let last = dims[dims.len() - 1].to_usize().expect("static");
    assert!(last.is_multiple_of(2));
    let half = last / 2;
    match dims.len() {
        2 => {
            let x1 = x.slice((.., ..half));
            let x2 = x.slice((.., half..));
            x1.silu() * x2
        }
        3 => {
            let x1 = x.slice((.., .., ..half));
            let x2 = x.slice((.., .., half..));
            x1.silu() * x2
        }
        n => panic!("swiglu: unsupported rank {n}"),
    }
}

// =============================================================================
// Sinusoidal timestep embedding
// =============================================================================

/// Build the sinusoidal positional embedding of `timestep` (a `(1,)` F32
/// tensor — caller sets the value at runtime via `runtime.set_data`).
/// Matches `diffusers.models.embeddings.Timesteps` with
/// `flip_sin_to_cos=True`, `downscale_freq_shift=0`, `max_period=10000`,
/// `scale=1`. Returns shape `(num_channels,)`.
///
/// Taking `timestep` as a graph tensor (not a Rust f32 constant) is what
/// lets the whole transformer forward be compiled **once** and re-executed
/// each diffusion step with a different timestep, instead of paying the
/// minutes-long search cost per step.
fn timesteps_proj(timestep: GraphTensor, num_channels: usize) -> GraphTensor {
    let cx = timestep.graph();
    let half = num_channels / 2;
    let exponents = cx.arange(half).cast(DType::F32) / half as f32;
    let log10000 = (10000.0_f32).ln();
    let freqs = (exponents * (-log10000)).exp(); // (half,)
    // Broadcast scalar timestep (shape (1,)) to (half,) by repeating along
    // the size-1 axis (stride substitution makes it a zero-stride broadcast).
    let t_broadcast = timestep.cast(DType::F32).repeat([half]);
    let arg = freqs * t_broadcast;
    // flip_sin_to_cos=True: cos first, then sin
    arg.cos().concat_along(arg.sin(), 0)
}

// =============================================================================
// Modulation
// =============================================================================

/// Modulation linear: `out = linear(silu(temb))`. Output dim = `dim * 3 * sets`.
fn modulation(temb: GraphTensor, weight: GraphTensor) -> GraphTensor {
    let act = temb.silu();
    linear_no_bias(act, weight)
}

/// Split modulation tensor (shape `(dim * 3 * sets,)`) into `sets` triples
/// of `(shift, scale, gate)`, each `(dim,)`.
fn split_modulation(
    mod_t: GraphTensor,
    sets: usize,
) -> Vec<(GraphTensor, GraphTensor, GraphTensor)> {
    let total = mod_t.dims()[0]
        .to_usize()
        .expect("mod tensor dim must be static");
    let dim = total / (3 * sets);
    let mut out = Vec::with_capacity(sets);
    for i in 0..sets {
        let base = 3 * i * dim;
        let shift = mod_t.slice((base..base + dim,));
        let scale = mod_t.slice((base + dim..base + 2 * dim,));
        let gate = mod_t.slice((base + 2 * dim..base + 3 * dim,));
        out.push((shift, scale, gate));
    }
    out
}

/// Apply (1 + scale) * x + shift, broadcasting scale/shift over the leading
/// sequence axis. `x: (S, D)`, `scale, shift: (D,)`.
fn ada_modulate(x: GraphTensor, scale: GraphTensor, shift: GraphTensor) -> GraphTensor {
    let s = x.dims()[0];
    let scale_b = scale.expand_lhs([s]);
    let shift_b = shift.expand_lhs([s]);
    x * (scale_b + 1.0_f32) + shift_b
}

// =============================================================================
// FeedForward (used by double-stream blocks)
// =============================================================================

struct FeedForward {
    linear_in: GraphTensor,  // (mlp_hidden*2, dim)
    linear_out: GraphTensor, // (dim, mlp_hidden)
}

impl FeedForward {
    fn new(prefix: &str, dim: usize, mlp_hidden: usize, cx: &mut Graph) -> Self {
        Self {
            linear_in: cx
                .named_tensor(format!("{prefix}.linear_in.weight"), (mlp_hidden * 2, dim))
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            linear_out: cx
                .named_tensor(format!("{prefix}.linear_out.weight"), (dim, mlp_hidden))
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let h = linear_no_bias(x, self.linear_in);
        let h = swiglu(h);
        linear_no_bias(h, self.linear_out)
    }
}

// =============================================================================
// Double-stream attention (img + txt joint attention)
// =============================================================================

struct DoubleStreamAttn {
    to_q: GraphTensor,
    to_k: GraphTensor,
    to_v: GraphTensor,
    add_q_proj: GraphTensor,
    add_k_proj: GraphTensor,
    add_v_proj: GraphTensor,
    norm_q: GraphTensor,       // (head_dim,)
    norm_k: GraphTensor,       // (head_dim,)
    norm_added_q: GraphTensor, // (head_dim,)
    norm_added_k: GraphTensor, // (head_dim,)
    to_out: GraphTensor,       // image-stream output projection
    to_add_out: GraphTensor,   // text-stream output projection
}

impl DoubleStreamAttn {
    fn new(prefix: &str, cx: &mut Graph) -> Self {
        let lin = |n: &str, cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(format!("{prefix}.{n}"), (HIDDEN, HIDDEN))
                .as_dtype(WEIGHT_DTYPE)
                .persist()
        };
        Self {
            to_q: lin("to_q.weight", cx),
            to_k: lin("to_k.weight", cx),
            to_v: lin("to_v.weight", cx),
            add_q_proj: lin("add_q_proj.weight", cx),
            add_k_proj: lin("add_k_proj.weight", cx),
            add_v_proj: lin("add_v_proj.weight", cx),
            norm_q: cx
                .named_tensor(format!("{prefix}.norm_q.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            norm_k: cx
                .named_tensor(format!("{prefix}.norm_k.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            norm_added_q: cx
                .named_tensor(format!("{prefix}.norm_added_q.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            norm_added_k: cx
                .named_tensor(format!("{prefix}.norm_added_k.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            to_out: lin("to_out.0.weight", cx),
            to_add_out: lin("to_add_out.weight", cx),
        }
    }

    /// Returns `(img_out, txt_out)`.
    /// img / txt: `(S_img, HIDDEN)` / `(S_txt, HIDDEN)`. RoPE: `(cos, sin)` of
    /// shape `(S_txt + S_img, HEAD_DIM)`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: GraphTensor,
        txt: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
    ) -> (GraphTensor, GraphTensor) {
        let s_img = img.dims()[0].to_usize().expect("S_img static");
        let s_txt = txt.dims()[0].to_usize().expect("S_txt static");

        // QKV projections.
        let q_img = linear_no_bias(img, self.to_q);
        let k_img = linear_no_bias(img, self.to_k);
        let v_img = linear_no_bias(img, self.to_v);
        let q_txt = linear_no_bias(txt, self.add_q_proj);
        let k_txt = linear_no_bias(txt, self.add_k_proj);
        let v_txt = linear_no_bias(txt, self.add_v_proj);

        // Reshape to (S, H, D).
        let q_img = q_img.split_dims(1, HEAD_DIM); // (S_img, HEADS, HEAD_DIM)
        let k_img = k_img.split_dims(1, HEAD_DIM);
        let v_img = v_img.split_dims(1, HEAD_DIM);
        let q_txt = q_txt.split_dims(1, HEAD_DIM);
        let k_txt = k_txt.split_dims(1, HEAD_DIM);
        let v_txt = v_txt.split_dims(1, HEAD_DIM);

        // QK norm per head.
        let q_img = rmsnorm(q_img, self.norm_q, RMS_NORM_HEAD_EPS);
        let k_img = rmsnorm(k_img, self.norm_k, RMS_NORM_HEAD_EPS);
        let q_txt = rmsnorm(q_txt, self.norm_added_q, RMS_NORM_HEAD_EPS);
        let k_txt = rmsnorm(k_txt, self.norm_added_k, RMS_NORM_HEAD_EPS);

        // Concat along sequence (txt first, then img — matches diffusers).
        let q = q_txt.concat_along(q_img, 0); // (S_txt + S_img, H, D)
        let k = k_txt.concat_along(k_img, 0);
        let v = v_txt.concat_along(v_img, 0);

        // RoPE on Q, K (V unchanged).
        let q = apply_rope(q, rope_cos, rope_sin);
        let k = apply_rope(k, rope_cos, rope_sin);

        // SDPA expects (H, S, D).
        let q = q.transpose(0, 1);
        let k = k.transpose(0, 1);
        let v = v.transpose(0, 1);

        let attn = sdpa(q, k, v); // (S_total, H, D)
        // `merge_dims(1, 2)` on (S, H, D) produces non-contiguous K
        // stride for the next matmul (the o_proj path). Without
        // `* 1.0` the cublaslt 2D rule can't match and the broadcast
        // Mul intermediate is ~36 GB BF16 at flux2 dimensions.
        let attn = attn.merge_dims(1, 2) * 1.0_f32; // (S_total, HIDDEN)

        // Split back into txt + img streams.
        let attn_txt = attn.slice((..s_txt, ..));
        let attn_img = attn.slice((s_txt..s_txt + s_img, ..));

        let img_out = linear_no_bias(attn_img, self.to_out);
        let txt_out = linear_no_bias(attn_txt, self.to_add_out);
        (img_out, txt_out)
    }
}

// =============================================================================
// Single-stream parallel attention (fused QKV + MLP-in, fused attn-out + MLP-out)
// =============================================================================

struct SingleStreamAttn {
    to_qkv_mlp_proj: GraphTensor, // (3*HIDDEN + 2*MLP_HIDDEN, HIDDEN)
    norm_q: GraphTensor,          // (HEAD_DIM,)
    norm_k: GraphTensor,
    to_out: GraphTensor, // (HIDDEN, HIDDEN + MLP_HIDDEN)
}

impl SingleStreamAttn {
    fn new(prefix: &str, cx: &mut Graph) -> Self {
        let qkv_mlp_out = 3 * HIDDEN + 2 * MLP_HIDDEN; // 18432 + 36864 = 55296
        Self {
            to_qkv_mlp_proj: cx
                .named_tensor(
                    format!("{prefix}.to_qkv_mlp_proj.weight"),
                    (qkv_mlp_out, HIDDEN),
                )
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            norm_q: cx
                .named_tensor(format!("{prefix}.norm_q.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            norm_k: cx
                .named_tensor(format!("{prefix}.norm_k.weight"), HEAD_DIM)
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
            to_out: cx
                .named_tensor(
                    format!("{prefix}.to_out.weight"),
                    (HIDDEN, HIDDEN + MLP_HIDDEN),
                )
                .as_dtype(WEIGHT_DTYPE)
                .persist(),
        }
    }

    /// `hidden`: `(S, HIDDEN)`, `rope_cos/sin`: `(S, HEAD_DIM)`.
    fn forward(
        &self,
        hidden: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
    ) -> GraphTensor {
        let projected = linear_no_bias(hidden, self.to_qkv_mlp_proj);
        let qkv_size = 3 * HIDDEN;
        let qkv = projected.slice((.., ..qkv_size));
        let mlp_in = projected.slice((.., qkv_size..));

        let q = qkv.slice((.., ..HIDDEN));
        let k = qkv.slice((.., HIDDEN..2 * HIDDEN));
        let v = qkv.slice((.., 2 * HIDDEN..));

        let q = q.split_dims(1, HEAD_DIM); // (S, H, D)
        let k = k.split_dims(1, HEAD_DIM);
        let v = v.split_dims(1, HEAD_DIM);

        let q = rmsnorm(q, self.norm_q, RMS_NORM_HEAD_EPS);
        let k = rmsnorm(k, self.norm_k, RMS_NORM_HEAD_EPS);

        let q = apply_rope(q, rope_cos, rope_sin);
        let k = apply_rope(k, rope_cos, rope_sin);

        let q = q.transpose(0, 1);
        let k = k.transpose(0, 1);
        let v = v.transpose(0, 1);
        // `merge_dims(1, 2)` on (S, H, D) produces non-contiguous K
        // stride; force materialization so cublaslt can match the
        // downstream `to_out` matmul. See dual-stream block above.
        let attn = sdpa(q, k, v).merge_dims(1, 2) * 1.0_f32; // (S, HIDDEN)

        let mlp = swiglu(mlp_in); // (S, MLP_HIDDEN)

        let combined = attn.concat_along(mlp, 1); // (S, HIDDEN + MLP_HIDDEN)
        linear_no_bias(combined, self.to_out)
    }
}

// =============================================================================
// Double-stream block
// =============================================================================

struct DoubleStreamBlock {
    attn: DoubleStreamAttn,
    ff: FeedForward,
    ff_context: FeedForward,
}

impl DoubleStreamBlock {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("transformer_blocks.{idx}");
        Self {
            attn: DoubleStreamAttn::new(&format!("{prefix}.attn"), cx),
            ff: FeedForward::new(&format!("{prefix}.ff"), HIDDEN, MLP_HIDDEN, cx),
            ff_context: FeedForward::new(&format!("{prefix}.ff_context"), HIDDEN, MLP_HIDDEN, cx),
        }
    }

    /// img/txt: `(S_*, HIDDEN)`. mod tensors `(36864,)`. Returns `(img_out, txt_out)`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        img: GraphTensor,
        txt: GraphTensor,
        mod_img: GraphTensor,
        mod_txt: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
    ) -> (GraphTensor, GraphTensor) {
        let img_mods = split_modulation(mod_img, 2);
        let txt_mods = split_modulation(mod_txt, 2);
        let (shift_msa, scale_msa, gate_msa) = img_mods[0];
        let (shift_mlp, scale_mlp, gate_mlp) = img_mods[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = txt_mods[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = txt_mods[1];

        // Pre-attn norms + adaLN modulation.
        let img_n = ada_modulate(layernorm_noaffine(img, RMS_EPS), scale_msa, shift_msa);
        let txt_n = ada_modulate(layernorm_noaffine(txt, RMS_EPS), c_scale_msa, c_shift_msa);

        let (attn_img, attn_txt) = self.attn.forward(img_n, txt_n, rope_cos, rope_sin);

        let img = img + ada_gate(attn_img, gate_msa);
        let txt = txt + ada_gate(attn_txt, c_gate_msa);

        // FF on each stream with second-set adaLN.
        let img_ff = self.ff.forward(ada_modulate(
            layernorm_noaffine(img, RMS_EPS),
            scale_mlp,
            shift_mlp,
        ));
        let img = img + ada_gate(img_ff, gate_mlp);

        let txt_ff = self.ff_context.forward(ada_modulate(
            layernorm_noaffine(txt, RMS_EPS),
            c_scale_mlp,
            c_shift_mlp,
        ));
        let txt = txt + ada_gate(txt_ff, c_gate_mlp);

        (img, txt)
    }
}

fn ada_gate(x: GraphTensor, gate: GraphTensor) -> GraphTensor {
    let s = x.dims()[0];
    x * gate.expand_lhs([s])
}

// =============================================================================
// Single-stream block
// =============================================================================

struct SingleStreamBlock {
    attn: SingleStreamAttn,
}

impl SingleStreamBlock {
    fn new(idx: usize, cx: &mut Graph) -> Self {
        let prefix = format!("single_transformer_blocks.{idx}");
        Self {
            attn: SingleStreamAttn::new(&format!("{prefix}.attn"), cx),
        }
    }

    /// hidden: `(S, HIDDEN)`. mod: `(18432,)`.
    fn forward(
        &self,
        hidden: GraphTensor,
        mod_t: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
    ) -> GraphTensor {
        let mods = split_modulation(mod_t, 1);
        let (shift, scale, gate) = mods[0];

        let h = ada_modulate(layernorm_noaffine(hidden, RMS_EPS), scale, shift);
        let attn_out = self.attn.forward(h, rope_cos, rope_sin);
        hidden + ada_gate(attn_out, gate)
    }
}

// =============================================================================
// Position-id construction + Flux2PosEmbed (RoPE freqs)
// =============================================================================

/// Build the 4D position-id tensor for the concatenated (txt, img) sequence,
/// matching `diffusers.pipelines.flux2.pipeline_flux2._prepare_text_ids` and
/// `_prepare_latent_ids` exactly.
///
/// The 4 axes are interpreted as **(time, h, w, layer/sequence)**.
///
///   * `txt_ids`: shape `(S_txt, 4)`. Row `l` is `(0, 0, 0, l)` — text tokens
///     vary only along the last axis (the "layer" / sequence index).
///   * `img_ids`: shape `(S_img, 4)` where `S_img = h_pack * w_pack`. Row at
///     `(hi, wi)` (cartesian product order) is `(0, hi, wi, 0)` — image
///     tokens vary along the spatial axes 1 and 2.
///
/// `h_pack` and `w_pack` are the **post-pack** spatial dims that the
/// transformer sees, i.e. `H/16` and `W/16` for an HxW pixel image. (The VAE
/// 8× downsample plus the channel-pack 2× spatial fold give 16× total.)
pub fn build_position_ids(s_txt: usize, h_pack: usize, w_pack: usize) -> (Vec<f32>, Vec<f32>) {
    let mut txt_ids = Vec::with_capacity(s_txt * 4);
    for l in 0..s_txt {
        txt_ids.extend_from_slice(&[0.0, 0.0, 0.0, l as f32]);
    }
    let mut img_ids = Vec::with_capacity(h_pack * w_pack * 4);
    for hi in 0..h_pack {
        for wi in 0..w_pack {
            img_ids.extend_from_slice(&[0.0, hi as f32, wi as f32, 0.0]);
        }
    }
    (txt_ids, img_ids)
}

/// Pre-compute `(cos, sin)` flat tables for the concatenated `(txt, img)`
/// position grid. Each is `S × HEAD_DIM` row-major. This mirrors
/// `Flux2PosEmbed.forward` (calls `get_1d_rotary_pos_embed` per axis with
/// `repeat_interleave_real=True`, then concatenates along the last dim).
pub fn build_rope_tables(s_txt: usize, h_pack: usize, w_pack: usize) -> (Vec<f32>, Vec<f32>) {
    let (txt_ids, img_ids) = build_position_ids(s_txt, h_pack, w_pack);
    let s_total = s_txt + h_pack * w_pack;
    let head_dim = HEAD_DIM;
    debug_assert_eq!(ROPE_AXES.iter().sum::<usize>(), head_dim);

    let mut cos_table = Vec::with_capacity(s_total * head_dim);
    let mut sin_table = Vec::with_capacity(s_total * head_dim);

    let row = |row_ids: &[f32]| -> (Vec<f32>, Vec<f32>) {
        // For each axis, generate cos/sin of length axes_dim[i] (with
        // repeat_interleave_real=True meaning each freq is repeated twice).
        let mut row_cos = Vec::with_capacity(head_dim);
        let mut row_sin = Vec::with_capacity(head_dim);
        for (i, &dim) in ROPE_AXES.iter().enumerate() {
            let pos = row_ids[i];
            let half = dim / 2;
            for j in 0..half {
                let exponent = (2 * j) as f32 / dim as f32;
                let freq = 1.0_f32 / ROPE_THETA.powf(exponent);
                let arg = pos * freq;
                let c = arg.cos();
                let s = arg.sin();
                // repeat_interleave_real: cos cos sin sin pattern
                row_cos.push(c);
                row_cos.push(c);
                row_sin.push(s);
                row_sin.push(s);
            }
        }
        (row_cos, row_sin)
    };

    for r in 0..s_txt {
        let (c, s) = row(&txt_ids[r * 4..(r + 1) * 4]);
        cos_table.extend(c);
        sin_table.extend(s);
    }
    for r in 0..h_pack * w_pack {
        let (c, s) = row(&img_ids[r * 4..(r + 1) * 4]);
        cos_table.extend(c);
        sin_table.extend(s);
    }
    (cos_table, sin_table)
}

// =============================================================================
// Top-level transformer
// =============================================================================

pub struct Flux2Transformer {
    // Embedders
    pub x_embedder: GraphTensor,       // (HIDDEN, IN_CHANNELS)
    pub context_embedder: GraphTensor, // (HIDDEN, JOINT_ATTENTION_DIM)

    // Time + guidance embedding
    pub time_t1_w: GraphTensor, // (HIDDEN, TIMESTEP_GUIDANCE_CHANNELS)
    pub time_t2_w: GraphTensor, // (HIDDEN, HIDDEN)
    pub guidance_t1_w: GraphTensor,
    pub guidance_t2_w: GraphTensor,

    // Modulation tables
    pub mod_img: GraphTensor,    // (HIDDEN*6, HIDDEN)
    pub mod_txt: GraphTensor,    // (HIDDEN*6, HIDDEN)
    pub mod_single: GraphTensor, // (HIDDEN*3, HIDDEN)

    // Output
    pub norm_out_lin: GraphTensor, // (HIDDEN*2, HIDDEN)  for AdaLayerNormContinuous
    pub proj_out: GraphTensor,     // (PATCH²*OUT_CHANNELS, HIDDEN)

    // Blocks
    transformer_blocks: Vec<DoubleStreamBlock>,
    single_transformer_blocks: Vec<SingleStreamBlock>,
}

impl Flux2Transformer {
    pub fn init(cx: &mut Graph) -> Self {
        let bf16 = WEIGHT_DTYPE;
        let mk = |name: &str, shape: (usize, usize), cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(name, shape).as_dtype(bf16).persist()
        };
        let mk1 = |name: &str, n: usize, cx: &mut Graph| -> GraphTensor {
            cx.named_tensor(name, n).as_dtype(bf16).persist()
        };

        let x_embedder = mk("x_embedder.weight", (HIDDEN, IN_CHANNELS), cx);
        let context_embedder = mk("context_embedder.weight", (HIDDEN, JOINT_ATTENTION_DIM), cx);

        let time_t1_w = mk(
            "time_guidance_embed.timestep_embedder.linear_1.weight",
            (HIDDEN, TIMESTEP_GUIDANCE_CHANNELS),
            cx,
        );
        let time_t2_w = mk(
            "time_guidance_embed.timestep_embedder.linear_2.weight",
            (HIDDEN, HIDDEN),
            cx,
        );
        let guidance_t1_w = mk(
            "time_guidance_embed.guidance_embedder.linear_1.weight",
            (HIDDEN, TIMESTEP_GUIDANCE_CHANNELS),
            cx,
        );
        let guidance_t2_w = mk(
            "time_guidance_embed.guidance_embedder.linear_2.weight",
            (HIDDEN, HIDDEN),
            cx,
        );

        let mod_img = mk(
            "double_stream_modulation_img.linear.weight",
            (HIDDEN * 6, HIDDEN),
            cx,
        );
        let mod_txt = mk(
            "double_stream_modulation_txt.linear.weight",
            (HIDDEN * 6, HIDDEN),
            cx,
        );
        let mod_single = mk(
            "single_stream_modulation.linear.weight",
            (HIDDEN * 3, HIDDEN),
            cx,
        );

        let norm_out_lin = mk("norm_out.linear.weight", (HIDDEN * 2, HIDDEN), cx);
        let proj_out = mk(
            "proj_out.weight",
            (PATCH_SIZE * PATCH_SIZE * IN_CHANNELS, HIDDEN),
            cx,
        );

        let transformer_blocks = (0..num_layers())
            .map(|i| DoubleStreamBlock::new(i, cx))
            .collect();
        let single_transformer_blocks = (0..num_single_layers())
            .map(|i| SingleStreamBlock::new(i, cx))
            .collect();

        let _ = mk1; // kept for parity if extra biases get added later
        Self {
            x_embedder,
            context_embedder,
            time_t1_w,
            time_t2_w,
            guidance_t1_w,
            guidance_t2_w,
            mod_img,
            mod_txt,
            mod_single,
            norm_out_lin,
            proj_out,
            transformer_blocks,
            single_transformer_blocks,
        }
    }

    /// Compute `temb = timestep_emb + guidance_emb`. Both `timestep` and
    /// `guidance` are `(1,)` F32 graph tensors set per denoising step at
    /// runtime; the caller is responsible for the `* 1000` scaling that
    /// diffusers does in its forward.
    fn embed_time(&self, timestep: GraphTensor, guidance: GraphTensor) -> GraphTensor {
        // Diffusers' Flux2Transformer2DModel.forward multiplies its
        // timestep + guidance inputs by 1000 before passing them to
        // `time_guidance_embed`. The pipeline upstream divides the raw
        // scheduler timestep by 1000 to give the transformer a 0..1
        // scalar; the transformer multiplies it back to 0..1000 here so
        // the sin/cos `time_proj` argument range matches what the
        // model was trained on.
        //
        // Our `main.rs` feeds the same 0..1 sigma scalar (matching the
        // pipeline-level interface) and we mirror the *1000 here.
        let timestep = timestep * 1000.0_f32;
        let guidance = guidance * 1000.0_f32;
        let t_proj = timesteps_proj(timestep, TIMESTEP_GUIDANCE_CHANNELS);
        let t1 = linear_no_bias(t_proj, self.time_t1_w).silu();
        let t_emb = linear_no_bias(t1, self.time_t2_w);
        let g_proj = timesteps_proj(guidance, TIMESTEP_GUIDANCE_CHANNELS);
        let g1 = linear_no_bias(g_proj, self.guidance_t1_w).silu();
        let g_emb = linear_no_bias(g1, self.guidance_t2_w);
        t_emb + g_emb
    }

    /// Single denoising-step forward, fully graph-tensorized so the same
    /// compiled graph runs for every step of the diffusion loop.
    ///
    /// - `latent`: `(S_img, IN_CHANNELS=128)` already patched, F32. Updated
    ///   each step.
    /// - `text_embed`: `(S_txt, JOINT_ATTENTION_DIM=15360)`, set once before
    ///   the loop.
    /// - `rope_cos`, `rope_sin`: `(S_txt + S_img, HEAD_DIM=128)`, also set once.
    /// - `timestep`, `guidance`: `(1,)` F32 scalars set per step (already
    ///   scaled by 1000).
    ///
    /// Returns the model's velocity prediction the scheduler integrates.
    pub fn forward(
        &self,
        latent: GraphTensor,
        text_embed: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        timestep: GraphTensor,
        guidance: GraphTensor,
    ) -> GraphTensor {
        let temb = self.embed_time(timestep, guidance);
        let mod_img = modulation(temb, self.mod_img);
        let mod_txt = modulation(temb, self.mod_txt);
        let mod_single = modulation(temb, self.mod_single);

        let mut img = linear_no_bias(latent, self.x_embedder);
        let mut txt = linear_no_bias(text_embed, self.context_embedder);

        for block in self.transformer_blocks.iter() {
            let (i, t) = block.forward(img, txt, mod_img, mod_txt, rope_cos, rope_sin);
            img = i;
            txt = t;
        }

        let s_img = img.dims()[0].to_usize().expect("S_img static");
        let s_txt = txt.dims()[0].to_usize().expect("S_txt static");
        let mut hidden = txt.concat_along(img, 0); // (S_txt + S_img, HIDDEN)

        for block in self.single_transformer_blocks.iter() {
            hidden = block.forward(hidden, mod_single, rope_cos, rope_sin);
        }

        // Drop text prefix.
        let img = hidden.slice((s_txt..s_txt + s_img, ..));

        // AdaLayerNormContinuous: scale, shift = chunk(linear(silu(temb)), 2).
        let emb = linear_no_bias(temb.silu(), self.norm_out_lin);
        let half = HIDDEN;
        let scale = emb.slice((..half,));
        let shift = emb.slice((half..,));
        let normed = layernorm_noaffine(img, RMS_EPS);
        let modulated = ada_modulate(normed, scale, shift);

        linear_no_bias(modulated, self.proj_out)
    }
}

// =============================================================================
// Tests
// =============================================================================
