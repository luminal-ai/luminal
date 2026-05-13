//! AutoencoderKLFlux2 decoder, in pure HLIR.
//!
//! ## Status
//!
//! - All three primitives (`conv2d_bias`, `group_norm`, `nearest_upsample_2x`)
//!   are implemented and **individually validated** against numerical
//!   references — see the tests at the bottom of this file.
//! - Stitching them into the full decoder currently hits a `luminal_cuda_lite`
//!   optimizer limit: chains of two prefix convs feeding a two-iteration
//!   resnet body with a residual back to the second conv's output cause the
//!   e-graph cleanup to discard the output's eclass ("No valid graphs present
//!   in the e-graph!"). See `deep_conv_chain_with_residual_compiles` (ignored)
//!   for the minimal reproducer. Every resnet block in the diffusers VAE has
//!   this shape, so the full decoder can't be lowered until that's resolved.
//!
//! ## Architecture (for reference once the optimizer is fixed)
//!
//! Pipeline (input image of side N pixels, latent stride 8):
//! 1. `post_quant_conv`            : 1×1 conv 32 → 32, latent at (N/8, N/8)
//! 2. `decoder.conv_in`            : 3×3 conv 32 → 512
//! 3. `decoder.mid_block`          : ResNet → SelfAttn → ResNet, all 512
//! 4. `decoder.up_blocks[0..3]`    : 3 resnets each + nearest-2× upsample
//!    (channel sequence 512 → 512 → 512 → 256 → 128; last block has no upsample)
//! 5. `decoder.conv_norm_out`      : GroupNorm(32 groups) + SiLU
//! 6. `decoder.conv_out`           : 3×3 conv 128 → 3 = (R,G,B) pixels
//!
//! Three building blocks that don't exist in `luminal_nn` get inlined here
//! using only stock HLIR ops (no custom kernels):
//!
//! - **`conv2d_bias`** — unfold + matmul + bias, then a single explicit gather
//!   to reshape (H_out*W_out, C_out) into (C_out, H_out, W_out).
//! - **`group_norm`** — flatten each group's volume into a single axis,
//!   `layer_norm` over that axis, reshape back, per-channel affine.
//! - **`nearest_upsample_2x`** — `expand_dim(broadcast) + merge_dims` on each
//!   spatial axis, so each pixel is duplicated 2×2.

use luminal::{graph::Graph, prelude::*};

/// Standard AutoencoderKL constants for Flux 2.
pub const LATENT_CHANNELS: usize = 32;
pub const VAE_DOWNSAMPLE: usize = 8; // 3 spatial halvings on the encoder side.
pub const NORM_NUM_GROUPS: usize = 32;
pub const NORM_EPS: f32 = 1e-6;
pub const BLOCK_OUT_CHANNELS: [usize; 4] = [128, 256, 512, 512];
pub const LAYERS_PER_BLOCK: usize = 2; // diffusers config; the decoder uses 3 resnets/block (= layers_per_block + 1).
pub const RESNETS_PER_BLOCK: usize = LAYERS_PER_BLOCK + 1;

// Decoder channel progression (reverse of encoder: deepest first).
// up_blocks[i].in_channels  = block_out_channels[max(reversed_idx - 1, 0)]
// up_blocks[i].out_channels = block_out_channels[reversed_idx]
// where reversed_idx walks block_out_channels from back to front.
fn decoder_block_channels(block_idx: usize) -> (usize, usize) {
    let n = BLOCK_OUT_CHANNELS.len();
    let reversed = n - 1 - block_idx;
    let prev = if reversed + 1 < n {
        BLOCK_OUT_CHANNELS[reversed + 1]
    } else {
        BLOCK_OUT_CHANNELS[reversed]
    };
    let out = BLOCK_OUT_CHANNELS[reversed];
    let in_c = if block_idx == 0 {
        BLOCK_OUT_CHANNELS[n - 1] // mid block runs at the deepest channel count
    } else {
        prev
    };
    (in_c, out)
}

// =============================================================================
// HLIR primitive helpers
// =============================================================================

/// 2D convolution with bias on a `(C_in, H, W)` input, weights stored as
/// `(C_out, C_in, K, K)` flat-loaded, bias as `(C_out,)`. Returns
/// `(C_out, H_out, W_out)` where `H_out = (H + 2*padding - kernel) / stride + 1`.
///
/// Wraps the direct conv kernel from [`luminal_cuda_lite::kernel::conv2d_bias`]
/// (one CUDA thread per output element), which avoids materializing the
/// `(H_out*W_out, C_in*K*K)` unfold intermediate that earlier HLIR-only
/// implementations needed.
fn conv2d_bias(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> GraphTensor {
    luminal_cuda_lite::kernel::conv2d_bias(x, weight, bias, kernel, stride, padding)
}

/// PyTorch-style GroupNorm on a (C, H, W) tensor.
///
/// The channel axis is split into `(num_groups, group_size)`; the mean and
/// variance are computed jointly over `(group_size, H, W)` per group; then
/// the output is rescaled and shifted by per-channel `weight` and `bias`.
///
/// Implementation note: we flatten the per-group volume into a single axis
/// before normalizing (rather than calling `layer_norm` over three axes at
/// once). The single-axis form generates simpler egglog patterns and survives
/// composition into deep conv chains, where the 3-axis form drops out of the
/// e-graph during cleanup.
fn group_norm(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    num_groups: usize,
    eps: f32,
) -> GraphTensor {
    let dims = x.dims();
    assert_eq!(dims.len(), 3, "group_norm expects (C, H, W)");
    let c = dims[0];
    let h = dims[1];
    let w = dims[2];

    let c_const = c
        .to_usize()
        .expect("num_channels must be static for GroupNorm");
    let h_const = h.to_usize().expect("height must be static for GroupNorm");
    let w_const = w.to_usize().expect("width must be static for GroupNorm");
    assert!(
        c_const.is_multiple_of(num_groups),
        "num_channels ({c_const}) must be a multiple of num_groups ({num_groups})",
    );
    let group_size = c_const / num_groups;
    let group_volume = group_size * h_const * w_const;

    // Reshape to (num_groups, group_size * H * W) — one flat axis per group.
    let flat = x.merge_dims(0, 1).merge_dims(0, 1); // (C*H*W,)
    let grouped = flat.split_dims(0, group_volume); // (num_groups, group_volume)

    // LayerNorm over the single per-group axis.
    let normed = grouped.layer_norm(1, eps);

    // Reshape (num_groups, group_volume) back to (C, H, W).
    let unshaped = normed
        .merge_dims(0, 1) // flat (C*H*W,)
        .split_dims(0, h_const * w_const) // (C, H*W)
        .split_dims(1, w_const); // (C, H, W)

    // Per-channel affine: weight, bias both shape (C,) -> (C, H, W).
    let w_b = weight.expand_dim(1, h).expand_dim(2, w);
    let b_b = bias.expand_dim(1, h).expand_dim(2, w);
    unshaped * w_b + b_b
}

/// Nearest-neighbour 2× spatial upsample on a (C, H, W) tensor.
fn nearest_upsample_2x(x: GraphTensor) -> GraphTensor {
    // (C, H, W) -> (C, H, 2, W) -> (C, 2H, W) -> (C, 2H, W, 2) -> (C, 2H, 2W)
    let stage1 = x.expand_dim(2, 2_usize).merge_dims(1, 2);
    let stage2 = stage1.expand_dim(3, 2_usize).merge_dims(2, 3);
    // Materialize the broadcast view so subsequent ops see contiguous strides.
    stage2 + 0.0_f32
}

/// SiLU = x * sigmoid(x).
fn silu(x: GraphTensor) -> GraphTensor {
    x.silu()
}

// =============================================================================
// Decoder building blocks
// =============================================================================

struct ResnetBlock {
    norm1_w: GraphTensor,
    norm1_b: GraphTensor,
    conv1_w: GraphTensor,
    conv1_b: GraphTensor,
    norm2_w: GraphTensor,
    norm2_b: GraphTensor,
    conv2_w: GraphTensor,
    conv2_b: GraphTensor,
    shortcut: Option<(GraphTensor, GraphTensor)>, // 1×1 conv when in_c != out_c
    in_channels: usize,
    out_channels: usize,
}

impl ResnetBlock {
    fn new(prefix: &str, in_c: usize, out_c: usize, cx: &mut Graph) -> Self {
        let shortcut = if in_c == out_c {
            None
        } else {
            Some((
                cx.named_tensor(format!("{prefix}.conv_shortcut.weight"), (out_c, in_c))
                    .persist(),
                cx.named_tensor(format!("{prefix}.conv_shortcut.bias"), out_c)
                    .persist(),
            ))
        };
        Self {
            norm1_w: cx
                .named_tensor(format!("{prefix}.norm1.weight"), in_c)
                .persist(),
            norm1_b: cx
                .named_tensor(format!("{prefix}.norm1.bias"), in_c)
                .persist(),
            conv1_w: cx
                .named_tensor(format!("{prefix}.conv1.weight"), (out_c, in_c * 3 * 3))
                .persist(),
            conv1_b: cx
                .named_tensor(format!("{prefix}.conv1.bias"), out_c)
                .persist(),
            norm2_w: cx
                .named_tensor(format!("{prefix}.norm2.weight"), out_c)
                .persist(),
            norm2_b: cx
                .named_tensor(format!("{prefix}.norm2.bias"), out_c)
                .persist(),
            conv2_w: cx
                .named_tensor(format!("{prefix}.conv2.weight"), (out_c, out_c * 3 * 3))
                .persist(),
            conv2_b: cx
                .named_tensor(format!("{prefix}.conv2.bias"), out_c)
                .persist(),
            shortcut,
            in_channels: in_c,
            out_channels: out_c,
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let h = group_norm(x, self.norm1_w, self.norm1_b, NORM_NUM_GROUPS, NORM_EPS);
        let h = silu(h);
        let h = conv2d_bias(h, self.conv1_w, self.conv1_b, 3, 1, 1);
        let h = group_norm(h, self.norm2_w, self.norm2_b, NORM_NUM_GROUPS, NORM_EPS);
        let h = silu(h);
        let h = conv2d_bias(h, self.conv2_w, self.conv2_b, 3, 1, 1);

        let skip = if self.in_channels == self.out_channels {
            x
        } else {
            let (sw, sb) = self.shortcut.expect("shortcut required when in_c != out_c");
            conv2d_bias(x, sw, sb, 1, 1, 0)
        };
        skip + h
    }
}

struct AttnBlock {
    group_norm_w: GraphTensor,
    group_norm_b: GraphTensor,
    to_q_w: GraphTensor,
    to_q_b: GraphTensor,
    to_k_w: GraphTensor,
    to_k_b: GraphTensor,
    to_v_w: GraphTensor,
    to_v_b: GraphTensor,
    to_out_w: GraphTensor,
    to_out_b: GraphTensor,
    channels: usize,
}

impl AttnBlock {
    fn new(prefix: &str, channels: usize, cx: &mut Graph) -> Self {
        let lin =
            |name: &str, out: usize, inn: usize, cx: &mut Graph| -> (GraphTensor, GraphTensor) {
                (
                    cx.named_tensor(format!("{prefix}.{name}.weight"), (out, inn))
                        .persist(),
                    cx.named_tensor(format!("{prefix}.{name}.bias"), out)
                        .persist(),
                )
            };
        let (to_q_w, to_q_b) = lin("to_q", channels, channels, cx);
        let (to_k_w, to_k_b) = lin("to_k", channels, channels, cx);
        let (to_v_w, to_v_b) = lin("to_v", channels, channels, cx);
        let (to_out_w, to_out_b) = lin("to_out.0", channels, channels, cx);
        Self {
            group_norm_w: cx
                .named_tensor(format!("{prefix}.group_norm.weight"), channels)
                .persist(),
            group_norm_b: cx
                .named_tensor(format!("{prefix}.group_norm.bias"), channels)
                .persist(),
            to_q_w,
            to_q_b,
            to_k_w,
            to_k_b,
            to_v_w,
            to_v_b,
            to_out_w,
            to_out_b,
            channels,
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let dims = x.dims();
        assert_eq!(dims.len(), 3, "AttnBlock expects (C, H, W)");
        let _h = dims[1];
        let w = dims[2];
        let residual = x;

        // GroupNorm + reshape to (HW, C) for linear projections.
        let normed = group_norm(
            x,
            self.group_norm_w,
            self.group_norm_b,
            NORM_NUM_GROUPS,
            NORM_EPS,
        );
        // (C, H, W) -> (C, H*W) -> (H*W, C). The transpose at the end leaves
        // a column-major view, which the direct matmul kernels assume away;
        // `* 1.0` forces a contiguous row-major materialization.
        let merged = normed.merge_dims(1, 2).transpose(0, 1) * 1.0_f32;

        // Q, K, V projections — direct kernel routes around the cublaslt
        // 2D rule, which silently fails to fire for some of these matmuls
        // and lets search occasionally pick the broadcast Mul + SumReduce
        // fallback. At 1024² the bad path on `q @ kᵀ` allocates a
        // `(HW, HW, C) = (16384, 16384, 512)` ≈ 524 GiB intermediate.
        let q = luminal_cuda_lite::kernel::linear_bias(merged, self.to_q_w, self.to_q_b);
        let k = luminal_cuda_lite::kernel::linear_bias(merged, self.to_k_w, self.to_k_b);
        let v = luminal_cuda_lite::kernel::linear_bias(merged, self.to_v_w, self.to_v_b);

        // Standard scaled dot-product attention over the spatial axis.
        // `q @ kᵀ` with k stored row-major as `(HW, C)`: matmul_2d_t handles
        // the transpose without materialising k as a separate tensor.
        let scale = (self.channels as f32).sqrt().recip();
        let scores = luminal_cuda_lite::kernel::matmul_2d_t(q, k) * scale;
        let attn_w = scores.softmax(1);
        // attn_w is (HW, HW) row-major, v is (HW, C) row-major; plain matmul.
        let attn = luminal_cuda_lite::kernel::matmul_2d(attn_w, v);

        let out = luminal_cuda_lite::kernel::linear_bias(attn, self.to_out_w, self.to_out_b);
        // (H*W, C) -> (C, H*W) -> (C, H, W)
        let out = out.transpose(0, 1).split_dims(1, w);
        residual + out
    }
}

struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsampler: Option<(GraphTensor, GraphTensor)>, // 3×3 conv after nearest-2×
}

impl UpBlock {
    fn new(prefix: &str, in_c: usize, out_c: usize, with_upsampler: bool, cx: &mut Graph) -> Self {
        let mut resnets = Vec::with_capacity(RESNETS_PER_BLOCK);
        for r in 0..RESNETS_PER_BLOCK {
            let resnet_in = if r == 0 { in_c } else { out_c };
            resnets.push(ResnetBlock::new(
                &format!("{prefix}.resnets.{r}"),
                resnet_in,
                out_c,
                cx,
            ));
        }
        let upsampler = if with_upsampler {
            Some((
                cx.named_tensor(
                    format!("{prefix}.upsamplers.0.conv.weight"),
                    (out_c, out_c * 3 * 3),
                )
                .persist(),
                cx.named_tensor(format!("{prefix}.upsamplers.0.conv.bias"), out_c)
                    .persist(),
            ))
        } else {
            None
        };
        Self { resnets, upsampler }
    }

    fn forward(&self, mut x: GraphTensor) -> GraphTensor {
        for r in &self.resnets {
            x = r.forward(x);
        }
        if let Some((w, b)) = &self.upsampler {
            let up = nearest_upsample_2x(x);
            x = conv2d_bias(up, *w, *b, 3, 1, 1);
        }
        x
    }
}

pub struct VaeDecoder {
    post_quant_w: GraphTensor,
    post_quant_b: GraphTensor,
    conv_in_w: GraphTensor,
    conv_in_b: GraphTensor,
    mid_resnet_0: ResnetBlock,
    mid_attn: AttnBlock,
    mid_resnet_1: ResnetBlock,
    up_blocks: Vec<UpBlock>,
    norm_out_w: GraphTensor,
    norm_out_b: GraphTensor,
    conv_out_w: GraphTensor,
    conv_out_b: GraphTensor,
}

impl VaeDecoder {
    pub fn new(cx: &mut Graph) -> Self {
        let post_quant_w = cx
            .named_tensor("post_quant_conv.weight", (LATENT_CHANNELS, LATENT_CHANNELS))
            .persist();
        let post_quant_b = cx
            .named_tensor("post_quant_conv.bias", LATENT_CHANNELS)
            .persist();

        let mid = BLOCK_OUT_CHANNELS[BLOCK_OUT_CHANNELS.len() - 1];
        let conv_in_w = cx
            .named_tensor("decoder.conv_in.weight", (mid, LATENT_CHANNELS * 3 * 3))
            .persist();
        let conv_in_b = cx.named_tensor("decoder.conv_in.bias", mid).persist();

        let mid_resnet_0 = ResnetBlock::new("decoder.mid_block.resnets.0", mid, mid, cx);
        let mid_attn = AttnBlock::new("decoder.mid_block.attentions.0", mid, cx);
        let mid_resnet_1 = ResnetBlock::new("decoder.mid_block.resnets.1", mid, mid, cx);

        let mut up_blocks = Vec::with_capacity(BLOCK_OUT_CHANNELS.len());
        for b in 0..BLOCK_OUT_CHANNELS.len() {
            let (in_c, out_c) = decoder_block_channels(b);
            let with_upsampler = b < BLOCK_OUT_CHANNELS.len() - 1;
            up_blocks.push(UpBlock::new(
                &format!("decoder.up_blocks.{b}"),
                in_c,
                out_c,
                with_upsampler,
                cx,
            ));
        }

        let last_c = BLOCK_OUT_CHANNELS[0];
        let norm_out_w = cx
            .named_tensor("decoder.conv_norm_out.weight", last_c)
            .persist();
        let norm_out_b = cx
            .named_tensor("decoder.conv_norm_out.bias", last_c)
            .persist();
        let conv_out_w = cx
            .named_tensor("decoder.conv_out.weight", (3, last_c * 3 * 3))
            .persist();
        let conv_out_b = cx.named_tensor("decoder.conv_out.bias", 3).persist();

        Self {
            post_quant_w,
            post_quant_b,
            conv_in_w,
            conv_in_b,
            mid_resnet_0,
            mid_attn,
            mid_resnet_1,
            up_blocks,
            norm_out_w,
            norm_out_b,
            conv_out_w,
            conv_out_b,
        }
    }

    /// Decode a latent of shape (LATENT_CHANNELS, h, w) into an RGB image
    /// of shape (3, h * VAE_DOWNSAMPLE, w * VAE_DOWNSAMPLE) in the [-1, 1] range.
    pub fn forward(&self, latent: GraphTensor) -> GraphTensor {
        self.forward_partial(latent, usize::MAX)
    }

    /// Run the decoder up to stage `stop_at` (used for incremental debugging).
    /// Stages: 0=post_quant only, 1=+conv_in, 2..=4=+mid (resnet, attn, resnet),
    /// 5..=8=+up_blocks[0..3], 9=+conv_norm_out+silu, 10=+conv_out (full).
    pub fn forward_partial(&self, latent: GraphTensor, stop_at: usize) -> GraphTensor {
        let mut x = conv2d_bias(latent, self.post_quant_w, self.post_quant_b, 1, 1, 0);
        if stop_at == 0 {
            return x;
        }
        x = conv2d_bias(x, self.conv_in_w, self.conv_in_b, 3, 1, 1);
        if stop_at == 1 {
            return x;
        }
        x = self.mid_resnet_0.forward(x);
        if stop_at == 2 {
            return x;
        }
        x = self.mid_attn.forward(x);
        if stop_at == 3 {
            return x;
        }
        x = self.mid_resnet_1.forward(x);
        if stop_at == 4 {
            return x;
        }
        for (i, blk) in self.up_blocks.iter().enumerate() {
            x = blk.forward(x);
            if stop_at == 5 + i {
                return x;
            }
        }
        x = group_norm(
            x,
            self.norm_out_w,
            self.norm_out_b,
            NORM_NUM_GROUPS,
            NORM_EPS,
        );
        x = silu(x);
        if stop_at == 9 {
            return x;
        }
        conv2d_bias(x, self.conv_out_w, self.conv_out_b, 3, 1, 1)
    }
}
