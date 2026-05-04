#![allow(dead_code)]
//! YOLO v11n model definition for Luminal.
//!
//! Mirrors the Ultralytics architecture for the `yolo11n` variant:
//!   * width_mult = 0.25, depth_mult = 0.5, max_channels = 1024
//!   * COCO classes (nc = 80), reg_max = 16, three detection scales (8, 16, 32 strides)
//!
//! Weights are loaded from a fused `weights.safetensors` artifact. The export
//! folds Conv + BatchNorm into a single bias-augmented Conv2d, so this model
//! expects only `conv.weight` and `conv.bias` per Conv block.
//!
//! Feature map shapes for a 640x640 input (NCHW, batch=1):
//!   layer 4 -> (1, 128, 80, 80)   (P3 features for the head, after concat layer 15)
//!   layer 6 -> (1, 128, 40, 40)   (P4 backbone features for the head)
//!   layer 10 -> (1, 256, 20, 20)  (P5 backbone features for the head)
//!   final detect output -> (1, 84, 8400)

use luminal::prelude::*;
use luminal::shape::ShapeTracker;

/// Materialize a (potentially non-contiguous) tensor into a contiguous buffer.
/// Mirrors the gather-via-index_expression trick used by [`GraphTensor::output`]
/// so downstream ops see a clean stride pattern.
pub fn make_contiguous(t: GraphTensor) -> GraphTensor {
    if t.shape.is_contiguous() {
        return t;
    }
    let dims = t.dims();
    let total = dims.iter().copied().reduce(|a, b| a * b).unwrap();
    let idx_expr = t.shape.index_expression();
    let idx = t.graph().iota(idx_expr, total);
    let mut gathered = t.gather(idx);
    gathered.shape = ShapeTracker::new(dims);
    gathered
}

fn canonicalize_static_shape(mut t: GraphTensor) -> GraphTensor {
    for dim in &mut t.shape.dims {
        *dim = dim
            .to_usize()
            .map(Expression::from)
            .unwrap_or_else(|| dim.simplify());
    }
    for stride in &mut t.shape.strides {
        *stride = stride.simplify();
    }
    t
}

pub const NC: usize = 80;
pub const REG_MAX: usize = 16;
pub const NO: usize = NC + REG_MAX * 4; // 84
pub const STRIDES: [usize; 3] = [8, 16, 32];
pub const IMG_SIZE: usize = 640;

// width_mult = 0.25 channel widths
pub const C0: usize = 16; // 64*0.25
pub const C1: usize = 32; // 128*0.25
pub const C2: usize = 64; // 256*0.25
pub const C3: usize = 128; // 512*0.25
pub const C4: usize = 256; // 1024*0.25

/// Conv2d (with bias) + optional SiLU activation. Operates on (1, C, H, W).
/// Weight has shape (c_out, c_in*k*k) flattened from PyTorch's (c_out, c_in, k, k).
pub struct Conv {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub k: usize,
    pub s: usize,
    pub p: usize,
    pub c_in: usize,
    pub c_out: usize,
}

impl Conv {
    pub fn new(
        name: &str,
        c_in: usize,
        c_out: usize,
        k: usize,
        s: usize,
        p: usize,
        cx: &mut Graph,
    ) -> Self {
        let weight = cx
            .named_tensor(format!("{name}.weight"), (c_out, c_in * k * k))
            .persist();
        let bias = cx.named_tensor(format!("{name}.bias"), c_out).persist();
        Self {
            weight,
            bias,
            k,
            s,
            p,
            c_in,
            c_out,
        }
    }

    /// Apply the convolution + bias (no activation). Closely mirrors the
    /// pt2-translator's `conv_unfold` so it exercises the same tested code
    /// paths in the luminal e-graph. Special-cases 1x1 convs to a plain matmul
    /// (no unfold) since they don't need spatial windowing.
    pub fn forward_no_act(&self, x: GraphTensor) -> GraphTensor {
        let x = canonicalize_static_shape(x);
        if self.k == 1 && self.s == 1 && self.p == 0 {
            return self.forward_1x1(x);
        }
        // x: (1, c_in, H, W) — keep the batch dim throughout.
        let rank = 4;
        let spatial = 2;

        // Pad spatial dims only.
        let zero = Expression::from(0);
        let pp = Expression::from(self.p);
        let padded = if self.p > 0 {
            x.pad(vec![(zero, zero), (zero, zero), (pp, pp), (pp, pp)], 0.0)
        } else {
            x
        };

        // Unfold with full-rank kernel [1, 1, k, k]
        let unfolded = padded.unfold(
            vec![1usize, 1, self.k, self.k],
            vec![1usize, 1, self.s, self.s],
            vec![1usize, 1, 1, 1],
        );

        // Permute to [N, win_spatial..., C_in, k_N, k_C, k_spatial...] (matches conv_unfold)
        let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
        perm.push(0);
        perm.extend(2..2 + spatial);
        perm.push(1);
        perm.extend(rank..2 * rank);
        let permuted = unfolded.permute(perm);

        let output_spatial_dims: Vec<Expression> = permuted.dims()[1..1 + spatial].to_vec();

        // Merge all channel+kernel dims into [N, spatial..., ch_in * kernel_product]
        let mut patches = permuted;
        let target = 2 + spatial;
        while patches.dims().len() > target {
            let last = patches.dims().len();
            patches = patches.merge_dims(last - 2, last - 1);
        }

        // Merge spatial dims into one
        for _ in 1..spatial {
            patches = patches.merge_dims(1, 2);
        }
        // patches: [N=1, spatial_product, ch_in * kernel_product]

        let mut out = patches.matmul(self.weight.permute((1, 0)));
        // out: [N=1, spatial_product, ch_out]

        // Restore spatial dimensions
        for i in (1..spatial).rev() {
            out = out.split_dims(1, output_spatial_dims[i]);
        }

        // Move ch_out from last to position 1: [N, ch_out, spatial...]
        let mut final_order: Vec<usize> = Vec::with_capacity(2 + spatial);
        final_order.push(0);
        final_order.push(1 + spatial);
        final_order.extend(1..1 + spatial);
        let out = out.permute(final_order);

        // Add bias broadcast across spatial dims.
        let out_dims = out.dims();
        let mut b_expanded = self.bias.expand_dim(0, 1);
        for i in 0..spatial {
            b_expanded = b_expanded.expand_dim(2 + i, out_dims[2 + i]);
        }
        out + b_expanded
    }

    /// Apply the convolution + bias + SiLU (default Conv block path).
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        self.forward_no_act(x).silu()
    }

    /// Specialized 1x1 conv path: equivalent to a per-spatial matmul with no
    /// unfold and no padding. Uses a 2D matmul so the e-graph can match
    /// luminal_cuda_lite's TileMatmulFullSplit specialization.
    fn forward_1x1(&self, x: GraphTensor) -> GraphTensor {
        // x: (1, c_in, H, W) -> drop batch dim, then permute to (H, W, c_in)
        let dims = x.dims();
        let h = dims[2];
        let w = dims[3];
        let x = x.squeeze(0); // (c_in, H, W)
        let xt = x.permute(&[1, 2, 0]); // (H, W, c_in)
                                        // 2D matmul matches the specialized kernel in cuda_lite.
        let xt = xt.merge_dims(0, 1); // (H*W, c_in)
        let out = xt.matmul(self.weight.t()); // (H*W, c_out)
        let out = out.split_dims(0, w); // (H, W, c_out)
        let out = out.permute(&[2, 0, 1]); // (c_out, H, W)
        let bias = self.bias.expand_dim(1, h).expand_dim(2, w);
        let out = out + bias;
        out.expand_dim(0, 1) // restore batch dim
    }
}

/// Depth-wise convolution (groups = in_channels = out_channels).
/// Weight has shape (C, K*K) when flattened from PyTorch's (C, 1, K, K).
pub struct DwConv {
    pub weight: GraphTensor,
    pub bias: GraphTensor,
    pub c: usize,
    pub k: usize,
    pub s: usize,
    pub p: usize,
}

impl DwConv {
    pub fn new(name: &str, c: usize, k: usize, s: usize, p: usize, cx: &mut Graph) -> Self {
        let weight = cx
            .named_tensor(format!("{name}.weight"), (c, k * k))
            .persist();
        let bias = cx.named_tensor(format!("{name}.bias"), c).persist();
        Self {
            weight,
            bias,
            c,
            k,
            s,
            p,
        }
    }

    pub fn forward_no_act(&self, x: GraphTensor) -> GraphTensor {
        let x = canonicalize_static_shape(x);
        let dims = x.dims();
        let h = dims[2];
        let w = dims[3];
        let h_out = (h + 2 * self.p - self.k) / self.s + 1;
        let w_out = (w + 2 * self.p - self.k) / self.s + 1;

        // Pad spatial dims only.
        let zero = Expression::from(0);
        let pp = Expression::from(self.p);
        let padded = x.pad(vec![(zero, zero), (zero, zero), (pp, pp), (pp, pp)], 0.0);
        // Unfold: kernel for batch and channel = 1, spatial = k.
        let unfolded = padded.unfold(
            vec![1usize, 1, self.k, self.k],
            vec![1usize, 1, self.s, self.s],
            vec![1usize, 1, 1, 1],
        );
        // Shape: (1, c, h_out, w_out, 1, 1, k, k). Squeeze the two size-1 kernel dims.
        let unfolded = unfolded.squeeze(4).squeeze(4);
        // Now (1, c, h_out, w_out, k, k). Reshape weight to broadcast: (1, c, 1, 1, k, k).
        let w_kk = self.weight.split_dims(1, self.k); // (c, k, k)
        let w_b = w_kk
            .expand_dim(0, 1)
            .expand_dim(2, h_out)
            .expand_dim(3, w_out);
        let mul = unfolded * w_b;
        let summed = mul.sum(&[5usize, 4]); // sum over kernel dims (descending)
        let bias = self
            .bias
            .expand_dim(0, 1)
            .expand_dim(2, h_out)
            .expand_dim(3, w_out);
        summed + bias
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        self.forward_no_act(x).silu()
    }
}

/// Standard YOLO Bottleneck.
pub struct Bottleneck {
    pub cv1: Conv,
    pub cv2: Conv,
    pub add: bool,
}

impl Bottleneck {
    pub fn new(
        name: &str,
        c1: usize,
        c2: usize,
        shortcut: bool,
        k: (usize, usize),
        e: f32,
        cx: &mut Graph,
    ) -> Self {
        // PyTorch impl: c_ = int(c2 * e).
        let c_ = (c2 as f32 * e) as usize;
        let cv1 = Conv::new(&format!("{name}.cv1.conv"), c1, c_, k.0, 1, k.0 / 2, cx);
        let cv2 = Conv::new(&format!("{name}.cv2.conv"), c_, c2, k.1, 1, k.1 / 2, cx);
        Self {
            cv1,
            cv2,
            add: shortcut && c1 == c2,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let y = self.cv2.forward(self.cv1.forward(x));
        if self.add {
            (x + y) * 1.0
        } else {
            y
        }
    }
}

/// `C3k` module: 3 convs + sequence of n bottlenecks (k=3 inside).
pub struct C3k {
    pub cv1: Conv,
    pub cv2: Conv,
    pub cv3: Conv,
    pub m: Vec<Bottleneck>,
    pub c_: usize,
}

impl C3k {
    pub fn new(name: &str, c1: usize, c2: usize, n: usize, shortcut: bool, cx: &mut Graph) -> Self {
        let c_ = c2 / 2; // e=0.5
        let cv1 = Conv::new(&format!("{name}.cv1.conv"), c1, c_, 1, 1, 0, cx);
        let cv2 = Conv::new(&format!("{name}.cv2.conv"), c1, c_, 1, 1, 0, cx);
        let cv3 = Conv::new(&format!("{name}.cv3.conv"), 2 * c_, c2, 1, 1, 0, cx);
        let m = (0..n)
            .map(|i| Bottleneck::new(&format!("{name}.m.{i}"), c_, c_, shortcut, (3, 3), 1.0, cx))
            .collect();
        Self {
            cv1,
            cv2,
            cv3,
            m,
            c_,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let mut a = self.cv1.forward(x);
        for b in &self.m {
            a = b.forward(a);
        }
        let b = self.cv2.forward(x);
        let cat = a.concat_along(b, 1);
        self.cv3.forward(cat)
    }
}

/// Two variants of the inner block in C3k2.
pub enum C3k2Inner {
    Bottleneck(Box<Bottleneck>),
    C3k(Box<C3k>),
}

impl C3k2Inner {
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        match self {
            C3k2Inner::Bottleneck(b) => b.forward(x),
            C3k2Inner::C3k(c) => c.forward(x),
        }
    }
}

/// `C3k2` module (faster CSP variant). We replace the chunk-style `cv1` with
/// two pre-split convs (`cv1a`, `cv1b`) so the model never needs to slice a
/// tensor on the channel dim — luminal's e-graph cleanup struggles with
/// `slice + residual_add` patterns and would cascade-cleanup roots.
pub struct C3k2 {
    pub cv1a: Conv,
    pub cv1b: Conv,
    pub cv2: Conv,
    pub m: Vec<C3k2Inner>,
    pub c: usize, // hidden channel size
}

#[derive(Clone, Copy)]
pub struct C3k2Config {
    pub c3k: bool,
    pub e: f32,
    pub shortcut: bool,
}

impl C3k2Config {
    pub const fn new(c3k: bool, e: f32, shortcut: bool) -> Self {
        Self { c3k, e, shortcut }
    }
}

impl C3k2 {
    pub fn new(
        name: &str,
        c1: usize,
        c2: usize,
        n: usize,
        config: C3k2Config,
        cx: &mut Graph,
    ) -> Self {
        let c = (c2 as f32 * config.e) as usize; // hidden
                                                 // Two halves of the original cv1 (channel-split). Saved as
                                                 // model.<L>.cv1{a,b}.conv.{weight,bias} by python/reference.py.
        let cv1a = Conv::new(&format!("{name}.cv1a.conv"), c1, c, 1, 1, 0, cx);
        let cv1b = Conv::new(&format!("{name}.cv1b.conv"), c1, c, 1, 1, 0, cx);
        let cv2 = Conv::new(&format!("{name}.cv2.conv"), (2 + n) * c, c2, 1, 1, 0, cx);
        let m: Vec<_> = (0..n)
            .map(|i| {
                if config.c3k {
                    C3k2Inner::C3k(Box::new(C3k::new(
                        &format!("{name}.m.{i}"),
                        c,
                        c,
                        2,
                        config.shortcut,
                        cx,
                    )))
                } else {
                    C3k2Inner::Bottleneck(Box::new(Bottleneck::new(
                        &format!("{name}.m.{i}"),
                        c,
                        c,
                        config.shortcut,
                        (3, 3),
                        0.5,
                        cx,
                    )))
                }
            })
            .collect();
        Self {
            cv1a,
            cv1b,
            cv2,
            m,
            c,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        // a, b are clean conv outputs — no slice required.
        let a = self.cv1a.forward(x);
        let b = self.cv1b.forward(x);

        let mut ys: Vec<GraphTensor> = vec![a, b];
        for inner in &self.m {
            let last = *ys.last().unwrap();
            ys.push(inner.forward(last));
        }
        let mut cat = ys[0];
        for y in &ys[1..] {
            cat = cat.concat_along(*y, 1);
        }
        let cat = make_contiguous(cat);
        self.cv2.forward(cat)
    }
}

/// Spatial Pyramid Pooling — Fast.
pub struct Sppf {
    pub cv1: Conv,
    pub cv2: Conv,
    pub k: usize,
}

impl Sppf {
    pub fn new(name: &str, c1: usize, c2: usize, k: usize, cx: &mut Graph) -> Self {
        let c_ = c1 / 2;
        let cv1 = Conv::new(&format!("{name}.cv1.conv"), c1, c_, 1, 1, 0, cx);
        let cv2 = Conv::new(&format!("{name}.cv2.conv"), c_ * 4, c2, 1, 1, 0, cx);
        Self { cv1, cv2, k }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let y0 = self.cv1.forward(x);
        let y1 = max_pool_2d(y0, self.k, 1, self.k / 2);
        let y2 = max_pool_2d(y1, self.k, 1, self.k / 2);
        let y3 = max_pool_2d(y2, self.k, 1, self.k / 2);
        let cat = y0
            .concat_along(y1, 1)
            .concat_along(y2, 1)
            .concat_along(y3, 1);
        self.cv2.forward(cat)
    }
}

/// MaxPool2d via pad (with -inf-equivalent) + unfold + max reduction.
pub fn max_pool_2d(x: GraphTensor, k: usize, s: usize, p: usize) -> GraphTensor {
    let x = canonicalize_static_shape(x);
    let dims = x.dims();
    let h = dims[2];
    let w = dims[3];
    let h_out = (h + 2 * p - k) / s + 1;
    let w_out = (w + 2 * p - k) / s + 1;

    let zero = Expression::from(0);
    let pp = Expression::from(p);
    let padded = x.pad(
        vec![(zero, zero), (zero, zero), (pp, pp), (pp, pp)],
        -1.0e30,
    );
    let unfolded = padded.unfold(
        vec![1usize, 1, k, k],
        vec![1usize, 1, s, s],
        vec![1usize, 1, 1, 1],
    );
    // Shape: (1, c, h_out, w_out, 1, 1, k, k); squeeze the two size-1 kernel dims.
    let unfolded = unfolded.squeeze(4).squeeze(4);
    let _ = h_out;
    let _ = w_out;
    unfolded.max(&[5usize, 4])
}

/// Nearest-neighbor 2x upsample using broadcast.
pub fn upsample_2x(x: GraphTensor) -> GraphTensor {
    // (1, C, H, W) -> (1, C, H, 2, W) -> (1, C, H, 2, W, 2) -> (1, C, 2H, W, 2) -> (1, C, 2H, 2W)
    x.expand_dim(3, 2usize)
        .expand_dim(5, 2usize)
        .merge_dims(2, 3)
        .merge_dims(3, 4)
}

/// Position-Sensitive Attention with depthwise positional encoding.
/// We replace the single qkv conv + chunk with three pre-split convs to avoid
/// the slice-on-channel cascade pitfall (see C3k2 comment).
pub struct Attention {
    pub q_split: Conv,
    pub k_split: Conv,
    pub v_split: Conv,
    pub proj: Conv,
    pub pe: DwConv,
    pub num_heads: usize,
    pub head_dim: usize,
    pub key_dim: usize,
    pub c: usize,
    pub scale: f32,
}

impl Attention {
    pub fn new(name: &str, c: usize, num_heads: usize, attn_ratio: f32, cx: &mut Graph) -> Self {
        let head_dim = c / num_heads;
        let key_dim = (head_dim as f32 * attn_ratio) as usize;
        let q_split = Conv::new(
            &format!("{name}.q_split.conv"),
            c,
            num_heads * key_dim,
            1,
            1,
            0,
            cx,
        );
        let k_split = Conv::new(
            &format!("{name}.k_split.conv"),
            c,
            num_heads * key_dim,
            1,
            1,
            0,
            cx,
        );
        let v_split = Conv::new(
            &format!("{name}.v_split.conv"),
            c,
            num_heads * head_dim,
            1,
            1,
            0,
            cx,
        );
        let proj = Conv::new(&format!("{name}.proj.conv"), c, c, 1, 1, 0, cx);
        let pe = DwConv::new(&format!("{name}.pe.conv"), c, 3, 1, 1, cx);
        Self {
            q_split,
            k_split,
            v_split,
            proj,
            pe,
            num_heads,
            head_dim,
            key_dim,
            c,
            scale: (key_dim as f32).powf(-0.5),
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        // x: (1, C, H, W); B=1
        let dims = x.dims();
        let _h_dim = dims[2];
        let w_dim = dims[3];

        // Three independent QKV convs.
        let q_full = self.q_split.forward_no_act(x); // (1, num_heads*key_dim, H, W)
        let k_full = self.k_split.forward_no_act(x);
        let v_full = self.v_split.forward_no_act(x); // (1, num_heads*head_dim, H, W)

        // Reshape to (1, num_heads, head/key_dim, N)
        let q = q_full.merge_dims(2, 3).split_dims(1, self.key_dim); // (1, nh, key_dim, N)
        let k = k_full.merge_dims(2, 3).split_dims(1, self.key_dim);
        let v = v_full.merge_dims(2, 3).split_dims(1, self.head_dim); // (1, nh, head_dim, N)

        // attn = (q.transpose(-2, -1) @ k) * scale  -> (1, num_heads, N, N)
        let q_t = q.transpose(2, 3); // (1, num_heads, N, key_dim)
        let attn = q_t.matmul(k) * self.scale; // (1, num_heads, N, N)
        let attn = attn.softmax(3);

        // x_attn = v @ attn.T -> (1, num_heads, head_dim, N)
        let attn_t = attn.transpose(2, 3);
        let x_attn = v.matmul(attn_t);

        // (1, num_heads, head_dim, N) -> (1, C, H, W)
        let x_attn = x_attn.merge_dims(1, 2).split_dims(2, w_dim);

        // pe runs on v reshaped back to (1, C, H, W) — v_full already has that layout.
        let pe_out = self.pe.forward_no_act(v_full);

        let combined = x_attn + pe_out;
        self.proj.forward_no_act(combined)
    }
}

pub struct Ffn {
    pub fc1: Conv, // SiLU activated
    pub fc2: Conv, // act=False
}

impl Ffn {
    pub fn new(name: &str, c: usize, cx: &mut Graph) -> Self {
        let fc1 = Conv::new(&format!("{name}.0.conv"), c, c * 2, 1, 1, 0, cx);
        let fc2 = Conv::new(&format!("{name}.1.conv"), c * 2, c, 1, 1, 0, cx);
        Self { fc1, fc2 }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        // First conv has SiLU, second has Identity (act=False)
        self.fc2.forward_no_act(self.fc1.forward(x))
    }
}

pub struct PsaBlock {
    pub attn: Attention,
    pub ffn: Ffn,
}

impl PsaBlock {
    pub fn new(name: &str, c: usize, attn_ratio: f32, num_heads: usize, cx: &mut Graph) -> Self {
        let attn = Attention::new(&format!("{name}.attn"), c, num_heads, attn_ratio, cx);
        let ffn = Ffn::new(&format!("{name}.ffn"), c, cx);
        Self { attn, ffn }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = x + self.attn.forward(x);
        x + self.ffn.forward(x)
    }
}

pub struct C2psa {
    pub cv1a: Conv,
    pub cv1b: Conv,
    pub cv2: Conv,
    pub m: Vec<PsaBlock>,
    pub c: usize,
}

impl C2psa {
    pub fn new(name: &str, c1: usize, c2: usize, n: usize, e: f32, cx: &mut Graph) -> Self {
        assert_eq!(c1, c2);
        let c = (c1 as f32 * e) as usize;
        let cv1a = Conv::new(&format!("{name}.cv1a.conv"), c1, c, 1, 1, 0, cx);
        let cv1b = Conv::new(&format!("{name}.cv1b.conv"), c1, c, 1, 1, 0, cx);
        let cv2 = Conv::new(&format!("{name}.cv2.conv"), 2 * c, c1, 1, 1, 0, cx);
        let num_heads = (c / 64).max(1);
        let m = (0..n)
            .map(|i| PsaBlock::new(&format!("{name}.m.{i}"), c, 0.5, num_heads, cx))
            .collect();
        Self {
            cv1a,
            cv1b,
            cv2,
            m,
            c,
        }
    }

    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let a = self.cv1a.forward(x);
        let mut b = self.cv1b.forward(x);
        for blk in &self.m {
            b = blk.forward(b);
        }
        let cat = make_contiguous(a.concat_along(b, 1));
        self.cv2.forward(cat)
    }
}

/// One detection scale's box + class branches (without DFL/decode — applied later).
pub struct DetectScale {
    // Box regression branch (cv2[i]):
    pub box1: Conv,    // 3x3 + SiLU
    pub box2: Conv,    // 3x3 + SiLU
    pub box_out: Conv, // 1x1 plain Conv2d with bias (no act)

    // Classification branch (cv3[i]):
    pub dw_a: DwConv,  // 3x3 depthwise + SiLU
    pub pw_a: Conv,    // 1x1 + SiLU
    pub dw_b: DwConv,  // 3x3 depthwise + SiLU
    pub pw_b: Conv,    // 1x1 + SiLU
    pub cls_out: Conv, // 1x1 plain Conv2d with bias (no act)

    pub c_in: usize,
}

impl DetectScale {
    pub fn new(
        name_prefix_cv2: &str,
        name_prefix_cv3: &str,
        c_in: usize,
        c2: usize,
        c3: usize,
        cx: &mut Graph,
    ) -> Self {
        // cv2: Sequential(Conv(c_in, c2, 3), Conv(c2, c2, 3), Conv2d(c2, 4*reg_max, 1))
        let box1 = Conv::new(&format!("{name_prefix_cv2}.0.conv"), c_in, c2, 3, 1, 1, cx);
        let box2 = Conv::new(&format!("{name_prefix_cv2}.1.conv"), c2, c2, 3, 1, 1, cx);
        let box_out = Conv::new(
            name_prefix_cv2_terminal(name_prefix_cv2).as_str(),
            c2,
            4 * REG_MAX,
            1,
            1,
            0,
            cx,
        );

        // cv3: Sequential(Sequential(DWConv(c_in, c_in, 3), Conv(c_in, c3, 1)),
        //                 Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
        //                 Conv2d(c3, nc, 1))
        let dw_a = DwConv::new(&format!("{name_prefix_cv3}.0.0.conv"), c_in, 3, 1, 1, cx);
        let pw_a = Conv::new(
            &format!("{name_prefix_cv3}.0.1.conv"),
            c_in,
            c3,
            1,
            1,
            0,
            cx,
        );
        let dw_b = DwConv::new(&format!("{name_prefix_cv3}.1.0.conv"), c3, 3, 1, 1, cx);
        let pw_b = Conv::new(&format!("{name_prefix_cv3}.1.1.conv"), c3, c3, 1, 1, 0, cx);
        let cls_out = Conv::new(
            name_prefix_cv3_terminal(name_prefix_cv3).as_str(),
            c3,
            NC,
            1,
            1,
            0,
            cx,
        );

        Self {
            box1,
            box2,
            box_out,
            dw_a,
            pw_a,
            dw_b,
            pw_b,
            cls_out,
            c_in,
        }
    }

    /// Returns (box_logits (1, 4*reg_max, H, W), cls_logits (1, nc, H, W)).
    pub fn forward(&self, x: GraphTensor) -> (GraphTensor, GraphTensor) {
        let b = self.box1.forward(x);
        let b = self.box2.forward(b);
        let b = self.box_out.forward_no_act(b);

        let c = self.dw_a.forward(x);
        let c = self.pw_a.forward(c);
        let c = self.dw_b.forward(c);
        let c = self.pw_b.forward(c);
        let c = self.cls_out.forward_no_act(c);
        (b, c)
    }
}

fn name_prefix_cv2_terminal(prefix: &str) -> String {
    // The third element (index 2) inside cv2[i] is a plain Conv2d (no .conv child).
    // Its weights live at `<cv2_prefix>.2.weight`/`.2.bias`.
    format!("{prefix}.2")
}

fn name_prefix_cv3_terminal(prefix: &str) -> String {
    format!("{prefix}.2")
}

pub struct Detect {
    pub scales: Vec<DetectScale>,
    pub dfl_weight: GraphTensor, // (16,) - constant arange(16)
    pub anchors: GraphTensor,    // (2, 8400) precomputed
    pub strides: GraphTensor,    // (1, 8400) precomputed
    pub feat_sizes: Vec<usize>,
}

impl Detect {
    pub fn new(name: &str, ch: &[usize], feat_sizes: &[usize], cx: &mut Graph) -> Self {
        // Detect head channel widths
        let c2 = (16usize).max(ch[0] / 4).max(REG_MAX * 4);
        let c3 = ch[0].max(NC.min(100));
        let scales = (0..ch.len())
            .map(|i| {
                DetectScale::new(
                    &format!("{name}.cv2.{i}"),
                    &format!("{name}.cv3.{i}"),
                    ch[i],
                    c2,
                    c3,
                    cx,
                )
            })
            .collect();
        let dfl_weight = cx
            .named_tensor(format!("{name}.dfl.conv.weight"), REG_MAX)
            .persist();

        // Anchors and strides aren't in the safetensors — we feed them as inputs.
        let total_anchors: usize = feat_sizes.iter().map(|s| s * s).sum();
        let anchors = cx
            .named_tensor("yolo.anchors", (2usize, total_anchors))
            .persist();
        let strides = cx
            .named_tensor("yolo.strides", (1usize, total_anchors))
            .persist();
        Self {
            scales,
            dfl_weight,
            anchors,
            strides,
            feat_sizes: feat_sizes.to_vec(),
        }
    }

    /// Final prediction tensor (1, 84, total_anchors).
    pub fn forward(&self, feats: &[GraphTensor]) -> GraphTensor {
        assert_eq!(feats.len(), self.scales.len());
        let mut box_flats = Vec::new();
        let mut cls_flats = Vec::new();
        for (i, x) in feats.iter().enumerate() {
            let (b, c) = self.scales[i].forward(*x);
            // (1, 4*reg_max, H, W) -> (1, 4*reg_max, H*W)
            let b = b.merge_dims(2, 3);
            let c = c.merge_dims(2, 3);
            box_flats.push(b);
            cls_flats.push(c);
        }
        let mut boxes = box_flats[0];
        for b in &box_flats[1..] {
            boxes = boxes.concat_along(*b, 2);
        }
        let mut scores = cls_flats[0];
        for c in &cls_flats[1..] {
            scores = scores.concat_along(*c, 2);
        }
        // boxes: (1, 64, A); scores: (1, 80, A)

        // DFL: PyTorch does view(b, 4, reg_max, A) which row-major splits 64=4*reg_max with
        // 4 outer and reg_max inner. luminal split_dims(axis, inner_size) places `inner_size`
        // as the new inner dim, so we must pass REG_MAX (not 4) to get (1, 4, REG_MAX, A).
        let dfl_in = boxes.split_dims(1, REG_MAX); // (1, 4, REG_MAX, A)
                                                   // Then transpose so the REG_MAX bin axis becomes the softmax channel axis.
        let dfl_in = dfl_in.transpose(1, 2); // (1, REG_MAX, 4, A)
        let dfl_in = dfl_in.softmax(1);

        // dfl_weight: (reg_max,). Broadcast over batch, box coords, and anchors.
        // Conv1x1 == channel-wise weighted sum:
        // (1, reg_max, 4, A) * (1, reg_max, 4, A) -> sum over reg_max -> (1, 4, A)
        let w = self
            .dfl_weight
            .expand_to_shape_on_axes(dfl_in.shape, &[0usize, 2, 3]);
        let dfl_out = (dfl_in * w).sum(&[1usize]); // (1, 4, A)

        // dist2bbox xywh: lt = dfl[:, :2, :], rb = dfl[:, 2:, :]
        let lt = make_contiguous(dfl_out.slice((.., 0..2, ..)));
        let rb = make_contiguous(dfl_out.slice((.., 2..4, ..)));

        // anchors: (2, A). Add batch dim.
        let anchors = self.anchors.expand_dim(0, 1); // (1, 2, A)
        let x1y1 = anchors - lt;
        let x2y2 = anchors + rb;
        let cxy = (x1y1 + x2y2) * 0.5;
        let wh = x2y2 - x1y1;
        let dbox = cxy.concat_along(wh, 1); // (1, 4, A)

        // Multiply by strides: (1, 1, A)
        let strides = self.strides.expand_dim(0, 1); // (1, 1, A)
        let dbox = dbox * strides.expand_dim(1, 4usize).squeeze(2); // broadcast across 4 box dims
                                                                    // (the squeeze removes the size-1 channel from the second expand)

        let scores_sig = scores.sigmoid();
        dbox.concat_along(scores_sig, 1)
    }
}

pub struct YoloV11 {
    // Backbone
    pub conv0: Conv,     // model.0
    pub conv1: Conv,     // model.1
    pub c3k2_2: C3k2,    // model.2
    pub conv3: Conv,     // model.3
    pub c3k2_4: C3k2,    // model.4
    pub conv5: Conv,     // model.5
    pub c3k2_6: C3k2,    // model.6
    pub conv7: Conv,     // model.7
    pub c3k2_8: C3k2,    // model.8
    pub sppf_9: Sppf,    // model.9
    pub c2psa_10: C2psa, // model.10
    // Head
    pub c3k2_13: C3k2,
    pub c3k2_16: C3k2,
    pub conv17: Conv,
    pub c3k2_19: C3k2,
    pub conv20: Conv,
    pub c3k2_22: C3k2,
    pub detect: Detect,
}

impl YoloV11 {
    pub fn init(cx: &mut Graph) -> Self {
        // Backbone
        let conv0 = Conv::new("model.0.conv", 3, C0, 3, 2, 1, cx);
        let conv1 = Conv::new("model.1.conv", C0, C1, 3, 2, 1, cx);
        let c3k2_2 = C3k2::new("model.2", C1, C2, 1, C3k2Config::new(false, 0.25, true), cx);
        let conv3 = Conv::new("model.3.conv", C2, C2, 3, 2, 1, cx);
        let c3k2_4 = C3k2::new("model.4", C2, C3, 1, C3k2Config::new(false, 0.25, true), cx);
        let conv5 = Conv::new("model.5.conv", C3, C3, 3, 2, 1, cx);
        let c3k2_6 = C3k2::new("model.6", C3, C3, 1, C3k2Config::new(true, 0.5, true), cx);
        let conv7 = Conv::new("model.7.conv", C3, C4, 3, 2, 1, cx);
        let c3k2_8 = C3k2::new("model.8", C4, C4, 1, C3k2Config::new(true, 0.5, true), cx);
        let sppf_9 = Sppf::new("model.9", C4, C4, 5, cx);
        let c2psa_10 = C2psa::new("model.10", C4, C4, 1, 0.5, cx);

        // Head
        let c3k2_13 = C3k2::new(
            "model.13",
            C4 + C3,
            C3,
            1,
            C3k2Config::new(false, 0.5, true),
            cx,
        );
        let c3k2_16 = C3k2::new(
            "model.16",
            C3 + C3,
            C2,
            1,
            C3k2Config::new(false, 0.5, true),
            cx,
        );
        let conv17 = Conv::new("model.17.conv", C2, C2, 3, 2, 1, cx);
        let c3k2_19 = C3k2::new(
            "model.19",
            C2 + C3,
            C3,
            1,
            C3k2Config::new(false, 0.5, true),
            cx,
        );
        let conv20 = Conv::new("model.20.conv", C3, C3, 3, 2, 1, cx);
        let c3k2_22 = C3k2::new(
            "model.22",
            C3 + C4,
            C4,
            1,
            C3k2Config::new(true, 0.5, true),
            cx,
        );

        // Detect head reads from layers (16, 19, 22) at (80, 40, 20) feature sizes
        let detect = Detect::new("model.23", &[C2, C3, C4], &[80, 40, 20], cx);
        Self {
            conv0,
            conv1,
            c3k2_2,
            conv3,
            c3k2_4,
            conv5,
            c3k2_6,
            conv7,
            c3k2_8,
            sppf_9,
            c2psa_10,
            c3k2_13,
            c3k2_16,
            conv17,
            c3k2_19,
            conv20,
            c3k2_22,
            detect,
        }
    }

    /// Run inference on `(1, 3, 640, 640)` input. Returns (1, 84, 8400) decoded predictions.
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.conv0.forward(x); // (1, 16, 320, 320)
        let x = self.conv1.forward(x); // (1, 32, 160, 160)
        let x = self.c3k2_2.forward(x); // (1, 64, 160, 160)
        let x = self.conv3.forward(x); // (1, 64, 80, 80)
        let l4 = self.c3k2_4.forward(x); // (1, 128, 80, 80)
        let x = self.conv5.forward(l4); // (1, 128, 40, 40)
        let l6 = self.c3k2_6.forward(x); // (1, 128, 40, 40)
        let x = self.conv7.forward(l6); // (1, 256, 20, 20)
        let x = self.c3k2_8.forward(x); // (1, 256, 20, 20)
        let x = self.sppf_9.forward(x); // (1, 256, 20, 20)
        let l10 = self.c2psa_10.forward(x); // (1, 256, 20, 20)

        // Head
        let up = upsample_2x(l10); // (1, 256, 40, 40)
        let x = make_contiguous(up.concat_along(l6, 1)); // (1, 384, 40, 40)
        let l13 = self.c3k2_13.forward(x); // (1, 128, 40, 40)
        let up = upsample_2x(l13); // (1, 128, 80, 80)
        let x = make_contiguous(up.concat_along(l4, 1)); // (1, 256, 80, 80)
        let l16 = self.c3k2_16.forward(x); // (1, 64, 80, 80) -> P3
        let down = self.conv17.forward(l16); // (1, 64, 40, 40)
        let x = make_contiguous(down.concat_along(l13, 1)); // (1, 192, 40, 40)
        let l19 = self.c3k2_19.forward(x); // (1, 128, 40, 40) -> P4
        let down = self.conv20.forward(l19); // (1, 128, 20, 20)
        let x = make_contiguous(down.concat_along(l10, 1)); // (1, 384, 20, 20)
        let l22 = self.c3k2_22.forward(x); // (1, 256, 20, 20) -> P5

        self.detect.forward(&[l16, l19, l22])
    }
}

/// Compute the (anchors, strides) flat tensors for the three YOLO scales.
/// Returns (anchors flat (2*A,), strides flat (A,)).
pub fn make_anchors_and_strides(
    feat_sizes: &[usize],
    stride_values: &[usize],
) -> (Vec<f32>, Vec<f32>) {
    let total_anchors: usize = feat_sizes.iter().map(|s| s * s).sum();
    // anchors stored as (2, A): row 0 = x, row 1 = y
    let mut anchors_x = Vec::with_capacity(total_anchors);
    let mut anchors_y = Vec::with_capacity(total_anchors);
    let mut strides_v = Vec::with_capacity(total_anchors);
    for (sz, st) in feat_sizes.iter().zip(stride_values.iter()) {
        for r in 0..*sz {
            for c in 0..*sz {
                anchors_x.push(c as f32 + 0.5);
                anchors_y.push(r as f32 + 0.5);
                strides_v.push(*st as f32);
            }
        }
    }
    let mut anchors = Vec::with_capacity(2 * total_anchors);
    anchors.extend(anchors_x);
    anchors.extend(anchors_y);
    (anchors, strides_v)
}

/// DFL constant weights (arange(reg_max) reshaped as (1, reg_max, 1, 1)).
pub fn dfl_weight() -> Vec<f32> {
    (0..REG_MAX as i32).map(|i| i as f32).collect()
}
