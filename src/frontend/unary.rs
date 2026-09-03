use itertools::Itertools;

use crate::prelude::*;
use std::ops::{Add, Mul, Neg};

/// Scatter `arange(ax_size)` into rank positions to convert per-element ranks
/// into sort indices. Handles multi-dim by computing flat scatter offsets.
fn scatter_ranks_to_sort_indices(
    ranks: GraphTensor,
    dims: Vec<IntExpr>,
    axis: usize,
    g: &mut Graph,
) -> GraphTensor {
    let ndim = dims.len();

    // Values: [0, 1, ..., ax_size-1] along axis over the full shape,
    // and the zero init — each a single full-shape coordinate-function
    // iota (P1 + ruling 2026-08-26): NO applies at all, where the old
    // scaffolding minted an expand chain per non-axis dim.
    let values = g.iota(dims.clone(), |c| c[axis]);
    let zeros = g.iota(dims.clone(), |_| IntExpr::from(0usize));

    if ndim == 1 {
        return values.scatter1d(ranks, zeros);
    }

    // Multi-dim: ranks are per-axis (0..ax_size) but scatter uses flat indices.
    // Compute: adjusted = base_offset + ranks * axis_stride
    let mut strides = vec![IntExpr::from(1usize); ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        // Frontend simplification restored (revert ruling 2026-08-27).
        strides[d] = (strides[d + 1] * dims[d + 1]).simplify();
    }
    let axis_stride = strides[axis];
    let ranks_scaled = ranks * axis_stride;

    // One coordinate-function iota replaces the per-dim iota +
    // expand_dim scaffolding (P1, 2026-08-07).
    let base_offset = g.iota(dims.to_vec(), |c| {
        (0..ndim)
            .filter(|d| *d != axis)
            .fold(IntExpr::from(0), |acc, d| acc + c[d] * strides[d])
    });
    let adjusted = base_offset + ranks_scaled;

    values.scatter1d(adjusted, zeros)
}

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        // Dtype-aware: an integer tensor negates through an Int
        // constant — the f32 literal would record a refused f32 -> Int
        // cast (typed-buffers cast policy, 2026-08-11). The narrow
        // integers joined the executable set on 2026-09-02 (main #399)
        // and take the same arm: on an unsigned type the -1 constant
        // casts to that type's all-ones code and the wrapping multiply
        // is two's-complement negation, which is what it means there.
        match self.dtype {
            DType::Int
            | DType::I64
            | DType::I4
            | DType::U4
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16 => self * IntExpr::from(-1),
            _ => self * -1.,
        }
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Log2,
                &[(self.id, self.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Exp2,
                &[(self.id, self.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Recip,
                &[(self.id, self.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Sin,
                &[(self.id, self.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Sqrt,
                &[(self.id, self.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes.to_axes())
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand_to_shape_on_axes(self.dims(), axes)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.dims(), axes)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        self.mean_norm(axes.to_axes()).std_norm(axes, epsilon)
    }

    /// Normalize the tensor along `axes` using an Lp norm.
    pub fn normalize(self, p: f32, axes: impl ToAxes, epsilon: f32) -> GraphTensor {
        let norm = self.abs().pow(p).sum(axes.to_axes()).pow(1.0 / p);
        self / norm
            .maximum_f32(epsilon)
            .expand_to_shape_on_axes(self.dims(), axes)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.dims(), axes.to_axes());
        let exp = m.exp();
        exp / exp
            .sum(axes.to_axes())
            .expand_to_shape_on_axes(self.dims(), axes)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.dims(), axes.to_axes());
        m - m
            .exp()
            .sum(axes.to_axes())
            .log()
            .expand_to_shape_on_axes(m.dims(), axes)
    }

    /// Get the indicies of the max elements along an axis
    pub fn argmax(self, axis: usize) -> GraphTensor {
        // Get one-hot along last dimension
        let x_equal = self
            .eq(self.max(axis).expand_dim(axis, self.dims()[axis]))
            .cast(DType::Int);
        // Create index arange for last dimension
        let r = self.graph().arange(self.dims()[axis]);
        let axes = (0..self.rank()).filter(|i| *i != axis).collect_vec();
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand_to_shape_on_axes(self.dims(), axes)).max(axis)
    }

    /// Get the indices of the min elements along an axis
    pub fn argmin(self, axis: usize) -> GraphTensor {
        (-self).argmax(axis)
    }

    /// Compute the sample variance along axes
    pub fn var(self, axes: impl ToAxes) -> GraphTensor {
        self.var_options(axes, 1)
    }

    /// Compute the sample variance along an axes with options
    pub fn var_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        let axes = axes.to_axes();
        let n = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<IntExpr>();
        let mean = self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.dims(), axes.to_axes());
        let centered = self - mean;
        (centered * centered).sum(axes) / (n - correction)
    }

    /// Compute the sample standard deviation along axes
    pub fn std(self, axes: impl ToAxes) -> GraphTensor {
        self.std_options(axes, 1)
    }

    /// Compute the standard deviation along axes with options
    pub fn std_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        self.var_options(axes, correction).sqrt()
    }

    /// Take the absolute value.
    ///
    /// DTYPE-AWARE (main #399, re-expressed 2026-09-02). The float path
    /// is unchanged: `relu(x) + relu(-x)`. Integers cannot take it —
    /// `relu` is `maximum_f32`, which builds its bound with
    /// `constant_float(0.0).cast(self.dtype)`, and an F32 -> Int cast is
    /// REFUSED at authoring by the cast policy of 2026-08-11, so
    /// `abs()` on an Int tensor used to panic before it recorded
    /// anything. So:
    ///
    /// * UNSIGNED integers are already their own absolute value —
    ///   identity, and no ops recorded at all.
    /// * SIGNED integers use main's identity `x * (1 - 2*(x < 0))`,
    ///   built from Int constants rather than float ones so no
    ///   float -> int cast appears. This is also CORRECT at the signed
    ///   minimum in the only sense available: `i32::MIN` has no
    ///   representable absolute value, and the multiplication reports
    ///   that as an overflow (the Int kernels are checked) instead of
    ///   returning `MIN` as `relu` + `relu` would.
    pub fn abs(self) -> GraphTensor {
        match self.dtype {
            DType::U4 | DType::U8 | DType::U16 => self,
            DType::I4 | DType::I8 | DType::I16 | DType::Int | DType::I64 => {
                let dims = self.dims();
                let zero = self
                    .graph()
                    .constant(0)
                    .cast(self.dtype)
                    .expand_rhs(dims.clone());
                let one = self.graph().constant(1).cast(self.dtype).expand_rhs(dims);
                let negative = self.lt(zero).cast(self.dtype);
                self * (one - negative * 2)
            }
            _ => self.relu() + (-self).relu(),
        }
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor {
        self / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor {
        self.maximum_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        (1. + (-self).exp()).reciprocal()
    }

    /// The swish (aka silu) activation function
    pub fn swish(self) -> GraphTensor {
        self * self.sigmoid()
    }

    /// The silu (aka swish) activation function
    pub fn silu(self) -> GraphTensor {
        self.swish()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor {
        self.relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    ///
    /// This is the exact (erf-based) form, matching PyTorch's default `aten.gelu`
    /// (`approximate="none"`). `erf` is composed from existing primitives via the
    /// Abramowitz & Stegun 7.1.26 rational+exp approximation (max abs error ~1.5e-7),
    /// so this is a single, deterministic elementwise composition with no
    /// special-cased kernel. For the cheaper tanh approximation, see
    /// [`gelu_fast_tanh_approximation`](Self::gelu_fast_tanh_approximation).
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor {
        // erf(u), u = x / sqrt(2), via A&S 7.1.26:
        //   t = 1 / (1 + p*|u|)
        //   erf(u) = sign(u) * (1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-u^2))
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        const P: f32 = 0.3275911;
        const A1: f32 = 0.254829592;
        const A2: f32 = -0.284496736;
        const A3: f32 = 1.421413741;
        const A4: f32 = -1.453152027;
        const A5: f32 = 1.061405429;

        let u = INV_SQRT2 * self;
        let t = (1. + P * u.abs()).reciprocal();
        // Horner form of the degree-5 polynomial in t.
        let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
        let erf = u.sign() * (1. - poly * (-(u * u)).exp());
        0.5 * self * (1. + erf)
    }

    /// The tanh approximation of GELU (PyTorch's `approximate="tanh"`):
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    ///
    /// Cheaper than the exact [`gelu`](Self::gelu) (fewer ops, ~3e-4 max error) and
    /// what cuBLASLt's GELU epilogue / the GLUMoE Gemma-GELU recognition match.
    #[allow(clippy::excessive_precision)]
    pub fn gelu_fast_tanh_approximation(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        let scaled = 1.5957691216 * self * (1. + 0.044715 * self * self);
        self * scaled.sigmoid()
    }

    /// Compute the sorted indexes of this tensor along a certian axis
    pub fn argsort(self, axis: usize, descending: bool) -> GraphTensor {
        // Compare all elements with all other elements by making an axis
        let ax_size = self.dims()[axis];
        let a = self.expand_dim(axis + 1, ax_size) + 0.0;
        let b = self.expand_dim(axis, ax_size) + 1e-9;
        let cmp = if descending { a.gt(b) } else { a.lt(b) };
        // ind[j] = rank of element j (how many elements are smaller/larger).
        // Int-native (2026-08-11): the Bool comparison converts through
        // the exact 0/1 indicator and SUMS in i32 — the old spelling
        // summed in f32 and cast back, a refused lossy read.
        let ranks = cmp.cast(DType::Int).sum(axis);
        // Scatter original indices into rank positions to get sort indices
        scatter_ranks_to_sort_indices(ranks, self.dims(), axis, self.graph())
    }

    /// Stable argsort: like `argsort`, but breaks ties by original index
    /// (lower index first). Guarantees unique ranks even when values are equal.
    pub fn stable_argsort(self, axis: usize, descending: bool) -> GraphTensor {
        let ax_size = self.dims()[axis];
        let dims = self.dims();

        // Expanded shape: original dims with ax_size inserted at axis
        let mut exp_dims = dims.clone();
        exp_dims.insert(axis, ax_size);

        // Pairwise value tensors (* 1.0 forces materialization to avoid stride issues)
        let a_val = self.expand_dim(axis + 1, ax_size) * 1.0;
        let b_val = self.expand_dim(axis, ax_size) * 1.0;

        // Index tensors for tiebreaking (Int-native comparisons) —
        // full-shape coordinate-function iotas (P1 + ruling 2026-08-26):
        // iota_a reads coordinate `axis`, iota_b coordinate `axis + 1`;
        // NO applies where the old scaffolding minted an expand chain
        // per surrounding dim.
        let iota_a = self.graph().iota(exp_dims.clone(), |c| c[axis]);
        let iota_b = self.graph().iota(exp_dims.clone(), |c| c[axis + 1]);

        // Lexicographic comparison with stable tiebreaking (lower index first):
        // ascending:  rank[j] = count of i where (x[i] < x[j]) || (x[i]==x[j] && i < j)
        // descending: rank[j] = count of i where (x[i] > x[j]) || (x[i]==x[j] && i < j)
        let primary = if descending {
            a_val.gt(b_val)
        } else {
            a_val.lt(b_val)
        };
        let idx_cmp = iota_a.lt(iota_b);
        // Int-native indicator arithmetic (2026-08-11): every Bool
        // converts through the exact 0/1 indicator, the counting sum
        // runs in i32, and no f32 -> Int cast (refused) ever appears.
        let primary_count = primary.cast(DType::Int);
        let idx_count = idx_cmp.cast(DType::Int);
        let lt_count = a_val.lt(b_val).cast(DType::Int);
        let gt_count = a_val.gt(b_val).cast(DType::Int);
        let one = self.graph().constant(1).expand_rhs(lt_count.dims());
        let val_eq = (one - lt_count) * (one - gt_count);
        let cmp = primary_count + val_eq * idx_count;

        // Scatter original indices into rank positions to get sort indices
        let ranks = cmp.sum(axis);
        scatter_ranks_to_sort_indices(ranks, dims, axis, self.graph())
    }

    /// Sort the tensor along a certian axis
    pub fn sort(self, axis: usize, descending: bool) -> GraphTensor {
        self.gather1d(self.argsort(axis, descending))
    }

    /// Sort and retrieve top-k **indexes**
    pub fn topk_indexes(self, k: usize, axis: usize) -> GraphTensor {
        // Stable sort is required: with exact value ties (e.g. softmax rows
        // where several entries underflow to exactly 0.0), the unstable
        // argsort produces duplicate ranks and the rank→index scatter emits
        // garbage — nondeterministically across search candidates, since tie
        // outcomes depend on each candidate's float details (FTZ, fusion).
        self.stable_argsort(axis, true).slice_along(..k, axis)
    }

    /// Sort and retrieve top-k **values** (largest first)
    pub fn topk_values(self, k: usize, axis: usize) -> GraphTensor {
        let top_k_idx = self.topk_indexes(k, axis);
        self.gather_elements(top_k_idx, axis)
    }

    /// Apply a cumulative reduction operation along dimensions
    ///
    /// See `cumsum` or `cummax` for usage examples.
    pub fn cumop(
        mut self,
        axes: impl ToAxes,
        op: impl Fn(GraphTensor, usize) -> GraphTensor,
        pad_elem: f32,
    ) -> Self {
        let n_dims = self.rank();
        for axis in axes.to_axes() {
            // Pad out length
            let mut kernel = vec![1.into(); n_dims];
            let mut padding = vec![(IntExpr::from(0), IntExpr::from(0)); n_dims];
            let orig_length = self.dims()[axis];
            padding[axis] = (orig_length - 1, 0.into());
            kernel[axis] = orig_length;
            self = self.pad(padding, pad_elem);
            // Unfold + removal of the non-cumulative kernel dimensions:
            // ONE macro construct, ONE apply (ruling 2026-08-26) — the
            // squeezes compose into the unfold's own map.
            let mut chain = self.unfold_view(kernel, vec![1; n_dims], vec![1; n_dims]);
            for i in (0..n_dims).rev() {
                if i != axis {
                    chain = chain.squeeze(n_dims + i);
                }
            }
            self = chain.finish();
            // apply operation along cumulative dimensions
            self = op(self, n_dims);
        }
        self
    }

    /// Apply a cumulative sum along dimensions
    pub fn cumsum(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.sum(axes), 0.)
    }

    /// Apply a cumulative max along dimensions
    pub fn cummax(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.max(axes), f32::MIN)
    }

    /// Apply a cumulative product along dimensions
    pub fn cumprod(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.prod(axes), 1.)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::tests::{assert_close, random_vec};
    use candle_core::{Device, Tensor};
    use candle_nn::ops::softmax;
    use itertools::Itertools;
    use luminal::prelude::*;
    use proptest::prelude::*;

    /// `abs()` ON AN UNSIGNED INTEGER RECORDS NOTHING (main #399's
    /// `DType::U4 | U8 | U16 => self` arm). Not a performance nicety: a
    /// recorded no-op would be a `lt` against zero, whose answer is
    /// constant-false for an unsigned type, followed by arithmetic that
    /// exists only to multiply by one.
    #[test]
    fn unsigned_abs_is_the_identity() {
        let mut cx = Graph::new();
        for dtype in [DType::U4, DType::U8, DType::U16] {
            let x = cx.tensor_dtyped(4, dtype);
            assert_eq!(
                x.abs().id,
                x.id,
                "abs() on {dtype:?} must be the identity, not a recorded op"
            );
        }
    }

    fn cummax_ref_2d(a: Tensor) -> Tensor {
        let v = a.to_vec2::<f32>().unwrap();
        let mut out = vec![vec![0.0; v[0].len()]; v.len()];
        for (i, row) in v.iter().enumerate() {
            let mut acc = f32::NEG_INFINITY;
            for (j, val) in row.iter().enumerate() {
                acc = acc.max(*val);
                out[i][j] = acc;
            }
        }
        Tensor::new(out, a.device()).unwrap()
    }

    fn cumprod_ref_2d(a: Tensor) -> Tensor {
        let v = a.to_vec2::<f32>().unwrap();
        let mut out = vec![vec![0.0; v[0].len()]; v.len()];
        for (i, row) in v.iter().enumerate() {
            let mut acc = 1.0;
            for (j, val) in row.iter().enumerate() {
                acc *= val;
                out[i][j] = acc;
            }
        }
        Tensor::new(out, a.device()).unwrap()
    }

    pub fn test_unary(
        shape: impl ToShape,
        func: impl Fn(GraphTensor) -> GraphTensor,
        ref_func: impl Fn(Tensor) -> Tensor,
    ) {
        let shape = shape
            .to_shape()
            .into_iter()
            .map(|e| e.to_usize().unwrap())
            .collect_vec();
        let mut cx = Graph::new();
        let a = cx.tensor(shape.clone());
        let b = func(a).output();

        let v = random_vec(shape.iter().copied().product());
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, v.clone().into())]);

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::new(v, &device).unwrap().reshape(shape).unwrap();
        let ref_b = ref_func(ref_a).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id).unwrap(), &ref_b.to_vec1::<f32>().unwrap())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_exp(size in 1usize..128) {
            test_unary(size, |a| a.exp(), |a| a.exp().unwrap());
        }

        #[test]
        fn test_log(size in 1usize..128) {
            test_unary(size, |a| a.log(), |a| a.log().unwrap());
        }

        #[test]
        fn test_sin(size in 1usize..128) {
            test_unary(size, |a| a.sin(), |a| a.sin().unwrap());
        }

        #[test]
        fn test_cos(size in 1usize..128) {
            test_unary(size, |a| a.cos(), |a| a.cos().unwrap());
        }

        #[test]
        fn test_relu(size in 1usize..128) {
            test_unary(size, |a| a.relu(), |a| a.relu().unwrap());
        }

        #[test]
        #[ignore = "SCALE-GATED (Step 4b): the exact-gelu model (~100 values: A&S degree-5 Horner + sign/abs + the to-Bool desugar) sends the unbounded (saturate (run)) schedule into a >1h AC/distributivity closure — needs bounded saturation scheduling before it can run as a test"]
        fn test_gelu_exact(size in 1usize..128) {
            // Exact GELU vs candle's exact erf GELU.
            test_unary(size, |a| a.gelu(), |a| a.gelu_erf().unwrap());
        }

        #[test]
        fn test_gelu_tanh_approximation(size in 1usize..128) {
            test_unary(
                size,
                |a| a.gelu_fast_tanh_approximation(),
                |a| a.gelu().unwrap(),
            );
        }

        #[test]
        fn test_swish(size in 1usize..128) {
            test_unary(size, |a| a.swish(), |a| a.silu().unwrap());
        }

        #[test]
        fn test_tanh(size in 1usize..128) {
            test_unary(size, |a| a.tanh(), |a| a.tanh().unwrap());
        }

        #[test]
        fn test_recip(size in 1usize..128) {
            test_unary(size, |a| a.reciprocal(), |a| a.recip().unwrap());
        }

        #[test]
        fn test_sqrt(size in 1usize..128) {
            test_unary(size, |a| a.sqrt(), |a| a.sqrt().unwrap());
        }

        #[test]
        fn test_square(size in 1usize..128) {
            test_unary(size, |a| a.square(), |a| a.powf(2.0).unwrap());
        }

        #[test]
        fn test_softmax(size in 1usize..128, rows in 1usize..16, cols in 1usize..16) {
            test_unary(size, |a| a.softmax(0), |a| softmax(&a, 0).unwrap());
            test_unary((rows, cols), |a| a.softmax(1), |a| softmax(&a, 1).unwrap());
        }

        #[test]
        fn test_layer_norm(size in 1usize..128) {
            test_unary(
                size,
                |a| a.layer_norm(0, 1e-5),
                |a| {
                    let meaned = (a.clone() - a.mean(0).unwrap().broadcast_as(size)).unwrap();
                    meaned
                        .powf(2.0)
                        .unwrap()
                        .mean(0)
                        .unwrap()
                        .add(&Tensor::new(1e-5_f32, a.device()).unwrap())
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .recip()
                        .unwrap()
                        .broadcast_as(size)
                        .unwrap()
                        .mul(&meaned)
                        .unwrap()
                },
            );
        }

        #[test]
        fn test_cumulative(rows in 1usize..16, cols in 1usize..16) {
            test_unary(rows, |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
            test_unary((rows, cols), |a| a.cumsum(1), |a| a.cumsum(1).unwrap());
            test_unary((rows, cols), |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
            test_unary(
                (rows, cols),
                |a| a.cumsum((0, 1)),
                |a| a.cumsum(0).unwrap().cumsum(1).unwrap(),
            );
            test_unary(
                (rows, cols),
                |a| a.cumsum((1, 0)),
                |a| a.cumsum(1).unwrap().cumsum(0).unwrap(),
            );
            test_unary((rows, cols), |a| a.cummax(1), cummax_ref_2d);
            test_unary((rows, cols), |a| a.cumprod(1), cumprod_ref_2d);
        }

        #[test]
        fn test_argmax(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.argmax(0).cast(DType::F32), |a| a.argmax(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary((rows, cols), |a| a.argmax(1).cast(DType::F32), |a| a.argmax(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_argmin(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.argmin(0).cast(DType::F32), |a| a.argmin(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary((rows, cols), |a| a.argmin(1).cast(DType::F32), |a| a.argmin(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_var(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.var(1), |a| a.var(1).unwrap());
            test_unary((rows, cols), |a| a.var(0), |a| a.var(0).unwrap());
        }

        #[test]
        fn test_std(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.std(1), |a| a.var(1).unwrap().sqrt().unwrap());
        }

    }

    /// topk_indexes rides stable_argsort → the rank scatter (the B-tail
    /// flat sugar). Concrete hand case with an exact value tie in row 1 to
    /// pin the STABLE tiebreak: first occurrence outranks its twin.
    #[test]
    fn test_topk_indexes() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 4));
        let out = x.topk_indexes(2, 1).cast(DType::F32).output();

        // row 0: [0.1, 3.0, 2.0, -1.0] → top-2 desc = idx 1, 2
        // row 1: [5.0, 5.0, 0.0, 7.0] → 7.0 at idx 3, then the 5.0 tie —
        //        stable → idx 0 before idx 1
        let x_vals = vec![0.1f32, 3.0, 2.0, -1.0, 5.0, 5.0, 0.0, 7.0];
        let expected = [1.0f32, 2.0, 3.0, 0.0];
        let rt = luminal_reference::harness::run_reference(&cx, &[(x.id, x_vals.into())]);
        let got = rt.get_f32(out.id).unwrap();
        assert_eq!(got.len(), expected.len());
        for (index, (a, b)) in got.iter().zip(expected).enumerate() {
            assert_eq!(*a, b, "element {index}: got {a} vs expected {b}");
        }
    }
}
