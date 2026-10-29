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
            | DType::U64
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
    fn elementary_constant(self, narrow: f32, wide: f64) -> GraphTensor {
        let constant = if self.dtype == DType::F64 {
            self.graph().constant_f64(wide)
        } else {
            self.graph().constant_f32(narrow).cast(self.dtype)
        };
        constant.expand_rhs(self.dims())
    }

    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self.graph().logical.op(
            LogicalOp::Log2,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self.graph().logical.op(
            LogicalOp::Exp2,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * self.elementary_constant(1.0 / f32::ln(2.), std::f64::consts::LOG2_E)).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * self.elementary_constant(f32::ln(2.), std::f64::consts::LN_2)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self.graph().logical.op(
            LogicalOp::Recip,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self.graph().logical.op(
            LogicalOp::Sin,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        (self.elementary_constant(std::f32::consts::PI / 2., std::f64::consts::FRAC_PI_2) - self)
            .sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self.graph().logical.op(
            LogicalOp::Sqrt,
            &[(self.id, self.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Round every element toward negative infinity. NaN and ±inf propagate.
    pub fn floor(self) -> GraphTensor {
        self.round_logical(LogicalOp::Floor)
    }

    /// Round every element toward positive infinity. NaN and ±inf propagate.
    pub fn ceil(self) -> GraphTensor {
        self.round_logical(LogicalOp::Ceil)
    }

    /// Round every element toward zero. NaN and ±inf propagate.
    pub fn trunc(self) -> GraphTensor {
        self.round_logical(LogicalOp::Trunc)
    }

    /// Round every element to the nearest integer; ties go to even.
    pub fn round(self) -> GraphTensor {
        self.round_logical(LogicalOp::Round)
    }

    /// Shared recorder path for the dtype-preserving rounding ops.
    fn round_logical(self, op: LogicalOp) -> GraphTensor {
        let new_id =
            self.graph()
                .logical
                .op(op, &[(self.id, self.dims())], self.dims(), self.dtype);
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

    /// Get the first index of the maximum along an axis. A NaN takes
    /// precedence over ordered values; multiple NaNs also choose the first.
    pub fn argmax(self, axis: usize) -> GraphTensor {
        let dims = self.dims();
        let extent = dims[axis];
        let mut winner = self
            .eq(self.max(vec![axis]).expand_dim(axis, extent))
            .cast(DType::Int);
        if !matches!(
            self.dtype,
            DType::Int
                | DType::I64
                | DType::U64
                | DType::I4
                | DType::U4
                | DType::I8
                | DType::U8
                | DType::I16
                | DType::U16
                | DType::Bool
        ) {
            // Ordered comparisons distinguish NaN without relying on eq's
            // less/greater decomposition. Widening here only classifies NaNs;
            // the maximum and equality above retain the original value dtype.
            let x = self.cast(DType::F32);
            let low = self
                .graph()
                .constant_f32(f32::NEG_INFINITY)
                .expand_rhs(dims.clone());
            let high = self
                .graph()
                .constant_f32(f32::INFINITY)
                .expand_rhs(dims.clone());
            let ordered = x.lt(high).cast(DType::Int) + low.lt(x).cast(DType::Int);
            let zero = self.graph().constant_i32(0).expand_rhs(dims.clone());
            let nan = ordered.eq(zero).cast(DType::Int);
            let any_nan = nan.max(vec![axis]).expand_dim(axis, extent);
            winner = any_nan.cast(DType::Bool).select(nan, winner);
        }
        // Larger positive ranks correspond to earlier indices. Non-winners
        // contribute zero; the largest winning rank yields the first index.
        // All index arithmetic stays integer, including values above 2^24.
        let ranks = self.graph().iota(dims, |c| extent - c[axis]);
        -(winner * ranks).max(vec![axis]) + extent
    }

    /// Get the indices of the min elements along an axis
    pub fn argmin(self, axis: usize) -> GraphTensor {
        (-self).argmax(axis)
    }

    /// Compute the sample variance along axes
    pub fn var(self, axes: impl ToAxes) -> GraphTensor {
        self.var_options(axes, 1)
    }

    /// Compute the variance along axes with an integer degrees-of-freedom
    /// `correction`. This is [`Self::var_correction`]: when `correction`
    /// reaches the reduced element count the divisor clamps to zero (the
    /// result is +inf or NaN), it never goes negative.
    pub fn var_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        self.var_correction(axes, correction as f64)
    }

    /// Compute the variance along `axes` with torch's real-valued
    /// degrees-of-freedom `correction`: 0 is the population variance, 1 is
    /// Bessel's sample variance, and fractional or negative values are legal.
    ///
    /// INVARIANT (torch's `WelfordOps::project`, plus `std_var_out`'s NaN
    /// fill for an empty reduction). With `N` the product of the reduced
    /// extents (possibly symbolic),
    ///
    /// ```text
    /// variance = sum((x - mean)^2) / d
    /// d = N - correction   if N > correction and N > 0
    /// d = 0                otherwise
    /// ```
    ///
    /// A zero divisor is the defined answer, not an error: a positive sum
    /// gives +inf and a zero sum (all elements equal, or no element at all)
    /// gives NaN. The divisor is never negative, so the variance never
    /// changes sign, and an empty reduction is NaN for every correction.
    /// The divisor is formed in `self`'s dtype; callers that follow torch's
    /// opmath widen half-precision operands first. An integer tensor needs
    /// an integral `correction`.
    pub fn var_correction(self, axes: impl ToAxes, correction: f64) -> GraphTensor {
        let axes = axes.to_axes();
        let n = axes.iter().map(|&i| self.dims()[i]).product::<IntExpr>();
        let mean = self
            .mean(axes.clone())
            .expand_to_shape_on_axes(self.dims(), axes.clone());
        let centered = self - mean;
        let squares = (centered * centered).sum(axes);
        let divisor = self
            .variance_divisor(n, correction)
            .expand_rhs(squares.dims());
        squares / divisor
    }

    /// The rank-0 divisor `d` of [`Self::var_correction`], in `self`'s dtype.
    fn variance_divisor(self, n: IntExpr, correction: f64) -> GraphTensor {
        let dtype = self.dtype;
        let float = matches!(
            dtype,
            DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
        );
        assert!(
            float || correction.fract() == 0.0,
            "variance of a {dtype:?} tensor needs an integral correction, got {correction}"
        );
        let scalar = |graph: &mut Graph, value: f64| match dtype {
            DType::F64 => graph.constant_f64(value),
            _ if float => graph.constant_f32(value as f32).cast(dtype),
            _ => graph.constant_i32(value as i64).cast(dtype),
        };
        // A static count evaluates the rule at authoring time, in f64.
        if let Some(count) = n.to_usize() {
            let count = count as f64;
            let degrees = if count > correction && count > 0.0 {
                count - correction
            } else {
                0.0
            };
            return scalar(self.graph(), degrees);
        }
        // A symbolic count evaluates the same rule at run time. No count
        // exceeds a NaN correction, so that divisor is zero outright; for
        // any other correction both conditions are `N > max(correction, 0)`.
        let zero = scalar(self.graph(), 0.0);
        if correction.is_nan() {
            return zero;
        }
        let count = self.graph().constant_i32(n).cast(dtype);
        let threshold = scalar(self.graph(), correction.max(0.0));
        let degrees = count - scalar(self.graph(), correction);
        count.gt(threshold).select(degrees, zero)
    }

    /// Compute the sample standard deviation along axes
    pub fn std(self, axes: impl ToAxes) -> GraphTensor {
        self.std_options(axes, 1)
    }

    /// Compute the standard deviation along axes with an integer
    /// degrees-of-freedom `correction` (see [`Self::var_options`]).
    pub fn std_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        self.var_options(axes, correction).sqrt()
    }

    /// Compute the standard deviation along axes with a real-valued
    /// degrees-of-freedom `correction` (see [`Self::var_correction`]).
    pub fn std_correction(self, axes: impl ToAxes, correction: f64) -> GraphTensor {
        self.var_correction(axes, correction).sqrt()
    }

    /// Take the absolute value.
    ///
    /// DTYPE-AWARE (main #399, re-expressed 2026-09-02). Both paths now
    /// lower to the native ternary `Select`:
    ///
    /// * UNSIGNED integers are already their own absolute value —
    ///   identity, and no ops recorded at all.
    /// * SIGNED integers use `(x < 0).select(-x, x)`. This is also CORRECT
    ///   at the signed minimum in the only sense available: `i32::MIN` has
    ///   no representable absolute value, and the negation reports that as
    ///   an overflow (the Int kernels are checked) instead of returning
    ///   `MIN` as a masked multiply would.
    /// * FLOATS also use `(x < 0).select(-x, x)`; `NaN < 0` is false, so a
    ///   NaN input passes through unchanged, matching torch.
    ///
    /// The old spellings (`x * (1 - 2*(x<0))` for ints,
    /// `relu(x) + relu(-x)` where `relu` is a masked `maximum` for floats)
    /// are sums of products: chained, they feed the integer
    /// associativity/commutativity/distributivity e-graph closure and
    /// explode saturation. `Select` is one node and is NaN/inf-safe.
    pub fn abs(self) -> GraphTensor {
        match self.dtype {
            DType::U4 | DType::U8 | DType::U16 | DType::U64 => self,
            DType::I4 | DType::I8 | DType::I16 | DType::Int | DType::I64 => self
                .lt(self
                    .graph()
                    .constant_i32(0)
                    .cast(self.dtype)
                    .expand_rhs(self.dims()))
                .select(-self, self),
            _ => self
                .lt(self
                    .graph()
                    .constant_f32(0.0)
                    .cast(self.dtype)
                    .expand_rhs(self.dims()))
                .select(-self, self),
        }
    }

    /// The sign of each element: `1` where `x > 0`, `-1` where `x < 0`, and
    /// `x` itself everywhere else.
    ///
    /// Invariant: EXACT for every input. Each nonzero value, subnormals and
    /// `±inf` included, maps to exactly `±1`; a zero keeps its own sign
    /// (`sign(-0.0) = -0.0`) and NaN propagates. It is the native `Select`
    /// over the ordered `gt`/`lt`, which are false for NaN and for both
    /// zeros, so exactly those inputs fall through to `x` unchanged. The
    /// old `x / (|x| + 1e-10)` was an approximation: NaN at `±inf`
    /// (`inf / inf`), `0.5` at `1e-10`, and not exactly `±1` anywhere
    /// below `|x| ~ 1e-3` in f32.
    ///
    /// This is the signum that odd-symmetric compositions (`erf`, `gelu`)
    /// multiply by, so `-0.0` and NaN come out of them correctly. It is NOT
    /// `torch.sign`, which is `(x > 0) - (x < 0)`: `+0` for both zeros and
    /// for NaN. The PT2 translator lowers `aten.sign` to that form itself.
    ///
    /// Integers take the same selects with integer constants. An unsigned
    /// type skips the `x < 0` arm (it is constant-false there), and Bool is
    /// its own sign (`torch.sign` on a bool tensor is the identity).
    pub fn sign(self) -> GraphTensor {
        let constant = |value: i32| -> GraphTensor {
            match self.dtype {
                DType::I4
                | DType::I8
                | DType::I16
                | DType::Int
                | DType::I64
                | DType::U4
                | DType::U8
                | DType::U16
                | DType::U64 => self
                    .graph()
                    .constant_i32(value)
                    .cast(self.dtype)
                    .expand_rhs(self.dims()),
                _ => self.elementary_constant(value as f32, value as f64),
            }
        };
        match self.dtype {
            DType::Bool => self,
            DType::U4 | DType::U8 | DType::U16 | DType::U64 => {
                self.gt(constant(0)).select(constant(1), self)
            }
            _ => {
                let zero = constant(0);
                let negative = self.lt(zero).select(constant(-1), self);
                self.gt(zero).select(constant(1), negative)
            }
        }
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
        let ranks = cmp.cast(DType::Int).sum(vec![axis]);
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
        let one = self.graph().constant_i32(1).expand_rhs(lt_count.dims());
        let val_eq = (one - lt_count) * (one - gt_count);
        let cmp = primary_count + val_eq * idx_count;

        // Scatter original indices into rank positions to get sort indices
        let ranks = cmp.sum(vec![axis]);
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

    /// Apply a cumulative sum along dimensions — one inclusive scan per axis.
    pub fn cumsum(self, axes: impl ToAxes) -> Self {
        self.scan_per_axis(axes, |axis_from_end| LogicalOp::UnspecifiedOrderScanSum {
            axis_from_end,
        })
    }

    /// Apply a cumulative max along dimensions — one inclusive scan per axis.
    pub fn cummax(self, axes: impl ToAxes) -> Self {
        self.scan_per_axis(axes, |axis_from_end| LogicalOp::UnspecifiedOrderScanMax {
            axis_from_end,
        })
    }

    /// Apply a cumulative product along dimensions — one inclusive scan per axis.
    pub fn cumprod(self, axes: impl ToAxes) -> Self {
        self.scan_per_axis(axes, |axis_from_end| LogicalOp::UnspecifiedOrderScanProd {
            axis_from_end,
        })
    }

    /// One recorded scan per axis, in the given order. A scan keeps the
    /// operand's dims and dtype, so no axis bookkeeping is needed.
    fn scan_per_axis(self, axes: impl ToAxes, op: impl Fn(usize) -> LogicalOp) -> Self {
        let (dims, mut id) = (self.dims(), self.id);
        let rank = dims.len();
        // A rank-0 scan is the value itself (torch parity): there is no axis
        // to fold along, and `rank - 1 - axis` would underflow.
        if rank == 0 {
            return self;
        }
        for axis in axes.to_axes() {
            assert!(axis < rank, "scan axis {axis} out of range for rank {rank}");
            id = self.graph().logical.op(
                op(rank - 1 - axis),
                &[(id, dims.clone())],
                dims.clone(),
                self.dtype,
            );
        }
        GraphTensor::from_id(id, dims, self.graph_ref, self.dtype)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::tests::{assert_close, assert_exact, random_vec};
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
            let x = cx.tensor(vec![4], dtype);
            assert_eq!(
                x.abs().id,
                x.id,
                "abs() on {dtype:?} must be the identity, not a recorded op"
            );
        }
    }

    /// `sign()` is EXACT: `±1` for every nonzero value, subnormals and
    /// `±inf` included; a zero keeps its own sign bit and NaN propagates.
    /// The old `x / (|x| + 1e-10)` gave NaN at `±inf`, `0.5` at `1e-10` and
    /// a fraction for every tiny value.
    #[test]
    fn sign_is_exact_at_zeros_subnormals_infinities_and_nan() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let tiny = [1e-45, 1e-30, 1e-10, 1e-4];
        let mut input = vec![-inf, -1.0, -0.0, 0.0, 1.0, inf, nan];
        let mut expected = vec![-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, nan];
        for t in tiny {
            input.extend([-t, t]);
            expected.extend([-1.0, 1.0]);
        }

        let mut cx = Graph::new();
        let x = cx.tensor(vec![input.len() + 1], DType::F32);
        let y = x.sign();
        let mut narrow: Vec<f32> = input.iter().map(|&v| v as f32).collect();
        // The smallest f32 subnormal, which `1e-45 as f32` rounds to.
        narrow.push(f32::from_bits(1));
        let rt = luminal_reference::harness::run_reference(&cx, &[(x.id, narrow.clone().into())]);
        let mut narrow_expected: Vec<f32> = expected.iter().map(|&v| v as f32).collect();
        narrow_expected.push(1.0);
        for ((&got, &want), &v) in rt
            .get_f32(y.id)
            .unwrap()
            .iter()
            .zip(&narrow_expected)
            .zip(&narrow)
        {
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "f32 sign({v:e}) = {got:e}, expected {want:e}"
            );
        }

        let mut cx = Graph::new();
        let x = cx.tensor(vec![input.len() + 1], DType::F64);
        let y = x.sign();
        let mut wide = input.clone();
        // The smallest f64 subnormal.
        wide.push(f64::from_bits(1));
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, luminal_reference::TypedBuffer::F64(wide.clone()))],
        );
        let mut wide_expected = expected;
        wide_expected.push(1.0);
        for ((&got, &want), &v) in rt
            .get_f64(y.id)
            .unwrap()
            .iter()
            .zip(&wide_expected)
            .zip(&wide)
        {
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "f64 sign({v:e}) = {got:e}, expected {want:e}"
            );
        }
    }

    /// Integer `sign()` selects integer constants: the extremes map to `±1`
    /// with no negation that could overflow, and Bool is its own sign.
    #[test]
    fn integer_sign_is_exact_and_bool_sign_is_the_identity() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![5], DType::Int);
        let y = x.sign();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, vec![i32::MIN, -7, 0, 9, i32::MAX].into())],
        );
        assert_exact(rt.get_i32(y.id).unwrap(), &[-1, -1, 0, 1, 1]);

        let mut cx = Graph::new();
        let x = cx.tensor(vec![5], DType::I64);
        let y = x.sign();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, vec![i64::MIN, -7, 0, 9, i64::MAX].into())],
        );
        assert_exact(rt.get_i64(y.id).unwrap(), &[-1, -1, 0, 1, 1]);

        let mut cx = Graph::new();
        let x = cx.tensor(vec![4], DType::Bool);
        assert_eq!(x.sign().id, x.id, "sign() on Bool must be the identity");
    }

    /// A cumulative product is exact: the sign of each prefix is carried,
    /// not reconstructed.
    #[test]
    fn cumprod_signs_are_exact() {
        let input = vec![-1.0f32, 2.0, -3.0, 4.0];
        let mut cx = Graph::new();
        let a = cx.tensor(vec![input.len()], DType::F32);
        let b = a.cumprod(vec![0]);
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, input.into())]);
        assert_exact(rt.get_f32(b.id).unwrap(), &[-1.0, -2.0, 6.0, 24.0]);
    }

    /// The reference runtime's scan IS the left-sequential fold, bit for bit.
    /// Once a NaN is seen, every later running maximum is NaN.
    #[test]
    fn cummax_propagates_nan() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![4], DType::F32);
        let b = a.cummax(vec![0]);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(a.id, vec![1.0, f32::NAN, 2.0, 3.0].into())],
        );
        let out = rt.get_f32(b.id).unwrap();
        assert_eq!(out[0], 1.0);
        assert!(out[1..].iter().all(|v| v.is_nan()), "got {out:?}");
    }

    #[test]
    fn cumsum_matches_the_sequential_fold() {
        let input = random_vec(64);
        let mut expected = Vec::with_capacity(input.len());
        let mut acc = 0.0f32;
        for x in &input {
            acc += *x;
            expected.push(acc);
        }
        let mut cx = Graph::new();
        let a = cx.tensor(vec![input.len()], DType::F32);
        let b = a.cumsum(vec![0]);
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, input.into())]);
        assert_exact(rt.get_f32(b.id).unwrap(), &expected);
    }

    /// The running maximum carries each prefix's largest value exactly.
    #[test]
    fn cummax_is_the_running_maximum() {
        let input = vec![-5.0f32, -3.0, -9.0, -1.0, -7.0, -2.0, -8.0, -4.0];
        let mut cx = Graph::new();
        let a = cx.tensor(vec![input.len()], DType::F32);
        let b = a.cummax(vec![0]);
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, input.into())]);
        assert_exact(
            rt.get_f32(b.id).unwrap(),
            &[-5.0, -3.0, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        );
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
        let a = cx.tensor(shape.clone(), DType::F32);
        let b = func(a);

        let v = random_vec(shape.iter().copied().product());
        // The returned value can be the input itself (zero padding, an
        // identity view, etc.). Bind that value explicitly instead of inferring
        // computed leaves, which need not exist for an identity operation.
        let rt = luminal_reference::harness::run_reference_bound(
            &cx,
            luminal_reference::ReferenceBindings::dense(&cx.logical, &[b.id]),
            &[(a.id, v.clone().into())],
            &[],
        );

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::new(v, &device).unwrap().reshape(shape).unwrap();
        let ref_b = ref_func(ref_a).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id).unwrap(), &ref_b.to_vec1::<f32>().unwrap())
    }

    #[test]
    fn unary_differential_harness_accepts_an_input_as_the_output() {
        test_unary([1, 1], |a| a, |a| a);
    }

    #[test]
    fn argmax_selects_first_ties_and_first_nan_on_any_axis() {
        let values = vec![
            1.,
            1.,
            1.,
            1.,
            -4.,
            -2.,
            -2.,
            -3.,
            -4.,
            -3.,
            -2.,
            -1.,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::INFINITY,
            2.,
            0.,
            -0.,
            -0.,
            0.,
            1.,
            f32::NAN,
            2.,
            f32::NAN,
            f32::NAN,
            1.,
            f32::NAN,
            3.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let expected = [0., 1., 3., 1., 0., 1., 0., 0.];
        let mut cx = Graph::new();
        let x = cx.tensor(vec![8, 4], DType::F32);
        let row = x.argmax(1).cast(DType::F32);
        let column = x.transpose(0, 1).argmax(0).cast(DType::F32);
        let rt = luminal_reference::harness::run_reference(&cx, &[(x.id, values.into())]);
        assert_exact(rt.get_f32(row.id).unwrap(), &expected);
        assert_exact(rt.get_f32(column.id).unwrap(), &expected);
    }

    #[test]
    fn argmax_keeps_integer_comparisons_exact() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![2, 4], DType::Int);
        let y = x.argmax(1).cast(DType::F32);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(
                x.id,
                vec![
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    i32::MIN,
                    i32::MAX - 1,
                    i32::MAX,
                    i32::MAX,
                    0,
                ]
                .into(),
            )],
        );
        assert_exact(rt.get_f32(y.id).unwrap(), &[0., 1.]);
        let mut cx = Graph::new();
        let x = cx.tensor(vec![1, 4], DType::I64);
        let y = x.argmax(1).cast(DType::F32);
        let base = 1_i64 << 54;
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, vec![base + 1, base + 2, base + 2, base].into())],
        );
        assert_exact(rt.get_f32(y.id).unwrap(), &[1.]);
    }

    #[test]
    fn binary64_elementary_compositions_use_binary64_constants() {
        let mut graph = Graph::new();
        let input = graph.tensor(vec![5], DType::F64);
        let outputs = [input.exp(), input.log(), input.cos()];
        let values = vec![0.125_f64, 0.5, 1.0, 2.0, 3.0];
        let runtime = luminal_reference::harness::run_reference(
            &graph,
            &[(
                input.id,
                luminal_reference::TypedBuffer::F64(values.clone()),
            )],
        );
        for (i, output) in outputs.iter().enumerate() {
            for (&actual, &x) in runtime.get_f64(output.id).unwrap().iter().zip(&values) {
                let expected = match i {
                    0 => x.exp(),
                    1 => x.ln(),
                    _ => x.cos(),
                };
                assert!(
                    (actual - expected).abs() <= 2e-14 * expected.abs().max(1.0),
                    "composition {i}, input {x}: {actual} != {expected}"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_exp(size in 1usize..128) {
            test_unary(vec![size], |a| a.exp(), |a| a.exp().unwrap());
        }

        #[test]
        fn test_log(size in 1usize..128) {
            test_unary(vec![size], |a| a.log(), |a| a.log().unwrap());
        }

        #[test]
        fn test_sin(size in 1usize..128) {
            test_unary(vec![size], |a| a.sin(), |a| a.sin().unwrap());
        }

        #[test]
        fn test_cos(size in 1usize..128) {
            test_unary(vec![size], |a| a.cos(), |a| a.cos().unwrap());
        }

        #[test]
        fn test_relu(size in 1usize..128) {
            test_unary(vec![size], |a| a.relu(), |a| a.relu().unwrap());
        }

        #[test]
        #[ignore = "SCALE-GATED (Step 4b): the exact-gelu model (~100 values: A&S degree-5 Horner + sign/abs + the to-Bool desugar) sends the unbounded (saturate (run)) schedule into a >1h AC/distributivity closure — needs bounded saturation scheduling before it can run as a test"]
        fn test_gelu_exact(size in 1usize..128) {
            // Exact GELU vs candle's exact erf GELU.
            test_unary(vec![size], |a| a.gelu(), |a| a.gelu_erf().unwrap());
        }

        #[test]
        fn test_gelu_tanh_approximation(size in 1usize..128) {
            test_unary(
                vec![size],
                |a| a.gelu_fast_tanh_approximation(),
                |a| a.gelu().unwrap(),
            );
        }

        #[test]
        fn test_swish(size in 1usize..128) {
            test_unary(vec![size], |a| a.swish(), |a| a.silu().unwrap());
        }

        #[test]
        fn test_tanh(size in 1usize..128) {
            test_unary(vec![size], |a| a.tanh(), |a| a.tanh().unwrap());
        }

        #[test]
        fn test_recip(size in 1usize..128) {
            test_unary(vec![size], |a| a.reciprocal(), |a| a.recip().unwrap());
        }

        #[test]
        fn test_sqrt(size in 1usize..128) {
            test_unary(vec![size], |a| a.sqrt(), |a| a.sqrt().unwrap());
        }

        #[test]
        fn test_square(size in 1usize..128) {
            test_unary(vec![size], |a| a.square(), |a| a.powf(2.0).unwrap());
        }

        #[test]
        fn test_softmax(size in 1usize..128, rows in 1usize..16, cols in 1usize..16) {
            test_unary(vec![size], |a| a.softmax(vec![0]), |a| softmax(&a, 0).unwrap());
            test_unary(vec![rows, cols], |a| a.softmax(vec![1]), |a| softmax(&a, 1).unwrap());
        }

        #[test]
        fn test_layer_norm(size in 1usize..128) {
            test_unary(
                vec![size],
                |a| a.layer_norm(vec![0], 1e-5),
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
            test_unary(vec![rows], |a| a.cumsum(vec![0]), |a| a.cumsum(0).unwrap());
            test_unary(vec![rows, cols], |a| a.cumsum(vec![1]), |a| a.cumsum(1).unwrap());
            test_unary(vec![rows, cols], |a| a.cumsum(vec![0]), |a| a.cumsum(0).unwrap());
            test_unary(
                vec![rows, cols],
                |a| a.cumsum(vec![0, 1]),
                |a| a.cumsum(0).unwrap().cumsum(1).unwrap(),
            );
            test_unary(
                vec![rows, cols],
                |a| a.cumsum(vec![1, 0]),
                |a| a.cumsum(1).unwrap().cumsum(0).unwrap(),
            );
            test_unary(vec![rows, cols], |a| a.cummax(vec![1]), cummax_ref_2d);
            test_unary(vec![rows, cols], |a| a.cumprod(vec![1]), cumprod_ref_2d);
        }

        #[test]
        fn test_argmax(rows in 1usize..16, cols in 1usize..16) {
            test_unary(vec![rows, cols], |a| a.argmax(0).cast(DType::F32), |a| a.argmax(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary(vec![rows, cols], |a| a.argmax(1).cast(DType::F32), |a| a.argmax(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_argmin(rows in 1usize..16, cols in 1usize..16) {
            test_unary(vec![rows, cols], |a| a.argmin(0).cast(DType::F32), |a| a.argmin(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary(vec![rows, cols], |a| a.argmin(1).cast(DType::F32), |a| a.argmin(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_var(rows in 1usize..16, cols in 1usize..16) {
            test_unary(vec![rows, cols], |a| a.var(vec![1]), |a| a.var(1).unwrap());
            test_unary(vec![rows, cols], |a| a.var(vec![0]), |a| a.var(0).unwrap());
        }

        #[test]
        fn test_std(rows in 1usize..16, cols in 1usize..16) {
            test_unary(vec![rows, cols], |a| a.std(vec![1]), |a| a.var(1).unwrap().sqrt().unwrap());
        }

    }

    /// topk_indexes rides stable_argsort → the rank scatter (the B-tail
    /// flat sugar). Concrete hand case with an exact value tie in row 1 to
    /// pin the STABLE tiebreak: first occurrence outranks its twin.
    #[test]
    fn test_topk_indexes() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![2, 4], DType::F32);
        let out = x.topk_indexes(2, 1).cast(DType::F32);

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

    /// Independent two-pass variance in f64 with torch's divisor rule.
    fn torch_variance(values: &[f32], correction: f64) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
        let squares: f64 = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
        let degrees = if n > correction && n > 0.0 {
            n - correction
        } else {
            0.0
        };
        squares / degrees
    }

    fn assert_same_float(got: f32, expected: f64, what: &str) {
        let expected = expected as f32;
        let same = (got.is_nan() && expected.is_nan())
            || got == expected
            || (got - expected).abs() <= 1e-5 * expected.abs().max(1.0);
        assert!(same, "{what}: got {got}, expected {expected}");
    }

    /// `var_correction` divides by torch's `max(0, N - correction)` for a
    /// real-valued correction (fractional, negative, N, past N, NaN), on a
    /// static extent and on a symbolic one, and the integer `var_options`
    /// wrapper clamps the same way instead of dividing by a negative count.
    #[test]
    fn var_correction_divides_by_clamped_degrees_of_freedom() {
        let rows = [
            [1.0f32, 2.0, 4.0, 8.0],
            [0.5, -1.0, 3.0, 2.5],
            [2.0, 2.0, 2.0, 5.0],
        ];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let corrections = [0.0, 0.5, 1.0, 1.5, -1.0, 4.0, 5.0, f64::NAN];
        for symbolic in [false, true] {
            let mut cx = Graph::new();
            let cols = if symbolic {
                cx.set_dim('a', 4);
                IntExpr::from('a')
            } else {
                IntExpr::from(4)
            };
            let x = cx.tensor(vec![IntExpr::from(3), cols], DType::F32);
            let outputs: Vec<_> = corrections
                .iter()
                .map(|&c| {
                    (
                        c,
                        x.var_correction(vec![1], c),
                        x.std_correction(vec![1], c),
                    )
                })
                .collect();
            let unsigned = x.var_options(vec![1], 6);
            let rt = luminal_reference::harness::run_reference(&cx, &[(x.id, flat.clone().into())]);
            for (c, var, std) in outputs {
                let var = rt.get_f32(var.id).unwrap();
                let std = rt.get_f32(std.id).unwrap();
                for (r, row) in rows.iter().enumerate() {
                    let expected = torch_variance(row, c);
                    let what = format!("symbolic={symbolic} correction={c} row {r}");
                    assert_same_float(var[r], expected, &format!("var {what}"));
                    assert_same_float(std[r], expected.sqrt(), &format!("std {what}"));
                }
            }
            let unsigned = rt.get_f32(unsigned.id).unwrap();
            assert!(unsigned.iter().all(|v| *v == f32::INFINITY), "{unsigned:?}");
        }
    }

    /// An empty reduction is NaN for every correction, including a negative
    /// one whose `N - correction` would be positive (torch fills NaN).
    #[test]
    fn var_correction_of_an_empty_reduction_is_nan() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![3, 0], DType::F32);
        let outputs: Vec<_> = [0.0, 1.0, 0.5, -1.0]
            .iter()
            .map(|&c| x.var_correction(vec![1], c))
            .collect();
        let rt =
            luminal_reference::harness::run_reference(&cx, &[(x.id, Vec::<f32>::new().into())]);
        for var in outputs {
            let var = rt.get_f32(var.id).unwrap();
            assert_eq!(var.len(), 3);
            assert!(var.iter().all(|v| v.is_nan()), "{var:?}");
        }
    }
}
