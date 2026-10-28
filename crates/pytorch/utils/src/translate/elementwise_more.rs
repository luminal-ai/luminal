//! Elementwise specials, squeeze, and triangle lowerings (port batch 7).
//!
//! Ported from the parked HLIR translator's `binary.rs`/`unary.rs`/
//! `tensor.rs`/`complex.rs` helpers on the recorder frontend.
//!
//! NaN/infinity-sensitive sites use the native ternary `LogicalSelect`
//! (`condition.select(a, b)`): it is a true selection, so an unselected NaN
//! or infinity is never multiplied by zero, and unlike the arithmetic mask
//! `a*mask + b*(1-mask)` it puts no sum of products into the e-graph.
//! The atan origin is additionally steered to a finite ratio (`0/1` rather
//! than `0/0`) so the intermediate angle is finite before the final select.

use anyhow::{Result, bail, ensure};
use luminal::prelude::*;

use super::{Translator, util};
use crate::pt2_schema::Node;

/// The axes torch's `squeeze` removes from a tensor with extents `dims`,
/// ascending and distinct. `requested` is `None` for `squeeze.default`
/// (every axis is a candidate), else the dims `squeeze.dim` / `squeeze.dims`
/// name.
///
/// INVARIANT (torch semantics): squeeze removes exactly the requested axes
/// whose extent is PROVABLY 1, and every other requested axis is a no-op.
/// "Provably 1" is `to_usize() == Some(1)`: the extent evaluates to 1 with
/// no dimension bound. A static extent other than 1 stays, as torch's
/// `squeeze(dim)` leaves it alone. A symbolic extent stays too. Export only
/// drops an axis it knows is 1, and then it records the literal 1, so a
/// symbol here names an axis torch kept (its size-oblivious reasoning
/// treats a dynamic extent as not 1). `Translator::squeeze_axes` checks
/// the result against the rank the export declares, so a squeeze that
/// broke this assumption would be refused rather than mistranslated.
///
/// Requested dims wrap against `max(rank, 1)`, as torch's `maybe_wrap_dim`
/// does for a scalar. A rank-0 tensor accepts dim 0 and -1 and comes back
/// unchanged. A dim out of that range, or one named twice, is an error, as
/// it is in torch.
pub(super) fn squeezable_axes(dims: &[IntExpr], requested: Option<&[i64]>) -> Result<Vec<usize>> {
    let rank = dims.len();
    let candidates: Vec<usize> = match requested {
        None => (0..rank).collect(),
        Some(requested) => {
            let wrap = rank.max(1) as i64;
            let mut axes = Vec::with_capacity(requested.len());
            for &dim in requested {
                let axis = if dim < 0 { dim + wrap } else { dim };
                ensure!(
                    (0..wrap).contains(&axis),
                    "squeeze dim {dim} out of range [{}, {}] for rank {rank}",
                    -wrap,
                    wrap - 1
                );
                let axis = axis as usize;
                ensure!(
                    !axes.contains(&axis),
                    "squeeze dim {axis} appears multiple times in the list of dims"
                );
                axes.push(axis);
            }
            axes
        }
    };
    let mut axes: Vec<usize> = candidates
        .into_iter()
        .filter(|&axis| axis < rank && dims[axis].to_usize() == Some(1))
        .collect();
    axes.sort_unstable();
    Ok(axes)
}

/// Remove `axes` (ascending and distinct, each of extent 1: see
/// [`squeezable_axes`]) from `tensor`. The highest goes first, so every
/// remaining index still names its original axis.
pub(super) fn remove_axes(tensor: GraphTensor, axes: &[usize]) -> GraphTensor {
    axes.iter().rev().fold(tensor, |t, &axis| t.squeeze(axis))
}

impl Translator<'_> {
    /// Non-arithmetic `select`: the native ternary `LogicalSelect`, preceded
    /// by the broadcast that establishes a common shape/dtype for the
    /// branches.
    fn real_select(
        &mut self,
        condition: GraphTensor,
        if_true: GraphTensor,
        if_false: GraphTensor,
    ) -> GraphTensor {
        let (if_true, condition) = util::broadcast_binary(if_true, condition);
        let (if_true, if_false) = util::broadcast_binary(if_true, if_false);
        let condition = if condition.dtype == DType::Bool {
            condition
        } else {
            condition.cast(DType::Bool)
        };
        condition.select(if_true, if_false)
    }

    /// `value == 0` that, unlike `util::is_zero`, is false for NaN.
    fn exact_is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let nonzero = self.bool_or(value.lt(zero), value.gt(zero));
        let nan = self.is_nan(value);
        let nonzero_or_nan = self.bool_or(nonzero, nan);
        self.bool_not(nonzero_or_nan)
    }

    /// Both operands promoted to the dtype the op computes in and
    /// broadcast; numeric literals are accepted on either side.
    fn promoted_binary_inputs(&mut self, node: &Node) -> Result<(GraphTensor, GraphTensor)> {
        let dtype = self.compute_dtype(node)?;
        let a = match self.optional_tensor_operand(&node.inputs[0])? {
            Some(tensor) => tensor,
            None => self.scalar(&node.inputs[0], dtype)?,
        };
        let b = match self.optional_tensor_operand(&node.inputs[1])? {
            Some(tensor) => tensor,
            None => self.scalar(&node.inputs[1], dtype)?,
        };
        let a = if a.dtype == dtype { a } else { a.cast(dtype) };
        let b = if b.dtype == dtype { b } else { b.cast(dtype) };
        Ok(util::broadcast_binary(a, b))
    }

    /// Range-reduced odd Taylor series for atan; mirrors `unary::real_atan`
    /// (private to that module) so `atan2` can share it.
    fn exact_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let one = self.constant_like(x, 1.0);
        // `x_safe` keeps the unselected reciprocal branch finite at `x = 0`.
        // Correctness does not depend on it: `real_select` is the native
        // `LogicalSelect`, a true selection, so an unselected `1/0 = inf`
        // never reaches the result, and the reciprocal is selected only
        // where `x > 1`, where `x_safe == x`.
        let x_safe = x + self.exact_is_zero(x).cast(x.dtype);
        let reciprocal_branch = x.gt(one);
        let reduced = self.real_select(reciprocal_branch, x_safe.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.real_select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.real_select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.real_select(reciprocal_branch, half_pi - base, base);
        self.copy_sign(angle, input)
    }

    /// `atan2(y, x)` with the full branch-cut, infinity and signed-zero
    /// cases. The origin is steered so `y/x` is `0/1` there rather than
    /// `0/0`, keeping the intermediate angle finite before the final select.
    fn exact_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let x_zero = self.exact_is_zero(x);
        let y_zero = self.exact_is_zero(y);
        let both_zero = self.bool_and(x_zero, y_zero);
        // Invariant: away from the origin the ratio divides by `x` itself,
        // so `y/x` and the quadrant correction below (`signbit(x)`) read
        // the same sign of `x`. Steer the origin by selection: adding the
        // `both_zero` indicator would compute `-0.0 + 0.0 = +0.0` and turn
        // `atan2(-1, -0.0)` into `-3pi/2`.
        let one = self.constant_like(x, 1.0);
        let safe_x = self.real_select(both_zero, one, x);
        let ratio = y / safe_x;
        let mut angle = self.exact_atan(ratio);
        let x_negative = self.signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.copy_sign(pi, y);
        angle = self.real_select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = self.bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.real_select(x_negative, three_quarters, quarter);
        let infinite_angle = self.copy_sign(infinite_angle, y);
        angle = self.real_select(both_inf, infinite_angle, angle);

        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.copy_sign(zero, y);
        let zero_angle = self.real_select(x_negative, signed_pi, signed_zero);
        self.real_select(both_zero, zero_angle, angle)
    }

    pub(super) fn translate_atan2(&mut self, node: &Node) -> Result<GraphTensor> {
        let (y, x) = self.promoted_binary_inputs(node)?;
        let output_dtype = y.dtype;
        let (y, x) = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            (y.cast(DType::F32), x.cast(DType::F32))
        } else {
            (y, x)
        };
        Ok(self.exact_atan2(y, x).cast(output_dtype))
    }

    pub(super) fn translate_copysign(&mut self, node: &Node, scalar: bool) -> Result<GraphTensor> {
        if scalar {
            let dtype = self.compute_dtype(node)?;
            let raw = self.operand(&node.inputs[0])?;
            let magnitude = if raw.dtype == dtype {
                raw
            } else {
                raw.cast(dtype)
            };
            // `ieee_abs`, not `abs()`: `abs(-0.0)` keeps the sign bit, so
            // `copysign(-0.0, 0.0)` would come out `-0.0`.
            let magnitude = self.ieee_abs(magnitude);
            let sign = node.inputs[1]
                .arg
                .as_int()
                .map(|value| value as f64)
                .or_else(|| node.inputs[1].arg.as_float())
                .ok_or_else(|| anyhow::anyhow!("{} requires a numeric RHS", node.target))?;
            // Egglog's scalar domain equates +0.0 and -0.0, so keep the
            // compile-time sign structurally instead of as a constant.
            return Ok(if sign.is_sign_negative() {
                magnitude * -1.0
            } else {
                magnitude
            });
        }
        let (magnitude, sign) = self.promoted_binary_inputs(node)?;
        Ok(self.copy_sign(magnitude, sign))
    }

    /// `torch.fmax` / `torch.fmin`: the larger / smaller operand, where a
    /// NaN operand is skipped rather than propagated.
    ///
    /// Invariant (what torch's CPU kernel returns for every dtype with IEEE
    /// NaN):
    /// - exactly one operand NaN: the other operand; both NaN: NaN;
    /// - otherwise the larger / smaller operand in the order
    ///   `-inf < ... < -0.0 < +0.0 < ... < +inf`, so a pair of zeros of
    ///   opposite sign gives `+0.0` from fmax and `-0.0` from fmin.
    ///
    /// For non-NaN operands, `a` is picked as the smaller (`b` as the larger)
    /// iff `a < b`, or `a` has its sign bit set and `b < a` does not hold.
    /// The second clause adds exactly the ties `a == b` with a sign-bit-set
    /// `a`: `(-0.0, +0.0)`, which it orders, and pairs with equal encodings
    /// (`(-0.0, -0.0)`, `(-2.0, -2.0)`), where both picks are the same
    /// value. Every choice is the native `LogicalSelect`, which never blends
    /// the branch it drops, so a NaN or infinite operand cannot leak into
    /// the result.
    /// Integer and Bool operands have neither NaN nor signed zeros, so the
    /// plain `maximum` / `minimum` already is the rule.
    pub(super) fn translate_fmax_fmin(&mut self, node: &Node, max: bool) -> Result<GraphTensor> {
        let (a, b) = self.promoted_binary_inputs(node)?;
        if !matches!(
            a.dtype,
            DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
        ) {
            return Ok(if max { a.maximum(b) } else { a.minimum(b) });
        }
        let a_negative_tie = self.signbit(a).logical_and(b.lt(a).logical_not());
        let a_first = a.lt(b).logical_or(a_negative_tie);
        let ordered = if max {
            a_first.select(b, a)
        } else {
            a_first.select(a, b)
        };
        let b_skipped = b.is_nan().select(a, ordered);
        Ok(a.is_nan().select(b, b_skipped))
    }

    /// `hypot(a, b) = sqrt(a^2 + b^2)` with the C99 Annex F special values
    /// torch follows, in this order of precedence:
    ///   1. if either operand is +-inf the result is +inf, even when the
    ///      other is NaN;
    ///   2. otherwise, if either operand is NaN the result is NaN;
    ///   3. otherwise `hypot(+-0, +-0) = +0`, and every other pair is the
    ///      overflow-safe `larger * sqrt(1 + (smaller / larger)^2)`.
    ///
    /// INVARIANT: each case is a native select (`real_select`, a true
    /// selection) on a predicate of the operands themselves, nested so the
    /// outer case wins. The scaled formula is therefore only observed on
    /// finite operands off the origin, where the ratio is in `[0, 1]`: no
    /// intermediate overflows (only the final product can, exactly when the
    /// true result is out of the dtype's range), and an underflowing
    /// `ratio^2` only drops a term below `1`'s precision. Its `inf / inf`
    /// and `0 / 0` elsewhere are selected away, never multiplied by a mask.
    /// The origin's `+0` is a constant rather than the formula's value:
    /// core `abs` is `select(x < 0, -x, x)`, which keeps `-0.0`.
    pub(super) fn translate_hypot(&mut self, node: &Node) -> Result<GraphTensor> {
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        let lhs = self.real_abs(lhs);
        let rhs = self.real_abs(rhs);
        let larger = lhs.maximum(rhs);
        let smaller = lhs.minimum(rhs);
        let zero = self.constant_like(larger, 0.0);
        let one = self.constant_like(larger, 1.0);

        // The ratio's divisor must have a finite, normal reciprocal: core
        // division below F64 is `smaller * (1 / larger)`, and in F32
        // `1 / larger` overflows for a subnormal `larger` (and is itself
        // subnormal, so inexact, above 2^126). The ratio is invariant under
        // scaling both magnitudes by one power of two, which is exact while
        // they stay normal, so scale before dividing: by 2^64 below 1, by
        // 2^-64 from 1 up, which puts an F32 `larger` in [2^-85, 2^64). A
        // `smaller` the down-scaling takes out of the normal range is below
        // 2^-62 * `larger`, where `ratio^2` is below `1`'s precision anyway.
        // F64 divides directly; the same exact scaling is harmless there.
        // The result multiplies the unscaled `larger`, so the scaling adds
        // no rounding.
        let up = self.constant_like(larger, 2f64.powi(64));
        let down = self.constant_like(larger, 2f64.powi(-64));
        let scale = self.real_select(larger.lt(one), up, down);
        let ratio = (smaller * scale) / (larger * scale);
        let scaled = larger * (one + ratio.square()).sqrt();
        // IEEE `==` (false for NaN): both magnitudes are zero.
        let origin = larger.eq(zero);
        let finite = self.real_select(origin, zero, scaled);

        let any_nan = self.bool_or(lhs.is_nan(), rhs.is_nan());
        let nan = self.constant_like(larger, f64::NAN);
        let result = self.real_select(any_nan, nan, finite);

        let lhs_inf = self.is_inf(lhs);
        let rhs_inf = self.is_inf(rhs);
        let any_inf = self.bool_or(lhs_inf, rhs_inf);
        let infinity = self.constant_like(larger, f64::INFINITY);
        Ok(self.real_select(any_inf, infinity, result))
    }

    pub(super) fn translate_gcd(&mut self, node: &Node) -> Result<GraphTensor> {
        // The old translator unrolled a 128-round Euclidean loop. Two things
        // had to change before it could plan on this branch:
        //   1. the arithmetic masks (`f*lhs + active*rhs`) became native
        //      selects — a sum of products fed the integer AC + distributivity
        //      e-graph closure and diverged saturation; and
        //   2. the divisor became `rhs.maximum(1)` rather than
        //      `finished.select(1, rhs)`, so its lower bound (`1`) is derivable
        //      from the structural max rule in `logical_op/select`, which is
        //      what discharges the proof gate on I64 `trunc_rem`.
        // Both are compile-time (egglog) facts; no runtime logic is added.
        self.translate_gcd_unrolled(node)
    }

    fn translate_gcd_unrolled(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        // The unroll is exact only if it runs every division Euclid can
        // need; an unfinished state is a wrong answer, not an approximation.
        // The round count is a function of the operands' dtype, taken
        // before the widening cast below (see `gcd_euclid_rounds`).
        let rounds = gcd_euclid_rounds(gcd_magnitude_bound(lhs.dtype));
        let mut lhs = lhs.cast(DType::I64);
        let mut rhs = rhs.cast(DType::I64);
        let one = self.constant_like(lhs, 1.0);
        let zero = self.constant_like(lhs, 0.0);
        for _ in 0..rounds {
            let finished = self.exact_is_zero(rhs);
            // Zero-free divisor: `maximum(|rhs|, 1)` is `|rhs|` whenever
            // `rhs != 0` and `1` when `rhs == 0`. Euclid is invariant under
            // the divisor's sign, so `|rhs|` is correct for signed inputs,
            // and the structural max bound rule derives its lower bound `1`
            // statically — the proof the I64 `trunc_rem` needs. (The previous
            // `finished.select(1, rhs)` was correct but had only the generic
            // union bounds, which included 0.)
            let safe_rhs = rhs.abs().maximum(one);
            let remainder = lhs.trunc_rem(safe_rhs);
            lhs = finished.select(lhs, rhs);
            rhs = finished.select(zero, remainder);
        }
        Ok(lhs.abs().cast(output_dtype))
    }

    pub(super) fn translate_exp2(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(x.dtype);
        let x = if x.dtype == dtype { x } else { x.cast(dtype) };
        Ok(x.exp2())
    }

    pub(super) fn translate_log2(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(x.dtype);
        let x = if x.dtype == dtype { x } else { x.cast(dtype) };
        Ok(x.log2())
    }

    pub(super) fn translate_isnan(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        Ok(self.is_nan(a))
    }

    pub(super) fn translate_leaky_relu(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(value.dtype);
        let value = if value.dtype == dtype {
            value
        } else {
            value.cast(dtype)
        };
        let slope = self.get_float_arg(node, 1).unwrap_or(0.01);
        let zero = self.constant_like(value, 0.0);
        let negative = value * self.constant_like(value, slope);
        Ok(self.real_select(value.gt(zero), value, negative))
    }

    pub(super) fn translate_bitwise(
        &mut self,
        node: &Node,
        or: bool,
        xor: bool,
    ) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(a.dtype);
        anyhow::ensure!(
            matches!(
                dtype,
                DType::Bool
                    | DType::I8
                    | DType::U8
                    | DType::I16
                    | DType::Int
                    | DType::I64
                    | DType::U64
            ),
            "bitwise operations require integer or Bool operands"
        );
        let b = if let Some(b) = self.optional_tensor_operand(&node.inputs[1])? {
            b
        } else {
            self.scalar(&node.inputs[1], dtype)?
        };
        let (a, b) = util::broadcast_binary(a.cast(dtype), b.cast(dtype));
        let dims = a.dims();
        let (a, b) = if dtype == DType::Bool {
            (a, b)
        } else {
            (a.bits_to_bools(), b.bits_to_bools())
        };
        let value = if or {
            a.logical_or(b)
        } else if xor {
            a.logical_or(b).logical_and(a.logical_and(b).logical_not())
        } else {
            a.logical_and(b)
        };
        Ok(if dtype == DType::Bool {
            value
        } else {
            value.bools_to_bits(dtype).squeeze(dims.len())
        })
    }

    /// `squeeze.default`, `squeeze.dim` and `squeeze.dims`: remove the
    /// axes [`Self::squeeze_axes`] selects.
    pub(super) fn translate_squeeze(&mut self, node: &Node, target: &str) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let axes = self.squeeze_axes(node, target, &x.dims())?;
        Ok(remove_axes(x, &axes))
    }

    /// The axes a squeeze overload removes from an operand with extents
    /// `dims`: the requested axes that [`squeezable_axes`] proves are 1.
    /// The real and the complex lowerings both take their axes from here,
    /// so all three overloads share one rule.
    ///
    /// The export's declared output rank is checked against the result. An
    /// axis torch removed that the rule keeps (or the reverse) is a loud
    /// error, not a silently different shape.
    pub(super) fn squeeze_axes(
        &self,
        node: &Node,
        target: &str,
        dims: &[IntExpr],
    ) -> Result<Vec<usize>> {
        let requested = match target {
            "squeeze.default" => None,
            "squeeze.dim" => Some(vec![self.get_int_arg(node, 1)?]),
            "squeeze.dims" => Some(self.get_ints_arg(node, 1)?),
            other => bail!("{other} is not a squeeze overload"),
        };
        let axes = squeezable_axes(dims, requested.as_deref())?;
        if let Some(name) = Self::tensor_output_names(node).first()
            && let Ok(meta) = self.tensor_meta(name)
        {
            let rank = dims.len() - axes.len();
            ensure!(
                rank == meta.sizes.len(),
                "{}: removing axes {axes:?} of extents {dims:?} gives rank {rank}, \
                 but the export declares rank {}",
                node.target,
                meta.sizes.len()
            );
        }
        Ok(axes)
    }

    pub(super) fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let diagonal = self.get_int_arg(node, 1).unwrap_or(0);
        let dims = x.dims();
        anyhow::ensure!(dims.len() >= 2, "tril/triu requires a matrix input");
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        // Compare coordinates inside the iota expression: extents remain symbolic,
        // and only the 0/1 predicate is materialized (no floating-point indices).
        let mut mask = self
            .cx
            .iota(vec![rows, cols], |c| {
                let offset = c[1] - c[0];
                if upper {
                    offset.gte(diagonal)
                } else {
                    IntExpr::from(diagonal).gte(offset)
                }
            })
            .cast(DType::Bool);
        for i in (0..dims.len() - 2).rev() {
            mask = mask.expand_dim(0, dims[i]);
        }
        let zero = self.cx.constant_i32(0).cast(x.dtype).expand_rhs(dims);
        // Multiplication would leak masked NaNs/infinities and lose signed zero.
        Ok(mask.select(x, zero))
    }
}

/// The largest magnitude an operand of `dtype` carries into gcd's I64
/// Euclid loop. An integer dtype is bounded by its own range. Every other
/// dtype (U64, and anything non-integer) reaches the loop through the cast
/// to I64, whose range bounds every magnitude by `2^63` (that of
/// `i64::MIN`); that bound is also the safe answer for any dtype added later.
pub(super) fn gcd_magnitude_bound(dtype: DType) -> u64 {
    match dtype {
        DType::Bool => 1,
        DType::I4 => 1 << 3,
        DType::U4 => (1 << 4) - 1,
        DType::I8 => 1 << 7,
        DType::U8 => u8::MAX as u64,
        DType::I16 => 1 << 15,
        DType::U16 => u16::MAX as u64,
        DType::Int => 1 << 31,
        _ => 1 << 63,
    }
}

/// The number of Euclid divisions after which every pair with
/// `|a|, |b| <= max_magnitude` has reached `b == 0`.
///
/// Lamé's theorem: if `a >= b > 0` and Euclid takes `n` divisions, then
/// `b >= F(n+1)` and `a >= F(n+2)`, with `F(1) = F(2) = 1`. When `|a| < |b|`
/// the first division only swaps the pair (`a rem b == a`), which costs one
/// more. So, with `K` the largest index such that `F(K) <= max_magnitude`,
/// every pair finishes within `K - 1` divisions, and the swapped consecutive
/// pair `(F(K-1), F(K))` needs exactly that many, so the bound is tight.
/// The truncated remainder by `|b|` has magnitude `|a| mod |b|`, so signs
/// never change the count.
///
/// I64 (`2^63`, `K = 92`) needs 91 rounds; Int (`2^31`, `K = 46`) 45;
/// I16 22; U8 12; I8 10.
pub(super) fn gcd_euclid_rounds(max_magnitude: u64) -> usize {
    // `(previous, current) = (F(index - 1), F(index))`, from `index = 2`.
    let (mut previous, mut current, mut index) = (1u64, 1u64, 2usize);
    loop {
        match previous.checked_add(current) {
            Some(next) if next <= max_magnitude => {
                (previous, current) = (current, next);
                index += 1;
            }
            _ => return index - 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntArg, IntsArg,
        Node, NodeInput, Signature, TensorArg, TensorMeta, TensorName, TensorRef,
    };
    use luminal::prelude::IntExpr;
    use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
    use rustc_hash::FxHashMap;

    use super::squeezable_axes;

    fn tensor_ref(name: &str) -> TensorRef {
        TensorRef {
            as_tensor: Some(TensorName {
                name: name.to_string(),
            }),
            as_tensors: None,
            as_sym_int: None,
            as_sym_float: None,
            as_sym_bool: None,
        }
    }

    fn tensor_arg(name: &str) -> Argument {
        Argument::Tensor(TensorArg {
            as_tensor: TensorName {
                name: name.to_string(),
            },
        })
    }

    fn input(name: &str, arg: Argument) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            arg,
            kind: 1,
        }
    }

    fn flt(name: &str, value: f64) -> NodeInput {
        input(name, Argument::Float(FloatArg { as_float: value }))
    }

    fn int(name: &str, value: i64) -> NodeInput {
        input(name, Argument::Int(IntArg { as_int: value }))
    }

    fn ints(name: &str, values: &[i64]) -> NodeInput {
        input(
            name,
            Argument::Ints(IntsArg {
                as_ints: values.to_vec(),
            }),
        )
    }

    fn sizes(values: &[i64]) -> Vec<DimSize> {
        values
            .iter()
            .map(|value| DimSize::Int(DimInt { as_int: *value }))
            .collect()
    }

    /// Build a one-node, one-input (`x`) / one-output (`y`) program.
    fn program(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        in_shape: &[i64],
        out_shape: &[i64],
    ) -> ParsedPT2 {
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "x".to_string(),
            TensorMeta {
                dtype,
                sizes: sizes(in_shape),
                strides: None,
                storage_offset: None,
                layout: None,
                device: None,
            },
        );
        tensor_values.insert(
            "y".to_string(),
            TensorMeta {
                dtype,
                sizes: sizes(out_shape),
                strides: None,
                storage_offset: None,
                layout: None,
                device: None,
            },
        );
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: Graph {
                    inputs: vec![tensor_ref("x")],
                    outputs: vec![tensor_ref("y")],
                    nodes: vec![Node {
                        target: target.to_string(),
                        inputs: args,
                        outputs: vec![tensor_ref("y")],
                    }],
                    tensor_values,
                    sym_int_values: HashMap::new(),
                },
                signature: Signature {
                    input_specs: Vec::new(),
                    output_specs: Vec::new(),
                },
            },
            range_constraints: HashMap::new(),
        };
        ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        }
    }

    /// Translate a single node with input `x` and output `y`, returning
    /// whether translation (including recording) succeeded.
    fn translates(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        in_shape: &[i64],
        out_shape: &[i64],
    ) -> bool {
        crate::translate::translate(&program(target, args, dtype, in_shape, out_shape)).is_ok()
    }

    /// Translate and execute on the reference runtime, returning the F32
    /// output for the staged F32 `x` input.
    fn run_f32(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        shape: &[i64],
        x: &[f32],
    ) -> Vec<f32> {
        let parsed = program(target, args, dtype, shape, shape);
        let translation = crate::translate::translate(&parsed).expect("translate");
        let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
        let dims: luminal::shape::DynMap = translation.dims.iter().map(|(s, v)| (*s, *v)).collect();
        let input = &translation.inputs[0];
        let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
        data.insert(input.tensor, TypedBuffer::F32(x.to_vec()));
        runtime
            .search(
                &luminal::shape::SymbolBounds::exact(&dims).unwrap(),
                &dims,
                &data,
                &harness_search_options(),
            )
            .expect("search");
        runtime.set_data(input.tensor, TypedBuffer::F32(x.to_vec()));
        runtime.execute().expect("execute");
        translation
            .outputs
            .first()
            .map(|output| runtime.get_f32(output.tensor).expect("f32 output").clone())
            .expect("one output")
    }

    /// Translate and execute a two-tensor binary node `target(x, w)` on the
    /// reference runtime, returning the F32 output.
    fn run_binary_f32(target: &str, shape: &[i64], x: &[f32], w: &[f32]) -> Vec<f32> {
        let mut parsed = program(
            target,
            binary(input("other", tensor_arg("w"))),
            F32,
            shape,
            shape,
        );
        let graph = &mut parsed.program.graph_module.graph;
        graph.inputs.push(tensor_ref("w"));
        let x_meta = graph.tensor_values["x"].clone();
        graph.tensor_values.insert("w".to_string(), x_meta);
        let translation = crate::translate::translate(&parsed).expect("translate");
        let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
        let dims: luminal::shape::DynMap = translation.dims.iter().map(|(s, v)| (*s, *v)).collect();
        let staged = |name: &str| match name {
            "x" => x.to_vec(),
            "w" => w.to_vec(),
            other => panic!("unexpected input {other}"),
        };
        let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
        for bound in &translation.inputs {
            data.insert(bound.tensor, TypedBuffer::F32(staged(&bound.graph_name)));
        }
        runtime
            .search(
                &luminal::shape::SymbolBounds::exact(&dims).unwrap(),
                &dims,
                &data,
                &harness_search_options(),
            )
            .expect("search");
        for bound in &translation.inputs {
            runtime.set_data(bound.tensor, TypedBuffer::F32(staged(&bound.graph_name)));
        }
        runtime.execute().expect("execute");
        translation
            .outputs
            .first()
            .map(|output| runtime.get_f32(output.tensor).expect("f32 output").clone())
            .expect("one output")
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tol, "index {i}: got {a}, expected {e}");
        }
    }

    const F32: u32 = 7;
    const I64: u32 = 5;

    fn binary(second: NodeInput) -> Vec<NodeInput> {
        vec![input("self", tensor_arg("x")), second]
    }

    #[test]
    fn elementwise_specials_record() {
        assert!(translates(
            "torch.ops.aten.atan2.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.copysign.Tensor",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.copysign.Scalar",
            binary(flt("other", -2.0)),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        for target in ["torch.ops.aten.fmax.default", "torch.ops.aten.fmin.default"] {
            assert!(translates(
                target,
                binary(input("other", tensor_arg("x"))),
                F32,
                &[2, 2],
                &[2, 2],
            ));
        }
        assert!(translates(
            "torch.ops.aten.hypot.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        // gcd translates through the select-based Euclidean unroll; it plans
        // when the caller attests non-negative input ranges.
        assert!(translates(
            "torch.ops.aten.gcd.default",
            binary(input("other", tensor_arg("x"))),
            I64,
            &[2, 2],
            &[2, 2],
        ));
    }

    #[test]
    fn small_unary_specials_record() {
        for target in ["torch.ops.aten.exp2.default", "torch.ops.aten.log2.default"] {
            assert!(translates(
                target,
                vec![input("self", tensor_arg("x"))],
                F32,
                &[2, 2],
                &[2, 2],
            ));
        }
        assert!(translates(
            "torch.ops.aten.isnan.default",
            vec![input("self", tensor_arg("x"))],
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.leaky_relu.default",
            binary(flt("negative_slope", 0.01)),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        for target in [
            "torch.ops.aten.bitwise_and.Tensor",
            "torch.ops.aten.bitwise_or.Tensor",
        ] {
            assert!(translates(
                target,
                binary(input("other", tensor_arg("x"))),
                I64,
                &[2, 2],
                &[2, 2],
            ));
        }
    }

    #[test]
    fn squeeze_and_triangular_record() {
        assert!(translates(
            "torch.ops.aten.squeeze.default",
            vec![input("self", tensor_arg("x"))],
            F32,
            &[1, 3, 1],
            &[3],
        ));
        assert!(translates(
            "torch.ops.aten.squeeze.dims",
            vec![input("self", tensor_arg("x")), ints("dim", &[0, 2])],
            F32,
            &[1, 3, 1],
            &[3],
        ));
        assert!(translates(
            "torch.ops.aten.tril.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 0)],
            F32,
            &[2, 3],
            &[2, 3],
        ));
        assert!(translates(
            "torch.ops.aten.triu.default",
            vec![input("self", tensor_arg("x")), int("diagonal", -1)],
            F32,
            &[4, 4],
            &[4, 4],
        ));
    }

    #[test]
    fn leaky_relu_matches_reference() {
        let x = [-2.0f32, -0.5, 0.0, 1.0, 3.5];
        let out = run_f32(
            "torch.ops.aten.leaky_relu.default",
            binary(flt("negative_slope", 0.25)),
            F32,
            &[5],
            &x,
        );
        let expected: Vec<f32> = x
            .iter()
            .map(|v| if *v > 0.0 { *v } else { 0.25 * *v })
            .collect();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn atan2_diagonal_matches_reference() {
        let x = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let out = run_f32(
            "torch.ops.aten.atan2.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[5],
            &x,
        );
        // atan2(x, x): pi/4 for x > 0, -3pi/4 for x < 0, 0 at the origin.
        let expected: Vec<f32> = x
            .iter()
            .map(|v| {
                if *v > 0.0 {
                    std::f32::consts::FRAC_PI_4
                } else if *v < 0.0 {
                    -3.0 * std::f32::consts::FRAC_PI_4
                } else {
                    0.0
                }
            })
            .collect();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn fmax_fmin_finite() {
        let x = [1.0f32, 5.0, -2.0];
        let max = run_f32(
            "torch.ops.aten.fmax.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&max, &[3.0, 5.0, 3.0], 1e-6);
        let min = run_f32(
            "torch.ops.aten.fmin.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&min, &[1.0, 3.0, -2.0], 1e-6);
    }

    /// torch.fmax/fmin skip a NaN operand.
    #[test]
    fn fmax_fmin_ignore_nan() {
        let x = [f32::NAN, 5.0, -2.0];
        let max = run_f32(
            "torch.ops.aten.fmax.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&max, &[3.0, 5.0, 3.0], 1e-6);
        let min = run_f32(
            "torch.ops.aten.fmin.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&min, &[3.0, 3.0, -2.0], 1e-6);
    }

    /// torch.fmax/fmin with tensors on both sides: a NaN on either side is
    /// skipped, two NaNs stay NaN, infinities order normally, and zeros
    /// order `-0.0 < +0.0` (torch's CPU kernel: fmax(-0.0, +0.0) = +0.0 and
    /// fmin(+0.0, -0.0) = -0.0, in both operand orders).
    #[test]
    fn fmax_fmin_skip_nan_and_order_signed_zeros() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let x = [
            nan, 1.0, nan, inf, nan, -inf, -0.0, 0.0, -0.0, 0.0, -2.0, 3.0,
        ];
        let w = [
            1.0, nan, nan, nan, -inf, 2.0, 0.0, -0.0, -0.0, 0.0, -2.0, -4.0,
        ];
        let expected_max = [
            1.0, 1.0, nan, inf, -inf, 2.0, 0.0, 0.0, -0.0, 0.0, -2.0, 3.0,
        ];
        let expected_min = [
            1.0, 1.0, nan, inf, -inf, -inf, -0.0, -0.0, -0.0, 0.0, -2.0, -4.0,
        ];
        for (target, expected) in [
            ("torch.ops.aten.fmax.default", expected_max),
            ("torch.ops.aten.fmin.default", expected_min),
        ] {
            let out = run_binary_f32(target, &[12], &x, &w);
            for (i, (got, want)) in out.iter().zip(expected).enumerate() {
                let same = if want.is_nan() {
                    got.is_nan()
                } else {
                    got.to_bits() == want.to_bits()
                };
                assert!(same, "{target} index {i}: got {got:?}, expected {want:?}");
            }
        }
    }

    #[test]
    fn hypot_matches_reference() {
        let x = [0.0f32, 3.0];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2],
            &x,
        );
        assert_close(&out, &[0.0, (18.0f32).sqrt()], 1e-6);
    }

    /// hypot against libm's `hypotf`: +inf wins over NaN, NaN wins over a
    /// finite value, the origin is +0 for either zero sign, and the scaled
    /// formula neither overflows below `f32::MAX` nor flushes subnormals.
    #[test]
    fn hypot_follows_c_special_values() {
        fn check(actual: &[f32], lhs: &[f32], rhs: &[f32]) {
            assert_eq!(actual.len(), lhs.len(), "length mismatch");
            for (i, (&got, (&a, &b))) in actual.iter().zip(lhs.iter().zip(rhs)).enumerate() {
                let want = a.hypot(b);
                let agrees = if want.is_finite() && want != 0.0 {
                    ((got - want) / want).abs() <= 2.0 * f32::EPSILON
                } else {
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
                };
                assert!(
                    agrees,
                    "index {i}: hypot({a}, {b}) got {got:e}, expected {want:e}"
                );
            }
        }
        let tiny = f32::from_bits(1);
        // hypot(x, x): the `inf / inf` and `0 / 0` ratios are selected away.
        let x = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -0.0,
            0.0,
            3.0,
            -1e38,
            3e38,
            tiny,
            f32::MIN_POSITIVE,
        ];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[x.len() as i64],
            &x,
        );
        check(&out, &x, &x);
        // An infinite operand gives +inf even beside a NaN.
        let x = [f32::NAN, 1.0, -0.0, f32::NEG_INFINITY];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(flt("other", f64::INFINITY)),
            F32,
            &[x.len() as i64],
            &x,
        );
        check(&out, &x, &[f32::INFINITY; 4]);
        // A NaN operand gives NaN unless the other is infinite.
        let x = [f32::INFINITY, f32::NEG_INFINITY, 1.0, -0.0, 1e38];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(flt("other", f64::NAN)),
            F32,
            &[x.len() as i64],
            &x,
        );
        check(&out, &x, &[f32::NAN; 5]);
        // A subnormal partner: the ratio's divisor may be subnormal, and a
        // ratio far below 1 must leave the largest finite magnitude intact.
        let partner = 1e-39f64;
        let x = [3e-39, tiny, 0.0, -0.0, -f32::MAX, 1.0];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(flt("other", partner)),
            F32,
            &[x.len() as i64],
            &x,
        );
        check(&out, &x, &[partner as f32; 6]);
    }

    #[test]
    fn tril_triu_match_reference() {
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let lower = run_f32(
            "torch.ops.aten.tril.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 0)],
            F32,
            &[2, 3],
            &x,
        );
        assert_close(&lower, &[0.0, 0.0, 0.0, 3.0, 4.0, 0.0], 1e-6);
        let upper = run_f32(
            "torch.ops.aten.triu.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 1)],
            F32,
            &[2, 3],
            &x,
        );
        assert_close(&upper, &[0.0, 1.0, 2.0, 0.0, 0.0, 5.0], 1e-6);
    }

    fn extents(values: &[i64]) -> Vec<IntExpr> {
        values.iter().map(|&value| IntExpr::from(value)).collect()
    }

    /// The squeeze rule: only requested axes whose extent is provably 1 go;
    /// every other requested axis is a no-op, as in torch.
    #[test]
    fn squeeze_removes_only_provable_unit_axes() {
        let dims = extents(&[1, 3, 1, 4]);
        let axes = |requested: Option<&[i64]>| squeezable_axes(&dims, requested).unwrap();
        // squeeze.dim on an extent != 1 (positive and negative) is a no-op.
        assert_eq!(axes(Some(&[1])), Vec::<usize>::new());
        assert_eq!(axes(Some(&[-1])), Vec::<usize>::new());
        assert_eq!(axes(Some(&[2])), vec![2]);
        assert_eq!(axes(Some(&[-2])), vec![2]);
        // squeeze.dims mixing unit and non-unit axes keeps the non-unit ones.
        assert_eq!(axes(Some(&[0, 1, 2])), vec![0, 2]);
        assert_eq!(axes(Some(&[-1, -4])), vec![0]);
        assert_eq!(axes(Some(&[2, 0])), vec![0, 2]);
        // squeeze.default: every unit axis.
        assert_eq!(axes(None), vec![0, 2]);
        assert_eq!(
            squeezable_axes(&extents(&[1, 1]), None).unwrap(),
            vec![0, 1]
        );
    }

    /// A symbolic extent is never squeezed: export only drops an axis it
    /// knows is 1, and then the extent is the literal 1.
    #[test]
    fn squeeze_keeps_a_symbolic_extent() {
        let dims = vec![IntExpr::from('a'), IntExpr::from(1i64), IntExpr::from(3i64)];
        assert_eq!(
            squeezable_axes(&dims, Some(&[0])).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(squeezable_axes(&dims, Some(&[0, 1])).unwrap(), vec![1]);
        assert_eq!(squeezable_axes(&dims, None).unwrap(), vec![1]);
    }

    /// A rank-0 tensor wraps dims against rank 1 (torch's `maybe_wrap_dim`
    /// for a scalar): dim 0 and -1 are accepted and remove nothing.
    #[test]
    fn squeeze_of_a_scalar_is_a_no_op() {
        let none = Vec::<usize>::new();
        assert_eq!(squeezable_axes(&[], Some(&[0])).unwrap(), none);
        assert_eq!(squeezable_axes(&[], Some(&[-1])).unwrap(), none);
        assert_eq!(squeezable_axes(&[], None).unwrap(), none);
        assert!(squeezable_axes(&[], Some(&[1])).is_err());
        assert!(squeezable_axes(&[], Some(&[-2])).is_err());
        assert!(squeezable_axes(&[], Some(&[0, -1])).is_err());
    }

    #[test]
    fn squeeze_refuses_what_torch_refuses() {
        let dims = extents(&[1, 3]);
        assert!(squeezable_axes(&dims, Some(&[2])).is_err());
        assert!(squeezable_axes(&dims, Some(&[-3])).is_err());
        assert!(squeezable_axes(&dims, Some(&[0, -2])).is_err());
    }

    /// Every overload, on non-unit axes and on a scalar, translates to
    /// the shape torch declares (`squeeze.dim` used to panic on both).
    #[test]
    fn squeeze_overloads_translate_to_the_declared_shape() {
        let check = |target: &str, dim: Option<NodeInput>, from: &[i64], to: &[i64]| {
            let target = format!("torch.ops.aten.{target}");
            let mut args = vec![input("self", tensor_arg("x"))];
            args.extend(dim);
            assert!(
                translates(&target, args, F32, from, to),
                "{target} {from:?} -> {to:?}"
            );
        };
        let shape = [1, 3, 1, 4];
        check("squeeze.dim", Some(int("dim", 1)), &shape, &shape);
        check("squeeze.dim", Some(int("dim", -1)), &shape, &shape);
        check("squeeze.dim", Some(int("dim", 2)), &shape, &[1, 3, 4]);
        check(
            "squeeze.dims",
            Some(ints("dim", &[0, 1])),
            &shape,
            &[3, 1, 4],
        );
        check("squeeze.default", None, &shape, &[3, 4]);
        check("squeeze.dim", Some(int("dim", 0)), &[], &[]);
        check("squeeze.dim", Some(int("dim", -1)), &[], &[]);
        check("squeeze.dims", Some(ints("dim", &[0])), &[], &[]);
        check("squeeze.default", None, &[], &[]);
    }

    /// The export's declared rank is checked: a squeeze that would give a
    /// different rank than torch recorded is refused, not translated.
    #[test]
    fn squeeze_refuses_a_rank_torch_did_not_declare() {
        assert!(!translates(
            "torch.ops.aten.squeeze.dim",
            vec![input("self", tensor_arg("x")), int("dim", 1)],
            F32,
            &[1, 3, 1, 4],
            &[1, 1, 4],
        ));
    }
}
