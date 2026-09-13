//! Real-domain special-function lowerings (port batch 5): the `special_*`
//! Bessel/Airy/erf/Polygamma family, `logcumsumexp`, and `angle`.
//!
//! Everything is built from recorder-frontend primitives. Helpers the parked
//! translator kept private to its own module (`real_atan`, `real_exp`,
//! `real_cos`, the polynomial evaluators, `real_atan2`) are duplicated here
//! under a `special_` prefix so no sibling port's method table is touched.
//! `translate_erfc`/`translate_erf` and the plain unary real-domain helpers
//! stay in `unary.rs`.
#![allow(dead_code)]

use anyhow::{Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::Node;

impl Translator<'_> {
    // -----------------------------------------------------------------
    // Shared real-domain helpers (private copies of the parked unary ones)
    // -----------------------------------------------------------------

    /// Input cast to the node's output dtype (the parked `unary_input`).
    fn special_input(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(x.dtype);
        Ok(if x.dtype == dtype { x } else { x.cast(dtype) })
    }

    /// Structural `where` that keeps the branches' dtype. Arithmetic masking
    /// (`a * mask + b * (1 - mask)`) turns an untaken NaN or infinity into
    /// `NaN * 0 = NaN`, poisoning the selected value. `concat_along` cannot
    /// be used either: its `pad` half is itself arithmetic-masked and leaks
    /// non-finite edge values into the pad region (LUM-804). Pack the two
    /// branches with a scatter into a fresh zero tensor — scatter only moves
    /// payload, it never computes on it — and select the branch with a
    /// gather index, the parked translator's construction.
    fn special_select(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = util::broadcast_binary(a, condition);
        let (a, b) = util::broadcast_binary(a, b);
        // If the `b` broadcast drove the shape, bring `condition` up to it.
        let (a, condition) = util::broadcast_binary(a, condition);
        let dims = a.dims();
        if dims.is_empty() {
            let selected =
                self.special_select(condition.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0));
            return selected.squeeze(0);
        }
        let rank = dims.len();
        // One coordinate tensor per data axis, shared by both writes and the
        // final read.
        let axis_coordinates: Vec<GraphTensor> = (0..rank)
            .map(|axis| self.cx.iota(dims.clone(), |c| c[axis]))
            .collect();
        // Branch 0 gets `b`, branch 1 gets `a` (the condition's 0/1 code).
        let mut false_coordinates = Vec::with_capacity(rank + 1);
        false_coordinates.push(self.cx.iota(dims.clone(), |_| IntExpr::from(0)));
        false_coordinates.extend(axis_coordinates.iter().copied());
        let mut true_coordinates = Vec::with_capacity(rank + 1);
        true_coordinates.push(self.cx.iota(dims.clone(), |_| IntExpr::from(1)));
        true_coordinates.extend(axis_coordinates.iter().copied());

        let mut stacked_dims = Vec::with_capacity(rank + 1);
        stacked_dims.push(IntExpr::from(2));
        stacked_dims.extend(dims.iter().copied());
        // A real materialized zero buffer: an expanded scalar would alias one
        // element and scatter copies its destination before writing.
        let scratch = self
            .cx
            .iota(stacked_dims, |_| IntExpr::from(0))
            .cast(a.dtype);
        let stacked = scratch
            .scatter(&false_coordinates, b)
            .scatter(&true_coordinates, a);

        let mut gather_coordinates = Vec::with_capacity(rank + 1);
        gather_coordinates.push(condition.cast(DType::Int));
        gather_coordinates.extend(axis_coordinates);
        stacked.gather(&gather_coordinates)
    }

    /// Keep log2(e) in the tensor's actual dtype (F64-safe `exp`).
    fn special_exp(&mut self, input: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(input, std::f64::consts::LOG2_E);
        (input * log2_e).exp2()
    }

    /// `cos(x) = sin(pi/2 - x)`, keeping pi/2 in the tensor's dtype.
    fn special_cos(&mut self, input: GraphTensor) -> GraphTensor {
        let half_pi = self.constant_like(input, std::f64::consts::FRAC_PI_2);
        (half_pi - input).sin()
    }

    fn special_sinh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.special_exp(input) - self.special_exp(input * -1.0))
    }

    fn special_cosh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.special_exp(input) + self.special_exp(input * -1.0))
    }

    fn special_acosh(&mut self, input: GraphTensor) -> GraphTensor {
        let reciprocal_squared = input.reciprocal().square();
        let one = self.constant_like(input, 1.0);
        input.log() + (one + (one - reciprocal_squared).sqrt()).log()
    }

    /// Range-reduced odd Taylor series for atan (same approximation as the
    /// recorder `unary.rs` private `real_atan`).
    fn special_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let one = self.constant_like(x, 1.0);
        let reciprocal_branch = x.gt(one);
        let reduced = self.special_select(reciprocal_branch, x.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.special_select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.special_select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.special_select(reciprocal_branch, half_pi - base, base);
        self.copy_sign(angle, input)
    }

    /// Parked real-domain `real_atan2` (from the complex port): quadrants
    /// from the sign of `x`, the two-infinite axis case, and signed zero.
    fn special_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let ratio = y / x;
        let mut angle = self.special_atan(ratio);
        let x_negative = self.signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.copy_sign(pi, y);
        angle = self.special_select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = self.bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.special_select(x_negative, three_quarters, quarter);
        let infinite_angle = self.copy_sign(infinite_angle, y);
        angle = self.special_select(both_inf, infinite_angle, angle);

        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);
        let both_zero = self.bool_and(x_zero, y_zero);
        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.copy_sign(zero, y);
        let zero_angle = self.special_select(x_negative, signed_pi, signed_zero);
        self.special_select(both_zero, zero_angle, angle)
    }

    /// Parked polynomial real `acos` (used only by the Chebyshev separators).
    #[allow(clippy::excessive_precision)]
    fn special_acos(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let polynomial =
            self.constant_like(x, 0.000_684_531_8) * x - self.constant_like(x, 0.003_974_577_8);
        let polynomial = polynomial * x + self.constant_like(x, 0.011_028_381);
        let polynomial = polynomial * x - self.constant_like(x, 0.020_727_666);
        let polynomial = polynomial * x + self.constant_like(x, 0.032_571_17);
        let polynomial = polynomial * x - self.constant_like(x, 0.050_593_574);
        let polynomial = polynomial * x + self.constant_like(x, 0.089_030_14);
        let polynomial = polynomial * x - self.constant_like(x, 0.214_601_16);
        let half_pi = self.constant_like(x, std::f64::consts::FRAC_PI_2);
        let polynomial = polynomial * x + half_pi;
        let one = self.constant_like(x, 1.0);
        let positive = polynomial * (one - x).sqrt();

        let zero = self.constant_like(input, 0.0);
        let negative = input.lt(zero).cast(input.dtype);
        let pi = self.constant_like(input, std::f64::consts::PI);
        let two = self.constant_like(input, 2.0);
        positive + negative * (pi - two * positive)
    }

    fn special_polynomial_evaluate(
        &mut self,
        value: GraphTensor,
        coefficients: &[f64],
    ) -> GraphTensor {
        let mut result = self.constant_like(value, coefficients[0]);
        for coefficient in coefficients.iter().copied().skip(1) {
            result = result * value + self.constant_like(value, coefficient);
        }
        result
    }

    fn special_polynomial_with_leading_one(
        &mut self,
        value: GraphTensor,
        coefficients: &[f64],
    ) -> GraphTensor {
        let mut result = value + self.constant_like(value, coefficients[0]);
        for coefficient in coefficients.iter().copied().skip(1) {
            result = result * value + self.constant_like(value, coefficient);
        }
        result
    }

    fn special_chebyshev_evaluate(
        &mut self,
        value: GraphTensor,
        coefficients: &[f64],
    ) -> GraphTensor {
        debug_assert!(coefficients.len() >= 2);
        let mut b0 = self.constant_like(value, coefficients[0]);
        let mut b1 = self.constant_like(value, 0.0);
        let mut b2 = b1;
        for coefficient in coefficients.iter().copied().skip(1) {
            b2 = b1;
            b1 = b0;
            b0 = value * b1 - b2 + self.constant_like(value, coefficient);
        }
        (b0 - b2) * 0.5
    }

    // -----------------------------------------------------------------
    // Modified Bessel I
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_modified_bessel_i0(&mut self, value: GraphTensor, scaled: bool) -> GraphTensor {
        #[rustfmt::skip]
        const SMALL: [f64; 30] = [-4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16, 1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14, -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11, 9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238635035E-9, -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7, 1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5, -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4, 1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2, -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2, 1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1,];
        #[rustfmt::skip]
        const LARGE: [f64; 25] = [-7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17, 3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16, 1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15, -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13, 7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11, -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10, 3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7, 2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3, 8.04490411014108831608E-1,];

        self.special_modified_bessel_i(value, &SMALL, &LARGE, 0, scaled)
    }

    #[allow(clippy::excessive_precision)]
    fn special_modified_bessel_i1(&mut self, value: GraphTensor, scaled: bool) -> GraphTensor {
        #[rustfmt::skip]
        const SMALL: [f64; 29] = [2.77791411276104639959E-18, -2.11142121435816608115E-17, 1.55363195773620046921E-16, -1.10559694773538630805E-15, 7.60068429473540693410E-15, -5.04218550472791168711E-14, 3.22379336594557470981E-13, -1.98397439776494371520E-12, 1.17361862988909016308E-11, -6.66348972350202774223E-11, 3.62559028155211703701E-10, -1.88724975172282928790E-9, 9.38153738649577178388E-9, -4.44505912879632808065E-8, 2.00329475355213526229E-7, -8.56872026469545474066E-7, 3.47025130813767847674E-6, -1.32731636560394358279E-5, 4.78156510755005422638E-5, -1.61760815825896745588E-4, 5.12285956168575772895E-4, -1.51357245063125314899E-3, 4.15642294431288815669E-3, -1.05640848946261981558E-2, 2.47264490306265168283E-2, -5.29459812080949914269E-2, 1.02643658689847095384E-1, -1.76416518357834055153E-1, 2.52587186443633654823E-1,];
        #[rustfmt::skip]
        const LARGE: [f64; 25] = [7.51729631084210481353E-18, 4.41434832307170791151E-18, -4.65030536848935832153E-17, -3.20952592199342395980E-17, 2.96262899764595013876E-16, 3.30820231092092828324E-16, -1.88035477551078244854E-15, -3.81440307243700780478E-15, 1.04202769841288027642E-14, 4.27244001671195135429E-14, -2.10154184277266431302E-14, -4.08355111109219731823E-13, -7.19855177624590851209E-13, 2.03562854414708950722E-12, 1.41258074366137813316E-11, 3.25260358301548823856E-11, -1.89749581235054123450E-11, -5.58974346219658380687E-10, -3.83538038596423702205E-9, -2.63146884688951950684E-8, -2.51223623787020892529E-7, -3.88256480887769039346E-6, -1.10588938762623716291E-4, -9.76109749136146840777E-3, 7.78576235018280120474E-1,];

        self.special_modified_bessel_i(value, &SMALL, &LARGE, 1, scaled)
    }

    #[allow(clippy::excessive_precision)]
    fn special_modified_bessel_i(
        &mut self,
        value: GraphTensor,
        small_coefficients: &[f64],
        large_coefficients: &[f64],
        order: usize,
        scaled: bool,
    ) -> GraphTensor {
        let absolute = self.real_abs(value);
        let small_argument = absolute * 0.5 - self.constant_like(value, 2.0);
        let mut small = self.special_chebyshev_evaluate(small_argument, small_coefficients);
        if order == 1 {
            small *= absolute;
        }
        let large_argument =
            self.constant_like(value, 32.0) / absolute - self.constant_like(value, 2.0);
        let mut large =
            self.special_chebyshev_evaluate(large_argument, large_coefficients) / absolute.sqrt();
        if !scaled {
            let exponential = self.special_exp(absolute);
            small *= exponential;
            large *= exponential;
        }
        let threshold = self.constant_like(value, 8.0);
        let magnitude = self.special_select(absolute.le(threshold), small, large);
        if order == 0 {
            magnitude
        } else {
            let zero = self.constant_like(value, 0.0);
            self.special_select(value.lt(zero), -magnitude, magnitude)
        }
    }

    pub(super) fn translate_modified_bessel(
        &mut self,
        node: &Node,
        order: usize,
        scaled: bool,
    ) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        match order {
            0 => Ok(self.special_modified_bessel_i0(value, scaled)),
            1 => Ok(self.special_modified_bessel_i1(value, scaled)),
            _ => bail!("modified_bessel order {order} is not ported"),
        }
    }

    // -----------------------------------------------------------------
    // Spherical and cylindrical Bessel
    // -----------------------------------------------------------------

    pub(super) fn translate_spherical_bessel_j0(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let finite = value.sin() / value;
        let is_zero = self.is_zero(value);
        let with_zero = self.special_select(is_zero, one, finite);
        let infinite = self.is_inf(value);
        Ok(self.special_select(infinite, zero, with_zero))
    }

    #[allow(clippy::excessive_precision)]
    fn special_cylindrical_bessel_asymptotic(
        &mut self,
        value: GraphTensor,
        order: usize,
        second_kind: bool,
    ) -> GraphTensor {
        #[rustfmt::skip]
        const J0_PP: [f64; 7] = [7.96936729297347051624e-04, 8.28352392107440799803e-02, 1.23953371646414299388e+00, 5.44725003058768775090e+00, 8.74716500199817011941e+00, 5.30324038235394892183e+00, 9.99999999999999997821e-01,];
        #[rustfmt::skip]
        const J0_PQ: [f64; 7] = [9.24408810558863637013e-04, 8.56288474354474431428e-02, 1.25352743901058953537e+00, 5.47097740330417105182e+00, 8.76190883237069594232e+00, 5.30605288235394617618e+00, 1.00000000000000000218e+00,];
        #[rustfmt::skip]
        const J0_QP: [f64; 8] = [-1.13663838898469149931e-02, -1.28252718670509318512e+00, -1.95539544257735972385e+01, -9.32060152123768231369e+01, -1.77681167980488050595e+02, -1.47077505154951170175e+02, -5.14105326766599330220e+01, -6.05014350600728481186e+00,];
        #[rustfmt::skip]
        const J0_QQ: [f64; 7] = [6.43178256118178023184e+01, 8.56430025976980587198e+02, 3.88240183605401609683e+03, 7.24046774195652478189e+03, 5.93072701187316984827e+03, 2.06209331660327847417e+03, 2.42005740240291393179e+02,];
        #[rustfmt::skip]
        const J1_PP: [f64; 7] = [7.62125616208173112003e-04, 7.31397056940917570436e-02, 1.12719608129684925192e+00, 5.11207951146807644818e+00, 8.42404590141772420927e+00, 5.21451598682361504063e+00, 1.00000000000000000254e+00,];
        #[rustfmt::skip]
        const J1_PQ: [f64; 7] = [5.71323128072548699714e-04, 6.88455908754495404082e-02, 1.10514232634061696926e+00, 5.07386386128601488557e+00, 8.39985554327604159757e+00, 5.20982848682361821619e+00, 9.99999999999999997461e-01,];
        #[rustfmt::skip]
        const J1_QP: [f64; 8] = [5.10862594750176621635e-02, 4.98213872951233449420e+00, 7.58238284132545283818e+01, 3.66779609360150777800e+02, 7.10856304998926107277e+02, 5.97489612400613639965e+02, 2.11688757100572135698e+02, 2.52070205858023719784e+01,];
        #[rustfmt::skip]
        const J1_QQ: [f64; 7] = [7.42373277035675149943e+01, 1.05644886038262816351e+03, 4.98641058337653607651e+03, 9.56231892404756170795e+03, 7.99704160447350683650e+03, 2.82619278517639096600e+03, 3.36093607810698293419e+02,];

        let (pp, pq, qp, qq, phase): (&[f64], &[f64], &[f64], &[f64], f64) = if order == 0 {
            (&J0_PP, &J0_PQ, &J0_QP, &J0_QQ, std::f64::consts::FRAC_PI_4)
        } else {
            (
                &J1_PP,
                &J1_PQ,
                &J1_QP,
                &J1_QQ,
                2.356194490192344928846982537459627163,
            )
        };
        let reciprocal_square = self.constant_like(value, 25.0) / value.square();
        let p = self.special_polynomial_evaluate(reciprocal_square, pp)
            / self.special_polynomial_evaluate(reciprocal_square, pq);
        let q = self.constant_like(value, 5.0) / value
            * self.special_polynomial_evaluate(reciprocal_square, qp)
            / self.special_polynomial_evaluate(reciprocal_square, qq);
        let angle = value - self.constant_like(value, phase);
        let cosine = self.special_cos(angle);
        let sine = angle.sin();
        let oscillation = if second_kind {
            p * sine + q * cosine
        } else {
            p * cosine - q * sine
        };
        oscillation * self.constant_like(value, 0.797884560802865355879892119868763737)
            / value.sqrt()
    }

    #[allow(clippy::excessive_precision)]
    fn special_cylindrical_bessel_j0(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const RP: [f64; 4] = [-4.79443220978201773821e+09, 1.95617491946556577543e+12, -2.49248344360967716204e+14, 9.70862251047306323952e+15,];
        #[rustfmt::skip]
        const RQ: [f64; 8] = [4.99563147152651017219e+02, 1.73785401676374683123e+05, 4.84409658339962045305e+07, 1.11855537045356834862e+10, 2.11277520115489217587e+12, 3.10518229857422583814e+14, 3.18121955943204943306e+16, 1.71086294081043136091e+18,];

        let absolute = self.real_abs(value);
        let squared = absolute.square();
        let ratio = self.special_polynomial_evaluate(squared, &RP)
            / self.special_polynomial_evaluate(squared, &RQ);
        let ordinary = (squared - self.constant_like(value, 5.78318596294678452118))
            * (squared - self.constant_like(value, 30.4712623436620863991))
            * ratio;
        let near_zero = self.constant_like(value, 1.0) - squared * 0.25;
        let tiny = self.constant_like(value, 1.0e-5);
        let small = self.special_select(absolute.lt(tiny), near_zero, ordinary);
        let large = self.special_cylindrical_bessel_asymptotic(absolute, 0, false);
        let threshold = self.constant_like(value, 5.0);
        self.special_select(absolute.le(threshold), small, large)
    }

    #[allow(clippy::excessive_precision)]
    fn special_cylindrical_bessel_j1(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const RP: [f64; 4] = [-8.99971225705559398224e+08, 4.52228297998194034323e+11, -7.27494245221818276015e+13, 3.68295732863852883286e+15,];
        #[rustfmt::skip]
        const RQ: [f64; 8] = [6.20836478118054335476e+02, 2.56987256757748830383e+05, 8.35146791431949253037e+07, 2.21511595479792499675e+10, 4.74914122079991414898e+12, 7.84369607876235854894e+14, 8.95222336184627338078e+16, 5.32278620332680085395e+18,];

        let absolute = self.real_abs(value);
        let squared = absolute.square();
        let small = self.special_polynomial_evaluate(squared, &RP)
            / self.special_polynomial_evaluate(squared, &RQ)
            * absolute
            * (squared - self.constant_like(value, 14.6819706421238932572))
            * (squared - self.constant_like(value, 49.2184563216946036703));
        let large = self.special_cylindrical_bessel_asymptotic(absolute, 1, false);
        let threshold = self.constant_like(value, 5.0);
        let magnitude = self.special_select(absolute.le(threshold), small, large);
        self.copy_sign(magnitude, value)
    }

    #[allow(clippy::excessive_precision)]
    fn special_cylindrical_bessel_y0(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const YP: [f64; 8] = [1.55924367855235737965e+04, -1.46639295903971606143e+07, 5.43526477051876500413e+09, -9.82136065717911466409e+11, 8.75906394395366999549e+13, -3.46628303384729719441e+15, 4.42733268572569800351e+16, -1.84950800436986690637e+16,];
        #[rustfmt::skip]
        const YQ: [f64; 7] = [1.04128353664259848412e+03, 6.26107330137134956842e+05, 2.68919633393814121987e+08, 8.64002487103935000337e+10, 2.02979612750105546709e+13, 3.17157752842975028269e+15, 2.50596256172653059228e+17,];

        let squared = value.square();
        let small = self.special_polynomial_evaluate(squared, &YP)
            / self.special_polynomial_evaluate(squared, &YQ)
            + self.constant_like(value, std::f64::consts::FRAC_2_PI)
                * value.log()
                * self.special_cylindrical_bessel_j0(value);
        let large = self.special_cylindrical_bessel_asymptotic(value, 0, true);
        let threshold = self.constant_like(value, 5.0);
        let ordinary = self.special_select(value.le(threshold), small, large);
        self.special_finish_cylindrical_bessel_y(value, ordinary)
    }

    #[allow(clippy::excessive_precision)]
    fn special_cylindrical_bessel_y1(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const YP: [f64; 6] = [1.26320474790178026440e+09, -6.47355876379160291031e+11, 1.14509511541823727583e+14, -8.12770255501325109621e+15, 2.02439475713594898196e+17, -7.78877196265950026825e+17,];
        #[rustfmt::skip]
        const YQ: [f64; 8] = [5.94301592346128195359e+02, 2.35564092943068577943e+05, 7.34811944459721705660e+07, 1.87601316108706159478e+10, 3.88231277496238566008e+12, 6.20557727146953693363e+14, 6.87141087355300489866e+16, 3.97270608116560655612e+18,];

        let squared = value.square();
        let small = value * self.special_polynomial_evaluate(squared, &YP)
            / self.special_polynomial_evaluate(squared, &YQ)
            + self.constant_like(value, std::f64::consts::FRAC_2_PI)
                * (self.special_cylindrical_bessel_j1(value) * value.log() - value.reciprocal());
        let large = self.special_cylindrical_bessel_asymptotic(value, 1, true);
        let threshold = self.constant_like(value, 5.0);
        let ordinary = self.special_select(value.le(threshold), small, large);
        self.special_finish_cylindrical_bessel_y(value, ordinary)
    }

    fn special_finish_cylindrical_bessel_y(
        &mut self,
        value: GraphTensor,
        ordinary: GraphTensor,
    ) -> GraphTensor {
        let negative_infinity = self.constant_like(value, f64::NEG_INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let zero = self.is_zero(value);
        let with_zero = self.special_select(zero, negative_infinity, ordinary);
        let zero_value = self.constant_like(value, 0.0);
        let negative = value.lt(zero_value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(negative, input_nan);
        self.special_select(invalid, nan, with_zero)
    }

    pub(super) fn translate_cylindrical_bessel(
        &mut self,
        node: &Node,
        order: usize,
        y: bool,
    ) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        match (order, y) {
            (0, false) => Ok(self.special_cylindrical_bessel_j0(value)),
            (1, false) => Ok(self.special_cylindrical_bessel_j1(value)),
            (0, true) => Ok(self.special_cylindrical_bessel_y0(value)),
            (1, true) => Ok(self.special_cylindrical_bessel_y1(value)),
            _ => bail!("cylindrical_bessel order {order} (y={y}) is not ported"),
        }
    }

    // -----------------------------------------------------------------
    // Modified Bessel K
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_modified_bessel_k(
        &mut self,
        value: GraphTensor,
        order: usize,
        scaled: bool,
    ) -> GraphTensor {
        #[rustfmt::skip]
        const K0_A: [f64; 10] = [1.37446543561352307156e-16, 4.25981614279661018399e-14, 1.03496952576338420167e-11, 1.90451637722020886025e-09, 2.53479107902614945675e-07, 2.28621210311945178607e-05, 1.26461541144692592338e-03, 3.59799365153615016266e-02, 3.44289899924628486886e-01, -5.35327393233902768720e-01,];
        #[rustfmt::skip]
        const K0_B: [f64; 25] = [5.30043377268626276149e-18, -1.64758043015242134646e-17, 5.21039150503902756861e-17, -1.67823109680541210385e-16, 5.51205597852431940784e-16, -1.84859337734377901440e-15, 6.34007647740507060557e-15, -2.22751332699166985548e-14, 8.03289077536357521100e-14, -2.98009692317273043925e-13, 1.14034058820847496303e-12, -4.51459788337394416547e-12, 1.85594911495471785253e-11, -7.95748924447710747776e-11, 3.57739728140030116597e-10, -1.69753450938905987466e-09, 8.57403401741422608519e-09, -4.66048989768794782956e-08, 2.76681363944501510342e-07, -1.83175552271911948767e-06, 1.39498137188764993662e-05, -1.28495495816278026384e-04, 1.56988388573005337491e-03, -3.14481013119645005427e-02, 2.44030308206595545468e+00,];
        #[rustfmt::skip]
        const K1_A: [f64; 11] = [-7.02386347938628759343e-18, -2.42744985051936593393e-15, -6.66690169419932900609e-13, -1.41148839263352776110e-10, -2.21338763073472585583e-08, -2.43340614156596823496e-06, -1.73028895751305206302e-04, -6.97572385963986435018e-03, -1.22611180822657148235e-01, -3.53155960776544875667e-01, 1.52530022733894777053e+00,];
        #[rustfmt::skip]
        const K1_B: [f64; 25] = [-5.75674448366501715755e-18, 1.79405087314755922667e-17, -5.68946255844285935196e-17, 1.83809354436663880070e-16, -6.05704724837331885336e-16, 2.03870316562433424052e-15, -7.01983709041831346144e-15, 2.47715442448130437068e-14, -8.97670518232499435011e-14, 3.34841966607842919884e-13, -1.28917396095102890680e-12, 5.13963967348173025100e-12, -2.12996783842756842877e-11, 9.21831518760500529508e-11, -4.19035475934189648750e-10, 2.01504975519703286596e-09, -1.03457624656780970260e-08, 5.74108412545004946722e-08, -3.50196060308781257119e-07, 2.40648494783721712015e-06, -1.93619797416608296024e-05, 1.95215518471351631108e-04, -2.85781685962277938680e-03, 1.03923736576817238437e-01, 2.72062619048444266945e+00,];

        let (small_coefficients, large_coefficients): (&[f64], &[f64]) = if order == 0 {
            (&K0_A, &K0_B)
        } else {
            (&K1_A, &K1_B)
        };
        let two = self.constant_like(value, 2.0);
        let small_argument = value.square() - two;
        let small_series = self.special_chebyshev_evaluate(small_argument, small_coefficients);
        let half = self.constant_like(value, 0.5);
        let small = if order == 0 {
            small_series - (value * half).log() * self.special_modified_bessel_i0(value, false)
        } else {
            (value * half).log() * self.special_modified_bessel_i1(value, false)
                + small_series / value
        };
        let eight = self.constant_like(value, 8.0);
        let large_argument = eight / value - two;
        let large =
            self.special_chebyshev_evaluate(large_argument, large_coefficients) / value.sqrt();
        let small = if scaled {
            small * self.special_exp(value)
        } else {
            small
        };
        let large = if scaled {
            large
        } else {
            self.special_exp(-value) * large
        };
        let ordinary = self.special_select(value.le(two), small, large);
        let infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let zero = self.is_zero(value);
        let with_zero = self.special_select(zero, infinity, ordinary);
        let zero_value = self.constant_like(value, 0.0);
        let negative = value.lt(zero_value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(negative, input_nan);
        self.special_select(invalid, nan, with_zero)
    }

    pub(super) fn translate_modified_bessel_k(
        &mut self,
        node: &Node,
        order: usize,
        scaled: bool,
    ) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        match order {
            0 | 1 => Ok(self.special_modified_bessel_k(value, order, scaled)),
            _ => bail!("modified_bessel_k order {order} is not ported"),
        }
    }

    // -----------------------------------------------------------------
    // Airy Ai
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_airy_ai(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const AN: [f64; 8] = [3.46538101525629032477e-01, 1.20075952739645805542e+01, 7.62796053615234516538e+01, 1.68089224934630576269e+02, 1.59756391350164413639e+02, 7.05360906840444183113e+01, 1.40264691163389668864e+01, 9.99999999999999995305e-01,];
        #[rustfmt::skip]
        const AD: [f64; 8] = [5.67594532638770212846e-01, 1.47562562584847203173e+01, 8.45138970141474626562e+01, 1.77318088145400459522e+02, 1.64234692871529701831e+02, 7.14778400825575695274e+01, 1.40959135607834029598e+01, 1.00000000000000000470e+00,];
        #[rustfmt::skip]
        const AFN: [f64; 9] = [-1.31696323418331795333e-01, -6.26456544431912369773e-01, -6.93158036036933542233e-01, -2.79779981545119124951e-01, -4.91900132609500318020e-02, -4.06265923594885404393e-03, -1.59276496239262096340e-04, -2.77649108155232920844e-06, -1.67787698489114633780e-08,];
        #[rustfmt::skip]
        const AFD: [f64; 9] = [1.33560420706553243746e+01, 3.26825032795224613948e+01, 2.67367040941499554804e+01, 9.18707402907259625840e+00, 1.47529146771666414581e+00, 1.15687173795188044134e-01, 4.40291641615211203805e-03, 7.54720348287414296618e-05, 4.51850092970580378464e-07,];
        #[rustfmt::skip]
        const AGN: [f64; 11] = [1.97339932091685679179e-02, 3.91103029615688277255e-01, 1.06579897599595591108e+00, 9.39169229816650230044e-01, 3.51465656105547619242e-01, 6.33888919628925490927e-02, 5.85804113048388458567e-03, 2.82851600836737019778e-04, 6.98793669997260967291e-06, 8.11789239554389293311e-08, 3.41551784765923618484e-10,];
        #[rustfmt::skip]
        const AGD: [f64; 10] = [9.30892908077441974853e+00, 1.98352928718312140417e+01, 1.55646628932864612953e+01, 5.47686069422975497931e+00, 9.54293611618961883998e-01, 8.64580826352392193095e-02, 4.12656523824222607191e-03, 1.01259085116509135510e-04, 1.17166733214413521882e-06, 4.91834570062930015649e-09,];

        let one = self.constant_like(value, 1.0);
        let negative_value = -value;
        let negative_root = negative_value.sqrt();
        let three = self.constant_like(value, 3.0);
        let negative_phase = self.constant_like(value, -2.0) * value * negative_root / three;
        let z = negative_phase.reciprocal();
        let z_squared = z.square();
        let f = one
            + z_squared * self.special_polynomial_evaluate(z_squared, &AFN)
                / self.special_polynomial_evaluate(z_squared, &AFD);
        let g = z * self.special_polynomial_evaluate(z_squared, &AGN)
            / self.special_polynomial_evaluate(z_squared, &AGD);
        let angle = negative_phase + self.constant_like(value, std::f64::consts::FRAC_PI_4);
        let negative = self.constant_like(value, 0.564189583547756286948) / negative_root.sqrt()
            * (angle.sin() * f - self.special_cos(angle) * g);

        let positive_root = value.sqrt();
        let two_thirds = self.constant_like(value, 2.0 / 3.0);
        let zeta = value * positive_root * two_thirds;
        let inverse_zeta = zeta.reciprocal();
        let positive = self.constant_like(value, 0.564189583547756286948)
            * (self.special_polynomial_evaluate(inverse_zeta, &AN)
                / self.special_polynomial_evaluate(inverse_zeta, &AD))
            * self.special_exp(-zeta)
            / (positive_root.sqrt() * 2.0);

        // The power-series branch is only selected on [-2.09, 2.09), so a
        // fixed 30 terms uniformly exceeds F64 precision without any runtime
        // convergence or control-flow requirement.
        let cubic = value * value * value;
        let mut series_f = one;
        let mut series_g = value;
        let mut m = one;
        let mut n = value;
        for iteration in 0..30 {
            let k = (iteration * 3) as f64;
            let m_denominator = self.constant_like(value, (k + 2.0) * (k + 3.0));
            let n_denominator = self.constant_like(value, (k + 3.0) * (k + 4.0));
            m = m * cubic / m_denominator;
            n = n * cubic / n_denominator;
            series_f += m;
            series_g += n;
        }
        let central = self.constant_like(value, 0.355028053887817239260) * series_f
            - self.constant_like(value, 0.258819403792806798405) * series_g;

        let negative_threshold = self.constant_like(value, -2.09);
        let positive_threshold = self.constant_like(value, 2.09);
        let lower_or_central = self.special_select(value.lt(negative_threshold), negative, central);
        let ordinary =
            self.special_select(value.ge(positive_threshold), positive, lower_or_central);
        let underflow_threshold = self.constant_like(value, 103.892);
        let zero = self.constant_like(value, 0.0);
        let ordinary = self.special_select(value.gt(underflow_threshold), zero, ordinary);
        let infinite = self.is_inf(value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(infinite, input_nan);
        let nan = self.constant_like(value, f64::NAN);
        self.special_select(invalid, nan, ordinary)
    }

    pub(super) fn translate_airy_ai(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        Ok(self.special_airy_ai(value))
    }

    // -----------------------------------------------------------------
    // Inverse erf / inverse normal CDF
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_accurate_erf(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const T: [f64; 5] = [9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3, 7.00332514112805075473E3, 5.55923013010394962768E4,];
        #[rustfmt::skip]
        const U: [f64; 5] = [3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3, 2.26290000613890934246E4, 4.92673942608635921086E4,];
        #[rustfmt::skip]
        const P: [f64; 9] = [2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0, 4.86371970985681366614E1, 1.96520832956077098242E2, 5.26445194995477358631E2, 9.34528527171957607540E2, 1.02755188689515710272E3, 5.57535335369399327526E2,];
        #[rustfmt::skip]
        const Q: [f64; 8] = [1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2, 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3, 1.65666309194161350182E3, 5.57535340817727675546E2,];
        #[rustfmt::skip]
        const R: [f64; 6] = [5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0, 6.16021097993053585195E0, 7.40974269950448939160E0, 2.97886665372100240670E0,];
        #[rustfmt::skip]
        const S: [f64; 6] = [2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1, 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0,];

        let absolute = self.real_abs(value);
        let squared = absolute.square();
        let central = absolute * self.special_polynomial_evaluate(squared, &T)
            / self.special_polynomial_with_leading_one(squared, &U);
        let moderate = self.special_polynomial_evaluate(absolute, &P)
            / self.special_polynomial_with_leading_one(absolute, &Q);
        let large = self.special_polynomial_evaluate(absolute, &R)
            / self.special_polynomial_with_leading_one(absolute, &S);
        let eight = self.constant_like(value, 8.0);
        let erfc_ratio = self.special_select(absolute.lt(eight), moderate, large);
        let erfc = self.special_exp(squared * -1.0) * erfc_ratio;
        let one = self.constant_like(value, 1.0);
        let tail = one - erfc;
        let magnitude = self.special_select(absolute.lt(one), central, tail);
        let infinite = self.is_inf(value);
        let magnitude = self.special_select(infinite, one, magnitude);
        self.copy_sign(magnitude, value)
    }

    #[allow(clippy::excessive_precision)]
    fn special_inverse_erf(&mut self, value: GraphTensor) -> GraphTensor {
        const A: [f64; 4] = [-0.140543331, 0.914624893, -1.645349621, 0.886226899];
        const B: [f64; 4] = [0.012229801, -0.329097515, 1.442710462, -2.118377725];
        const C: [f64; 4] = [1.641345311, 3.429567803, -1.624906493, -1.970840454];
        const D: [f64; 2] = [3.543889200, 1.637067800];

        let absolute = self.real_abs(value);
        let squared = value.square();
        let central_numerator = self.special_polynomial_evaluate(squared, &A);
        let central_denominator = squared * self.special_polynomial_evaluate(squared, &B)
            + self.constant_like(value, 1.0);
        let central = value * central_numerator / central_denominator;

        let one = self.constant_like(value, 1.0);
        let tail_root = (((one - absolute) * 0.5).log() * -1.0).sqrt();
        let tail_numerator = self.special_polynomial_evaluate(tail_root, &C);
        let tail_denominator = tail_root
            * (tail_root * self.constant_like(value, D[1]) + self.constant_like(value, D[0]))
            + one;
        let tail = self.copy_sign(tail_numerator, value) / tail_denominator;
        let threshold = self.constant_like(value, 0.7);
        let mut result = self.special_select(absolute.le(threshold), central, tail);

        let derivative_scale = self.constant_like(value, std::f64::consts::FRAC_2_SQRT_PI);
        for _ in 0..2 {
            let error = self.special_accurate_erf(result) - value;
            let derivative = derivative_scale * self.special_exp(result.square() * -1.0);
            result -= error / derivative;
        }

        let infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let endpoint = self.is_zero(absolute - one);
        let signed_infinity = self.copy_sign(infinity, value);
        result = self.special_select(endpoint, signed_infinity, result);
        let outside = absolute.gt(one);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(outside, input_nan);
        self.special_select(invalid, nan, result)
    }

    #[allow(clippy::excessive_precision)]
    fn special_inverse_normal_cdf(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const P0: [f64; 5] = [-5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1, 1.39312609387279679503E1, -1.23916583867381258016E0,];
        #[rustfmt::skip]
        const Q0: [f64; 9] = [1.00000000000000000000E0, 1.95448858338141759834E0, 4.67627912898881538453E0, 8.63602421390890590575E1, -2.25462687854119370527E2, 2.00260212380060660359E2, -8.20372256168333339912E1, 1.59056225126211695515E1, -1.18331621121330003142E0,];
        #[rustfmt::skip]
        const P1: [f64; 9] = [4.05544892305962419923E0, 3.15251094599893866154E1, 5.71628192246421288162E1, 4.40805073893200834700E1, 1.46849561928858024014E1, 2.18663306850790267539E0, -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4,];
        #[rustfmt::skip]
        const Q1: [f64; 9] = [1.00000000000000000000E0, 1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1, 1.50425385692907503408E1, 2.50464946208309415979E0, -1.42182922854787788574E-1, -3.80806407691578277194E-2, -9.33259480895457427372E-4,];
        #[rustfmt::skip]
        const P2: [f64; 9] = [3.23774891776946035970E0, 6.91522889068984211695E0, 3.93881025292474443415E0, 1.33303460815807542389E0, 2.01485389549179081538E-1, 1.23716634817820021358E-2, 3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9,];
        #[rustfmt::skip]
        const Q2: [f64; 9] = [1.00000000000000000000E0, 6.02427039364742014255E0, 3.67983563856160859403E0, 1.37702099489081330271E0, 2.16236993594496635890E-1, 1.34204006088543189037E-2, 3.28014464682127739104E-4, 2.89247864745380683936E-6, 6.79019408009981274425E-9,];

        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let tail_threshold = self.constant_like(value, 0.13533528323661269189);
        let upper_tail = value.gt(one - tail_threshold);
        let tail_probability = self.special_select(upper_tail, one - value, value);

        let centered = tail_probability - self.constant_like(value, 0.5);
        let centered_squared = centered.square();
        let central = centered
            + centered * centered_squared * self.special_polynomial_evaluate(centered_squared, &P0)
                / self.special_polynomial_evaluate(centered_squared, &Q0);
        let central = central * self.constant_like(value, 2.50662827463100050242);

        let root = (tail_probability.log() * -2.0).sqrt();
        let root_reciprocal = root.reciprocal();
        let leading = root - root.log() / root;
        let first = root_reciprocal * self.special_polynomial_evaluate(root_reciprocal, &P1)
            / self.special_polynomial_evaluate(root_reciprocal, &Q1);
        let second = root_reciprocal * self.special_polynomial_evaluate(root_reciprocal, &P2)
            / self.special_polynomial_evaluate(root_reciprocal, &Q2);
        let eight = self.constant_like(value, 8.0);
        let correction = self.special_select(root.lt(eight), first, second);
        let magnitude = leading - correction;
        let tail = self.special_select(upper_tail, magnitude, -magnitude);
        let ordinary = self.special_select(tail_probability.gt(tail_threshold), central, tail);

        let negative_infinity = self.constant_like(value, f64::NEG_INFINITY);
        let positive_infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let is_zero = self.is_zero(value);
        let is_one = self.is_zero(value - one);
        let outside = self.bool_or(value.lt(zero), value.gt(one));
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(outside, input_nan);
        let endpoints = self.special_select(is_zero, negative_infinity, ordinary);
        let endpoints = self.special_select(is_one, positive_infinity, endpoints);
        self.special_select(invalid, nan, endpoints)
    }

    pub(super) fn translate_ndtri(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        Ok(self.special_inverse_normal_cdf(value))
    }

    pub(super) fn translate_erfinv(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        Ok(self.special_inverse_erf(value))
    }

    // -----------------------------------------------------------------
    // Chebyshev polynomials
    // -----------------------------------------------------------------

    /// A tensor operand or a numeric literal as a tensor of `dtype`.
    fn special_numeric_tensor_arg(
        &mut self,
        node: &Node,
        index: usize,
        dtype: DType,
    ) -> Result<GraphTensor> {
        let input = node
            .inputs
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("{} missing input {index}", node.target))?;
        if let Some(name) = input.arg.as_value_name()
            && let Some(&value) = self.values.get(name)
        {
            return Ok(if value.dtype == dtype {
                value
            } else {
                value.cast(dtype)
            });
        }
        let value = input
            .arg
            .as_int()
            .map(|value| value as f64)
            .or_else(|| input.arg.as_float())
            .or_else(|| {
                input
                    .arg
                    .as_bool()
                    .map(|value| if value { 1.0 } else { 0.0 })
            })
            .ok_or_else(|| anyhow::anyhow!("{} input {index} must be numeric", node.target))?;
        Ok(self.floating_scalar(value, dtype))
    }

    pub(super) fn translate_chebyshev_polynomial(
        &mut self,
        node: &Node,
        kind: u8,
        shifted: bool,
    ) -> Result<GraphTensor> {
        if kind > 3 {
            bail!("chebyshev kind {kind} is not ported");
        }
        let dtype = self.output_meta_dtype(node)?;
        let value = self.special_numeric_tensor_arg(node, 0, dtype)?;
        let degree = self.special_numeric_tensor_arg(node, 1, dtype)?;
        let (mut value, degree) = util::broadcast_binary(value, degree);
        if shifted {
            let two = self.constant_like(value, 2.0);
            let one = self.constant_like(value, 1.0);
            value = value * two - one;
        }
        // Casting to an integer dtype only truncates the degree; `trunc`
        // keeps the computation in the tensor's float dtype.
        let degree = degree.trunc();

        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let two = self.constant_like(value, 2.0);
        let half = self.constant_like(value, 0.5);
        let degree_plus_one = degree + one;
        let twice_degree_plus_one = degree * two + one;
        let odd = (degree % two).ne(zero);

        let absolute = self.real_abs(value);
        let angle = self.special_acos(value);
        let inside = match kind {
            0 => self.special_cos(degree * angle),
            1 => (degree_plus_one * angle).sin() / angle.sin(),
            2 => self.special_cos((degree + half) * angle) / self.special_cos(angle * 0.5),
            3 => ((degree + half) * angle).sin() / (angle * 0.5).sin(),
            _ => unreachable!(),
        };

        let hyperbolic_angle = self.special_acosh(absolute);
        let positive_outside = match kind {
            0 => self.special_cosh(degree * hyperbolic_angle),
            1 => {
                self.special_sinh(degree_plus_one * hyperbolic_angle)
                    / self.special_sinh(hyperbolic_angle)
            }
            2 => {
                self.special_cosh((degree + half) * hyperbolic_angle)
                    / self.special_cosh(hyperbolic_angle * 0.5)
            }
            3 => {
                self.special_sinh((degree + half) * hyperbolic_angle)
                    / self.special_sinh(hyperbolic_angle * 0.5)
            }
            _ => unreachable!(),
        };
        let negative_outside_magnitude = match kind {
            0 => self.special_cosh(degree * hyperbolic_angle),
            1 => {
                self.special_sinh(degree_plus_one * hyperbolic_angle)
                    / self.special_sinh(hyperbolic_angle)
            }
            2 => {
                self.special_sinh((degree + half) * hyperbolic_angle)
                    / self.special_sinh(hyperbolic_angle * 0.5)
            }
            3 => {
                self.special_cosh((degree + half) * hyperbolic_angle)
                    / self.special_cosh(hyperbolic_angle * 0.5)
            }
            _ => unreachable!(),
        };
        let negative_outside =
            self.special_select(odd, -negative_outside_magnitude, negative_outside_magnitude);
        let outside = self.special_select(value.lt(zero), negative_outside, positive_outside);
        let mut result = self.special_select(absolute.lt(one), inside, outside);

        let positive_endpoint = match kind {
            0 | 2 => one,
            1 => degree_plus_one,
            3 => twice_degree_plus_one,
            _ => unreachable!(),
        };
        let negative_endpoint_magnitude = match kind {
            0 | 3 => one,
            1 => degree_plus_one,
            2 => twice_degree_plus_one,
            _ => unreachable!(),
        };
        let negative_endpoint = self.special_select(
            odd,
            -negative_endpoint_magnitude,
            negative_endpoint_magnitude,
        );
        let endpoint = self.special_select(value.lt(zero), negative_endpoint, positive_endpoint);
        let is_endpoint = self.is_zero(absolute - one);
        result = self.special_select(is_endpoint, endpoint, result);

        // ATen uses the three-term recurrence for ordinary degrees. Keeping
        // that path for the common finite range reproduces its rounding while
        // the closed forms above cover arbitrary runtime degrees without an
        // unbounded compiler-side loop.
        let mut previous = one;
        let mut current = match kind {
            0 => value,
            1 => value * two,
            2 => value * two - one,
            3 => value * two + one,
            _ => unreachable!(),
        };
        let degree_zero = self.is_zero(degree);
        result = self.special_select(degree_zero, previous, result);
        let degree_one = self.is_zero(degree - one);
        result = self.special_select(degree_one, current, result);
        for index in 2..=20 {
            let next = value * two * current - previous;
            let index_value = self.constant_like(value, index as f64);
            let at_index = self.is_zero(degree - index_value);
            result = self.special_select(at_index, next, result);
            previous = current;
            current = next;
        }

        let negative_degree = degree.lt(zero);
        Ok(self.special_select(negative_degree, zero, result))
    }

    // -----------------------------------------------------------------
    // Gamma family
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_lanczos_lgamma_positive(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const COEFFICIENTS: [f64; 9] = [0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8, 771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905, -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7,];
        let shifted = value - self.constant_like(value, 1.0);
        let mut series = self.constant_like(value, COEFFICIENTS[0]);
        for (index, coefficient) in COEFFICIENTS.iter().copied().enumerate().skip(1) {
            let denominator = shifted + self.constant_like(value, index as f64);
            series += self.constant_like(value, coefficient) / denominator;
        }
        let t = shifted + self.constant_like(value, 7.5);
        let half = self.constant_like(value, 0.5);
        self.constant_like(value, 0.5 * (2.0 * std::f64::consts::PI).ln())
            + (shifted + half) * t.log()
            - t
            + series.log()
    }

    pub(super) fn translate_lgamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        let direct = self.special_lanczos_lgamma_positive(value);
        let one = self.constant_like(value, 1.0);
        let reflected_positive = self.special_lanczos_lgamma_positive(one - value);
        let pi = self.constant_like(value, std::f64::consts::PI);
        let reflection = pi.log() - self.real_abs((value * pi).sin()).log() - reflected_positive;
        let half = self.constant_like(value, 0.5);
        let ordinary = self.special_select(value.lt(half), reflection, direct);
        let infinity = self.constant_like(value, f64::INFINITY);
        let infinite = self.is_inf(value);
        let zero = self.constant_like(value, 0.0);
        let nonpositive = value.le(zero);
        let integral = value.eq(value.floor());
        let pole = self.bool_and(nonpositive, integral);
        let exceptional = self.bool_or(infinite, pole);
        let one = self.constant_like(value, 1.0);
        let two = self.constant_like(value, 2.0);
        let equals_one = value.eq(one);
        let equals_two = value.eq(two);
        let one_or_two = self.bool_or(equals_one, equals_two);
        let ordinary = self.special_select(one_or_two, zero, ordinary);
        let result = self.special_select(exceptional, infinity, ordinary);
        let nan = self.is_nan(value);
        let nan_value = self.constant_like(value, f64::NAN);
        Ok(self.special_select(nan, nan_value, result))
    }

    fn special_positive_digamma(&mut self, value: GraphTensor) -> GraphTensor {
        let mut shifted = value;
        let mut recurrence = self.constant_like(value, 0.0);
        let threshold = self.constant_like(value, 8.0);
        let zero = self.constant_like(value, 0.0);
        for _ in 0..8 {
            let active = shifted.lt(threshold);
            let term = shifted.reciprocal() * -1.0;
            recurrence += self.special_select(active, term, zero);
            shifted += active.cast(value.dtype);
        }
        let inverse = shifted.reciprocal();
        let inverse_squared = inverse.square();
        let correction = self.constant_like(value, 1.0 / 12.0)
            - inverse_squared
                * (self.constant_like(value, 1.0 / 120.0)
                    - inverse_squared
                        * (self.constant_like(value, 1.0 / 252.0)
                            - inverse_squared
                                * (self.constant_like(value, 1.0 / 240.0)
                                    - inverse_squared
                                        * (self.constant_like(value, 5.0 / 660.0)
                                            - inverse_squared
                                                * self.constant_like(value, 691.0 / 32_760.0)))));
        recurrence + shifted.log() - inverse * 0.5 - inverse_squared * correction
    }

    fn special_positive_polygamma(&mut self, value: GraphTensor, order: usize) -> GraphTensor {
        debug_assert!(order >= 1);
        let factorial = |n: usize| (1..=n).fold(1.0, |product, value| product * value as f64);
        let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
        let order_factorial = factorial(order);
        let mut shifted = value;
        let mut recurrence = self.constant_like(value, 0.0);
        let threshold = self.constant_like(value, 8.0);
        let zero = self.constant_like(value, 0.0);
        for _ in 0..8 {
            let active = shifted.lt(threshold);
            let term = shifted.pow(-((order + 1) as f32))
                * self.constant_like(value, sign * order_factorial);
            recurrence += self.special_select(active, term, zero);
            shifted += active.cast(value.dtype);
        }

        const BERNOULLI: [(usize, f64); 6] = [
            (2, 1.0 / 6.0),
            (4, -1.0 / 30.0),
            (6, 1.0 / 42.0),
            (8, -1.0 / 30.0),
            (10, 5.0 / 66.0),
            (12, -691.0 / 2730.0),
        ];
        let mut asymptotic =
            shifted.pow(-(order as f32)) * self.constant_like(value, factorial(order - 1));
        asymptotic +=
            shifted.pow(-((order + 1) as f32)) * self.constant_like(value, 0.5 * order_factorial);
        for (degree, bernoulli) in BERNOULLI {
            let coefficient = bernoulli * factorial(order + degree - 1) / factorial(degree);
            asymptotic +=
                shifted.pow(-((order + degree) as f32)) * self.constant_like(value, coefficient);
        }
        recurrence + asymptotic * self.constant_like(value, sign)
    }

    #[allow(clippy::excessive_precision)]
    fn special_real_polygamma(&mut self, value: GraphTensor, order: usize) -> GraphTensor {
        let positive = if order == 0 {
            self.special_positive_digamma(value)
        } else {
            self.special_positive_polygamma(value, order)
        };
        let one = self.constant_like(value, 1.0);
        let reflected_positive = if order == 0 {
            self.special_positive_digamma(one - value)
        } else {
            self.special_positive_polygamma(one - value, order)
        };
        let pi = self.constant_like(value, std::f64::consts::PI);
        let angle = value * pi;
        let cotangent = self.special_cos(angle) / angle.sin();
        let coefficients = cotangent_derivative_polynomial(order);
        let mut polynomial = self.constant_like(value, *coefficients.last().unwrap_or(&0.0));
        for coefficient in coefficients.iter().rev().skip(1) {
            polynomial = polynomial * cotangent + self.constant_like(value, *coefficient);
        }
        let reflected = if order == 1 {
            // Avoid forming 1 + cot(x)^2 with the cosine approximation. The
            // equivalent csc(x)^2 form is materially more accurate near poles.
            let sine = angle.sin();
            -reflected_positive + pi.square() / sine.square()
        } else {
            let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
            reflected_positive * sign
                - polynomial
                    * self.constant_like(value, std::f64::consts::PI.powi((order + 1) as i32))
        };
        let zero = self.constant_like(value, 0.0);
        let nonpositive = value.le(zero);
        let mut result = self.special_select(nonpositive, reflected, positive);

        let is_zero = self.is_zero(value);
        if order >= 1 {
            let pole = self.constant_like(
                value,
                if order % 2 == 1 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
            );
            result = self.special_select(is_zero, pole, result);
            if order >= 2 {
                let negative = value.lt(zero);
                let integral = value.eq(value.floor());
                let negative_pole = self.bool_and(negative, integral);
                result = self.special_select(negative_pole, pole, result);
            }
        } else {
            let negative = value.lt(zero);
            let integral = value.eq(value.floor());
            let negative_pole = self.bool_and(negative, integral);
            let nan = self.constant_like(value, f64::NAN);
            result = self.special_select(negative_pole, nan, result);
        }

        let infinite = self.is_inf(value);
        let negative_sign = self.signbit(value);
        let positive_sign = self.bool_not(negative_sign);
        let negative_infinite = self.bool_and(infinite, negative_sign);
        let positive_infinite = self.bool_and(infinite, positive_sign);
        let nan = self.constant_like(value, f64::NAN);
        let infinity_result = if order == 0 {
            self.constant_like(value, f64::INFINITY)
        } else if order == 1 {
            zero
        } else {
            nan
        };
        result = self.special_select(positive_infinite, infinity_result, result);
        let negative_infinity_result = if order >= 2 {
            self.constant_like(
                value,
                if order % 2 == 1 {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                },
            )
        } else {
            nan
        };
        result = self.special_select(negative_infinite, negative_infinity_result, result);
        let input_nan = self.is_nan(value);
        self.special_select(input_nan, nan, result)
    }

    pub(super) fn translate_digamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        // ATen evaluates digamma in the input dtype; the reference runtime
        // has no F64 arithmetic or comparisons, so an internal F64 promotion
        // cannot execute. F32 is well inside the op's tolerance.
        Ok(self.special_real_polygamma(value, 0))
    }

    pub(super) fn translate_polygamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let order = self.get_int_arg(node, 0)?;
        anyhow::ensure!(order >= 0, "polygamma order must be nonnegative");
        let value = self.operand(&node.inputs[1])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(value.dtype);
        let value = if value.dtype == dtype {
            value
        } else {
            value.cast(dtype)
        };
        let output_dtype = value.dtype;
        if order == 1 && output_dtype != DType::F64 {
            // ATen evaluates trigamma reflection in the output dtype. Near
            // negative poles that dtype-specific sin(pi*x) rounding is visible
            // in the result, so an internal F64 promotion would not conform.
            return Ok(self.special_real_polygamma(value, order as usize));
        }
        Ok(self
            .special_real_polygamma(value.cast(DType::F64), order as usize)
            .cast(output_dtype))
    }

    // -----------------------------------------------------------------
    // Scaled complementary error function
    // -----------------------------------------------------------------

    #[allow(clippy::excessive_precision)]
    fn special_erfcx_positive(&mut self, value: GraphTensor) -> GraphTensor {
        // This is the positive branch of the Numerical Recipes erfc
        // approximation after cancelling exp(-x^2). Keeping erfcx in this
        // scaled form avoids both the underflow and the growing relative error
        // of computing erfc(x) * exp(x^2) separately.
        let t =
            (value * self.constant_like(value, 0.5) + self.constant_like(value, 1.0)).reciprocal();
        let polynomial = t
            * (t * (t
                * (t * (t
                    * (t * (t
                        * (t * self.constant_like(value, 0.170_872_77)
                            + self.constant_like(value, -0.822_152_23))
                        + self.constant_like(value, 1.488_515_87))
                        + self.constant_like(value, -1.135_203_98))
                    + self.constant_like(value, 0.278_868_07))
                    + self.constant_like(value, -0.186_288_06))
                + self.constant_like(value, 0.096_784_18))
                + self.constant_like(value, 0.374_091_96))
            + self.constant_like(value, 1.000_023_68);
        let offset = self.constant_like(value, -1.265_512_23);
        t * self.special_exp(polynomial * t + offset)
    }

    pub(super) fn translate_erfcx(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        let absolute = self.real_abs(value);
        let positive = self.special_erfcx_positive(absolute);
        let negative = self.constant_like(value, 2.0) * self.special_exp(value.square()) - positive;
        let zero = self.constant_like(value, 0.0);
        Ok(self.special_select(value.lt(zero), negative, positive))
    }

    // -----------------------------------------------------------------
    // logcumsumexp
    // -----------------------------------------------------------------

    pub(super) fn translate_logcumsumexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        if value.rank() == 0 {
            anyhow::ensure!(
                matches!(self.get_int_arg(node, 1)?, -1 | 0),
                "logcumsumexp dimension is out of range for a scalar"
            );
            return Ok(value);
        }
        let rank = value.rank();
        let axis = util::normalize_dim(self.get_int_arg(node, 1)?, rank);
        let length = value.dims()[axis];
        let mut padding = vec![(IntExpr::from(0), IntExpr::from(0)); rank];
        padding[axis] = (length - 1, IntExpr::from(0));
        // The window fill must sit below every real value but must stay
        // FINITE: `pad`'s arithmetic mask multiplies the fill by zero on the
        // real elements, and `-inf * 0` is NaN. The lowest finite float is
        // harmless — it never wins a max and `exp(lowest - max)` underflows
        // to zero.
        let lowest = match value.dtype {
            DType::F64 => f64::MIN,
            _ => f32::MIN as f64,
        };
        let window_fill = self.floating_scalar(lowest, value.dtype);
        let padded = value.pad_with(padding, window_fill);
        let mut kernel = vec![IntExpr::from(1); rank];
        kernel[axis] = length;
        let mut windows = padded.unfold(kernel, vec![1usize; rank], vec![1usize; rank]);
        for kernel_axis in (0..rank).rev() {
            if kernel_axis != axis {
                windows = windows.squeeze(rank + kernel_axis);
            }
        }
        let reduction_axis = rank;
        let maximum = windows.max(reduction_axis);
        let expanded = maximum.expand_dim(reduction_axis, windows.dims()[reduction_axis]);
        let ordinary = maximum + (windows - expanded).exp().sum(reduction_axis).log();

        // Positive infinity is detected directly: `util::signbit` reads the
        // sign through `reciprocal().lt(0)`, and `reciprocal(-inf)` is `-0.0`
        // which is not `< 0`, so the `-inf` window fill would otherwise be
        // counted as a positive infinity.
        let largest = match windows.dtype {
            DType::F64 => f64::MAX,
            _ => f32::MAX as f64,
        };
        let positive_infinite = windows.gt(self.constant_like(windows, largest));
        let positive_count = positive_infinite.cast(DType::F32).sum(reduction_axis);
        let zero = self.constant_like(positive_count, 0.0);
        let has_positive_infinity = positive_count.gt(zero);
        let infinity = self.constant_like(ordinary, f64::INFINITY);
        let result = self.special_select(has_positive_infinity, infinity, ordinary);

        let nan_count = self.is_nan(windows).cast(DType::F32).sum(reduction_axis);
        let zero = self.constant_like(nan_count, 0.0);
        let has_nan = nan_count.gt(zero);
        let nan = self.constant_like(result, f64::NAN);
        Ok(self.special_select(has_nan, nan, result))
    }

    // -----------------------------------------------------------------
    // angle
    // -----------------------------------------------------------------

    pub(super) fn translate_angle(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.special_input(node)?;
        let zero = self.constant_like(value, 0.0);
        Ok(self.special_atan2(zero, value))
    }
}

fn cotangent_derivative_polynomial(order: usize) -> Vec<f64> {
    // If u = cot(pi*x), then D^n u = pi^n P_n(u), with
    // P_{n+1}(u) = -(1 + u^2) P'_n(u).
    let mut coefficients = vec![0.0, 1.0];
    for _ in 0..order {
        let mut derivative = vec![0.0; coefficients.len().saturating_sub(1)];
        for (degree, coefficient) in coefficients.iter().copied().enumerate().skip(1) {
            derivative[degree - 1] = degree as f64 * coefficient;
        }
        let mut next = vec![0.0; derivative.len() + 2];
        for (degree, coefficient) in derivative.into_iter().enumerate() {
            next[degree] -= coefficient;
            next[degree + 2] -= coefficient;
        }
        coefficients = next;
    }
    coefficients
}
