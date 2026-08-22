use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::{broadcast_binary, torch_dtype_int_to_luminal};

use super::Translator;

const ARGSORT_INPUT_ARG: usize = 0;
const ARGSORT_DIM_ARG: usize = 1;
const ARGSORT_DESCENDING_ARG: usize = 2;

const MASKED_FILL_INPUT_ARG: usize = 0;
const MASKED_FILL_MASK_ARG: usize = 1;
const MASKED_FILL_VALUE_ARG: usize = 2;

const FLOOR_DIVIDE_INPUT_ARG: usize = 0;
const FLOOR_DIVIDE_OTHER_ARG: usize = 1;

const DIV_MODE_INPUT_ARG: usize = 0;
const DIV_MODE_OTHER_ARG: usize = 1;

#[derive(Clone, Copy)]
pub(crate) enum ChebyshevKind {
    First,
    Second,
    Third,
    Fourth,
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_argsort(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, ARGSORT_INPUT_ARG)?;
        let dim = if node.inputs.len() > ARGSORT_DIM_ARG {
            self.get_int_arg(node, ARGSORT_DIM_ARG).unwrap_or(-1)
        } else {
            -1
        };
        let descending = if node.inputs.len() > ARGSORT_DESCENDING_ARG {
            self.get_bool_arg(node, ARGSORT_DESCENDING_ARG)
                .unwrap_or(false)
        } else {
            false
        };
        let dim = crate::pt2_util::normalize_dim(dim, a.shape.len());
        // PyTorch's `torch.argsort` returns int64 unconditionally;
        // luminal's frontend `stable_argsort` returns i32 (storage-
        // efficient default for native Rust callers). Cast at the
        // PT2↔luminal boundary so the strict output-read path sees
        // an I64 buffer.
        let sort_key = if a.dtype == DType::Bool {
            a.cast(DType::F32)
        } else {
            a
        };
        Ok(sort_key.stable_argsort(dim, descending).cast(DType::I64))
    }

    pub(crate) fn translate_unary_op(
        &mut self,
        node: &Node,
        f: impl Fn(GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let a = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(f(a))
    }

    pub(crate) fn floor_tensor(&mut self, value: GraphTensor) -> GraphTensor {
        let truncated = value.cast(DType::I64).cast(value.dtype);
        truncated - value.lt(truncated).cast(value.dtype)
    }

    /// Round to nearest with ties to even, using only casts, comparisons and
    /// arithmetic. Integral inputs are already rounded and remain unchanged.
    pub(crate) fn round_to_even(&mut self, value: GraphTensor) -> GraphTensor {
        if matches!(
            value.dtype,
            DType::Int
                | DType::I64
                | DType::I4
                | DType::U4
                | DType::I8
                | DType::U8
                | DType::I16
                | DType::U16
                | DType::Bool
        ) {
            return value;
        }

        let lower = self.floor_tensor(value);
        let half = self.constant_like(value, 0.5);
        let fraction = value - lower;
        let above_half = fraction.gt(half);
        let exactly_half = fraction.eq(half);
        let two = self.constant_like(value, 2.0);
        let odd_lower = (lower % two).ne(self.constant_like(value, 0.0));
        let increment = self.bool_or(above_half, self.bool_and(exactly_half, odd_lower));
        let rounded = lower + increment.cast(value.dtype);

        // Integer casts are not meaningful for IEEE exceptional values.
        let nan = self.is_nan(value);
        let infinite = self.is_inf(value);
        let exceptional = self.bool_or(nan, infinite);
        self.select(exceptional, value, rounded)
    }

    pub(crate) fn translate_round(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        let decimals = self.named_int_arg(node, "decimals").unwrap_or(0);
        if decimals == 0
            || !matches!(
                value.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::F64
            )
        {
            return Ok(self.round_to_even(value));
        }
        let scale = 10_f64.powi(decimals.unsigned_abs().min(i32::MAX as u64) as i32);
        let scale = self.constant_like(value, scale);
        Ok(if decimals > 0 {
            self.round_to_even(value * scale) / scale
        } else {
            self.round_to_even(value / scale) * scale
        })
    }

    pub(crate) fn translate_hardtanh(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        let minimum = self.get_float_arg(node, 1).unwrap_or(-1.0);
        let maximum = self.get_float_arg(node, 2).unwrap_or(1.0);
        let minimum = self.constant_like(value, minimum);
        let maximum = self.constant_like(value, maximum);
        let lower_clamped = self.select(value.lt(minimum), minimum, value);
        Ok(self.select(lower_clamped.gt(maximum), maximum, lower_clamped))
    }

    pub(crate) fn translate_leaky_relu(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let slope = self.get_float_arg(node, 1).unwrap_or(0.01);
        let zero = self.constant_like(value, 0.0);
        let negative = value * self.constant_like(value, slope);
        Ok(self.select(value.gt(zero), value, negative))
    }

    pub(crate) fn translate_elu(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let alpha = self.get_float_arg(node, 1).unwrap_or(1.0);
        let scale = self.get_float_arg(node, 2).unwrap_or(1.0);
        let input_scale = self.get_float_arg(node, 3).unwrap_or(1.0);
        let zero = self.constant_like(value, 0.0);
        let input_scale = self.constant_like(value, input_scale);
        let one = self.constant_like(value, 1.0);
        let negative =
            (self.real_exp(value * input_scale) - one) * self.constant_like(value, alpha);
        let selected = self.select(value.gt(zero), value, negative);
        Ok(selected * self.constant_like(value, scale))
    }

    pub(crate) fn translate_signbit(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        Ok(
            if matches!(
                value.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::F64
            ) {
                self.signbit(value)
            } else {
                value.lt(self.constant_like(value, 0.0))
            },
        )
    }

    #[allow(clippy::excessive_precision)]
    pub(crate) fn real_erf(&mut self, value: GraphTensor) -> GraphTensor {
        // Abramowitz & Stegun 7.1.28 (maximum error about 1.5e-7).
        let absolute = self.real_abs(value);
        let t = (absolute * self.constant_like(value, 0.327_591_1)
            + self.constant_like(value, 1.0))
        .reciprocal();
        let polynomial = t
            * (t * (t
                * (t * (t * self.constant_like(value, 1.061_405_429)
                    + self.constant_like(value, -1.453_152_027))
                    + self.constant_like(value, 1.421_413_741))
                + self.constant_like(value, -0.284_496_736))
                + self.constant_like(value, 0.254_829_592));
        let magnitude =
            self.constant_like(value, 1.0) - polynomial * self.real_exp(value.square() * -1.0);
        self.copy_sign(magnitude, value)
    }

    pub(crate) fn translate_erf(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.real_erf(value))
    }

    pub(crate) fn translate_erfc(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.real_erfc(value))
    }

    fn erfcx_positive(&mut self, value: GraphTensor) -> GraphTensor {
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
        t * self.real_exp(polynomial * t + offset)
    }

    pub(crate) fn real_erfc(&mut self, value: GraphTensor) -> GraphTensor {
        let absolute = self.real_abs(value);
        let positive = self.erfcx_positive(absolute) * self.real_exp(absolute.square() * -1.0);
        let negative = self.constant_like(value, 2.0) - positive;
        let zero = self.constant_like(value, 0.0);
        self.select(value.lt(zero), negative, positive)
    }

    pub(crate) fn translate_erfcx(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let absolute = self.real_abs(value);
        let positive = self.erfcx_positive(absolute);
        let negative = self.constant_like(value, 2.0) * self.real_exp(value.square()) - positive;
        let zero = self.constant_like(value, 0.0);
        Ok(self.select(value.lt(zero), negative, positive))
    }

    fn chebyshev_evaluate(&mut self, value: GraphTensor, coefficients: &[f64]) -> GraphTensor {
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

    #[allow(clippy::excessive_precision)]
    fn modified_bessel_i0(&mut self, value: GraphTensor, scaled: bool) -> GraphTensor {
        #[rustfmt::skip]
        const SMALL: [f64; 30] = [-4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16, 1.71539128555513303061E-15, -1.16853328779934516808E-14, 7.67618549860493561688E-14, -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11, 9.67580903537323691224E-11, -5.18979560163526290666E-10, 2.65982372468238635035E-9, -1.30002500998624804212E-8, 6.04699502254191894932E-8, -2.67079385394061173391E-7, 1.11738753912010371815E-6, -4.41673835845875056359E-6, 1.64484480707288970893E-5, -5.75419501008210370398E-5, 1.88502885095841655729E-4, -5.76375574538582365885E-4, 1.63947561694133579842E-3, -4.32430999505057594430E-3, 1.05464603945949983183E-2, -2.37374148058994688156E-2, 4.93052842396707084878E-2, -9.49010970480476444210E-2, 1.71620901522208775349E-1, -3.04682672343198398683E-1, 6.76795274409476084995E-1,];
        #[rustfmt::skip]
        const LARGE: [f64; 25] = [-7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17, 3.46122286769746109310E-17, -2.82762398051658348494E-16, -3.42548561967721913462E-16, 1.77256013305652638360E-15, 3.81168066935262242075E-15, -9.55484669882830764870E-15, -4.15056934728722208663E-14, 1.54008621752140982691E-14, 3.85277838274214270114E-13, 7.18012445138366623367E-13, -1.79417853150680611778E-12, -1.32158118404477131188E-11, -3.14991652796324136454E-11, 1.18891471078464383424E-11, 4.94060238822496958910E-10, 3.39623202570838634515E-9, 2.26666899049817806459E-8, 2.04891858946906374183E-7, 2.89137052083475648297E-6, 6.88975834691682398426E-5, 3.36911647825569408990E-3, 8.04490411014108831608E-1,];

        self.modified_bessel_i(value, &SMALL, &LARGE, 0, scaled)
    }

    #[allow(clippy::excessive_precision)]
    fn modified_bessel_i1(&mut self, value: GraphTensor, scaled: bool) -> GraphTensor {
        #[rustfmt::skip]
        const SMALL: [f64; 29] = [2.77791411276104639959E-18, -2.11142121435816608115E-17, 1.55363195773620046921E-16, -1.10559694773538630805E-15, 7.60068429473540693410E-15, -5.04218550472791168711E-14, 3.22379336594557470981E-13, -1.98397439776494371520E-12, 1.17361862988909016308E-11, -6.66348972350202774223E-11, 3.62559028155211703701E-10, -1.88724975172282928790E-9, 9.38153738649577178388E-9, -4.44505912879632808065E-8, 2.00329475355213526229E-7, -8.56872026469545474066E-7, 3.47025130813767847674E-6, -1.32731636560394358279E-5, 4.78156510755005422638E-5, -1.61760815825896745588E-4, 5.12285956168575772895E-4, -1.51357245063125314899E-3, 4.15642294431288815669E-3, -1.05640848946261981558E-2, 2.47264490306265168283E-2, -5.29459812080949914269E-2, 1.02643658689847095384E-1, -1.76416518357834055153E-1, 2.52587186443633654823E-1,];
        #[rustfmt::skip]
        const LARGE: [f64; 25] = [7.51729631084210481353E-18, 4.41434832307170791151E-18, -4.65030536848935832153E-17, -3.20952592199342395980E-17, 2.96262899764595013876E-16, 3.30820231092092828324E-16, -1.88035477551078244854E-15, -3.81440307243700780478E-15, 1.04202769841288027642E-14, 4.27244001671195135429E-14, -2.10154184277266431302E-14, -4.08355111109219731823E-13, -7.19855177624590851209E-13, 2.03562854414708950722E-12, 1.41258074366137813316E-11, 3.25260358301548823856E-11, -1.89749581235054123450E-11, -5.58974346219658380687E-10, -3.83538038596423702205E-9, -2.63146884688951950684E-8, -2.51223623787020892529E-7, -3.88256480887769039346E-6, -1.10588938762623716291E-4, -9.76109749136146840777E-3, 7.78576235018280120474E-1,];

        self.modified_bessel_i(value, &SMALL, &LARGE, 1, scaled)
    }

    fn modified_bessel_i(
        &mut self,
        value: GraphTensor,
        small_coefficients: &[f64],
        large_coefficients: &[f64],
        order: usize,
        scaled: bool,
    ) -> GraphTensor {
        let absolute = self.real_abs(value);
        let small_argument = absolute * 0.5 - self.constant_like(value, 2.0);
        let mut small = self.chebyshev_evaluate(small_argument, small_coefficients);
        if order == 1 {
            small *= absolute;
        }
        let large_argument =
            self.constant_like(value, 32.0) / absolute - self.constant_like(value, 2.0);
        let mut large =
            self.chebyshev_evaluate(large_argument, large_coefficients) / absolute.sqrt();
        if !scaled {
            let exponential = self.real_exp(absolute);
            small *= exponential;
            large *= exponential;
        }
        let threshold = self.constant_like(value, 8.0);
        let magnitude = self.select(absolute.le(threshold), small, large);
        if order == 0 {
            magnitude
        } else {
            let zero = self.constant_like(value, 0.0);
            self.select(value.lt(zero), -magnitude, magnitude)
        }
    }

    pub(crate) fn translate_modified_bessel(
        &mut self,
        node: &Node,
        order: usize,
        scaled: bool,
    ) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(if order == 0 {
            self.modified_bessel_i0(value, scaled)
        } else {
            self.modified_bessel_i1(value, scaled)
        })
    }

    pub(crate) fn translate_spherical_bessel_j0(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let finite = value.sin() / value;
        let is_zero = self.is_zero(value);
        let with_zero = self.select(is_zero, one, finite);
        let infinite = self.is_inf(value);
        Ok(self.select(infinite, zero, with_zero))
    }

    fn polynomial_evaluate(&mut self, value: GraphTensor, coefficients: &[f64]) -> GraphTensor {
        let mut result = self.constant_like(value, coefficients[0]);
        for coefficient in coefficients.iter().copied().skip(1) {
            result = result * value + self.constant_like(value, coefficient);
        }
        result
    }

    fn polynomial_with_leading_one(
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

    #[allow(clippy::excessive_precision)]
    fn cylindrical_bessel_asymptotic(
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
        let p = self.polynomial_evaluate(reciprocal_square, pp)
            / self.polynomial_evaluate(reciprocal_square, pq);
        let q = self.constant_like(value, 5.0) / value
            * self.polynomial_evaluate(reciprocal_square, qp)
            / self.polynomial_evaluate(reciprocal_square, qq);
        let angle = value - self.constant_like(value, phase);
        let cosine = self.real_cos(angle);
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
    fn cylindrical_bessel_j0(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const RP: [f64; 4] = [-4.79443220978201773821e+09, 1.95617491946556577543e+12, -2.49248344360967716204e+14, 9.70862251047306323952e+15,];
        #[rustfmt::skip]
        const RQ: [f64; 8] = [4.99563147152651017219e+02, 1.73785401676374683123e+05, 4.84409658339962045305e+07, 1.11855537045356834862e+10, 2.11277520115489217587e+12, 3.10518229857422583814e+14, 3.18121955943204943306e+16, 1.71086294081043136091e+18,];

        let absolute = self.real_abs(value);
        let squared = absolute.square();
        let ratio = self.polynomial_evaluate(squared, &RP) / self.polynomial_evaluate(squared, &RQ);
        let ordinary = (squared - self.constant_like(value, 5.78318596294678452118))
            * (squared - self.constant_like(value, 30.4712623436620863991))
            * ratio;
        let near_zero = self.constant_like(value, 1.0) - squared * 0.25;
        let tiny = self.constant_like(value, 1.0e-5);
        let small = self.select(absolute.lt(tiny), near_zero, ordinary);
        let large = self.cylindrical_bessel_asymptotic(absolute, 0, false);
        let threshold = self.constant_like(value, 5.0);
        self.select(absolute.le(threshold), small, large)
    }

    #[allow(clippy::excessive_precision)]
    fn cylindrical_bessel_j1(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const RP: [f64; 4] = [-8.99971225705559398224e+08, 4.52228297998194034323e+11, -7.27494245221818276015e+13, 3.68295732863852883286e+15,];
        #[rustfmt::skip]
        const RQ: [f64; 8] = [6.20836478118054335476e+02, 2.56987256757748830383e+05, 8.35146791431949253037e+07, 2.21511595479792499675e+10, 4.74914122079991414898e+12, 7.84369607876235854894e+14, 8.95222336184627338078e+16, 5.32278620332680085395e+18,];

        let absolute = self.real_abs(value);
        let squared = absolute.square();
        let small = self.polynomial_evaluate(squared, &RP) / self.polynomial_evaluate(squared, &RQ)
            * absolute
            * (squared - self.constant_like(value, 14.6819706421238932572))
            * (squared - self.constant_like(value, 49.2184563216946036703));
        let large = self.cylindrical_bessel_asymptotic(absolute, 1, false);
        let threshold = self.constant_like(value, 5.0);
        let magnitude = self.select(absolute.le(threshold), small, large);
        self.copy_sign(magnitude, value)
    }

    #[allow(clippy::excessive_precision)]
    fn cylindrical_bessel_y0(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const YP: [f64; 8] = [1.55924367855235737965e+04, -1.46639295903971606143e+07, 5.43526477051876500413e+09, -9.82136065717911466409e+11, 8.75906394395366999549e+13, -3.46628303384729719441e+15, 4.42733268572569800351e+16, -1.84950800436986690637e+16,];
        #[rustfmt::skip]
        const YQ: [f64; 7] = [1.04128353664259848412e+03, 6.26107330137134956842e+05, 2.68919633393814121987e+08, 8.64002487103935000337e+10, 2.02979612750105546709e+13, 3.17157752842975028269e+15, 2.50596256172653059228e+17,];

        let squared = value.square();
        let small = self.polynomial_evaluate(squared, &YP) / self.polynomial_evaluate(squared, &YQ)
            + self.constant_like(value, std::f64::consts::FRAC_2_PI)
                * value.log()
                * self.cylindrical_bessel_j0(value);
        let large = self.cylindrical_bessel_asymptotic(value, 0, true);
        let threshold = self.constant_like(value, 5.0);
        let ordinary = self.select(value.le(threshold), small, large);
        self.finish_cylindrical_bessel_y(value, ordinary)
    }

    #[allow(clippy::excessive_precision)]
    fn cylindrical_bessel_y1(&mut self, value: GraphTensor) -> GraphTensor {
        #[rustfmt::skip]
        const YP: [f64; 6] = [1.26320474790178026440e+09, -6.47355876379160291031e+11, 1.14509511541823727583e+14, -8.12770255501325109621e+15, 2.02439475713594898196e+17, -7.78877196265950026825e+17,];
        #[rustfmt::skip]
        const YQ: [f64; 8] = [5.94301592346128195359e+02, 2.35564092943068577943e+05, 7.34811944459721705660e+07, 1.87601316108706159478e+10, 3.88231277496238566008e+12, 6.20557727146953693363e+14, 6.87141087355300489866e+16, 3.97270608116560655612e+18,];

        let squared = value.square();
        let small = value * self.polynomial_evaluate(squared, &YP)
            / self.polynomial_evaluate(squared, &YQ)
            + self.constant_like(value, std::f64::consts::FRAC_2_PI)
                * (self.cylindrical_bessel_j1(value) * value.log() - value.reciprocal());
        let large = self.cylindrical_bessel_asymptotic(value, 1, true);
        let threshold = self.constant_like(value, 5.0);
        let ordinary = self.select(value.le(threshold), small, large);
        self.finish_cylindrical_bessel_y(value, ordinary)
    }

    fn finish_cylindrical_bessel_y(
        &mut self,
        value: GraphTensor,
        ordinary: GraphTensor,
    ) -> GraphTensor {
        let negative_infinity = self.constant_like(value, f64::NEG_INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let zero = self.is_zero(value);
        let with_zero = self.select(zero, negative_infinity, ordinary);
        let zero_value = self.constant_like(value, 0.0);
        let negative = value.lt(zero_value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(negative, input_nan);
        self.select(invalid, nan, with_zero)
    }

    pub(crate) fn translate_cylindrical_bessel(
        &mut self,
        node: &Node,
        order: usize,
        second_kind: bool,
    ) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(match (order, second_kind) {
            (0, false) => self.cylindrical_bessel_j0(value),
            (1, false) => self.cylindrical_bessel_j1(value),
            (0, true) => self.cylindrical_bessel_y0(value),
            (1, true) => self.cylindrical_bessel_y1(value),
            _ => unreachable!(),
        })
    }

    #[allow(clippy::excessive_precision)]
    fn modified_bessel_k(&mut self, value: GraphTensor, order: usize, scaled: bool) -> GraphTensor {
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
        let small_series = self.chebyshev_evaluate(small_argument, small_coefficients);
        let half = self.constant_like(value, 0.5);
        let small = if order == 0 {
            small_series - (value * half).log() * self.modified_bessel_i0(value, false)
        } else {
            (value * half).log() * self.modified_bessel_i1(value, false) + small_series / value
        };
        let eight = self.constant_like(value, 8.0);
        let large_argument = eight / value - two;
        let large = self.chebyshev_evaluate(large_argument, large_coefficients) / value.sqrt();
        let small = if scaled {
            small * self.real_exp(value)
        } else {
            small
        };
        let large = if scaled {
            large
        } else {
            self.real_exp(-value) * large
        };
        let ordinary = self.select(value.le(two), small, large);
        let infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let zero = self.is_zero(value);
        let with_zero = self.select(zero, infinity, ordinary);
        let zero_value = self.constant_like(value, 0.0);
        let negative = value.lt(zero_value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(negative, input_nan);
        self.select(invalid, nan, with_zero)
    }

    pub(crate) fn translate_modified_bessel_k(
        &mut self,
        node: &Node,
        order: usize,
        scaled: bool,
    ) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.modified_bessel_k(value, order, scaled))
    }

    #[allow(clippy::excessive_precision)]
    fn airy_ai(&mut self, value: GraphTensor) -> GraphTensor {
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
            + z_squared * self.polynomial_evaluate(z_squared, &AFN)
                / self.polynomial_evaluate(z_squared, &AFD);
        let g = z * self.polynomial_evaluate(z_squared, &AGN)
            / self.polynomial_evaluate(z_squared, &AGD);
        let angle = negative_phase + self.constant_like(value, std::f64::consts::FRAC_PI_4);
        let negative = self.constant_like(value, 0.564189583547756286948) / negative_root.sqrt()
            * (angle.sin() * f - self.real_cos(angle) * g);

        let positive_root = value.sqrt();
        let two_thirds = self.constant_like(value, 2.0 / 3.0);
        let zeta = value * positive_root * two_thirds;
        let inverse_zeta = zeta.reciprocal();
        let positive = self.constant_like(value, 0.564189583547756286948)
            * (self.polynomial_evaluate(inverse_zeta, &AN)
                / self.polynomial_evaluate(inverse_zeta, &AD))
            * self.real_exp(-zeta)
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
        let lower_or_central = self.select(value.lt(negative_threshold), negative, central);
        let ordinary = self.select(value.ge(positive_threshold), positive, lower_or_central);
        let underflow_threshold = self.constant_like(value, 103.892);
        let zero = self.constant_like(value, 0.0);
        let ordinary = self.select(value.gt(underflow_threshold), zero, ordinary);
        let infinite = self.is_inf(value);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(infinite, input_nan);
        let nan = self.constant_like(value, f64::NAN);
        self.select(invalid, nan, ordinary)
    }

    pub(crate) fn translate_airy_ai(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.airy_ai(value))
    }

    #[allow(clippy::excessive_precision)]
    fn accurate_real_erf(&mut self, value: GraphTensor) -> GraphTensor {
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
        let central = absolute * self.polynomial_evaluate(squared, &T)
            / self.polynomial_with_leading_one(squared, &U);
        let moderate =
            self.polynomial_evaluate(absolute, &P) / self.polynomial_with_leading_one(absolute, &Q);
        let large =
            self.polynomial_evaluate(absolute, &R) / self.polynomial_with_leading_one(absolute, &S);
        let eight = self.constant_like(value, 8.0);
        let erfc_ratio = self.select(absolute.lt(eight), moderate, large);
        let erfc = self.real_exp(squared * -1.0) * erfc_ratio;
        let one = self.constant_like(value, 1.0);
        let tail = one - erfc;
        let magnitude = self.select(absolute.lt(one), central, tail);
        let infinite = self.is_inf(value);
        let magnitude = self.select(infinite, one, magnitude);
        self.copy_sign(magnitude, value)
    }

    #[allow(clippy::excessive_precision)]
    fn inverse_erf(&mut self, value: GraphTensor) -> GraphTensor {
        const A: [f64; 4] = [-0.140543331, 0.914624893, -1.645349621, 0.886226899];
        const B: [f64; 4] = [0.012229801, -0.329097515, 1.442710462, -2.118377725];
        const C: [f64; 4] = [1.641345311, 3.429567803, -1.624906493, -1.970840454];
        const D: [f64; 2] = [3.543889200, 1.637067800];

        let absolute = self.real_abs(value);
        let squared = value.square();
        let central_numerator = self.polynomial_evaluate(squared, &A);
        let central_denominator =
            squared * self.polynomial_evaluate(squared, &B) + self.constant_like(value, 1.0);
        let central = value * central_numerator / central_denominator;

        let one = self.constant_like(value, 1.0);
        let tail_root = (((one - absolute) * 0.5).log() * -1.0).sqrt();
        let tail_numerator = self.polynomial_evaluate(tail_root, &C);
        let tail_denominator = tail_root
            * (tail_root * self.constant_like(value, D[1]) + self.constant_like(value, D[0]))
            + one;
        let tail = self.copy_sign(tail_numerator, value) / tail_denominator;
        let threshold = self.constant_like(value, 0.7);
        let mut result = self.select(absolute.le(threshold), central, tail);

        let derivative_scale = self.constant_like(value, std::f64::consts::FRAC_2_SQRT_PI);
        for _ in 0..2 {
            let error = self.accurate_real_erf(result) - value;
            let derivative = derivative_scale * self.real_exp(result.square() * -1.0);
            result -= error / derivative;
        }

        let infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let endpoint = self.is_zero(absolute - one);
        let signed_infinity = self.copy_sign(infinity, value);
        result = self.select(endpoint, signed_infinity, result);
        let outside = absolute.gt(one);
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(outside, input_nan);
        self.select(invalid, nan, result)
    }

    #[allow(clippy::excessive_precision)]
    fn inverse_normal_cdf(&mut self, value: GraphTensor) -> GraphTensor {
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
        let tail_probability = self.select(upper_tail, one - value, value);

        let centered = tail_probability - self.constant_like(value, 0.5);
        let centered_squared = centered.square();
        let central = centered
            + centered * centered_squared * self.polynomial_evaluate(centered_squared, &P0)
                / self.polynomial_evaluate(centered_squared, &Q0);
        let central = central * self.constant_like(value, 2.50662827463100050242);

        let root = (tail_probability.log() * -2.0).sqrt();
        let root_reciprocal = root.reciprocal();
        let leading = root - root.log() / root;
        let first = root_reciprocal * self.polynomial_evaluate(root_reciprocal, &P1)
            / self.polynomial_evaluate(root_reciprocal, &Q1);
        let second = root_reciprocal * self.polynomial_evaluate(root_reciprocal, &P2)
            / self.polynomial_evaluate(root_reciprocal, &Q2);
        let eight = self.constant_like(value, 8.0);
        let correction = self.select(root.lt(eight), first, second);
        let magnitude = leading - correction;
        let tail = self.select(upper_tail, magnitude, -magnitude);
        let ordinary = self.select(tail_probability.gt(tail_threshold), central, tail);

        let negative_infinity = self.constant_like(value, f64::NEG_INFINITY);
        let positive_infinity = self.constant_like(value, f64::INFINITY);
        let nan = self.constant_like(value, f64::NAN);
        let is_zero = self.is_zero(value);
        let is_one = self.is_zero(value - one);
        let outside = self.bool_or(value.lt(zero), value.gt(one));
        let input_nan = self.is_nan(value);
        let invalid = self.bool_or(outside, input_nan);
        let endpoints = self.select(is_zero, negative_infinity, ordinary);
        let endpoints = self.select(is_one, positive_infinity, endpoints);
        self.select(invalid, nan, endpoints)
    }

    pub(crate) fn translate_ndtri(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.inverse_normal_cdf(value))
    }

    pub(crate) fn translate_erfinv(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        Ok(self.inverse_erf(value))
    }

    fn numeric_tensor_arg(
        &mut self,
        node: &Node,
        index: usize,
        dtype: DType,
    ) -> Result<GraphTensor> {
        if let Some(name) = node.inputs[index].arg.as_tensor_name() {
            return Ok(self.get_tensor(name)?.cast(dtype));
        }
        let value = node.inputs[index]
            .arg
            .as_int()
            .map(|value| value as f64)
            .or_else(|| node.inputs[index].arg.as_float())
            .or_else(|| {
                node.inputs[index]
                    .arg
                    .as_bool()
                    .map(|value| if value { 1.0 } else { 0.0 })
            })
            .ok_or_else(|| anyhow::anyhow!("{} input {index} must be numeric", node.target))?;
        Ok(self.floating_scalar(value, dtype))
    }

    pub(crate) fn translate_chebyshev_polynomial(
        &mut self,
        node: &Node,
        kind: ChebyshevKind,
        shifted: bool,
    ) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        let value = self.numeric_tensor_arg(node, 0, dtype)?;
        let degree = self.numeric_tensor_arg(node, 1, dtype)?;
        let (mut value, mut degree) = broadcast_binary(value, degree);
        if shifted {
            value = value * 2.0 - self.constant_like(value, 1.0);
        }

        let integer_dtype = if dtype == DType::F64 {
            DType::I64
        } else {
            DType::Int
        };
        degree = degree.cast(integer_dtype).cast(dtype);
        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let two = self.constant_like(value, 2.0);
        let half = self.constant_like(value, 0.5);
        let degree_plus_one = degree + one;
        let twice_degree_plus_one = degree * 2.0 + one;
        let odd = (degree % two).ne(zero);

        let absolute = self.real_abs(value);
        let angle = self.real_acos(value);
        let inside = match kind {
            ChebyshevKind::First => self.real_cos(degree * angle),
            ChebyshevKind::Second => (degree_plus_one * angle).sin() / angle.sin(),
            ChebyshevKind::Third => {
                self.real_cos((degree + half) * angle) / self.real_cos(angle * 0.5)
            }
            ChebyshevKind::Fourth => ((degree + half) * angle).sin() / (angle * 0.5).sin(),
        };

        let hyperbolic_angle = self.real_acosh(absolute);
        let positive_outside = match kind {
            ChebyshevKind::First => self.real_cosh(degree * hyperbolic_angle),
            ChebyshevKind::Second => {
                self.real_sinh(degree_plus_one * hyperbolic_angle)
                    / self.real_sinh(hyperbolic_angle)
            }
            ChebyshevKind::Third => {
                self.real_cosh((degree + half) * hyperbolic_angle)
                    / self.real_cosh(hyperbolic_angle * 0.5)
            }
            ChebyshevKind::Fourth => {
                self.real_sinh((degree + half) * hyperbolic_angle)
                    / self.real_sinh(hyperbolic_angle * 0.5)
            }
        };
        let negative_outside_magnitude = match kind {
            ChebyshevKind::First => self.real_cosh(degree * hyperbolic_angle),
            ChebyshevKind::Second => {
                self.real_sinh(degree_plus_one * hyperbolic_angle)
                    / self.real_sinh(hyperbolic_angle)
            }
            ChebyshevKind::Third => {
                self.real_sinh((degree + half) * hyperbolic_angle)
                    / self.real_sinh(hyperbolic_angle * 0.5)
            }
            ChebyshevKind::Fourth => {
                self.real_cosh((degree + half) * hyperbolic_angle)
                    / self.real_cosh(hyperbolic_angle * 0.5)
            }
        };
        let negative_outside =
            self.select(odd, -negative_outside_magnitude, negative_outside_magnitude);
        let outside = self.select(value.lt(zero), negative_outside, positive_outside);
        let mut result = self.select(absolute.lt(one), inside, outside);

        let positive_endpoint = match kind {
            ChebyshevKind::First | ChebyshevKind::Third => one,
            ChebyshevKind::Second => degree_plus_one,
            ChebyshevKind::Fourth => twice_degree_plus_one,
        };
        let negative_endpoint_magnitude = match kind {
            ChebyshevKind::First | ChebyshevKind::Fourth => one,
            ChebyshevKind::Second => degree_plus_one,
            ChebyshevKind::Third => twice_degree_plus_one,
        };
        let negative_endpoint = self.select(
            odd,
            -negative_endpoint_magnitude,
            negative_endpoint_magnitude,
        );
        let endpoint = self.select(value.lt(zero), negative_endpoint, positive_endpoint);
        let is_endpoint = self.is_zero(absolute - one);
        result = self.select(is_endpoint, endpoint, result);

        // ATen uses the three-term recurrence for ordinary degrees. Keeping
        // that path for the common finite range reproduces its rounding while
        // the closed forms above cover arbitrary runtime degrees without an
        // unbounded compiler-side loop.
        let mut previous = one;
        let mut current = match kind {
            ChebyshevKind::First => value,
            ChebyshevKind::Second => value * 2.0,
            ChebyshevKind::Third => value * 2.0 - one,
            ChebyshevKind::Fourth => value * 2.0 + one,
        };
        let degree_zero = self.is_zero(degree);
        result = self.select(degree_zero, previous, result);
        let degree_one = self.is_zero(degree - one);
        result = self.select(degree_one, current, result);
        for index in 2..=20 {
            let next = value * 2.0 * current - previous;
            let index_value = self.constant_like(value, index as f64);
            let at_index = self.is_zero(degree - index_value);
            result = self.select(at_index, next, result);
            previous = current;
            current = next;
        }

        let negative_degree = degree.lt(zero);
        Ok(self.select(negative_degree, zero, result))
    }

    #[allow(clippy::excessive_precision)]
    fn lanczos_lgamma_positive(&mut self, value: GraphTensor) -> GraphTensor {
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

    pub(crate) fn translate_lgamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let direct = self.lanczos_lgamma_positive(value);
        let one = self.constant_like(value, 1.0);
        let reflected_positive = self.lanczos_lgamma_positive(one - value);
        let pi = self.constant_like(value, std::f64::consts::PI);
        let reflection = pi.log() - self.real_abs((value * pi).sin()).log() - reflected_positive;
        let half = self.constant_like(value, 0.5);
        let ordinary = self.select(value.lt(half), reflection, direct);
        let infinity = self.constant_like(value, f64::INFINITY);
        let infinite = self.is_inf(value);
        let zero = self.constant_like(value, 0.0);
        let nonpositive = value.le(zero);
        let integral = value.eq(self.floor_tensor(value));
        let pole = self.bool_and(nonpositive, integral);
        let exceptional = self.bool_or(infinite, pole);
        let one = self.constant_like(value, 1.0);
        let two = self.constant_like(value, 2.0);
        let equals_one = value.eq(one);
        let equals_two = value.eq(two);
        let one_or_two = self.bool_or(equals_one, equals_two);
        let ordinary = self.select(one_or_two, zero, ordinary);
        let result = self.select(exceptional, infinity, ordinary);
        let nan = self.is_nan(value);
        let nan_value = self.constant_like(value, f64::NAN);
        Ok(self.select(nan, nan_value, result))
    }

    fn positive_digamma(&mut self, value: GraphTensor) -> GraphTensor {
        let mut shifted = value;
        let mut recurrence = self.constant_like(value, 0.0);
        let threshold = self.constant_like(value, 8.0);
        let zero = self.constant_like(value, 0.0);
        for _ in 0..8 {
            let active = shifted.lt(threshold);
            let term = shifted.reciprocal() * -1.0;
            recurrence += self.select(active, term, zero);
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

    fn positive_polygamma(&mut self, value: GraphTensor, order: usize) -> GraphTensor {
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
            recurrence += self.select(active, term, zero);
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

    fn real_polygamma(&mut self, value: GraphTensor, order: usize) -> GraphTensor {
        let positive = if order == 0 {
            self.positive_digamma(value)
        } else {
            self.positive_polygamma(value, order)
        };
        let one = self.constant_like(value, 1.0);
        let reflected_positive = if order == 0 {
            self.positive_digamma(one - value)
        } else {
            self.positive_polygamma(one - value, order)
        };
        let pi = self.constant_like(value, std::f64::consts::PI);
        let angle = value * pi;
        let cotangent = self.real_cos(angle) / angle.sin();
        let coefficients = Self::cotangent_derivative_polynomial(order);
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
            reflected_positive * if order.is_multiple_of(2) { 1.0 } else { -1.0 }
                - polynomial
                    * self.constant_like(value, std::f64::consts::PI.powi((order + 1) as i32))
        };
        let zero = self.constant_like(value, 0.0);
        let nonpositive = value.le(zero);
        let mut result = self.select(nonpositive, reflected, positive);

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
            result = self.select(is_zero, pole, result);
            if order >= 2 {
                let negative = value.lt(zero);
                let integral = value.eq(self.floor_tensor(value));
                let negative_pole = self.bool_and(negative, integral);
                result = self.select(negative_pole, pole, result);
            }
        } else {
            let negative = value.lt(zero);
            let integral = value.eq(self.floor_tensor(value));
            let negative_pole = self.bool_and(negative, integral);
            let nan = self.constant_like(value, f64::NAN);
            result = self.select(negative_pole, nan, result);
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
        result = self.select(positive_infinite, infinity_result, result);
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
        result = self.select(negative_infinite, negative_infinity_result, result);
        let input_nan = self.is_nan(value);
        self.select(input_nan, nan, result)
    }

    pub(crate) fn translate_digamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let output_dtype = value.dtype;
        Ok(self
            .real_polygamma(value.cast(DType::F64), 0)
            .cast(output_dtype))
    }

    pub(crate) fn translate_polygamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let order = self.get_int_arg(node, 0)?;
        anyhow::ensure!(order >= 0, "polygamma order must be nonnegative");
        let value = self
            .get_input_tensor(node, 1)?
            .cast(self.output_meta_dtype(node)?);
        let output_dtype = value.dtype;
        if order == 1 && output_dtype != DType::F64 {
            // ATen evaluates trigamma reflection in the output dtype. Near
            // negative poles that dtype-specific sin(pi*x) rounding is visible
            // in the result, so an internal F64 promotion would not conform.
            return Ok(self.real_polygamma(value, order as usize));
        }
        let accurate = self
            .real_polygamma(value.cast(DType::F64), order as usize)
            .cast(output_dtype);
        Ok(accurate)
    }

    pub(crate) fn translate_logcumsumexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        if value.shape.is_empty() {
            anyhow::ensure!(
                matches!(self.get_int_arg(node, 1)?, -1 | 0),
                "logcumsumexp dimension is out of range for a scalar"
            );
            return Ok(value);
        }
        let axis = crate::pt2_util::normalize_dim(self.get_int_arg(node, 1)?, value.shape.len());
        let rank = value.shape.len();
        let length = value.dims()[axis];
        let mut padding = vec![(Expression::from(0), Expression::from(0)); rank];
        padding[axis] = (length - 1, Expression::from(0));
        let negative_infinity = self.floating_scalar(f64::NEG_INFINITY, value.dtype);
        let padded = value.pad_with(padding, negative_infinity);
        let mut kernel = vec![Expression::from(1); rank];
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

        let infinite = self.is_inf(windows);
        let negative = self.signbit(windows);
        let positive_infinite = self.bool_and(infinite, self.bool_not(negative));
        let positive_count = positive_infinite.cast(DType::Int).sum(reduction_axis);
        let zero = self.graph.constant(0).expand_rhs(positive_count.shape);
        let has_positive_infinity = positive_count.gt(zero);
        let infinity = self.constant_like(ordinary, f64::INFINITY);
        let result = self.select(has_positive_infinity, infinity, ordinary);

        let nan_count = self.is_nan(windows).cast(DType::Int).sum(reduction_axis);
        let zero = self.graph.constant(0).expand_rhs(nan_count.shape);
        let has_nan = nan_count.gt(zero);
        let nan = self.constant_like(result, f64::NAN);
        Ok(self.select(has_nan, nan, result))
    }

    /// Translate `aten.acos.default` into existing elementwise HLIR primitives.
    ///
    /// For `x >= 0`, approximate `acos(x)` as `sqrt(1 - x) * P(x)`, where
    /// `P` is a degree-8 Chebyshev approximation of
    /// `acos(x) / sqrt(1 - x)` on `[0, 1]`. Extend it to negative inputs with
    /// `acos(-x) = pi - acos(x)`. Factoring out the square-root endpoint
    /// behavior keeps the polynomial smooth and also makes out-of-domain real
    /// inputs produce NaN through `sqrt(1 - abs(x))`, matching PyTorch.
    ///
    /// PyTorch promotes integral and bool inputs to its default floating dtype.
    #[allow(clippy::excessive_precision)]
    pub(crate) fn translate_acos(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(self.real_acos(input))
    }

    /// Elementwise real acos used by both real ATen dispatch and compound
    /// complex inverse functions.
    #[allow(clippy::excessive_precision)]
    pub(crate) fn real_acos(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();

        // Horner form, highest-order coefficient first. The maximum absolute
        // approximation error in F32 is below 3e-7 over the real acos domain.
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

    /// Translate `aten.acosh.default` into existing elementwise HLIR primitives.
    ///
    /// The textbook `log(x + sqrt(x * x - 1))` form overflows before the log
    /// for large finite inputs, especially in F16. Use the equivalent form
    ///
    /// `log(x) + log(1 + sqrt(1 - 1 / x^2))`
    ///
    /// instead. For real inputs below one, either the square root or `log(x)`
    /// naturally produces NaN, matching PyTorch's real-domain behavior.
    /// PyTorch promotes integral and bool inputs to its default floating dtype,
    /// while floating inputs retain their dtype.
    pub(crate) fn translate_acosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(self.real_acosh(input))
    }

    /// Elementwise real acosh used by both real ATen dispatch and compound
    /// complex inverse functions.
    pub(crate) fn real_acosh(&mut self, input: GraphTensor) -> GraphTensor {
        let reciprocal_squared = input.reciprocal().square();
        let one = self.constant_like(input, 1.0);
        input.log() + (one + (one - reciprocal_squared).sqrt()).log()
    }

    fn unary_input(&mut self, node: &Node) -> Result<GraphTensor> {
        Ok(self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?))
    }

    pub(crate) fn translate_exp(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_exp(input))
    }

    /// Keep log2(e) in the tensor's actual dtype. `GraphTensor::exp` uses its
    /// historical F32 scalar API, which is not precise enough for F64 PT2.
    pub(crate) fn real_exp(&mut self, input: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(input, std::f64::consts::LOG2_E);
        (input * log2_e).exp2()
    }

    pub(crate) fn translate_expm1(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let one = self.constant_like(input, 1.0);
        Ok(self.real_exp(input) - one)
    }

    pub(crate) fn translate_log1p(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let one = self.constant_like(input, 1.0);
        Ok((input + one).log())
    }

    pub(crate) fn translate_log10(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let ln_ten = self.constant_like(input, std::f64::consts::LN_10);
        Ok(input.log() / ln_ten)
    }

    pub(crate) fn translate_sinh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let output_dtype = input.dtype;
        let opmath = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            input.cast(DType::F32)
        } else {
            input
        };
        Ok(self.real_sinh(opmath).cast(output_dtype))
    }

    pub(crate) fn real_tan(&mut self, input: GraphTensor) -> GraphTensor {
        let cosine = self.real_cos(input);
        input.sin() / cosine
    }

    pub(crate) fn translate_tan(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_tan(input))
    }

    pub(crate) fn translate_angle(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let zero = self.constant_like(input, 0.0);
        Ok(self.real_atan2(zero, input))
    }

    pub(crate) fn translate_isinf(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        if matches!(
            input.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            Ok(self.is_inf(input))
        } else {
            Ok(self
                .graph
                .constant(0)
                .cast(DType::Bool)
                .expand_rhs(input.shape))
        }
    }

    pub(crate) fn translate_ldexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        let exponent = self.get_input_tensor(node, 1)?.cast(input.dtype);
        let (input, exponent) = broadcast_binary(input, exponent);
        Ok(input * exponent.exp2())
    }

    pub(crate) fn translate_cos(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_cos(input))
    }

    /// Keep pi/2 in the tensor's actual dtype rather than narrowing F64.
    pub(crate) fn real_cos(&mut self, input: GraphTensor) -> GraphTensor {
        (self.constant_like(input, std::f64::consts::FRAC_PI_2) - input).sin()
    }

    pub(crate) fn translate_asin(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_asin(input))
    }

    /// `asin(x) = atan(x / sqrt(1 - x^2))` is accurate around zero and reaches
    /// the correct signed infinities at both endpoints. The square root also
    /// supplies PyTorch's NaN for real inputs outside [-1, 1].
    pub(crate) fn real_asin(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let denominator = (one - input.square()).sqrt();
        let ratio = input / denominator;
        self.real_atan(ratio)
    }

    pub(crate) fn translate_asinh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_asinh(input))
    }

    /// Stable real asinh. A short odd series preserves tiny values that would
    /// be rounded away by `log(|x| + hypot(x, 1))`; the logarithmic form covers
    /// the rest, with `log(|x|) + log(2)` preventing finite overflow near the
    /// largest representable value.
    pub(crate) fn real_asinh(&mut self, input: GraphTensor) -> GraphTensor {
        let x = self.real_abs(input);
        let x2 = x.square();
        let mut series = self.constant_like(x, 35.0 / 1152.0);
        series = series * x2 - self.constant_like(x, 5.0 / 112.0);
        series = series * x2 + self.constant_like(x, 3.0 / 40.0);
        series = series * x2 - self.constant_like(x, 1.0 / 6.0);
        let series = x + x * x2 * series;

        let one = self.constant_like(x, 1.0);
        let magnitude = x + self.real_hypot(x, one);
        let regular = magnitude.log();
        let log_two = self.constant_like(x, std::f64::consts::LN_2);
        let large = x.log() + log_two;
        let large_threshold = self.constant_like(
            x,
            match x.dtype {
                DType::F16 => 32_752.0,
                DType::F32 => (f32::MAX / 2.0) as f64,
                DType::F64 => f64::MAX / 2.0,
                DType::Bf16 => (f32::MAX / 2.0) as f64,
                other => unreachable!("asinh has non-floating dtype {other:?}"),
            },
        );
        let nonsmall = self.select(x.gt(large_threshold), large, regular);
        let threshold = self.constant_like(x, 0.125);
        let result = self.select(x.le(threshold), series, nonsmall);
        self.copy_sign(result, input)
    }

    pub(crate) fn translate_atan(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_atan(input))
    }

    /// Range-reduced odd Taylor series for atan. Reciprocal reduction maps the
    /// full real line to [0, 1], then a pi/4 transform bounds the polynomial
    /// argument by sqrt(2)-1. Degree 27 keeps F64 error below 1e-12 while using
    /// only ordinary real HLIR primitives.
    pub(crate) fn real_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = self.real_abs(input);
        let one = self.constant_like(x, 1.0);
        let reciprocal_branch = x.gt(one);
        let reduced = self.select(reciprocal_branch, x.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.select(reciprocal_branch, half_pi - base, base);
        self.copy_sign(angle, input)
    }

    pub(crate) fn translate_atanh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_atanh(input))
    }

    /// Stable real atanh. The series avoids cancellation near zero; the log
    /// difference supplies infinities at +/-1 and NaN outside the real domain.
    pub(crate) fn real_atanh(&mut self, input: GraphTensor) -> GraphTensor {
        let x2 = input.square();
        let mut series = self.constant_like(input, 1.0 / 11.0);
        series = series * x2 + self.constant_like(input, 1.0 / 9.0);
        series = series * x2 + self.constant_like(input, 1.0 / 7.0);
        series = series * x2 + self.constant_like(input, 1.0 / 5.0);
        series = series * x2 + self.constant_like(input, 1.0 / 3.0);
        let series = input + input * x2 * series;

        let one = self.constant_like(input, 1.0);
        let half = self.constant_like(input, 0.5);
        let regular = half * ((one + input).log() - (one - input).log());
        let small = self.real_abs(input).le(self.constant_like(input, 0.125));
        self.select(small, series, regular)
    }

    pub(crate) fn translate_cosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let output_dtype = input.dtype;
        let opmath = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            input.cast(DType::F32)
        } else {
            input
        };
        Ok(self.real_cosh(opmath).cast(output_dtype))
    }

    pub(crate) fn real_cosh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.real_exp(input) + self.real_exp(input * -1.0))
    }

    pub(crate) fn real_sinh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        let result = half * (self.real_exp(input) - self.real_exp(input * -1.0));
        let zero = self.is_zero(input);
        self.select(zero, input, result)
    }

    fn real_hypot(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let a = self.real_abs(a);
        let b = self.real_abs(b);
        let a_is_large = a.ge(b);
        let large = self.select(a_is_large, a, b);
        let small = self.select(a_is_large, b, a);
        let one = self.constant_like(large, 1.0);
        let large_is_zero = self.is_zero(large);
        let safe_large = self.select(large_is_zero, one, large);
        let ratio = small / safe_large;
        let finite = large * (one + ratio.square()).sqrt();
        let a_inf = self.is_inf(a);
        let b_inf = self.is_inf(b);
        let any_inf = self.bool_or(a_inf, b_inf);
        let infinity = self.constant_like(finite, f64::INFINITY);
        self.select(any_inf, infinity, finite)
    }

    pub(crate) fn translate_trunc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let integer_dtype = if input.dtype == DType::F64 {
            DType::I64
        } else {
            DType::Int
        };
        let truncated = input.cast(integer_dtype).cast(input.dtype);
        let threshold = self.constant_like(
            input,
            if input.dtype == DType::F64 {
                9_223_372_036_854_775_808.0
            } else {
                2_147_483_648.0
            },
        );
        // Beyond the integer range every representable float is already
        // integral. Preserve those values, non-finites, and signed zero.
        let at_limit = self.real_abs(input).ge(threshold);
        let is_inf = self.is_inf(input);
        let is_nan = self.is_nan(input);
        let nonfinite = self.bool_or(is_inf, is_nan);
        let preserve = self.bool_or(at_limit, nonfinite);
        let is_zero = self.is_zero(input);
        let preserve = self.bool_or(preserve, is_zero);
        Ok(self.select(preserve, input, truncated))
    }

    /// Translate `aten.gelu`, honoring the `approximate` kwarg. PyTorch's default
    /// (`approximate="none"`) is the exact erf form; `"tanh"` selects the tanh
    /// approximation. Mapping both to a single `gelu()` (as before) silently used the
    /// tanh approximation even when the model asked for exact, which accumulates
    /// visible error in deep GELU-heavy stacks (ViT, Whisper).
    pub(crate) fn translate_gelu(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        // PT2 serializes string args as {"as_string": "<value>"}; drill into the JSON.
        let approximate = node.inputs.iter().find_map(|input| {
            if input.name == "approximate"
                && let Argument::Other(val) = &input.arg
            {
                if let Some(s) = val.as_str() {
                    return Some(s.to_string());
                }
                if let Some(s) = val.get("as_string").and_then(|v| v.as_str()) {
                    return Some(s.to_string());
                }
            }
            None
        });
        Ok(match approximate.as_deref() {
            Some("tanh") => a.gelu_fast_tanh_approximation(),
            _ => a.gelu(),
        })
    }

    pub(crate) fn translate_to_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        for input in &node.inputs {
            if input.name == "dtype" {
                let dtype_int = input
                    .arg
                    .as_int()
                    .map(|i| i as u32)
                    .or_else(|| input.arg.as_scalar_type());
                if let Some(d) = dtype_int {
                    let dtype = torch_dtype_int_to_luminal(d);
                    // Skip emitting a Cast op when the dtype already matches —
                    // PT2 graphs frequently emit `_to_copy` purely as a clone hint
                    // (e.g. dtype=float32 on a tensor that is already F32), and
                    // every redundant Cast inflates the graph and survives until
                    // optimization passes can prove it as a no-op.
                    return Ok(if a.dtype == dtype { a } else { a.cast(dtype) });
                }
            }
        }
        Ok(a)
    }

    pub(crate) fn translate_layer_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let normalized_shape = self.get_ints_arg(node, 1)?;

        // Axes to normalize over = last N dims where N = len(normalized_shape)
        let ndim = input.shape.len();
        let num_norm_dims = normalized_shape.len();
        let axes: Vec<usize> = ((ndim - num_norm_dims)..ndim).collect();

        // eps is arg 4 (after input, normalized_shape, weight, bias), default 1e-5
        let eps = self.get_float_arg(node, 4).unwrap_or(1e-5) as f32;

        // torch computes LN statistics in fp32 (opmath) even for fp16/bf16
        // inputs — "For FP16 or BFloat16 inputs, ops should perform internal
        // math in FP32" (aten/src/ATen/OpMathType.h; used by
        // layer_norm_kernel.cpp as `opmath_t`). fp16 statistics overflow on
        // outlier activations (x^2 > 65504 at |x| > ~256 — the OPT-family
        // residual-stream profile).
        // Mirror translate_fused_rms_norm: normalize + affine in F32, cast
        // the result back to the input dtype.
        let out_dtype = input.dtype;
        let mut result = input.cast(DType::F32).layer_norm(axes, eps);

        // Apply weight (arg 2) if present and not None
        if let Some(weight_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let (r, w) = broadcast_binary(result, w);
            result = r * w;
        }

        // Apply bias (arg 3) if present and not None
        if let Some(bias_name) = node.inputs.get(3).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?.cast(DType::F32);
            let (r, b) = broadcast_binary(result, b);
            result = r + b;
        }

        Ok(result.cast(out_dtype))
    }

    /// `aten._fused_rms_norm` (F.rms_norm on CUDA): frontend `std_norm` +
    /// optional affine. Only `out` is consumed; `rstd` is DCE'd.
    pub(crate) fn translate_fused_rms_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let normalized_shape = self.get_ints_arg(node, 1)?;

        let ndim = input.shape.len();
        let num_norm_dims = normalized_shape.len();
        anyhow::ensure!(
            num_norm_dims <= ndim,
            "rms_norm normalized_shape rank {num_norm_dims} exceeds input rank {ndim}"
        );
        let axes: Vec<usize> = ((ndim - num_norm_dims)..ndim).collect();

        // eps (arg 3): eager resolves None to the fp32 machine epsilon
        // regardless of input dtype.
        let eps = self.get_float_arg(node, 3).unwrap_or(f32::EPSILON as f64) as f32;

        // Eager's fused kernel computes entirely in fp32 and casts the result
        // to the input dtype; matching it also handles mixed-dtype weights.
        let out_dtype = input.dtype;
        let mut result = input.cast(DType::F32).std_norm(axes, eps);

        // Apply weight (arg 2) if present and not None.
        if let Some(weight_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let (r, w) = broadcast_binary(result, w);
            result = r * w;
        }

        Ok(result.cast(out_dtype))
    }

    /// Translate `aten.native_group_norm.default`.
    ///
    /// Schema: `native_group_norm(input, weight?, bias?, N, C, HxW, num_groups, eps)
    /// -> (out, mean, rstd)`. We only produce the normalized `out`; the `mean`/`rstd`
    /// outputs exist solely for the backward pass and are never consumed by inference
    /// graphs, so (like `translate_layer_norm`) we return a single tensor and let the
    /// dispatcher assign it to output[0] while the unused outputs are DCE'd.
    ///
    /// GroupNorm splits the `C` channels into `num_groups` groups and normalizes each
    /// `(batch, group)` slice jointly over its `group_size * spatial` elements, then
    /// applies a per-channel affine. We compose this from existing primitives (no new
    /// op): reshape so each group's volume is a single contiguous axis, `layer_norm`
    /// over that one axis, reshape back, then the affine.
    ///
    /// The per-group volume is flattened into ONE axis before normalizing rather than
    /// reducing over multiple axes: the multi-axis reduction form is dropped by the
    /// e-graph during cleanup when composed into deep conv chains (see the note in
    /// `examples/flux2/src/vae.rs`). Reshapes use `Expression` extents throughout, so
    /// dynamic batch and dynamic spatial dims are preserved.
    pub(crate) fn translate_group_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let num_groups = self.get_int_arg(node, 6)? as usize;
        let eps = self.get_float_arg(node, 7).unwrap_or(1e-5) as f32;

        let orig_dims = input.dims();
        let ndim = orig_dims.len();
        anyhow::ensure!(
            ndim >= 2,
            "group_norm expects input rank >= 2 (N, C, ...), got {ndim}"
        );

        // Channel count must be static to size the groups (it always is — channel
        // count is a model-config constant).
        let c = orig_dims[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("group_norm requires a static channel dim"))?;
        anyhow::ensure!(
            num_groups != 0 && c % num_groups == 0,
            "group_norm: num_channels ({c}) must be a positive multiple of num_groups ({num_groups})"
        );
        let group_size = c / num_groups;

        // Per-group volume V = group_size * (product of spatial dims). Spatial extents
        // stay symbolic so dynamic spatial dims flow through.
        let spatial: Expression = orig_dims[2..].iter().cloned().product();
        let group_volume = spatial * Expression::from(group_size);

        // torch computes group-norm statistics in fp32 (opmath); fp16 stats
        // overflow on outlier activations. Normalize + affine in F32 and
        // cast back at the end (mirrors translate_fused_rms_norm).
        let out_dtype = input.dtype;
        // Flatten everything after the batch dim into one axis: (N, C, ...) -> (N, M),
        // where M = C * spatial. Group volumes are contiguous in this layout.
        let mut t = input.cast(DType::F32);
        while t.shape.len() > 2 {
            t = t.merge_dims(1, 2);
        }
        // (N, M) -> (N, num_groups, group_volume): M / group_volume == num_groups.
        t = t.split_dims(1, group_volume);

        // Normalize over the single per-group axis (matches PyTorch: biased variance,
        // eps inside the sqrt).
        t = t.layer_norm(2, eps);

        // Reshape back to the original (N, C, ...spatial).
        t = t.merge_dims(1, 2); // (N, num_groups, V) -> (N, M)
        // Peel the trailing (non-batch) dims back off one at a time, left to right.
        let trailing = &orig_dims[1..];
        for i in 0..trailing.len().saturating_sub(1) {
            let suffix: Expression = trailing[i + 1..].iter().cloned().product();
            t = t.split_dims(1 + i, suffix);
        }

        // Per-channel affine on the channel axis (axis 1). weight/bias are shape (C,);
        // broadcast them onto every axis except the channel axis.
        let non_channel_axes: Vec<usize> = (0..ndim).filter(|&a| a != 1).collect();
        if let Some(weight_name) = node.inputs.get(1).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let w = w.expand_to_shape_on_axes(t.shape, non_channel_axes.clone());
            let (r, w) = broadcast_binary(t, w);
            t = r * w;
        }
        if let Some(bias_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?.cast(DType::F32);
            let b = b.expand_to_shape_on_axes(t.shape, non_channel_axes);
            let (r, b) = broadcast_binary(t, b);
            t = r + b;
        }

        Ok(t.cast(out_dtype))
    }

    pub(crate) fn translate_sign(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let zero = self
            .graph
            .constant_float(0.0)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        let pos = a.gt(zero).cast(DType::Int);
        let neg = a.lt(zero).cast(DType::Int);
        let signed = pos - neg;
        Ok(if a.dtype == DType::Int {
            signed
        } else {
            signed.cast(a.dtype)
        })
    }

    pub(crate) fn translate_bitwise_not(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        Ok(match a.dtype {
            DType::Bool => {
                let one = self
                    .graph
                    .constant_float(1.0)
                    .cast(DType::Int)
                    .expand_rhs(a.shape);
                (one - a.cast(DType::Int)).cast(DType::Bool)
            }
            DType::Int => (a + 1) * -1.0,
            other => {
                anyhow::bail!("bitwise_not only supports Bool/Int routing tensors, got {other:?}")
            }
        })
    }

    pub(crate) fn translate_masked_fill_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        // `masked_fill(input, mask, fill)` = `where(mask, fill, input)`.
        // Routes through the shared `where_formula` helper so we exercise
        // the exact same code path as `aten.where.self`, which is verified
        // to handle the bf16 cast-back correctly. Hand-rolling the same
        // formula directly here used to drift (egglog made different
        // rewrite choices on the rebuilt-locally graph), so we deliberately
        // re-use the helper.
        // `aten.masked_fill.Scalar(input, mask, fill)` ≡
        // `aten.where.self(mask, full_like(input, fill), input)`. The
        // `full_like + where` sequence is the verified-working path
        // (test: `where(mask, torch.zeros_like(x), x)` round-trips with
        // max_diff = 0); we reproduce its exact graph-build order here.
        // Hand-rolling the formula in any other shape (single-mul, F32
        // throughout, alternative constant-cast orderings) routes egglog
        // through a rewrite that returns an F32 buffer downstream-read as
        // bf16 — the every-other-element-zero pattern.
        let input = self.get_input_tensor(node, MASKED_FILL_INPUT_ARG)?;
        let mask = self.get_input_tensor(node, MASKED_FILL_MASK_ARG)?;
        let fill = self.get_float_arg(node, MASKED_FILL_VALUE_ARG)? as f32;
        let out_dtype = input.dtype;
        // Build fill_t exactly like translate_full_like does:
        //   constant_float(val).cast(dtype).expand_rhs(reference.shape)
        let fill_t = self
            .graph
            .constant_float(fill)
            .cast(out_dtype)
            .expand_rhs(input.shape);
        Ok(self.where_formula(mask, fill_t, input, out_dtype))
    }

    pub(crate) fn translate_floor_divide(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, FLOOR_DIVIDE_INPUT_ARG)?;
        let b = if let Some(name) = node
            .inputs
            .get(FLOOR_DIVIDE_OTHER_ARG)
            .and_then(|i| i.arg.as_tensor_name())
        {
            self.get_tensor(name)?
        } else {
            let scalar = self.get_float_arg(node, FLOOR_DIVIDE_OTHER_ARG)? as f32;
            self.graph
                .constant_float(scalar)
                .cast(a.dtype)
                .expand_rhs(a.shape)
        };
        let (a, b) = crate::pt2_util::ensure_same_dtype(a, b);
        let (a, b) = broadcast_binary(a, b);
        let quotient = a.cast(DType::F32) / b.cast(DType::F32);
        let trunc = quotient.cast(DType::Int).cast(DType::F32);
        let adjust = quotient.lt(trunc).cast(DType::F32);
        let floored = trunc - adjust;
        Ok(if a.dtype == DType::Int {
            floored.cast(DType::Int)
        } else {
            floored.cast(a.dtype)
        })
    }

    pub(crate) fn translate_div_tensor_mode(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, DIV_MODE_INPUT_ARG)?;
        let b = if let Some(name) = node
            .inputs
            .get(DIV_MODE_OTHER_ARG)
            .and_then(|i| i.arg.as_tensor_name())
        {
            self.get_tensor(name)?
        } else {
            let scalar = self.get_float_arg(node, DIV_MODE_OTHER_ARG)? as f32;
            self.graph
                .constant_float(scalar)
                .cast(a.dtype)
                .expand_rhs(a.shape)
        };
        let (a, b) = crate::pt2_util::ensure_same_dtype(a, b);
        let (a, b) = broadcast_binary(a, b);

        // Check rounding_mode kwarg. PT2 serializes string args as
        // {"as_string": "<value>"}, so we have to drill into the JSON.
        let rounding_mode = node.inputs.iter().find_map(|input| {
            if input.name == "rounding_mode"
                && let Argument::Other(val) = &input.arg
            {
                if let Some(s) = val.as_str() {
                    return Some(s.to_string());
                }
                if let Some(s) = val.get("as_string").and_then(|v| v.as_str()) {
                    return Some(s.to_string());
                }
            }
            None
        });

        let quotient = a.cast(DType::F32) / b.cast(DType::F32);
        match rounding_mode.as_deref() {
            Some("floor") => {
                let trunc = quotient.cast(DType::Int).cast(DType::F32);
                let adjust = quotient.lt(trunc).cast(DType::F32);
                let floored = trunc - adjust;
                Ok(if a.dtype == DType::Int {
                    floored.cast(DType::Int)
                } else {
                    floored.cast(a.dtype)
                })
            }
            Some("trunc") => Ok(if a.dtype == DType::Int {
                quotient.cast(DType::Int)
            } else {
                quotient.cast(DType::Int).cast(a.dtype)
            }),
            _ => {
                // No rounding mode is true division, which ATen builds with
                // build_borrowing_binary_float_op — so an integral input comes
                // back float, not cast back to `a.dtype`.
                Ok(match self.recorded_output_dtype(node) {
                    Some(dtype) => quotient.cast(dtype),
                    None => quotient,
                })
            }
        }
    }

    pub(crate) fn translate_clamp(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let min_val = if node.inputs.len() > 1 {
            self.get_float_arg(node, 1).ok().map(|f| f as f32)
        } else {
            None
        };
        let max_val = if node.inputs.len() > 2 {
            self.get_float_arg(node, 2).ok().map(|f| f as f32)
        } else {
            None
        };

        let mut result = a;
        if let Some(min) = min_val {
            result = result.maximum_f32(min);
        }
        if let Some(max) = max_val {
            result = result.minimum_f32(max);
        }
        Ok(result)
    }

    /// `aten.clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None)`
    ///
    /// Unlike `clamp.default` (which takes Python scalar bounds), the `.Tensor`
    /// overload takes tensor bounds that appear as separate input nodes in the
    /// FX graph. PyTorch supports any NumPy-broadcastable bound shape:
    ///
    ///   - rank-0 (scalar wrapped in a tensor) — most common
    ///   - same shape as self (per-element clamp, e.g. learned bounds)
    ///   - any shape that broadcasts to self via right-align + size-1 expand
    ///     (e.g. `(3, 1)` against `(3, 4)` for per-row clamp; `(4,)` against
    ///     `(3, 4)` for per-column clamp; `(3, 4)` against `(2, 3, 4)`)
    ///
    /// We use `broadcast_binary` to right-align and expand both operands to a
    /// common shape before the elementwise max/min, matching PyTorch semantics
    /// across all three modes.
    ///
    /// Either bound may be absent (FX represents this as a non-tensor argument
    /// at the corresponding input slot), in which case we clamp to one side
    /// only.
    pub(crate) fn translate_clamp_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let min_tensor = node
            .inputs
            .get(1)
            .and_then(|i| i.arg.as_tensor_name())
            .map(|n| self.get_tensor(n))
            .transpose()?;
        let max_tensor = node
            .inputs
            .get(2)
            .and_then(|i| i.arg.as_tensor_name())
            .map(|n| self.get_tensor(n))
            .transpose()?;

        let mut result = a;
        if let Some(lo) = min_tensor {
            let lo = lo.cast(result.dtype);
            let (r, lo) = broadcast_binary(result, lo);
            result = r.maximum(lo);
        }
        if let Some(hi) = max_tensor {
            let hi = hi.cast(result.dtype);
            let (r, hi) = broadcast_binary(result, hi);
            result = r.minimum(hi);
        }
        Ok(result)
    }
}
