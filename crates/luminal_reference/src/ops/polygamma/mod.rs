use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolygammaFunctional {
    pub order: i64,
}

impl OpSlotNames for PolygammaFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for PolygammaFunctional {
    fn label(&self) -> &str {
        "PolygammaFunctionalGeneric"
    }
}

impl Bufferizable for PolygammaFunctional {}

impl ToDps for PolygammaFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(PolygammaFunctionalDps { order: self.order }))
    }
}

impl LayoutIrOp for PolygammaFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolygammaFunctionalDps {
    pub order: i64,
}

impl OpSlotNames for PolygammaFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for PolygammaFunctionalDps {
    fn label(&self) -> &str {
        "PolygammaFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1
    }
}

impl Bufferizable for PolygammaFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for PolygammaFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for PolygammaFunctionalDps {}

#[derive(Debug, Clone, Copy, Default)]
pub struct PolygammaFunctionalMatcher;

impl OpMatcher for PolygammaFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpPolygammaFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("order", 1), ("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(PolygammaFunctional {
            order: site.child_i64(1),
        })
    }
}

const ZETA_COEFFICIENTS: [f64; 12] = [
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.892_437_580_318_379_2e9,
    7.47242496e10,
    -2.950_130_727_918_164e12,
    1.164_678_281_435_006_7e14,
    -4.597_978_722_407_473e15,
    1.815_210_540_194_354_6e17,
    -7.166_165_256_175_667e18,
];

fn zeta_f32(x: f32, q: f32) -> f32 {
    if x == 1.0 {
        return f32::INFINITY;
    }
    if x < 1.0 {
        return f32::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f32::INFINITY;
        }
        if x != x.floor() {
            return f32::NAN;
        }
    }
    let machep = 1.1102230246251565e-16_f64;
    let mut sum = q.powf(-x) as f64;
    let mut a = q as f64;
    let mut b = 0.0_f64;
    let mut i = 0;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = a.powf(-(x as f64));
        sum += b;
        if -machep * sum < b && b < machep * sum {
            return sum as f32;
        }
    }
    let w = a;
    sum += b * w / (x as f64 - 1.0);
    sum -= 0.5 * b;
    a = 1.0;
    let mut k = 0.0;
    for coefficient in ZETA_COEFFICIENTS {
        a *= x as f64 + k;
        b /= w;
        let mut term = a * b / coefficient;
        sum += term;
        term = (term / sum).abs();
        if term < machep {
            return sum as f32;
        }
        k += 1.0;
        a *= x as f64 + k;
        b /= w;
        k += 1.0;
    }
    sum as f32
}

fn zeta_f64(x: f64, q: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return f64::NAN;
    }
    if q <= 0.0 {
        if q == q.floor() {
            return f64::INFINITY;
        }
        if x != x.floor() {
            return f64::NAN;
        }
    }
    let machep = 1.1102230246251565e-16_f64;
    let mut sum = q.powf(-x);
    let mut a = q;
    let mut b = 0.0;
    let mut i = 0;
    while i < 9 || a <= 9.0 {
        i += 1;
        a += 1.0;
        b = a.powf(-x);
        sum += b;
        if -machep * sum < b && b < machep * sum {
            return sum;
        }
    }
    let w = a;
    sum += b * w / (x - 1.0);
    sum -= 0.5 * b;
    a = 1.0;
    let mut k = 0.0;
    for coefficient in ZETA_COEFFICIENTS {
        a *= x + k;
        b /= w;
        let mut term = a * b / coefficient;
        sum += term;
        term = (term / sum).abs();
        if term < machep {
            return sum;
        }
        k += 1.0;
        a *= x + k;
        b /= w;
        k += 1.0;
    }
    sum
}

fn digamma_f32(mut x: f32) -> f32 {
    if x == 0.0 {
        return f32::INFINITY.copysign(-x);
    }
    if x < 0.0 {
        if x == x.trunc() {
            return f32::NAN;
        }
        let fraction = (x as f64).fract();
        return digamma_f32(1.0 - x)
            - (std::f64::consts::PI / (std::f64::consts::PI * fraction).tan()) as f32;
    }
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    if x == 10.0 {
        return result + 2.2517526;
    }
    let coefficients = [
        8.3333336e-2_f32,
        -2.1092797e-2,
        7.575_758e-3,
        -4.166667e-3,
        3.968254e-3,
        -8.333334e-3,
        8.3333336e-2,
    ];
    let mut correction = 0.0;
    if x < 1.0e17 {
        let z = 1.0 / (x * x);
        let polynomial = coefficients
            .into_iter()
            .fold(0.0, |value, coefficient| value * z + coefficient);
        correction = z * polynomial;
    }
    result + x.ln() - 0.5 / x - correction
}

fn digamma_f64(mut x: f64) -> f64 {
    if x == 0.0 {
        return f64::INFINITY.copysign(-x);
    }
    if x < 0.0 {
        if x == x.trunc() {
            return f64::NAN;
        }
        let fraction = x.fract();
        return digamma_f64(1.0 - x)
            - std::f64::consts::PI / (std::f64::consts::PI * fraction).tan();
    }
    let mut result = 0.0;
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    if x == 10.0 {
        return result + 2.251752589066721;
    }
    let coefficients = [
        8.333333333333333e-2_f64,
        -2.1092796092796093e-2,
        7.575757575757576e-3,
        -4.166666666666667e-3,
        3.968253968253968e-3,
        -8.333333333333333e-3,
        8.333333333333333e-2,
    ];
    let mut correction = 0.0;
    if x < 1.0e17 {
        let z = 1.0 / (x * x);
        let polynomial = coefficients
            .into_iter()
            .fold(0.0, |value, coefficient| value * z + coefficient);
        correction = z * polynomial;
    }
    result + x.ln() - 0.5 / x - correction
}

fn trigamma_f32(mut x: f32) -> f32 {
    let mut sign = 1.0;
    let mut result = 0.0;
    if x < 0.5 {
        sign = -1.0;
        let sine = (std::f32::consts::PI * x).sin();
        result -= std::f32::consts::PI.powi(2) / sine.powi(2);
        x = 1.0 - x;
    }
    for _ in 0..6 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inverse_square = 1.0 / (x * x);
    result += (1.0
        + 1.0 / (2.0 * x)
        + inverse_square
            * (1.0 / 6.0 - inverse_square * (1.0 / 30.0 - inverse_square * (1.0 / 42.0))))
        / x;
    sign * result
}

fn trigamma_f64(mut x: f64) -> f64 {
    let mut sign = 1.0;
    let mut result = 0.0;
    if x < 0.5 {
        sign = -1.0;
        let sine = (std::f64::consts::PI * x).sin();
        result -= std::f64::consts::PI.powi(2) / sine.powi(2);
        x = 1.0 - x;
    }
    for _ in 0..6 {
        result += 1.0 / (x * x);
        x += 1.0;
    }
    let inverse_square = 1.0 / (x * x);
    result += (1.0
        + 1.0 / (2.0 * x)
        + inverse_square
            * (1.0 / 6.0 - inverse_square * (1.0 / 30.0 - inverse_square * (1.0 / 42.0))))
        / x;
    sign * result
}

fn polygamma_f32(x: f32, order: i32) -> f32 {
    match order {
        0 => digamma_f32(x),
        1 => trigamma_f32(x),
        _ => {
            let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
            sign * libm::lgammaf(order as f32 + 1.0).exp() * zeta_f32(order as f32 + 1.0, x)
        }
    }
}

fn polygamma_f64(x: f64, order: i32) -> f64 {
    match order {
        0 => digamma_f64(x),
        1 => trigamma_f64(x),
        _ => {
            let sign = if order % 2 == 1 { 1.0 } else { -1.0 };
            sign * libm::lgamma(order as f64 + 1.0).exp() * zeta_f64(order as f64 + 1.0, x)
        }
    }
}

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = op
        .as_any()
        .downcast_ref::<PolygammaFunctionalDps>()
        .expect("polygamma kernel received the wrong op type");
    let order = op.order as i32;
    match &ctx.operands[0] {
        TypedBuffer::F16(_) => {
            ctx.unary_elementwise_f16(|x| half::f16::from_f32(polygamma_f32(x.to_f32(), order)))
        }
        TypedBuffer::Bf16(input) => {
            let input = input.clone();
            let dest = ctx.dests[0].as_bf16_mut()?;
            anyhow::ensure!(input.len() == dest.len(), "unary kernel length mismatch");
            for (out, x) in dest.iter_mut().zip(input) {
                *out = half::bf16::from_f32(polygamma_f32(x.to_f32(), order));
            }
            Ok(())
        }
        TypedBuffer::F32(_) => ctx.unary_elementwise(|x| polygamma_f32(x, order)),
        TypedBuffer::F64(_) => {
            ctx.unary_elementwise_typed(|x| polygamma_f32(x, order), |x| polygamma_f64(x, order))
        }
        other => anyhow::bail!("polygamma has no {} arm", other.type_name()),
    }
}
