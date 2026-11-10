use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, cuda_type, unary};
use anyhow::{Result, bail};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

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

fn helper_source(dtype: PlanDtype, order: i64) -> Result<String> {
    let (ty, sin, log, exp, lgamma, pow, floor, trunc, copysign, fabs) = match dtype {
        PlanDtype::F16 | PlanDtype::Bf16 | PlanDtype::F32 => (
            "float",
            "sinf",
            "logf",
            "expf",
            "lgammaf",
            "powf",
            "floorf",
            "truncf",
            "copysignf",
            "fabsf",
        ),
        PlanDtype::F64 => (
            "double", "sin", "log", "exp", "lgamma", "pow", "floor", "trunc", "copysign", "fabs",
        ),
        other => bail!("cuda-lite polygamma has no {other:?} arm"),
    };
    let source = r#"
__device__ __forceinline__ TYPE luminal_zeta(TYPE x, TYPE q) {
    const TYPE machep = (TYPE)1.11022302462515654042e-16;
    const TYPE a_coeff[12] = {
        (TYPE)12.0, (TYPE)-720.0, (TYPE)30240.0, (TYPE)-1209600.0,
        (TYPE)47900160.0, (TYPE)-1.8924375803183791606e9,
        (TYPE)7.47242496e10, (TYPE)-2.950130727918164224e12,
        (TYPE)1.1646782814350067249e14, (TYPE)-4.5979787224074726105e15,
        (TYPE)1.8152105401943546773e17, (TYPE)-7.1661652561756670113e18
    };
    const TYPE inf = (TYPE)__uint_as_float(0x7f800000u);
    const TYPE nan = (TYPE)__uint_as_float(0x7fc00000u);
    if (x == (TYPE)1) return inf;
    if (x < (TYPE)1) return nan;
    if (q <= (TYPE)0) {
        if (q == FLOOR(q)) return inf;
        if (x != FLOOR(x)) return nan;
    }
    TYPE s = POW(q, -x);
    TYPE a = q;
    TYPE b = (TYPE)0;
    int i = 0;
    while ((i < 9) || (a <= (TYPE)9)) {
        i += 1;
        a += (TYPE)1;
        b = POW(a, -x);
        s += b;
        if ((-machep * s < b) && (b < machep * s)) return s;
    }
    TYPE w = a;
    s += b * w / (x - (TYPE)1);
    s -= (TYPE)0.5 * b;
    a = (TYPE)1;
    TYPE k = (TYPE)0;
    for (i = 0; i < 12; i++) {
        a *= x + k;
        b /= w;
        TYPE t = a * b / a_coeff[i];
        s += t;
        t = FABS(t / s);
        if (t < machep) return s;
        k += (TYPE)1;
        a *= x + k;
        b /= w;
        k += (TYPE)1;
    }
    return s;
}

__device__ __forceinline__ TYPE luminal_digamma(TYPE x) {
    const double pi = 3.14159265358979323846;
    const TYPE inf = (TYPE)__uint_as_float(0x7f800000u);
    const TYPE nan = (TYPE)__uint_as_float(0x7fc00000u);
    if (x == (TYPE)0) return COPYSIGN(inf, -x);
    TYPE result = (TYPE)0;
    if (x < (TYPE)0) {
        if (x == TRUNC(x)) return nan;
        double integral;
        double fraction = modf((double)x, &integral);
        result = (TYPE)(-pi / tan(pi * fraction));
        x = (TYPE)1 - x;
    }
    while (x < (TYPE)10) {
        result -= (TYPE)1 / x;
        x += (TYPE)1;
    }
    if (x == (TYPE)10) return result + (TYPE)2.25175258906672110764;
    TYPE y = (TYPE)0;
    if (x < (TYPE)1.0e17) {
        const TYPE coeff[7] = {
            (TYPE)8.33333333333333333333e-2,
            (TYPE)-2.10927960927960927961e-2,
            (TYPE)7.57575757575757575758e-3,
            (TYPE)-4.16666666666666666667e-3,
            (TYPE)3.96825396825396825397e-3,
            (TYPE)-8.33333333333333333333e-3,
            (TYPE)8.33333333333333333333e-2
        };
        TYPE z = (TYPE)1 / (x * x);
        TYPE polynomial = (TYPE)0;
        for (int i = 0; i < 7; i++) polynomial = polynomial * z + coeff[i];
        y = z * polynomial;
    }
    return LOG(x) - (TYPE)0.5 / x - y + result;
}

__device__ __forceinline__ TYPE luminal_trigamma(TYPE x) {
    const TYPE pi = (TYPE)3.14159265358979323846;
    TYPE sign = (TYPE)1;
    TYPE result = (TYPE)0;
    if (x < (TYPE)0.5) {
        sign = (TYPE)-1;
        TYPE sine = SIN(pi * x);
        result -= (pi * pi) / (sine * sine);
        x = (TYPE)1 - x;
    }
    for (int i = 0; i < 6; i++) {
        result += (TYPE)1 / (x * x);
        x += (TYPE)1;
    }
    TYPE inverse_square = (TYPE)1 / (x * x);
    result += ((TYPE)1 + (TYPE)1 / ((TYPE)2 * x)
        + inverse_square * ((TYPE)1 / (TYPE)6
        - inverse_square * ((TYPE)1 / (TYPE)30
        - inverse_square * ((TYPE)1 / (TYPE)42)))) / x;
    return sign * result;
}

__device__ __forceinline__ TYPE luminal_polygamma_impl(TYPE x) {
    const int order = ORDER;
    if (order == 0) return luminal_digamma(x);
    if (order == 1) return luminal_trigamma(x);
    TYPE one = (TYPE)1;
    TYPE sign = (order % 2) ? one : -one;
    return sign * EXP(LGAMMA((TYPE)order + one))
        * luminal_zeta((TYPE)(order + 1), x);
}
"#;
    Ok(source
        .replace("TYPE", ty)
        .replace("ORDER", &(order as i32).to_string())
        .replace("SIN", sin)
        .replace("LOG", log)
        .replace("EXP", exp)
        .replace("LGAMMA", lgamma)
        .replace("POW", pow)
        .replace("FLOOR", floor)
        .replace("TRUNC", trunc)
        .replace("COPYSIGN", copysign)
        .replace("FABS", fabs))
}

impl KernelOp for PolygammaFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let dtype = ctx.operand_dtypes[0];
        let value = if matches!(dtype, PlanDtype::F16 | PlanDtype::Bf16) {
            format!(
                "({})(luminal_polygamma_impl((float)a[i]))",
                cuda_type(dtype)?
            )
        } else {
            "luminal_polygamma_impl(a[i])".to_string()
        };
        let helper = helper_source(dtype, self.order)?;
        let mut kernels = unary(ctx, &value)?;
        for kernel in &mut kernels {
            kernel.source.insert_str(0, &helper);
        }
        Ok(kernels)
    }
}

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
