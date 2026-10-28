//! An approved read-modify-write GEMM. C is the destination; there is no
//! independent D operand, D descriptor, or conditional alias permission.
use super::CublasLtForm;
use super::approved::{
    Bias, DECLARATIONS, Matrix, RULES, Scalars, WHITELIST, activation, bias, capture_dimensions,
    computation, dtype, matrix, member, scalars,
};
use super::exec::{LtActivation, LtBiasInput, LtCMatrixInput, LtCall, LtComputation};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
use luminal::prelude::egraph_serialize::ClassId;

pub const CONSTRUCTOR: &str = "LayoutTensorOpCublasLtAccumulateInPlace";

#[derive(Debug)]
pub struct CublasLtAccumulateInPlaceMatcher;

impl OpMatcher for CublasLtAccumulateInPlaceMatcher {
    fn egglog_constructor(&self) -> &'static str {
        CONSTRUCTOR
    }
    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: DECLARATIONS,
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: RULES,
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/scalar_rules.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/strided_batch_rules.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: WHITELIST,
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/call_inputs.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/accumulate_in_place_call.egg"),
            },
        ]
    }
    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[
            ("configuration", 0),
            ("A_matrix_descriptor", 1),
            ("B_matrix_descriptor", 2),
            ("C_and_D_matrix_descriptor", 3),
            ("bias", 4),
            ("semantics", 6),
            ("scalars", 7),
        ]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let configuration = member(site, &site.child_class(0), "CublasLtConfigurationOf");
        let child = |n| site.class_of_child(&configuration, n).unwrap();
        let computation = computation(site, &child(0));
        let frame = member(site, &site.child_class(6), "CublasLtExpressionFrameOf");
        let semantic = |n| site.class_of_child(&frame, n).unwrap();
        let bias = bias(site, &site.child_class(4));
        let enum_is = |n, name| site.nodes_in_class_value(&child(n), name).next().is_some();
        Box::new(CublasLtAccumulateInPlace {
            configuration: site.child_class(0),
            computation,
            logical_a: semantic(2),
            logical_b: semantic(3),
            logical_out: semantic(1),
            logical_root: semantic(0),
            a: matrix(site, &site.child_class(1)),
            b: matrix(site, &site.child_class(2)),
            c_and_d: matrix(site, &site.child_class(3)),
            bias,
            scale: dtype(site, &child(1)),
            scalars: scalars(site, &site.child_class(7), &child(7)),
            trans_a: enum_is(12, "CublasLtTranspose"),
            trans_b: enum_is(13, "CublasLtTranspose"),
            activation: activation(site, &child(8)),
        })
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtAccumulateInPlace {
    pub configuration: ClassId,
    pub computation: LtComputation,
    pub logical_a: ClassId,
    pub logical_b: ClassId,
    pub logical_out: ClassId,
    pub logical_root: ClassId,
    pub a: Matrix,
    pub b: Matrix,
    pub c_and_d: Matrix,
    pub bias: Option<Bias>,
    pub scale: PlanDtype,
    pub scalars: Scalars,
    pub trans_a: bool,
    pub trans_b: bool,
    pub activation: LtActivation,
}

impl CublasLtAccumulateInPlace {
    pub fn prototype() -> Self {
        let base = super::approved::ApprovedCublasLt::prototype();
        Self {
            configuration: base.configuration,
            computation: base.computation,
            logical_a: base.logical_a,
            logical_b: base.logical_b,
            logical_out: base.logical_out,
            logical_root: base.logical_root,
            a: base.a,
            b: base.b,
            c_and_d: base.d,
            bias: None,
            scale: base.scale,
            scalars: Scalars::host(1.0, 1.0),
            trans_a: false,
            trans_b: false,
            activation: LtActivation::Identity,
        }
    }
    /// Evaluate the shared descriptor once. The vendor receives identical C/D
    /// geometry, and bufferization ties C's read pointer to the result pointer.
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtCall {
        let a = self.a.resolve(dims);
        let b = self.b.resolve(dims);
        let c_and_d = self.c_and_d.resolve(dims);
        LtCall {
            computation: self.computation,
            form: if self.bias.is_some() {
                CublasLtForm::AccumulateBias
            } else {
                CublasLtForm::Accumulate
            },
            m: c_and_d.rows,
            n: c_and_d.cols,
            k: if self.trans_a { a.rows } else { a.cols },
            trans_a: self.trans_a,
            trans_b: self.trans_b,
            a_dtype: self.a.dtype,
            b_dtype: self.b.dtype,
            d_dtype: self.c_and_d.dtype,
            a,
            b,
            c: Some(LtCMatrixInput {
                descriptor: c_and_d,
                data_type: self.c_and_d.dtype,
                operand: self.destination_operand(),
            }),
            d: c_and_d,
            scale_dtype: self.scale,
            scalars: self.scalars.resolve(),
            activation: self.activation,
            bias: self.bias.as_ref().map(|bias| LtBiasInput {
                operand: 2 + self.scalars.resolve().input_count(),
                data_type: bias.dtype,
            }),
        }
    }
}
impl CublasLtAccumulateInPlace {
    pub fn destination_operand(&self) -> usize {
        2 + self.scalars.resolve().input_count() + usize::from(self.bias.is_some())
    }
}
impl OpSlotNames for CublasLtAccumulateInPlace {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "A".into(),
            1 => "B".into(),
            n if n == self.destination_operand() => "C_destination".into(),
            2 if matches!(self.scalars, Scalars::Device { .. }) => "alpha".into(),
            3 if matches!(self.scalars, Scalars::Device { .. }) => "beta".into(),
            n if self.bias.is_some() && n == 2 + self.scalars.resolve().input_count() => {
                "bias".into()
            }
            _ => format!("slot{operand}"),
        }
    }
}
impl BufferTensorIrOp for CublasLtAccumulateInPlace {
    fn label(&self) -> &str {
        "CublasLtAccumulateInPlace"
    }
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::host::<Self>())
    }
}
impl Bufferizable for CublasLtAccumulateInPlace {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: self.destination_operand(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for CublasLtAccumulateInPlace {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}
impl LayoutIrOp for CublasLtAccumulateInPlace {}
impl crate::HostOp for CublasLtAccumulateInPlace {
    fn capture_dims(&self) -> Option<Vec<luminal::shape::Symbol>> {
        Some(capture_dimensions([&self.a, &self.b, &self.c_and_d]))
    }
    fn workspace_bytes(&self, _: &crate::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(32 * 1024 * 1024)
    }
    #[cfg(feature = "device")]
    unsafe fn prepare(
        &self,
        ctx: &crate::host::HostOpContext<'_>,
    ) -> anyhow::Result<Box<dyn crate::host::PreparedHostOp>> {
        anyhow::ensure!(
            ctx.inputs
                .get(self.destination_operand())
                .is_some_and(|c| c.ptr == ctx.dest.ptr),
            "in-place cuBLASLt accumulation requires C and the result to share storage"
        );
        let call = self.resolve(ctx.dims);
        crate::ops::cublaslt::device_call::prepare(&call, ctx.inputs, ctx.dest, ctx.workspace)
            .map(|prepared| Box::new(prepared) as Box<dyn crate::host::PreparedHostOp>)
    }
}
