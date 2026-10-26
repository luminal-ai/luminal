//! An approved read-modify-write GEMM. C is the destination; there is no
//! independent D operand, D descriptor, or conditional alias permission.
use super::CublasLtForm;
use super::approved::{
    DECLARATIONS, Matrix, RULES, WHITELIST, capture_dimensions, dtype, matrix, member,
};
use super::exec::{CSource, LtCall};
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
                text: WHITELIST,
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
        ]
    }
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let configuration = member(site, &site.child_class(0), "CublasLtConfigurationOf");
        let child = |n| site.class_of_child(&configuration, n).unwrap();
        member(site, &child(0), "CublasLtFloat32Computation");
        member(site, &child(7), "CublasLtHostScalarsAlphaOneBetaOne");
        let frame = member(site, &site.child_class(6), "CublasLtExpressionFrameOf");
        let semantic = |n| site.class_of_child(&frame, n).unwrap();
        let bias = site
            .nodes_in_class_value(&site.child_class(4), "CublasLtBiasInput")
            .next()
            .map(|node| site.class_of_child(node, 0).unwrap());
        let enum_is = |n, name| site.nodes_in_class_value(&child(n), name).next().is_some();
        Box::new(CublasLtAccumulateInPlace {
            configuration: site.child_class(0),
            logical_a: semantic(2),
            logical_b: semantic(3),
            logical_out: semantic(1),
            logical_root: semantic(0),
            a: matrix(site, &site.child_class(1)),
            b: matrix(site, &site.child_class(2)),
            c_and_d: matrix(site, &site.child_class(3)),
            bias,
            scale: dtype(site, &child(1)),
            storage: [
                dtype(site, &child(2)),
                dtype(site, &child(3)),
                dtype(site, &child(4)),
                dtype(site, &child(5)),
            ],
            trans_a: enum_is(12, "CublasLtTranspose"),
            trans_b: enum_is(13, "CublasLtTranspose"),
            relu: enum_is(8, "CublasLtRectifiedLinearEpilogue")
                || enum_is(8, "CublasLtRectifiedLinearBiasEpilogue"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtAccumulateInPlace {
    pub configuration: ClassId,
    pub logical_a: ClassId,
    pub logical_b: ClassId,
    pub logical_out: ClassId,
    pub logical_root: ClassId,
    pub a: Matrix,
    pub b: Matrix,
    pub c_and_d: Matrix,
    pub bias: Option<ClassId>,
    pub scale: PlanDtype,
    pub storage: [PlanDtype; 4],
    pub trans_a: bool,
    pub trans_b: bool,
    pub relu: bool,
}

impl CublasLtAccumulateInPlace {
    pub fn prototype() -> Self {
        let base = super::approved::ApprovedCublasLt::prototype();
        Self {
            configuration: base.configuration,
            logical_a: base.logical_a,
            logical_b: base.logical_b,
            logical_out: base.logical_out,
            logical_root: base.logical_root,
            a: base.a,
            b: base.b,
            c_and_d: base.d,
            bias: None,
            scale: base.scale,
            storage: base.storage,
            trans_a: false,
            trans_b: false,
            relu: false,
        }
    }
    /// Evaluate the shared descriptor once. The vendor receives identical C/D
    /// geometry, and bufferization ties C's read pointer to the result pointer.
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtCall {
        let a = self.a.resolve(dims);
        let b = self.b.resolve(dims);
        let c_and_d = self.c_and_d.resolve(dims);
        LtCall {
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
            c: c_and_d,
            d: c_and_d,
            c_source: CSource::Operand(2),
            beta_is_one: true,
            relu: self.relu,
            bias_operand: self.bias.as_ref().map(|_| 3),
        }
    }
}
impl OpSlotNames for CublasLtAccumulateInPlace {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "A".into(),
            1 => "B".into(),
            2 => "C_destination".into(),
            3 => "bias".into(),
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
            operand: 2,
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
            ctx.inputs.get(2).is_some_and(|c| c.ptr == ctx.dest.ptr),
            "in-place cuBLASLt accumulation requires C and the result to share storage"
        );
        let call = self.resolve(ctx.dims);
        crate::ops::cublaslt::device_call::prepare(&call, ctx.inputs, ctx.dest, ctx.workspace)
            .map(|prepared| Box::new(prepared) as Box<dyn crate::host::PreparedHostOp>)
    }
}
