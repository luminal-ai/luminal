//! A complete call is admitted in egglog before extraction. This module only
//! decodes the elected configuration and evaluates its descriptor expressions.
use super::CublasLtForm;
use super::exec::{
    LtActivation, LtBiasInput, LtCMatrixInput, LtCall, LtComputation, LtDesc, LtOrder, LtScalars,
};
use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
use luminal::index_expr::IotaExpr;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};
use luminal::prelude::egraph_serialize::{ClassId, Node};

pub const CONSTRUCTOR: &str = "LayoutTensorOpCublasLtCall";
pub const DECLARATIONS: &str = include_str!("egg/approved_constructors.egg");
pub const RULES: &str = include_str!("egg/approved_rules.egg");
pub const WHITELIST: &str = include_str!("egg/approved_whitelist.egg");

#[derive(Debug)]
pub struct ApprovedCublasLtMatcher;

impl OpMatcher for ApprovedCublasLtMatcher {
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
                text: include_str!("egg/separate_destination_call.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[
            ("configuration", 0),
            ("A_matrix_descriptor", 1),
            ("B_matrix_descriptor", 2),
            ("C_matrix_argument", 3),
            ("bias", 4),
            ("D_matrix_descriptor", 5),
            ("semantics", 7),
            ("scalars", 8),
        ]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let config = member(site, &site.child_class(0), "CublasLtConfigurationOf");
        let child = |n: usize| site.class_of_child(&config, n).unwrap();
        // This decodes a closed enum, not an applicability decision or fallback.
        let computation = computation(site, &child(0));
        let a = matrix(site, &site.child_class(1));
        let b = matrix(site, &site.child_class(2));
        let d = matrix(site, &site.child_class(5));
        let c_arg = site.child_class(3);
        let c = if let Some(node) = site
            .nodes_in_class_value(&c_arg, "CublasLtCMatrixInput")
            .next()
        {
            Some(matrix(site, &site.class_of_child(node, 0).unwrap()))
        } else {
            member(site, &c_arg, "CublasLtCMatrixAbsent");
            None
        };
        let bias = bias(site, &site.child_class(4));
        let enum_is =
            |class: &ClassId, name: &str| site.nodes_in_class_value(class, name).next().is_some();
        let frame = member(site, &site.child_class(7), "CublasLtExpressionFrameOf");
        let semantic = |n| site.class_of_child(&frame, n).unwrap();
        Box::new(ApprovedCublasLt {
            configuration: site.child_class(0),
            computation,
            logical_a: semantic(2),
            logical_b: semantic(3),
            logical_out: semantic(1),
            logical_root: semantic(0),
            a,
            b,
            c,
            d,
            bias,
            scale: dtype(site, &child(1)),
            trans_a: enum_is(&child(12), "CublasLtTranspose"),
            trans_b: enum_is(&child(13), "CublasLtTranspose"),
            scalars: scalars(site, &site.child_class(8), &child(7)),
            activation: activation(site, &child(8)),
            dps: false,
        })
    }
}

pub(super) fn member(site: &ExtractionSite<'_>, class: &ClassId, name: &str) -> Node {
    site.nodes_in_class_value(class, name)
        .next()
        .unwrap_or_else(|| panic!("cuBLASLt approved contract: {class} lacks {name}"))
        .clone()
}

pub(super) fn dtype(site: &ExtractionSite<'_>, class: &ClassId) -> PlanDtype {
    for (name, dtype) in [
        ("F32", PlanDtype::F32),
        ("F16", PlanDtype::F16),
        ("Bf16", PlanDtype::Bf16),
    ] {
        if site.nodes_in_class_value(class, name).next().is_some() {
            return dtype;
        }
    }
    panic!("cuBLASLt approved contract: unsupported dtype constructor in {class}");
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bias {
    pub tensor: ClassId,
    pub dtype: PlanDtype,
}

/// Scalar arguments are decoded independently of the matrix element types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scalars {
    Host { alpha_bits: u32, beta_bits: u32 },
    Device { alpha: ClassId, beta: ClassId },
}

impl Scalars {
    pub fn host(alpha: f32, beta: f32) -> Self {
        Self::Host {
            alpha_bits: alpha.to_bits(),
            beta_bits: beta.to_bits(),
        }
    }
    pub fn resolve(&self) -> LtScalars {
        match self {
            Self::Host {
                alpha_bits,
                beta_bits,
            } => LtScalars::Host {
                alpha_bits: *alpha_bits,
                beta_bits: *beta_bits,
            },
            Self::Device { .. } => LtScalars::Device {
                alpha_operand: 2,
                beta_operand: 3,
            },
        }
    }
}

pub(super) fn scalars(site: &ExtractionSite<'_>, class: &ClassId, mode: &ClassId) -> Scalars {
    if let Some(node) = site
        .nodes_in_class_value(class, "CublasLtHostScalarArguments")
        .next()
    {
        member(site, mode, "CublasLtHostScalarPointerMode");
        let value = |index| {
            let child = site.class_of_child(node, index).unwrap();
            site.index
                .nodes_of(&child)
                .iter()
                .find_map(|id| site.egraph.nodes[id].op.parse::<f64>().ok())
                .expect("cuBLASLt host scalar literal") as f32
        };
        Scalars::host(value(0), value(1))
    } else {
        member(site, mode, "CublasLtDeviceScalarPointerMode");
        let node = member(site, class, "CublasLtDeviceScalarArguments");
        Scalars::Device {
            alpha: site.class_of_child(&node, 0).unwrap(),
            beta: site.class_of_child(&node, 1).unwrap(),
        }
    }
}

pub(super) fn bias(site: &ExtractionSite<'_>, class: &ClassId) -> Option<Bias> {
    if let Some(node) = site.nodes_in_class_value(class, "CublasLtBiasInput").next() {
        Some(Bias {
            tensor: site.class_of_child(node, 0).unwrap(),
            dtype: dtype(site, &site.class_of_child(node, 1).unwrap()),
        })
    } else {
        member(site, class, "CublasLtBiasAbsent");
        None
    }
}

pub(super) fn computation(site: &ExtractionSite<'_>, class: &ClassId) -> LtComputation {
    for mode in [LtComputation::Float32, LtComputation::TensorFloat32] {
        if site
            .nodes_in_class_value(class, mode.egglog_constructor())
            .next()
            .is_some()
        {
            return mode;
        }
    }
    panic!("cuBLASLt approved contract: unsupported computation constructor in {class}");
}

pub(super) fn activation(site: &ExtractionSite<'_>, class: &ClassId) -> LtActivation {
    if site
        .nodes_in_class_value(class, "CublasLtRectifiedLinearEpilogue")
        .next()
        .is_some()
        || site
            .nodes_in_class_value(class, "CublasLtRectifiedLinearBiasEpilogue")
            .next()
            .is_some()
    {
        LtActivation::RectifiedLinear
    } else if site
        .nodes_in_class_value(class, "CublasLtGaussianErrorLinearEpilogue")
        .next()
        .is_some()
        || site
            .nodes_in_class_value(class, "CublasLtGaussianErrorLinearBiasEpilogue")
            .next()
            .is_some()
    {
        LtActivation::GaussianErrorLinear
    } else {
        LtActivation::Identity
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub tensor: ClassId,
    pub dtype: PlanDtype,
    pub order: LtOrder,
    pub rows: IotaExpr,
    pub cols: IotaExpr,
    pub ld: IotaExpr,
    pub batch_count: IotaExpr,
    pub batch_stride_in_elements: IotaExpr,
}

pub(super) fn matrix(site: &ExtractionSite<'_>, class: &ClassId) -> Matrix {
    let node = member(site, class, "CublasLtMatrixDescriptorOf");
    let layout = member(
        site,
        &site.class_of_child(&node, 1).unwrap(),
        "CublasLtMatrixLayoutLit",
    );
    let child = |n| site.class_of_child(&layout, n).unwrap();
    let expr = |n| {
        luminal::index_expr::parse_int_expr(site, &child(n), 128, None)
            .expect("descriptor integer expression")
    };
    Matrix {
        tensor: site.class_of_child(&node, 0).unwrap(),
        dtype: dtype(site, &child(0)),
        order: if site
            .nodes_in_class_value(&child(1), "CublasLtRowMajor")
            .next()
            .is_some()
        {
            LtOrder::Row
        } else {
            LtOrder::Col
        },
        rows: expr(2),
        cols: expr(3),
        ld: expr(4),
        batch_count: expr(5),
        batch_stride_in_elements: expr(6),
    }
}

impl Matrix {
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtDesc {
        LtDesc {
            rows: self.rows.eval_with_dims(&[], dims),
            cols: self.cols.eval_with_dims(&[], dims),
            ld: self.ld.eval_with_dims(&[], dims),
            order: self.order,
            batch_count: self.batch_count.eval_with_dims(&[], dims),
            batch_stride_in_elements: self.batch_stride_in_elements.eval_with_dims(&[], dims),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApprovedCublasLt {
    pub configuration: ClassId,
    pub computation: LtComputation,
    pub logical_a: ClassId,
    pub logical_b: ClassId,
    pub logical_out: ClassId,
    pub logical_root: ClassId,
    pub a: Matrix,
    pub b: Matrix,
    pub c: Option<Matrix>,
    pub d: Matrix,
    pub bias: Option<Bias>,
    pub scale: PlanDtype,
    pub trans_a: bool,
    pub trans_b: bool,
    pub scalars: Scalars,
    pub activation: LtActivation,
    dps: bool,
}

impl ApprovedCublasLt {
    pub fn prototype() -> Self {
        let matrix = Matrix {
            tensor: ClassId::from("prototype"),
            dtype: PlanDtype::F32,
            order: LtOrder::Row,
            rows: IotaExpr::Lit(1),
            cols: IotaExpr::Lit(1),
            ld: IotaExpr::Lit(1),
            batch_count: IotaExpr::Lit(1),
            batch_stride_in_elements: IotaExpr::Lit(0),
        };
        Self {
            configuration: ClassId::from("prototype"),
            computation: LtComputation::Float32,
            logical_a: matrix.tensor.clone(),
            logical_b: matrix.tensor.clone(),
            logical_out: matrix.tensor.clone(),
            logical_root: matrix.tensor.clone(),
            a: matrix.clone(),
            b: matrix.clone(),
            c: None,
            d: matrix,
            bias: None,
            scale: PlanDtype::F32,
            trans_a: false,
            trans_b: false,
            scalars: Scalars::host(1.0, 0.0),
            activation: LtActivation::Identity,
            dps: false,
        }
    }
    fn input_count(&self) -> usize {
        2 + self.scalars.resolve().input_count()
            + usize::from(self.c.is_some())
            + usize::from(self.bias.is_some())
    }

    /// Evaluate already-approved descriptor expressions. No layout inference,
    /// transpose flip, legality filter, destination rebinding, or dtype fallback.
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtCall {
        let a = self.a.resolve(dims);
        let b = self.b.resolve(dims);
        let d = self.d.resolve(dims);
        LtCall {
            computation: self.computation,
            form: match (self.c.is_some(), self.bias.is_some()) {
                (false, false) => CublasLtForm::Base,
                (false, true) => CublasLtForm::Bias,
                (true, false) => CublasLtForm::Accumulate,
                (true, true) => CublasLtForm::AccumulateBias,
            },
            m: d.rows,
            n: d.cols,
            k: if self.trans_a { a.rows } else { a.cols },
            trans_a: self.trans_a,
            trans_b: self.trans_b,
            a_dtype: self.a.dtype,
            b_dtype: self.b.dtype,
            d_dtype: self.d.dtype,
            a,
            b,
            c: self.c.as_ref().map(|c| LtCMatrixInput {
                descriptor: c.resolve(dims),
                data_type: c.dtype,
                operand: 2
                    + self.scalars.resolve().input_count()
                    + usize::from(self.bias.is_some()),
            }),
            d,
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

impl OpSlotNames for ApprovedCublasLt {
    fn operand_name(&self, operand: usize) -> String {
        format!("slot{operand}")
    }
}
impl BufferTensorIrOp for ApprovedCublasLt {
    fn label(&self) -> &str {
        "CublasLtCall"
    }
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::host::<Self>())
    }
    fn operand_reads_memory(&self, operand: usize) -> bool {
        !self.dps || operand < self.input_count()
    }
}
impl Bufferizable for ApprovedCublasLt {
    fn alias_info(&self) -> Vec<AliasInfo> {
        if !self.dps {
            return vec![];
        }
        vec![AliasInfo {
            operand: self.input_count(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}
impl ToDps for ApprovedCublasLt {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        if self.dps {
            None
        } else {
            let mut dps = self.clone();
            dps.dps = true;
            Some(Box::new(dps))
        }
    }
}
impl LayoutIrOp for ApprovedCublasLt {}

impl crate::HostOp for ApprovedCublasLt {
    fn capture_dims(&self) -> Option<Vec<luminal::shape::Symbol>> {
        Some(capture_dimensions(
            [&self.a, &self.b, &self.d].into_iter().chain(self.c.iter()),
        ))
    }

    fn workspace_bytes(&self, _: &crate::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(32 * 1024 * 1024)
    }

    #[cfg(feature = "device")]
    unsafe fn prepare(
        &self,
        ctx: &crate::host::HostOpContext<'_>,
    ) -> anyhow::Result<Box<dyn crate::host::PreparedHostOp>> {
        let call = self.resolve(ctx.dims);
        crate::ops::cublaslt::device_call::prepare(&call, ctx.inputs, ctx.dest, ctx.workspace)
            .map(|prepared| Box::new(prepared) as Box<dyn crate::host::PreparedHostOp>)
    }
}

/// Capture dependencies of the already-selected descriptor expressions.
pub(super) fn capture_dimensions<'a>(
    matrices: impl IntoIterator<Item = &'a Matrix>,
) -> Vec<luminal::shape::Symbol> {
    fn visit(expr: &IotaExpr, out: &mut std::collections::BTreeSet<luminal::shape::Symbol>) {
        use IotaExpr::*;
        match expr {
            Var(name) => {
                out.insert(luminal::shape::Symbol::from(name.as_str()));
            }
            Lit(_) | Coord(_) => {}
            Add(a, b)
            | Mul(a, b)
            | TruncDiv(a, b)
            | TruncRem(a, b)
            | CeilDiv(a, b)
            | Min(a, b)
            | Max(a, b)
            | LessThanCast(a, b) => {
                visit(a, out);
                visit(b, out);
            }
        }
    }
    let mut out = std::collections::BTreeSet::new();
    for matrix in matrices {
        for expr in [
            &matrix.rows,
            &matrix.cols,
            &matrix.ld,
            &matrix.batch_count,
            &matrix.batch_stride_in_elements,
        ] {
            visit(expr, &mut out);
        }
    }
    out.into_iter().collect()
}
