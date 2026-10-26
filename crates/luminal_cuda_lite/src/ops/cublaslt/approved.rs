//! A complete call is admitted in egglog before extraction. This module only
//! decodes the elected configuration and evaluates its descriptor expressions.
use super::CublasLtForm;
use super::exec::{CSource, LtCall, LtDesc, LtOrder};
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
                text: WHITELIST,
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
        ]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let config = member(site, &site.child_class(0), "CublasLtConfigurationOf");
        let child = |n: usize| site.class_of_child(&config, n).unwrap();
        // This decodes a closed enum, not an applicability decision or fallback.
        member(site, &child(0), "CublasLtFloat32Computation");
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
            member(site, &c_arg, "CublasLtZeroCMatrix");
            None
        };
        let bias_arg = site.child_class(4);
        let bias = site
            .nodes_in_class_value(&bias_arg, "CublasLtBiasInput")
            .next()
            .map(|node| site.class_of_child(node, 0).unwrap());
        let enum_is =
            |class: &ClassId, name: &str| site.nodes_in_class_value(class, name).next().is_some();
        let frame = member(site, &site.child_class(7), "CublasLtExpressionFrameOf");
        let semantic = |n| site.class_of_child(&frame, n).unwrap();
        Box::new(ApprovedCublasLt {
            configuration: site.child_class(0),
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
            storage: [
                dtype(site, &child(2)),
                dtype(site, &child(3)),
                dtype(site, &child(4)),
                dtype(site, &child(5)),
            ],
            trans_a: enum_is(&child(12), "CublasLtTranspose"),
            trans_b: enum_is(&child(13), "CublasLtTranspose"),
            beta_one: enum_is(&child(7), "CublasLtHostScalarsAlphaOneBetaOne"),
            relu: enum_is(&child(8), "CublasLtRectifiedLinearEpilogue")
                || enum_is(&child(8), "CublasLtRectifiedLinearBiasEpilogue"),
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

#[derive(Debug, Clone)]
pub struct Matrix {
    pub tensor: ClassId,
    pub dtype: PlanDtype,
    pub order: LtOrder,
    pub rows: IotaExpr,
    pub cols: IotaExpr,
    pub ld: IotaExpr,
}

pub(super) fn matrix(site: &ExtractionSite<'_>, class: &ClassId) -> Matrix {
    let node = member(site, class, "CublasLtMatrixDescriptorOf");
    let child = |n| site.class_of_child(&node, n).unwrap();
    let expr = |n| {
        luminal::index_expr::parse_int_expr(site, &child(n), 128, None)
            .expect("descriptor integer expression")
    };
    Matrix {
        tensor: child(0),
        dtype: dtype(site, &child(1)),
        order: if site
            .nodes_in_class_value(&child(2), "CublasLtRowMajor")
            .next()
            .is_some()
        {
            LtOrder::Row
        } else {
            LtOrder::Col
        },
        rows: expr(3),
        cols: expr(4),
        ld: expr(5),
    }
}

impl Matrix {
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtDesc {
        LtDesc {
            rows: self.rows.eval_with_dims(&[], dims),
            cols: self.cols.eval_with_dims(&[], dims),
            ld: self.ld.eval_with_dims(&[], dims),
            order: self.order,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApprovedCublasLt {
    pub configuration: ClassId,
    pub logical_a: ClassId,
    pub logical_b: ClassId,
    pub logical_out: ClassId,
    pub logical_root: ClassId,
    pub a: Matrix,
    pub b: Matrix,
    pub c: Option<Matrix>,
    pub d: Matrix,
    pub bias: Option<ClassId>,
    pub scale: PlanDtype,
    pub storage: [PlanDtype; 4],
    pub trans_a: bool,
    pub trans_b: bool,
    pub beta_one: bool,
    pub relu: bool,
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
        };
        Self {
            configuration: ClassId::from("prototype"),
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
            storage: [PlanDtype::F32; 4],
            trans_a: false,
            trans_b: false,
            beta_one: false,
            relu: false,
            dps: false,
        }
    }
    fn input_count(&self) -> usize {
        2 + usize::from(self.c.is_some()) + usize::from(self.bias.is_some())
    }

    /// Evaluate already-approved descriptor expressions. No layout inference,
    /// transpose flip, legality filter, destination rebinding, or dtype fallback.
    pub fn resolve(&self, dims: &luminal::shape::DynMap) -> LtCall {
        let a = self.a.resolve(dims);
        let b = self.b.resolve(dims);
        let d = self.d.resolve(dims);
        LtCall {
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
            c: self.c.as_ref().map_or(d, |c| c.resolve(dims)),
            d,
            c_source: if self.c.is_some() {
                CSource::Operand(2)
            } else {
                CSource::AliasD
            },
            beta_is_one: self.beta_one,
            relu: self.relu,
            bias_operand: self
                .bias
                .as_ref()
                .map(|_| 2 + usize::from(self.c.is_some())),
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
        for expr in [&matrix.rows, &matrix.cols, &matrix.ld] {
            visit(expr, &mut out);
        }
    }
    out.into_iter().collect()
}
