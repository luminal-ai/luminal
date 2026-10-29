//! The type contract selected by egglog, separate from layout and epilogue.
//! Supported tuples live in `egg/cublaslt_config.egg`; Rust decodes the
//! elected term and dispatches it without selecting another implementation.

use luminal::dtype::PlanDtype;
use luminal::layout_ir::ExtractionSite;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtComputeType {
    /// `CUBLAS_COMPUTE_32F`: F32 multiply/accumulate, independent of D storage.
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtConfig {
    pub a_dtype: PlanDtype,
    pub b_dtype: PlanDtype,
    pub c_dtype: PlanDtype,
    pub d_dtype: PlanDtype,
    pub compute_type: LtComputeType,
    pub scale_dtype: PlanDtype,
    pub bias_dtype: PlanDtype,
}

impl LtConfig {
    /// The existing F32 contract, also useful for direct-call test fixtures.
    pub const F32: Self = Self {
        a_dtype: PlanDtype::F32,
        b_dtype: PlanDtype::F32,
        c_dtype: PlanDtype::F32,
        d_dtype: PlanDtype::F32,
        compute_type: LtComputeType::F32,
        scale_dtype: PlanDtype::F32,
        bias_dtype: PlanDtype::F32,
    };

    pub(super) fn parse(site: &ExtractionSite<'_>, slot: usize) -> Option<Self> {
        let class = site.child_class(slot);
        let node = site
            .nodes_in_class_value(&class, "CublasLtConfigLit")
            .next()?;
        let dtype = |index| {
            let class = site.class_of_child(node, index)?;
            site.index
                .nodes_of(&class)
                .iter()
                .find_map(|id| PlanDtype::from_egglog_name(&site.egraph.nodes[id].op))
        };
        let compute = site.class_of_child(node, 4)?;
        site.nodes_in_class_value(&compute, "CublasLtCompute32F")
            .next()?;
        Some(Self {
            a_dtype: dtype(0)?,
            b_dtype: dtype(1)?,
            c_dtype: dtype(2)?,
            d_dtype: dtype(3)?,
            compute_type: LtComputeType::F32,
            scale_dtype: dtype(5)?,
            bias_dtype: dtype(6)?,
        })
    }
}
