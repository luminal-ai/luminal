//! Backend-owned operations, matchers, and configurable registry.
//! Kernel execution is claimed through the DPS operation interface.

pub mod add;
pub mod cast;
pub mod constant;

pub mod div;
pub mod exp;
pub mod exp2;
pub mod gather;
pub mod index_map_apply_materialize;
pub mod index_map_apply_view;
pub mod iota;
pub mod less_than;
pub mod log2;
pub mod materialize_layout_copy;
pub mod modulo;
pub mod mul;
pub mod mul_reduce_sum;
pub mod recip;
pub mod reduce_max;
pub mod reduce_sum;
pub mod scatter;
pub mod sin;
pub mod sqrt;
pub mod trunc_div;
pub mod trunc_rem;

use luminal::layout_ir::{LayoutIrOp, OpMatcher};

pub struct RegisteredOp {
    pub matcher: Box<dyn OpMatcher>,
    pub prototype: Box<dyn LayoutIrOp>,
}

impl RegisteredOp {
    pub fn new(matcher: Box<dyn OpMatcher>, prototype: Box<dyn LayoutIrOp>) -> Self {
        Self { matcher, prototype }
    }

    pub fn decoders(&self) -> Vec<luminal::egglog_utils::eclass::ConstructorDecoder> {
        self.matcher.decoders()
    }

    pub fn constructor(&self) -> &'static str {
        self.matcher.egglog_constructor()
    }

    pub fn label(&self) -> &'static str {
        let ctor = self.constructor();
        ctor.strip_prefix("LayoutTensorOp").unwrap_or(ctor)
    }
}

pub fn metal_registry() -> Vec<RegisteredOp> {
    fn reg(
        matcher: impl OpMatcher + 'static,
        prototype: impl LayoutIrOp + 'static,
    ) -> RegisteredOp {
        RegisteredOp {
            matcher: Box::new(matcher),
            prototype: Box::new(prototype),
        }
    }
    vec![
        reg(add::AddFunctionalMatcher, add::AddFunctional),
        reg(
            mul_reduce_sum::MulReduceSumMatcher,
            mul_reduce_sum::MulReduceSum { axis: 0 },
        ),
        reg(
            materialize_layout_copy::MaterializeLayoutCopyMatcher,
            materialize_layout_copy::MaterializeLayoutCopy,
        ),
        reg(sqrt::SqrtFunctionalMatcher, sqrt::SqrtFunctional),
        reg(exp::ExpFunctionalMatcher, exp::ExpFunctional),
        reg(mul::MulFunctionalMatcher, mul::MulFunctional),
        reg(div::DivFunctionalMatcher, div::DivFunctional),
        reg(
            trunc_div::TruncDivFunctionalMatcher,
            trunc_div::TruncDivFunctional,
        ),
        reg(
            trunc_rem::TruncRemFunctionalMatcher,
            trunc_rem::TruncRemFunctional,
        ),
        reg(
            reduce_sum::ReduceSumMatcher,
            reduce_sum::ReduceSum { axis: 0 },
        ),
        reg(
            reduce_max::ReduceMaxMatcher,
            reduce_max::ReduceMax { axis: 0 },
        ),
        reg(iota::IotaMatcher, iota::Iota { expr: None }),
        reg(gather::GatherMatcher, gather::Gather { rank: 1 }),
        reg(constant::ConstantMatcher, constant::Constant { value: 0.0 }),
        reg(
            scatter::ScatterFunctionalMatcher,
            scatter::ScatterFunctional { rank: 1 },
        ),
        reg(exp2::Exp2FunctionalMatcher, exp2::Exp2Functional),
        reg(log2::Log2FunctionalMatcher, log2::Log2Functional),
        reg(sin::SinFunctionalMatcher, sin::SinFunctional),
        reg(recip::RecipFunctionalMatcher, recip::RecipFunctional),
        reg(modulo::ModFunctionalMatcher, modulo::ModFunctional),
        reg(less_than::LessThanMatcher, less_than::LessThan),
        reg(cast::CastMatcher, cast::Cast),
        reg(
            index_map_apply_materialize::IndexMapApplyMaterializeMatcher,
            index_map_apply_materialize::IndexMapApplyMaterialize { entries: None },
        ),
        reg(
            index_map_apply_view::IndexMapApplyViewMatcher,
            index_map_apply_view::IndexMapApplyView { entries: None },
        ),
    ]
}

pub fn metal_registry_filtered(keep: impl Fn(&RegisteredOp) -> bool) -> Vec<RegisteredOp> {
    metal_registry()
        .into_iter()
        .filter(|entry| keep(entry))
        .collect()
}

pub fn metal_matchers() -> Vec<Box<dyn OpMatcher>> {
    metal_registry()
        .into_iter()
        .map(|entry| entry.matcher)
        .collect()
}
