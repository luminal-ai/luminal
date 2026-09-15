//! THE PLAIN FUNCTIONAL OPS — forked from the reference registry.
//!
//! One folder per op, matcher and `.egg` rewrites together, kernels
//! omitted (this runtime never executes). These are the same 23 spellings
//! the reference runtime ships, carried here so the TestRuntime depends
//! on no other runtime crate.
//!
//! THE FORK IS REAL, and so is its cost: these rewrite rules now exist in
//! two places and nothing keeps them in step. `functional_rules_match_the_reference`
//! in `tests/fork_drift.rs` is the tripwire — it diffs this tree's `.egg`
//! text against the reference's and fails on any divergence, so a rule
//! edited on one side is a loud test failure rather than a silent
//! semantic split. Delete that test only when this op list is
//! deliberately whittled down to something that no longer mirrors the
//! reference.

pub mod add;
pub mod cast;
pub mod constant;
pub mod div;
pub mod exp;
pub mod exp2;
pub mod gather;
pub mod index_map_apply_materialize;
pub mod iota;
pub mod less_than;
pub mod log2;
pub mod materialize_layout_copy;
pub mod modulo;
pub mod mul;
pub mod recip;
pub mod reduce_max;
pub mod reduce_sum;
pub mod scatter;
pub mod select;
pub mod sin;
pub mod sqrt;
pub mod trunc_div;
pub mod trunc_rem;

/// Every forked functional matcher.
pub fn functional_matchers() -> Vec<Box<dyn luminal::layout_ir::OpMatcher>> {
    vec![
        Box::new(add::AddFunctionalMatcher),
        Box::new(cast::CastMatcher),
        Box::new(constant::ConstantMatcher),
        Box::new(div::DivFunctionalMatcher),
        Box::new(exp::ExpFunctionalMatcher),
        Box::new(exp2::Exp2FunctionalMatcher),
        Box::new(gather::GatherMatcher),
        Box::new(index_map_apply_materialize::IndexMapApplyMaterializeMatcher),
        Box::new(iota::IotaMatcher),
        Box::new(less_than::LessThanMatcher),
        Box::new(log2::Log2FunctionalMatcher),
        Box::new(materialize_layout_copy::MaterializeLayoutCopyMatcher),
        Box::new(modulo::ModFunctionalMatcher),
        Box::new(mul::MulFunctionalMatcher),
        Box::new(recip::RecipFunctionalMatcher),
        Box::new(reduce_max::ReduceMaxMatcher),
        Box::new(reduce_sum::ReduceSumMatcher),
        Box::new(scatter::ScatterFunctionalMatcher),
        Box::new(select::SelectFunctionalMatcher),
        Box::new(sin::SinFunctionalMatcher),
        Box::new(sqrt::SqrtFunctionalMatcher),
        Box::new(trunc_div::TruncDivFunctionalMatcher),
        Box::new(trunc_rem::TruncRemFunctionalMatcher),
    ]
}
