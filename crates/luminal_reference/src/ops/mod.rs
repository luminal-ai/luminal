//! The built-in LayoutIR op set, one file per op pair (functional form and
//! its destination-passing form, defined side by side — the pairing is part
//! of each op's definition).
//!
//! Everything here is deliberately hand-written and verbose: no macro, no
//! shared index arithmetic. Each op spells out its operand signature slot by
//! slot — which operands are read, and which destination operand is tied to
//! which result — the way a function signature spells out its returns.
//!
//! Label policy: an op's label is its egglog constructor name with only the
//! `LayoutTensorOp` namespace prefix removed — nothing else is edited (the
//! `Generic` suffix stays, and NO suffix is ever appended). The IR name is the
//! op's identity: future op sets will encode calling convention in the name
//! itself (e.g. FunctionalAdd vs InPlaceAdd), so tooling must never do name
//! surgery. The DPS form of an op therefore displays the SAME label; DPS-ness
//! is visible in the operand list (trailing destinations), not the name.

// The instance re-exports are the op-authoring surface: since the extractor
// went registry-driven it constructs instances only through matchers, so the
// non-test build no longer names these types — their remaining consumers are
// tests and the upcoming reference-backend/DPS work.
#![allow(unused_imports)]

// Kernel-bearing op modules are visible to the rest of this crate: the
// kernel registry's table rows point at
// `<op_module>::kernel` (op-folder ruling 2026-08-13).
pub(crate) mod add;
pub(crate) mod cast;
pub(crate) mod constant;
pub(crate) mod div;
pub(crate) mod exp;
pub(crate) mod exp2;
pub(crate) mod gather;
pub(crate) mod index_map_apply_materialize;
pub(crate) mod iota;
pub(crate) mod less_than;
pub(crate) mod log2;
pub(crate) mod modulo;
pub(crate) mod mul;
pub(crate) mod trunc_div;
pub(crate) mod trunc_rem;
// (poison moved to core `luminal::poison` in Step B; re-exported below.)
pub(crate) mod recip;
pub(crate) mod reduce_max;
pub(crate) mod reduce_sum;
pub(crate) mod scatter;
pub(crate) mod sin;
pub(crate) mod sqrt;

// The functional forms, the mutating forms, and Poison are exported: a DPS
// form is an implementation detail of its op pair, entering the world solely
// as a `Box<dyn LayoutIrOp>` through `to_dps()`. (A mutating form is NOT a
// DPS detail — it is its own extractable implementation.) If a pass ever
// needs a concrete DPS type, re-export it explicitly here.
//
// Instance structs deliberately do NOT derive `Default`: an instance is
// created by its matcher's `extract` (or derived via `to_dps` / synthesized
// by the DPS pass, for Poison) — a hygiene convention, not a privacy fence.
pub use add::AddFunctional;
pub use cast::Cast;
pub use constant::Constant;
pub use div::DivFunctional;
pub use exp::ExpFunctional;
pub use exp2::Exp2Functional;
pub use gather::Gather;
pub use index_map_apply_materialize::IndexMapApplyMaterialize;
pub use iota::Iota;
pub use less_than::LessThan;
pub use log2::Log2Functional;
pub use luminal::poison::Poison;
pub use modulo::ModFunctional;
pub use mul::MulFunctional;
pub use recip::RecipFunctional;
pub use reduce_max::ReduceMax;
pub use reduce_sum::ReduceSum;
pub use scatter::ScatterFunctional;
pub use sin::SinFunctional;
pub use sqrt::SqrtFunctional;

// DPS forms, re-exported for the reference runtime's kernel registry
// (reference::kernels downcasts plan ops to these concrete types —
// ops carry no execution of their own, ruling 2026-08-06).
pub use add::AddFunctionalDps;
pub use cast::CastDps;
pub use constant::ConstantDps;
pub use div::DivFunctionalDps;
pub use exp::ExpFunctionalDps;
pub use exp2::Exp2FunctionalDps;
pub use gather::GatherDps;
pub use index_map_apply_materialize::IndexMapApplyMaterializeDps;
pub use iota::{IotaDps, IotaExpr};
pub use less_than::LessThanDps;
pub use log2::Log2FunctionalDps;
pub use modulo::ModFunctionalDps;
pub use mul::MulFunctionalDps;
pub use recip::RecipFunctionalDps;
pub use reduce_max::ReduceMaxDps;
pub use reduce_sum::ReduceSumDps;
pub use scatter::ScatterFunctionalDps;
pub use sin::SinFunctionalDps;
pub use sqrt::SqrtFunctionalDps;

pub use add::AddFunctionalMatcher;
pub use cast::CastMatcher;
pub use constant::ConstantMatcher;
pub use div::DivFunctionalMatcher;
pub use exp::ExpFunctionalMatcher;
pub use exp2::Exp2FunctionalMatcher;
pub use gather::GatherMatcher;
pub use index_map_apply_materialize::IndexMapApplyMaterializeMatcher;
pub use iota::IotaMatcher;
pub use less_than::LessThanMatcher;
pub use log2::Log2FunctionalMatcher;
pub use modulo::ModFunctionalMatcher;
pub use mul::MulFunctionalMatcher;
pub use recip::RecipFunctionalMatcher;
pub use reduce_max::ReduceMaxMatcher;
pub use reduce_sum::ReduceSumMatcher;
pub use scatter::ScatterFunctionalMatcher;
pub use sin::SinFunctionalMatcher;
pub use sqrt::SqrtFunctionalMatcher;
pub use trunc_div::{TruncDivFunctional, TruncDivFunctionalDps, TruncDivFunctionalMatcher};
pub use trunc_rem::{TruncRemFunctional, TruncRemFunctionalDps, TruncRemFunctionalMatcher};

/// The matcher half of [`reference_ops`] — what the extractor builds its
/// constructor-name registry from. Not a list in its own right: every
/// entry comes from a row that also carries the op's kernel.
pub fn built_in_matchers() -> Vec<Box<dyn luminal::layout_ir::OpMatcher>> {
    reference_ops().iter().map(|op| (op.matcher)()).collect()
}

/// ONE ROW PER OP: how the e-graph produces it, and how the executor runs
/// it, in a single value that cannot be half-written.
///
/// This pairing is the whole point. The matcher list and the kernel table
/// used to be two independent `vec![…]`s, and "every registered op is
/// executable" was a property you had to *check* — the whole Mutating
/// family sat registered-but-kernel-less for months, minting e-nodes on
/// every saturation that selection then discarded. A row here carries
/// both halves, so that state is no longer expressible: you cannot
/// register an op without writing its kernel, and a kernel with no
/// matcher never reaches the executor because dispatch only ever walks
/// this table.
pub struct ReferenceOp {
    /// Builds the matcher. A fn pointer, not a value, because
    /// `Box<dyn OpMatcher>` is not `Clone` and callers want owned lists.
    pub matcher: fn() -> Box<dyn luminal::layout_ir::OpMatcher>,
    /// The executable form: the DPS type this op lowers to, its IR label,
    /// and the kernel body from the op's own folder.
    pub kernel: crate::kernels::ReferenceKernel,
}

/// THE REGISTRATION LIST: every built-in op, matcher and kernel together.
/// Adding an op = writing its module (instances + traits + matcher +
/// kernel, all in one folder) and adding its row here; nothing else in
/// the tree changes.
///
/// Deliberately hand-written and verbose, like the op modules themselves:
/// no macro, each row spelling out its own matcher, DPS type, label and
/// kernel. (Poison has no row — it is synthesized by the DPS pass, never
/// extracted and never dispatched. BufferAlloc/BufferFree have no row
/// either: they are bufferizer-minted plan infrastructure, not ops, and
/// live in the kernel table's infrastructure section.)
pub fn reference_ops() -> &'static [ReferenceOp] {
    use crate::kernels::entry;
    static OPS: std::sync::OnceLock<Vec<ReferenceOp>> = std::sync::OnceLock::new();
    OPS.get_or_init(|| {
        vec![
            // ── elementwise binary ──
            ReferenceOp {
                matcher: || Box::new(AddFunctionalMatcher),
                kernel: entry::<AddFunctionalDps>("AddFunctionalGeneric", add::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(MulFunctionalMatcher),
                kernel: entry::<MulFunctionalDps>("MulFunctionalGeneric", mul::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(DivFunctionalMatcher),
                kernel: entry::<DivFunctionalDps>("DivFunctionalGeneric", div::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(TruncDivFunctionalMatcher),
                kernel: entry::<TruncDivFunctionalDps>(
                    "TruncDivFunctionalGeneric",
                    trunc_div::kernel,
                ),
            },
            ReferenceOp {
                matcher: || Box::new(TruncRemFunctionalMatcher),
                kernel: entry::<TruncRemFunctionalDps>(
                    "TruncRemFunctionalGeneric",
                    trunc_rem::kernel,
                ),
            },
            ReferenceOp {
                matcher: || Box::new(ModFunctionalMatcher),
                kernel: entry::<ModFunctionalDps>("ModFunctionalGeneric", modulo::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(LessThanMatcher),
                kernel: entry::<LessThanDps>("LessThanGeneric", less_than::kernel),
            },
            // ── elementwise unary ──
            ReferenceOp {
                matcher: || Box::new(SqrtFunctionalMatcher),
                kernel: entry::<SqrtFunctionalDps>("SqrtFunctionalGeneric", sqrt::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(ExpFunctionalMatcher),
                kernel: entry::<ExpFunctionalDps>("ExpFunctionalGeneric", exp::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(Exp2FunctionalMatcher),
                kernel: entry::<Exp2FunctionalDps>("Exp2FunctionalGeneric", exp2::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(Log2FunctionalMatcher),
                kernel: entry::<Log2FunctionalDps>("Log2FunctionalGeneric", log2::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(SinFunctionalMatcher),
                kernel: entry::<SinFunctionalDps>("SinFunctionalGeneric", sin::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(RecipFunctionalMatcher),
                kernel: entry::<RecipFunctionalDps>("RecipFunctionalGeneric", recip::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(CastMatcher),
                kernel: entry::<CastDps>("CastGeneric", cast::kernel),
            },
            // ── sources ──
            ReferenceOp {
                matcher: || Box::new(ConstantMatcher),
                kernel: entry::<ConstantDps>("ConstantGeneric", constant::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(IotaMatcher),
                kernel: entry::<IotaDps>("IotaGeneric", iota::kernel),
            },
            // ── reductions ──
            ReferenceOp {
                matcher: || Box::new(ReduceSumMatcher),
                kernel: entry::<ReduceSumDps>("ReduceSumGeneric", reduce_sum::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(ReduceMaxMatcher),
                kernel: entry::<ReduceMaxDps>("ReduceMaxGeneric", reduce_max::kernel),
            },
            // ── data movement ──
            ReferenceOp {
                matcher: || Box::new(GatherMatcher),
                kernel: entry::<GatherDps>("GatherGeneric", gather::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(ScatterFunctionalMatcher),
                kernel: entry::<ScatterFunctionalDps>("ScatterFunctionalGeneric", scatter::kernel),
            },
            ReferenceOp {
                matcher: || Box::new(IndexMapApplyMaterializeMatcher),
                kernel: entry::<IndexMapApplyMaterializeDps>(
                    "IndexMapApplyMaterialize",
                    index_map_apply_materialize::kernel,
                ),
            },
        ]
    })
}

// =============================================================================
// Registry-contract pins (moved from core `bufferize` tests in Step B:
// they assert on THIS registry's declarations, not on the engine).
// =============================================================================
#[cfg(test)]
mod registry_contract {
    use luminal::layout_ir::{LayoutIrOp, Sharing, permits_sharing};

    /// The built-in functional ops take the conservative out-of-place
    /// defaults: no ties, results written into planner-allocated storage
    /// (never op-internal allocations), and never undefined contents.
    #[test]
    fn builtin_ops_declare_out_of_place_defaults() {
        use super::{AddFunctional, IndexMapApplyMaterialize, ReduceSum, SqrtFunctional};
        let ops: Vec<Box<dyn LayoutIrOp>> = vec![
            Box::new(AddFunctional),
            Box::new(SqrtFunctional),
            Box::new(ReduceSum { axis: 0 }),
            Box::new(IndexMapApplyMaterialize { entries: None }),
        ];
        for op in &ops {
            assert!(op.alias_info().is_empty());
            assert!(!op.result_is_allocated_internally(0));
            assert!(op.result_writes_memory(0));
            assert!(!op.result_is_undefined(0));
        }
    }

    /// RANK 9: NO built-in declares the unconditional sharing permit — ops
    /// whose in-place safety depends on preconditions get matched with those
    /// preconditions in egglog instead of asserting a blanket permit the
    /// engine would have to trust.
    ///
    /// The `Sharing::May` half of this pin moved with the only op that
    /// declares one: `AddMutatingInputAliasSafe` now lives in the
    /// TestRuntime, and `permit_is_declared_and_directional` there is the
    /// positive case. What stays here is the claim about THIS registry —
    /// that nothing in it declares a permit at all.
    #[test]
    fn builtin_ops_declare_no_unconditional_permits() {
        use super::{
            AddFunctional, DivFunctional, ExpFunctional, IndexMapApplyMaterialize, MulFunctional,
            ReduceMax, ReduceSum, SqrtFunctional,
        };
        let ops: Vec<Box<dyn LayoutIrOp>> = vec![
            Box::new(SqrtFunctional),
            Box::new(ExpFunctional),
            Box::new(AddFunctional),
            Box::new(MulFunctional),
            Box::new(DivFunctional),
            Box::new(ReduceSum { axis: 0 }),
            Box::new(ReduceMax { axis: 0 }),
            Box::new(IndexMapApplyMaterialize { entries: None }),
        ];
        for op in &ops {
            assert!(
                op.alias_info()
                    .iter()
                    .all(|info| info.sharing == Sharing::Must),
                "{}",
                op.label()
            );
        }
    }
}
