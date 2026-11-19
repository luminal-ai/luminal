//! THE MUTATING FAMILY — every in-place spelling of a logical op.
//!
//! Each declares `Must(0 -> 0)` with operand 0 READ, and `to_dps() ->
//! None`: the destination IS operand 0, so there is nothing to pass. That
//! declaration shape — a tie that writes storage it also reads — is what
//! drives the bufferizer's read-after-write conflict engine, its
//! relocation repair, and (for the alias-safe add) the only
//! `Sharing::May` permit in the tree.
//!
//! They live here because the reference runtime cannot execute them: its
//! kernel table carries only `*FunctionalDps` rows, and
//! `reference_allow_list()` is derived from that table, so every one of
//! these was extractable-but-never-selectable there while still costing
//! its egglog rules on each saturation.

pub mod add;
pub mod cos;
pub mod div;
pub mod exp;
pub mod exp2;
pub mod log2;
pub mod modulo;
pub mod mul;
pub mod recip;
pub mod scatter;
pub mod sin;
pub mod sqrt;

pub use add::{
    AddMutating, AddMutatingInputAliasSafe, AddMutatingInputAliasSafeMatcher, AddMutatingMatcher,
};
pub use cos::{CosMutating, CosMutatingMatcher};
pub use div::{DivMutating, DivMutatingMatcher};
pub use exp::{ExpMutating, ExpMutatingMatcher};
pub use exp2::{Exp2Mutating, Exp2MutatingMatcher};
pub use log2::{Log2Mutating, Log2MutatingMatcher};
pub use modulo::{ModMutating, ModMutatingMatcher};
pub use mul::{MulMutating, MulMutatingMatcher};
pub use recip::{RecipMutating, RecipMutatingMatcher};
pub use scatter::{ScatterMutating, ScatterMutatingMatcher};
pub use sin::{SinMutating, SinMutatingMatcher};
pub use sqrt::{SqrtMutating, SqrtMutatingMatcher};

/// Every mutating matcher, for the TestRuntime's vocabulary.
pub fn mutating_matchers() -> Vec<Box<dyn luminal::layout_ir::OpMatcher>> {
    vec![
        Box::new(add::AddMutatingMatcher),
        Box::new(add::AddMutatingInputAliasSafeMatcher),
        Box::new(cos::CosMutatingMatcher),
        Box::new(div::DivMutatingMatcher),
        Box::new(exp::ExpMutatingMatcher),
        Box::new(exp2::Exp2MutatingMatcher),
        Box::new(log2::Log2MutatingMatcher),
        Box::new(modulo::ModMutatingMatcher),
        Box::new(mul::MulMutatingMatcher),
        Box::new(recip::RecipMutatingMatcher),
        Box::new(scatter::ScatterMutatingMatcher),
        Box::new(sin::SinMutatingMatcher),
        Box::new(sqrt::SqrtMutatingMatcher),
    ]
}
