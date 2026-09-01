//! The poison-value producer synthesized by the DPS rewrite.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{Bufferizable, LayoutIrOp, ToDps};

/// `Poison() -> out0` — no operands, one result whose contents are undefined
/// (garbage). It exists only to give an appended destination operand a
/// producer in value-SSA; the bufferizer folds it away (its storage is the
/// destination's allocation, or — after seeding — the pinned output buffer).
///
/// Not an egglog op: it is synthesized by `dps_rewrite`, never extracted, and
/// never survives into a plan, so its label is outside the label policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poison;

impl OpSlotNames for Poison {}

impl BufferTensorIrOp for Poison {
    fn label(&self) -> &str {
        "Poison"
    }

    fn result_is_undefined(&self, _result: usize) -> bool {
        true
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // contents unspecified — nothing meaningful is written
    }
}

impl Bufferizable for Poison {}

impl ToDps for Poison {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for Poison {}
