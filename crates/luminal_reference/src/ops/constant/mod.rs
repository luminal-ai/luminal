//! The scalar constant: a ZERO-INPUT source op that writes ONE value into a
//! RANK-0 output (user ruling 2026-07-22: a true rank-0 tensor, never a
//! length-1 vector). Full-sized constant tensors are broadcast VIEWS of
//! this op's result — the empty index map at the target shape — so the
//! e-graph sees every broadcast of one constant as views over a single
//! rank-0 entity.
//!
//! Structurally the smallest op in the set: iota's zero-input shape with a
//! rank-0 destination. No mutating form — there is nothing to mutate.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `ConstantGeneric() -> out`
///
/// Functional source form: no operands, one freshly-written rank-0 result.
/// Reference semantics: write the value (F32 — the dtype rule pins this
/// in egglog) into the single element the layout addresses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant {
    /// The literal value (the term's f64 metadata).
    pub value: f64,
}

impl OpSlotNames for Constant {}

impl BufferTensorIrOp for Constant {
    fn label(&self) -> &str {
        "ConstantGeneric"
    }
}

impl Bufferizable for Constant {}

impl ToDps for Constant {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ConstantDps { value: self.value }))
    }
}

impl LayoutIrOp for Constant {}

/// Destination-passing form of [`Constant`], signature spelled slot by slot:
///
/// ```text
/// ConstantGeneric(dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantDps {
    /// The literal value (the term's f64 metadata).
    pub value: f64,
}

impl OpSlotNames for ConstantDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ConstantDps {
    fn label(&self) -> &str {
        "ConstantGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for ConstantDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ConstantDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ConstantDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpConstantGeneric` enodes and produces [`Constant`]
/// instances. Metadata children: `value` at child 0, `out_layout` at
/// child 1 — all metadata, no tensor operands.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstantMatcher;

impl OpMatcher for ConstantMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpConstantGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("value", 0), ("out_layout", 1)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Constant {
            value: site.child_f64(0),
        })
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::kernels::expect_op;
use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// Constant fill — LogicalConstant is F32 by its dtype rule.
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ConstantDps>(op)?;
    match &mut ctx.dests[0] {
        TypedBuffer::F32(dest) => dest.fill(op.value as f32),
        // LogicalConstant is F32 by its dtype rule; integer dests would
        // mean the plan annotated something the op cannot mean.
        other => anyhow::bail!(
            "constant fill has no {} arm (LogicalConstant is F32)",
            other.type_name()
        ),
    }
    Ok(())
}
