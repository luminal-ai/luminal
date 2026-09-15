//! Select: elementwise ternary choice over two same-shape/same-dtype value
//! branches by a boolean condition. See the match rules for the backend
//! story.
//!
//! This is the non-arithmetic selection primitive. Expressing a select as
//! `cond*a + (1-cond)*b` puts a sum of products into the e-graph, which the
//! integer associativity/commutativity/distributivity closure then expands
//! combinatorially; a real ternary op keeps the expression a single node.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `SelectFunctionalGeneric(condition, if_true, if_false) -> out` — functional form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectFunctional;

impl OpSlotNames for SelectFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "condition".to_string(),
            1 => "if_true".to_string(),
            2 => "if_false".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for SelectFunctional {
    fn label(&self) -> &str {
        "SelectFunctionalGeneric"
    }
}

impl Bufferizable for SelectFunctional {}

impl ToDps for SelectFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(SelectFunctionalDps))
    }
}

impl LayoutIrOp for SelectFunctional {}

/// Destination-passing form of [`SelectFunctional`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectFunctionalDps;

impl OpSlotNames for SelectFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "condition".to_string(),
            1 => "if_true".to_string(),
            2 => "if_false".to_string(),
            3 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for SelectFunctionalDps {
    fn label(&self) -> &str {
        "SelectFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0..=2 => true,
            3 => false, // dest0: write-only destination
            _ => true,
        }
    }
}

impl Bufferizable for SelectFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 3,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for SelectFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for SelectFunctionalDps {}

/// Matches `LayoutTensorOpSelectFunctionalGeneric` enodes.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelectFunctionalMatcher;

impl OpMatcher for SelectFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpSelectFunctionalGeneric"
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
        &[("out_layout", 3)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(SelectFunctional)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling 2026-08-13:
// everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// `dest0[i] = if condition[i] != 0 { if_true[i] } else { if_false[i] }`.
///
/// The condition is Bool8; the two value branches share one storage dtype,
/// which may be any supported type. The output is one of the branches
/// verbatim, so no arithmetic (and no width/overflow concern) is involved.
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let condition = match &ctx.operands[0] {
        TypedBuffer::Bool8(values) => values.clone(),
        other => anyhow::bail!("select condition must be Bool8, got {}", other.type_name()),
    };
    anyhow::ensure!(
        condition.len() == ctx.dests[0].len(),
        "select condition length {} != output length {}",
        condition.len(),
        ctx.dests[0].len()
    );

    macro_rules! arm {
        ($as:ident, $as_mut:ident) => {{
            let if_true = ctx.operands[1].$as()?.clone();
            let if_false = ctx.operands[2].$as()?.clone();
            anyhow::ensure!(
                if_true.len() == if_false.len() && if_true.len() == condition.len(),
                "select branch length mismatch"
            );
            let dest = ctx.dests[0].$as_mut()?;
            for (index, out) in dest.iter_mut().enumerate() {
                *out = if condition[index] != 0 {
                    if_true[index]
                } else {
                    if_false[index]
                };
            }
        }};
    }

    match &ctx.operands[1] {
        TypedBuffer::F32(_) => arm!(as_f32, as_f32_mut),
        TypedBuffer::F64(_) => arm!(as_f64, as_f64_mut),
        TypedBuffer::I32(_) => arm!(as_i32, as_i32_mut),
        TypedBuffer::I64(_) => arm!(as_i64, as_i64_mut),
        TypedBuffer::I8(_) => arm!(as_i8, as_i8_mut),
        TypedBuffer::U8(_) => arm!(as_u8, as_u8_mut),
        TypedBuffer::I16(_) => arm!(as_i16, as_i16_mut),
        TypedBuffer::Bool8(_) => arm!(as_bool8, as_bool8_mut),
        other => anyhow::bail!("select has no {} arm", other.type_name()),
    }
    Ok(())
}
