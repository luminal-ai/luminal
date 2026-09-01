//! Elementwise addition.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `AddFunctionalGeneric(lhs, rhs) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Elementwise: element
/// `i` of each input is read before element `i` of `out` is written (op-level
/// all-pairs claim — see the NOTE on `bufferizes_to_elementwise_access`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFunctional;

impl OpSlotNames for AddFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddFunctional {
    fn label(&self) -> &str {
        "AddFunctionalGeneric"
    }
}

impl Bufferizable for AddFunctional {}

impl ToDps for AddFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(AddFunctionalDps))
    }
}

impl LayoutIrOp for AddFunctional {}

/// Destination-passing form of [`AddFunctional`], signature spelled slot by slot:
///
/// ```text
/// AddGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddFunctionalDps;

impl OpSlotNames for AddFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddFunctionalDps {
    fn label(&self) -> &str {
        "AddFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // lhs
            1 => true,  // rhs
            2 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for AddFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for AddFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for AddFunctionalDps {}

/// `AddMutatingGeneric(lhs: read+write, rhs: read) -> out`
///
/// Mutating form: the kernel reads and overwrites ONE storage — its
/// first operand's. Matched in egglog only when the output layout equals
/// that operand's layout AND the written tensor is provably injective, so an
/// admitted tie is descriptor-exact by construction. The tie is `May` in the
/// relocation sense: a rejected mutation relocates the operand into the tied
/// result's fresh buffer (copy-in) and mutates there — the kernel's
/// one-buffer contract is invariant under relocation, never a hard error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMutating;

impl OpSlotNames for AddMutating {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMutating {
    fn label(&self) -> &str {
        "AddMutatingGeneric"
    }
}

impl Bufferizable for AddMutating {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for AddMutating {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for AddMutating {}

/// `AddMutatingInputAliasSafeGeneric(lhs: read+write, rhs: read) -> out`
///
/// Mutating form whose egglog match requires BOTH inputs and the output to
/// share one layout — which is exactly what makes it safe for `rhs` to alias
/// the mutated storage: element `i` coincides everywhere by construction,
/// and an elementwise kernel reads element `i` before writing it. The op
/// therefore declares the may-share permit for `rhs` against the mutated
/// result. The permit is unconditional here because its precondition was
/// discharged at match time; the engine trusts it and checks no layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddMutatingInputAliasSafe;

impl OpSlotNames for AddMutatingInputAliasSafe {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for AddMutatingInputAliasSafe {
    fn label(&self) -> &str {
        "AddMutatingInputAliasSafeGeneric"
    }
}

impl Bufferizable for AddMutatingInputAliasSafe {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![
            AliasInfo {
                operand: 0,
                result: 0,
                sharing: Sharing::Must,
            },
            AliasInfo {
                operand: 1,
                result: 0,
                sharing: Sharing::May,
            },
        ]
    }
}

impl ToDps for AddMutatingInputAliasSafe {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already destination-form: the destination IS operand 0
    }
}

impl LayoutIrOp for AddMutatingInputAliasSafe {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpAddFunctionalGeneric` enodes and produces
/// [`AddFunctional`] instances. Metadata children: `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddFunctionalMatcher;

impl OpMatcher for AddFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddFunctionalGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddFunctional)
    }
}

/// Matches `LayoutTensorOpAddMutatingGeneric` enodes and produces
/// [`AddMutating`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddMutatingMatcher;

impl OpMatcher for AddMutatingMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddMutatingGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_mutating_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_mutating.egg"),
            },
        ]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddMutating)
    }
}

/// Matches `LayoutTensorOpAddMutatingInputAliasSafeGeneric` enodes and produces
/// [`AddMutatingInputAliasSafe`] instances. No metadata children: the output layout IS
/// the mutated operand's, by the match rule's precondition.
#[derive(Debug, Clone, Copy, Default)]
pub struct AddMutatingInputAliasSafeMatcher;

impl OpMatcher for AddMutatingInputAliasSafeMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpAddMutatingInputAliasSafeGeneric"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_mutating_alias_safe_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_mutating_alias_safe.egg"),
            },
        ]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(AddMutatingInputAliasSafe)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::reference::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

/// Int arithmetic is CHECKED (non-wrapping by ruling 2026-08-11): an
/// overflow is a loud kernel error, never a wrapped value — until the
/// landing-D bounds proofs gate Int ops statically, this dynamic check
/// is the soundness floor.
pub(in crate::reference) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => ctx.binary_elementwise(|a, b| a + b),
        TypedBuffer::I32(_) => ctx.binary_elementwise_i32(|a, b| {
            a.checked_add(b).ok_or_else(|| {
                anyhow::anyhow!("i32 add overflow: {a} + {b} (ints are non-wrapping)")
            })
        }),
        TypedBuffer::I64(_) => ctx.binary_elementwise_i64(|a, b| {
            a.checked_add(b).ok_or_else(|| {
                anyhow::anyhow!("i64 add overflow: {a} + {b} (ints are non-wrapping)")
            })
        }),
        other => anyhow::bail!("add has no {} arm", other.type_name()),
    }
}
