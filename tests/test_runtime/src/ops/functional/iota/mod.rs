//! FORKED from `luminal_reference::ops::iota` — the TestRuntime owns its
//! whole op set outright and depends on no other runtime.
//!
//! The KERNEL is deliberately not carried over. This runtime is
//! plan-level: it asserts on `ExtractedGraph`s and `BufferIrGraph`s and
//! never executes, so a kernel here would be dead code demanding a
//! dispatch table to sit in. What it needs is the matcher, the instance
//! and the DPS form — the declarations the bufferizer reads.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `IotaGeneric() -> out`
///
/// Functional source form: no operands, one freshly-written result.
/// Reference semantics: `out[c0..ck] = expr(c0..ck)` evaluated in the
/// canonical right-major order, Int element type (the value IS an index
/// expression, never a float; the dtype rule pins this in egglog).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iota {
    /// The window expression, numerically — `None` for forms beyond the
    /// parsed subset (extraction stays infallible; the reference kernel
    /// refuses loudly instead).
    pub expr: Option<IotaExpr>,
}

pub use luminal::index_expr::{IotaExpr, ParseMemo, parse_int_expr, parse_int_expr_memo};

impl OpSlotNames for Iota {}

impl BufferTensorIrOp for Iota {
    fn label(&self) -> &str {
        "IotaGeneric"
    }
}

impl Bufferizable for Iota {}

impl ToDps for Iota {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IotaDps {
            expr: self.expr.clone(),
        }))
    }
}

impl LayoutIrOp for Iota {}

/// Destination-passing form of [`Iota`], signature spelled slot by slot:
///
/// ```text
/// IotaGeneric(dest0: write-only ↔ out0) -> out0
/// ```
///
/// The destination is the op's ONLY operand — the zero-input source's DPS
/// form is pure write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IotaDps {
    /// See [`Iota::expr`].
    pub expr: Option<IotaExpr>,
}

impl OpSlotNames for IotaDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IotaDps {
    fn label(&self) -> &str {
        "IotaGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
    }
}

impl Bufferizable for IotaDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IotaDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for IotaDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIotaGeneric` enodes and produces [`Iota`]
/// instances. Metadata children: `expr` at child 0, `shape` at child 1,
/// `out_layout` at child 2 — the whole constructor is metadata; there are
/// no tensor operands.
#[derive(Debug, Clone, Copy, Default)]
pub struct IotaMatcher;

impl OpMatcher for IotaMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIotaGeneric"
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
        &[("expr", 0), ("shape", 1), ("out_layout", 2)]
    }

    /// Pure structure — the bounds story lives in egglog (user ruling
    /// 2026-07-23): the iota-int32-certified gate on the op match makes a
    /// missing-bounds iota unimplementable (fail-open), and the
    /// fixpoint-invariants stratum panics on a PROVEN violation. An enode
    /// reaching this matcher is certified by construction.
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Iota {
            expr: parse_int_expr(site, &site.child_class(0), 64, Some(&site.child_class(1))),
        })
    }
}
