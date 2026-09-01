//! The index-expression generator: a ZERO-INPUT source op that writes
//! `expr(c0..ck)` at every output coordinate — the tensor form of a single
//! `IntExpr` over the output shape's coordinate variables.
//!
//! Iota is the first op with no tensor operands, which makes it the proof
//! case for the zero-input path through extraction (empty operand list),
//! DPS rewriting (the appended destination is the ONLY operand), and
//! bufferization (nothing to alias, one fresh write). There is no mutating
//! form — with no input there is nothing to mutate.

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
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

pub use crate::index_expr::{IotaExpr, ParseMemo, parse_int_expr, parse_int_expr_memo};

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

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::reference::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};
use crate::reference::kernels::expect_op;

/// Iota coordinate generator. Typed dests 2026-08-11: the exact i64
/// evaluation lands in NATIVE integer storage (the old `as f32` store
/// was the producer side of the Int-in-f32 smuggling and its 2^24
/// exactness cliff).
pub(in crate::reference) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<IotaDps>(op)?;
    let Some(expr) = &op.expr else {
        anyhow::bail!("iota reference kernel supports Lit/Coord/Add/Mul expressions only");
    };
    let out_dims = ctx.operand_dims.last().cloned().unwrap_or_default();
    let rank = out_dims.len();
    let mut coords = vec![0usize; rank];
    let eval_at = |flat: usize, coords: &mut Vec<usize>| {
        let mut remainder = flat;
        for axis in (0..rank).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        expr.eval(coords)
    };
    match &mut ctx.dests[0] {
        // Iota is Int by its dtype rule; the i64 evaluation lands
        // checked in i32 (loud on overflow — non-wrapping ruling).
        TypedBuffer::I32(dest) => {
            for (flat, output) in dest.iter_mut().enumerate() {
                let value = eval_at(flat, &mut coords);
                *output = i32::try_from(value).map_err(|_| {
                    anyhow::anyhow!("iota value {value} overflows i32 (ints are non-wrapping)")
                })?;
            }
        }
        TypedBuffer::I64(dest) => {
            for (flat, output) in dest.iter_mut().enumerate() {
                *output = eval_at(flat, &mut coords);
            }
        }
        other => anyhow::bail!(
            "iota has no {} arm (iota output is Int by its dtype rule)",
            other.type_name()
        ),
    }
    Ok(())
}
