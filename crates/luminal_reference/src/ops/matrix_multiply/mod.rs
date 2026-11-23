//! Dense matrix product of two rank-2 operands.
//!
//! Matches the logical `LogicalHelperMatrixMultiply` node, so the broadcast
//! product `[m, n, k]` of the decomposed chain is never materialized. Each
//! output element starts at zero and adds `a * b` in increasing `k`, which is
//! the same order and rounding as the decomposed `Mul` followed by `ReduceSum`.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// `MatrixMultiplyFunctionalGeneric(lhs, rhs) -> out`, with `lhs: [m, k]`,
/// `rhs: [k, n]` and `out: [m, n]`.
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixMultiplyFunctional;

impl OpSlotNames for MatrixMultiplyFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatrixMultiplyFunctional {
    fn label(&self) -> &str {
        "MatrixMultiplyGeneric"
    }
}

impl Bufferizable for MatrixMultiplyFunctional {}

impl ToDps for MatrixMultiplyFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MatrixMultiplyFunctionalDps))
    }
}

impl LayoutIrOp for MatrixMultiplyFunctional {}

/// Destination-passing form of [`MatrixMultiplyFunctional`], signature spelled
/// slot by slot:
///
/// ```text
/// MatrixMultiplyGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixMultiplyFunctionalDps;

impl OpSlotNames for MatrixMultiplyFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatrixMultiplyFunctionalDps {
    fn label(&self) -> &str {
        "MatrixMultiplyGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
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

impl Bufferizable for MatrixMultiplyFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MatrixMultiplyFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MatrixMultiplyFunctionalDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpMatrixMultiplyGeneric` enodes and produces
/// [`MatrixMultiplyFunctional`] instances. Metadata children: `out_layout` at
/// child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixMultiplyFunctionalMatcher;

impl OpMatcher for MatrixMultiplyFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpMatrixMultiplyGeneric"
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
        &[("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MatrixMultiplyFunctional)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

/// Only F32 has an arm. The matcher mints this op for F32 operands alone, so
/// any other dtype keeps the decomposed chain and never reaches this kernel.
pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    match &ctx.operands[0] {
        TypedBuffer::F32(_) => matrix_multiply_f32(ctx),
        other => anyhow::bail!("matrix multiply has no {} arm", other.type_name()),
    }
}

fn matrix_multiply_f32(ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let (lhs_dims, rhs_dims) = (&ctx.operand_dims[0], &ctx.operand_dims[1]);
    anyhow::ensure!(
        lhs_dims.len() == 2 && rhs_dims.len() == 2 && lhs_dims[1] == rhs_dims[0],
        "matrix multiply needs [m, k] x [k, n] operands, got {lhs_dims:?} x {rhs_dims:?}"
    );
    let (m, k, n) = (lhs_dims[0], lhs_dims[1], rhs_dims[1]);
    let lhs = ctx.operands[0].as_f32()?;
    let rhs = ctx.operands[1].as_f32()?;
    let out = ctx.dests[0].as_f32_mut()?;
    anyhow::ensure!(
        lhs.len() == m * k && rhs.len() == k * n && out.len() == m * n,
        "matrix multiply buffers disagree with dims [{m}, {k}] x [{k}, {n}]: \
         lhs {}, rhs {}, out {}",
        lhs.len(),
        rhs.len(),
        out.len()
    );
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    Ok(())
}
