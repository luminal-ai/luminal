//! Matrix product without materializing the broadcast product.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `MatrixMultiplyGeneric(lhs, rhs) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (both operands read, the result freshly allocated).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixMultiply;

impl OpSlotNames for MatrixMultiply {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatrixMultiply {
    fn label(&self) -> &str {
        "MatrixMultiplyGeneric"
    }
}

impl Bufferizable for MatrixMultiply {}

impl ToDps for MatrixMultiply {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MatrixMultiplyDps))
    }
}

impl LayoutIrOp for MatrixMultiply {}

/// Destination-passing form of [`MatrixMultiply`], signature spelled slot by slot:
///
/// ```text
/// MatrixMultiplyGeneric(lhs: read, rhs: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixMultiplyDps;

impl OpSlotNames for MatrixMultiplyDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "lhs".to_string(),
            1 => "rhs".to_string(),
            2 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for MatrixMultiplyDps {
    fn label(&self) -> &str {
        "MatrixMultiplyGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 | 1 => true, // lhs and rhs
            2 => false,    // dest0: write-only destination
            _ => true,     // outside the signature: conservative default
        }
    }
}

impl Bufferizable for MatrixMultiplyDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 2,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MatrixMultiplyDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for MatrixMultiplyDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpMatrixMultiplyGeneric` enodes and produces
/// [`MatrixMultiply`] instances. Metadata children: `layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixMultiplyMatcher;

impl OpMatcher for MatrixMultiplyMatcher {
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
        &[("layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MatrixMultiply)
    }
}

use crate::kernels::expect_op;
use crate::typed_buffer::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    expect_op::<MatrixMultiplyDps>(op)?;
    let a = &ctx.operand_dims[0];
    let b = &ctx.operand_dims[1];
    anyhow::ensure!(
        a.len() == 2 && b.len() == 2 && a[1] == b[0],
        "matrix product geometry mismatch"
    );
    let (m, k, n) = (a[0], a[1], b[1]);
    let lhs_len = m
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("matrix lhs size overflow"))?;
    let rhs_len = k
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("matrix rhs size overflow"))?;
    let out_len = m
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("matrix output size overflow"))?;
    let k_stride = isize::try_from(k)?;
    let n_stride = isize::try_from(n)?;
    macro_rules! product {
        ($lhs:expr, $rhs:expr, $out:expr, $gemm:path) => {{
            let (lhs, rhs, out) = ($lhs, $rhs, $out);
            anyhow::ensure!(
                lhs.len() == lhs_len && rhs.len() == rhs_len && out.len() == out_len,
                "matrix product buffer size mismatch"
            );
            // The runtime gathered each operand into dense row-major storage.
            // The checked dimensions above cover every address GEMM touches;
            // the fresh output is disjoint from both immutable inputs.
            unsafe {
                $gemm(
                    m,
                    k,
                    n,
                    1.0,
                    lhs.as_ptr(),
                    k_stride,
                    1,
                    rhs.as_ptr(),
                    n_stride,
                    1,
                    0.0,
                    out.as_mut_ptr(),
                    n_stride,
                    1,
                );
            }
        }};
    }
    match (&ctx.operands[0], &ctx.operands[1], &mut ctx.dests[0]) {
        (TypedBuffer::F32(a), TypedBuffer::F32(b), TypedBuffer::F32(out)) => {
            product!(a, b, out, matrixmultiply::sgemm)
        }
        (TypedBuffer::F64(a), TypedBuffer::F64(b), TypedBuffer::F64(out)) => {
            product!(a, b, out, matrixmultiply::dgemm)
        }
        (TypedBuffer::F16(a), TypedBuffer::F16(b), TypedBuffer::F32(out)) => {
            let a: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
            let b: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
            product!(&a, &b, out, matrixmultiply::sgemm);
        }
        (TypedBuffer::Bf16(a), TypedBuffer::Bf16(b), TypedBuffer::F32(out)) => {
            let a: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
            let b: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
            product!(&a, &b, out, matrixmultiply::sgemm);
        }
        _ => anyhow::bail!("matrix product requires matching floating-point storage"),
    }
    Ok(())
}
