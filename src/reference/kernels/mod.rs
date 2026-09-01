//! The REFERENCE RUNTIME's kernel registry. Kernel BODIES live in their
//! op's folder as `crate::reference::ops::<op>::kernel` (op-folder ruling
//! 2026-08-13: everything about an op lives in the op's folder); this file
//! keeps what is not any single op's business — the label→fn dispatch
//! TABLE, the helpers shared by several op kernels, and the kernels for
//! bufferizer-synthesized plan infrastructure (which has no op folder).
//! The table is still the single implementation inventory: because
//! [`super::reference_allow_list`] is DERIVED from this table, the runtime
//! can neither over-claim (an allow-listed op with no kernel) nor
//! under-claim (a kernel the search is not offered).
//!
//! BUNDLE-LEVEL CLAIMS, TYPE-LEVEL DISPATCH (ruling 2026-08-06): the
//! runtime's CLAIM is per op FAMILY — one matcher constructor covers
//! every rank, layout, and instance its matcher can match; nothing is
//! ever enumerated per variant. One registry row = one executable FORM
//! of a family (today always the DPS form), and instance flexibility
//! (rank, axis, expression, map entries) flows into the kernel through
//! the downcast. The row key is the concrete op TYPE (TypeId), not the
//! label, because labels are shared: DPS forms keep their functional
//! form's IR name, and only DPS forms are executable (plans are
//! post-`dps_rewrite`) — TypeId dispatch turns that from an assumption
//! into a checked invariant. Any op type absent from the table refuses
//! loudly at execution ("no reference kernel for ..."). A future
//! layout-specialized kernel changes nothing here: the family still
//! claims once; its kernel branches internally on instance data.

use crate::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};
use std::any::TypeId;

pub struct ReferenceKernel {
    /// The op's IR label (DPS forms keep the functional form's label);
    /// the allow-list derivation matches this against matcher constructors.
    pub label: &'static str,
    /// The concrete op type this kernel downcasts to.
    pub op_type: TypeId,
    pub execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
}

fn entry<T: 'static>(
    label: &'static str,
    execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
) -> ReferenceKernel {
    ReferenceKernel {
        label,
        op_type: TypeId::of::<T>(),
        execute,
    }
}

/// The full kernel table. One entry per executable op type; labels repeat
/// only if two executable types legitimately share one (none do today —
/// functional forms are not executable and have no entries).
pub fn reference_kernels() -> &'static [ReferenceKernel] {
    use crate::reference::ops::*;
    static KERNELS: std::sync::OnceLock<Vec<ReferenceKernel>> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        vec![
            // ── elementwise ──
            entry::<AddFunctionalDps>("AddFunctionalGeneric", add::kernel),
            entry::<MulFunctionalDps>("MulFunctionalGeneric", mul::kernel),
            entry::<DivFunctionalDps>("DivFunctionalGeneric", div::kernel),
            entry::<TruncDivFunctionalDps>("TruncDivFunctionalGeneric", trunc_div::kernel),
            entry::<TruncRemFunctionalDps>("TruncRemFunctionalGeneric", trunc_rem::kernel),
            entry::<ModFunctionalDps>("ModFunctionalGeneric", modulo::kernel),
            entry::<SqrtFunctionalDps>("SqrtFunctionalGeneric", sqrt::kernel),
            entry::<ExpFunctionalDps>("ExpFunctionalGeneric", exp::kernel),
            entry::<Exp2FunctionalDps>("Exp2FunctionalGeneric", exp2::kernel),
            entry::<Log2FunctionalDps>("Log2FunctionalGeneric", log2::kernel),
            entry::<SinFunctionalDps>("SinFunctionalGeneric", sin::kernel),
            entry::<RecipFunctionalDps>("RecipFunctionalGeneric", recip::kernel),
            entry::<LessThanDps>("LessThanGeneric", less_than::kernel),
            entry::<CastDps>("CastGeneric", cast::kernel),
            // ── sources ──
            entry::<ConstantDps>("ConstantGeneric", constant::kernel),
            entry::<IotaDps>("IotaGeneric", iota::kernel),
            // ── reductions ──
            entry::<ReduceSumDps>("ReduceSumGeneric", reduce_sum::kernel),
            entry::<ReduceMaxDps>("ReduceMaxGeneric", reduce_max::kernel),
            // ── data movement ──
            entry::<GatherDps>("GatherGeneric", gather::kernel),
            entry::<ScatterFunctionalDps>("ScatterFunctionalGeneric", scatter::kernel),
            entry::<IndexMapApplyMaterializeDps>(
                "IndexMapApplyMaterialize",
                index_map_apply_materialize::kernel,
            ),
            entry::<MaterializeLayoutCopyDps>("CopyGeneric", materialize_layout_copy::kernel),
            // ── plan infrastructure (bufferizer-synthesized, no matchers) ──
            entry::<crate::buffer_tensor_ir::BufferAlloc>("BufferAlloc", buffer_alloc),
            entry::<crate::buffer_tensor_ir::BufferFree>("BufferFree", buffer_free),
        ]
    })
}

/// Look up the kernel for a plan op by its concrete type.
pub fn kernel_for(op: &dyn BufferTensorIrOp) -> Option<&'static ReferenceKernel> {
    let op_type = op.as_any().type_id();
    reference_kernels()
        .iter()
        .find(|kernel| kernel.op_type == op_type)
}

/// Downcast the dispatched op to the kernel's concrete type — a mismatch
/// means the registry row and the kernel disagree, which refuses loudly.
pub(in crate::reference) fn expect_op<T: 'static>(op: &dyn BufferTensorIrOp) -> anyhow::Result<&T> {
    op.as_any().downcast_ref::<T>().ok_or_else(|| {
        anyhow::anyhow!(
            "kernel dispatched to a different op type than its registry row declares (registry drift)"
        )
    })
}

// ---------------------------------------------------------------------------
// Shared kernel helpers — used by SEVERAL op modules' kernels (gather,
// scatter, index-map materialize, layout copy), so they live here rather
// than in any one op's folder.
// ---------------------------------------------------------------------------

/// Read a rank of coordinate operands as native i32, promoted to i64
/// for index arithmetic (gather + scatter). TYPED (2026-08-11):
/// coordinates are read from NATIVE i32 buffers — the old `as_f32` read
/// plus `as i64` truncation was the consumer side of the Int-in-f32
/// smuggling (a 4.9999995 truncating to 4 while passing the bounds
/// check).
pub(in crate::reference) fn coordinate_columns(
    operands: &[TypedBuffer],
) -> anyhow::Result<Vec<Vec<i64>>> {
    operands
        .iter()
        .map(|operand| {
            Ok(operand
                .as_i32()?
                .iter()
                .map(|value| i64::from(*value))
                .collect())
        })
        .collect()
}

/// dest[flat] = source[index[flat]] for every flat, whatever the payload
/// dtype — variants must match (the plan annotated both sides). Shared by
/// gather, index-map materialize, and the dense layout copy: the index
/// computation differs per op, the element moves do not.
pub(in crate::reference) fn move_gathered(
    source: &TypedBuffer,
    dest: &mut TypedBuffer,
    index_of: &[usize],
) -> anyhow::Result<()> {
    match (source, dest) {
        (TypedBuffer::F32(data), TypedBuffer::F32(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::I32(data), TypedBuffer::I32(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::I64(data), TypedBuffer::I64(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::Bool8(data), TypedBuffer::Bool8(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (TypedBuffer::F8E4M3(data), TypedBuffer::F8E4M3(dest)) => {
            for (flat, index) in index_of.iter().enumerate() {
                dest[flat] = data[*index];
            }
        }
        (source, dest) => anyhow::bail!(
            "payload move between {} and {} buffers",
            source.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Plan-infrastructure kernels — for bufferizer-synthesized ops (no
// matchers, never in the allow list, and no folder under `reference::ops`:
// BufferAlloc/BufferFree are not LayoutTensor ops). They exist so the
// executor's dispatch stays uniform.
// ---------------------------------------------------------------------------

/// No computation: storage is pre-materialized; contents start undefined
/// (zeros here).
fn buffer_alloc(_op: &dyn BufferTensorIrOp, _ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    Ok(())
}

/// No computation: liveness bookkeeping only; the reference executor
/// keeps storage.
fn buffer_free(_op: &dyn BufferTensorIrOp, _ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    Ok(())
}
