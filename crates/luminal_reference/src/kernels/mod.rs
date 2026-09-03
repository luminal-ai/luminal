//! The REFERENCE RUNTIME's kernel registry. Kernel BODIES live in their
//! op's folder as `crate::ops::<op>::kernel` (op-folder ruling
//! 2026-08-13: everything about an op lives in the op's folder); this file
//! keeps what is not any single op's business — the label→fn dispatch
//! TABLE, the helpers shared by several op kernels, and the kernels for
//! bufferizer-synthesized plan infrastructure (which has no op folder).
//! The table is DERIVED from `crate::ops::reference_ops()`, where each
//! op's matcher and kernel are one row: the runtime cannot over-claim
//! (a registered op with no kernel) or under-claim (a kernel the search
//! is not offered) because neither half can be written without the other.
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

use luminal::buffer_tensor_ir::{BufferTensorIrOp, ReferenceKernelCtx, TypedBuffer};
use std::any::TypeId;

#[derive(Clone, Copy)]
pub struct ReferenceKernel {
    /// The op's IR label (DPS forms keep the functional form's label).
    pub label: &'static str,
    /// The concrete op type this kernel downcasts to.
    pub op_type: TypeId,
    pub execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
}

pub(crate) fn entry<T: 'static>(
    label: &'static str,
    execute: fn(&dyn BufferTensorIrOp, &mut ReferenceKernelCtx) -> anyhow::Result<()>,
) -> ReferenceKernel {
    ReferenceKernel {
        label,
        op_type: TypeId::of::<T>(),
        execute,
    }
}

/// THE DISPATCH TABLE, derived from [`crate::ops::reference_ops`] plus
/// the bufferizer-minted plan infrastructure.
///
/// Nothing is enumerated here that is not already an op row: the op
/// registration list carries each op's kernel alongside its matcher, so
/// "every registered op is executable" and "every executable op is
/// registered" hold BY CONSTRUCTION rather than by a test that checks two
/// hand-maintained lists agree. The only rows this adds are BufferAlloc
/// and BufferFree, which have no matcher because they are not ops — the
/// storage-lifetime pass mints them.
pub fn reference_kernels() -> &'static [ReferenceKernel] {
    static KERNELS: std::sync::OnceLock<Vec<ReferenceKernel>> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        let mut table: Vec<ReferenceKernel> = crate::ops::reference_ops()
            .iter()
            .map(|op| op.kernel)
            .collect();
        table.push(entry::<luminal::buffer_tensor_ir::BufferAlloc>(
            "BufferAlloc",
            buffer_alloc,
        ));
        table.push(entry::<luminal::buffer_tensor_ir::BufferFree>(
            "BufferFree",
            buffer_free,
        ));
        table
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
pub(crate) fn expect_op<T: 'static>(op: &dyn BufferTensorIrOp) -> anyhow::Result<&T> {
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
pub(crate) fn coordinate_columns(operands: &[TypedBuffer]) -> anyhow::Result<Vec<Vec<i64>>> {
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
pub(crate) fn move_gathered(
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
        (TypedBuffer::F64(data), TypedBuffer::F64(dest)) => {
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
