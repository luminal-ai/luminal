//! The CUDA-lite runtime's PLAN-LAYOUT vocabulary and its read-back
//! helpers.
//!
//! There is no CUDA-flavored layout type and no CUDA-flavored decoder
//! any more (ruling D9, 2026-09-03): core owns the decoder and publishes
//! the struct it produces ([`luminal::layouts::DecodedLayout`] — the
//! elected class's decoded spellings plus the value's `dtype-of` fact),
//! and the runtimes import it directly. A backend that wants a layout
//! core has never heard of is still free to bring its own type; this one
//! does not.
//!
//! The read-back helpers below evaluate through
//! [`luminal::layouts::LayoutFacts::element_index`] — the constructor's
//! own read function, asked of the layout rather than matched on.

use anyhow::Result;
pub use luminal::layouts::DecodedLayout;

/// The CUDA-lite bufferized plan.
pub type CudaPlan = luminal::bufferize::BufferIrGraph<DecodedLayout>;

// ===========================================================================
// READING BACK through a returned layout — this runtime evaluating its OWN
// vocabulary. Not planner machinery, and not a core capability: core
// transports `L` opaquely and never reads through it. The canonical
// cross-runtime version of this lives in the testing crate
// (`test_runtime::test_equality`); CL cannot depend on it (that crate
// depends on CL), so the CL device suites and examples share this one.
// ===========================================================================

/// Evaluate a layout term at concrete coordinates (front-indexed;
/// `Coord{axis_from_end}` reads `coords[rank-1-axis_from_end]`).
/// Symbolic vars refuse loudly. Core owns the evaluator
/// ([`luminal::layouts::IntExprTerm::eval_at`]); this keeps the name the
/// CL device suites, `view_admission` and the examples already call.
pub fn eval_term(expr: &luminal::layouts::IntExprTerm, coords: &[usize]) -> Result<i64> {
    expr.eval_at(coords)
}

/// Element `coords` of a value interpreted under `layout`, down to the
/// flat ELEMENT index into the backing buffer. Fail-closed on symbolic
/// extents, foreign-rank coordinates, out-of-domain coordinates, a
/// mid-element bit offset, and a negative result.
pub fn element_index(layout: &DecodedLayout, coords: &[usize]) -> Result<usize> {
    layout.element_index(coords)
}

/// Read every element of a value out of `backing` through `layout`, in
/// row-major coordinate order over the layout's own domain — the "ask for
/// it in dense" comparison helper. THE BACKING LENGTH IS THE ONLY REACH
/// AUTHORITY (offset-form layouts disclose none), so every index is
/// bounds-checked against it.
pub fn dense_f32(backing: &[f32], layout: &DecodedLayout) -> Result<Vec<f32>> {
    let dims = layout
        .literal_extents()
        .ok_or_else(|| anyhow::anyhow!("dense read: symbolic extents"))?;
    let numel: usize = dims.iter().product();
    let rank = dims.len();
    let mut coords = vec![0usize; rank];
    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let flat = element_index(layout, &coords)?;
        anyhow::ensure!(
            flat < backing.len(),
            "element index {flat} exceeds the backing buffer ({} elements)",
            backing.len()
        );
        out.push(backing[flat]);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Ok(out)
}
