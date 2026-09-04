//! The CUDA-lite runtime's PLAN-LAYOUT vocabulary and its read-back
//! helpers.
//!
//! There is no CUDA-flavored layout type and no CUDA-flavored decoder
//! any more (ruling D9, 2026-09-03): core owns the decoder and publishes
//! the struct it produces ([`luminal::layouts::DecodedLayout`] — the
//! shared mirror vocabulary plus the value's `dtype-of` fact), and the
//! runtimes import it directly. A backend that wants a layout core has
//! never heard of is still free to bring its own type; this one does
//! not.

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

/// Evaluate a mirror term at concrete coordinates (front-indexed;
/// `Coord{axis_from_end}` reads `coords[rank-1-axis_from_end]`).
/// Symbolic vars and unsupported forms refuse loudly.
pub fn eval_term(expr: &luminal::layouts::IntExprTerm, coords: &[usize]) -> Result<i64> {
    use luminal::layouts::IntExprTerm as T;
    let rank = coords.len();
    Ok(match expr {
        T::Lit(v) => *v,
        T::Var(name) => anyhow::bail!("layout read: symbolic dim `{name}` cannot evaluate"),
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| {
                    anyhow::anyhow!("coordinate axis {axis_from_end} out of rank {rank}")
                })?;
            coords[rank - 1 - axis] as i64
        }
        T::Add(a, b) => eval_term(a, coords)? + eval_term(b, coords)?,
        T::Mul(a, b) => eval_term(a, coords)? * eval_term(b, coords)?,
        T::TruncDiv(a, b) => {
            let (a, b) = (eval_term(a, coords)?, eval_term(b, coords)?);
            anyhow::ensure!(b != 0, "division by zero in a layout expression");
            a / b
        }
        T::TruncRem(a, b) => {
            let (a, b) = (eval_term(a, coords)?, eval_term(b, coords)?);
            anyhow::ensure!(b != 0, "remainder by zero in a layout expression");
            a % b
        }
        T::Min(a, b) => eval_term(a, coords)?.min(eval_term(b, coords)?),
        T::Max(a, b) => eval_term(a, coords)?.max(eval_term(b, coords)?),
        T::LessThanCast(a, b) => i64::from(eval_term(a, coords)? < eval_term(b, coords)?),
        other => anyhow::bail!("layout read: {other:?} has no evaluation here (fail-closed)"),
    })
}

/// Element `coords` of a value interpreted under `layout`, down to the
/// flat ELEMENT index into the backing buffer. Fail-closed on symbolic
/// extents, foreign-rank coordinates, out-of-domain coordinates, a
/// mid-element bit offset, and a negative result.
pub fn element_index(layout: &DecodedLayout, coords: &[usize]) -> Result<usize> {
    use luminal::layouts::MirrorLayout as M;
    let extents = layout
        .mirror
        .literal_extents()
        .ok_or_else(|| anyhow::anyhow!("layout read: symbolic extents"))?;
    anyhow::ensure!(
        coords.len() == extents.len(),
        "{} coordinates for a rank-{} layout",
        coords.len(),
        extents.len()
    );
    for (axis, (&c, &d)) in coords.iter().zip(&extents).enumerate() {
        anyhow::ensure!(c < d, "coordinate {c} out of extent {d} (axis {axis})");
    }
    let flat = match &layout.mirror {
        M::RightMajor(_) => coords
            .iter()
            .zip(&extents)
            .fold(0usize, |acc, (&c, &d)| acc * d + c) as i64,
        M::LeftMajor(_) => {
            let mut stride = 1usize;
            let mut acc = 0usize;
            for (&c, &d) in coords.iter().zip(&extents) {
                acc += c * stride;
                stride *= d;
            }
            acc as i64
        }
        M::Strided(st) => {
            let mut total = 0i64;
            for summand in &st.chain {
                total += eval_term(summand, coords)?;
            }
            total
        }
        M::ElementOffset(eo) => eval_term(&eo.offset, coords)?,
        M::BitOffset(bo) => {
            let bits = eval_term(&bo.offset, coords)?;
            anyhow::ensure!(bo.width.0 > 0, "non-positive bit width {}", bo.width.0);
            anyhow::ensure!(
                bits % bo.width.0 == 0,
                "bit offset {bits} is not element-aligned to width {}",
                bo.width.0
            );
            bits / bo.width.0
        }
    };
    usize::try_from(flat).map_err(|_| anyhow::anyhow!("negative element index {flat}"))
}

/// Read every element of a value out of `backing` through `layout`, in
/// row-major coordinate order over the layout's own domain — the "ask for
/// it in dense" comparison helper. THE BACKING LENGTH IS THE ONLY REACH
/// AUTHORITY (offset-form layouts disclose none), so every index is
/// bounds-checked against it.
pub fn dense_f32(backing: &[f32], layout: &DecodedLayout) -> Result<Vec<f32>> {
    let dims = layout
        .mirror
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
