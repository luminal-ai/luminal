//! TEST EQUALITY (Option B prototype) — element readback through a
//! returned (buffer, layout) pair, TEST-CRATE-SIDE with no core
//! involvement (Austin's ruling: `walk_layout_index` "probably gets
//! deleted / moved to something called 'test equality' or something in
//! the testing crate"; walked_dense dies — "If you need it dense, ask
//! for it in dense", and a comparison that wants element access uses
//! THIS, bufferizer-independent).
//!
//! ONE reader lives here: [`element_index`] — evaluate the runtime's
//! OWN layout vocabulary (the `luminal::layouts` mirror structs, the
//! exact value `fetch` returns on the binding) at concrete coordinates,
//! down to the flat element index into the BACKING buffer. The legacy
//! hop-chain walker (`walk_hop_index`, ex-core `walk_layout_index`)
//! died with the hop machinery — the corrected contract deleted
//! `ComposedAccess` from the plan entirely.

use anyhow::{anyhow, bail, ensure, Result};
use luminal::layouts::{IntExprTerm, MirrorLayout, ShapeTerm};

/// Evaluate a mirror [`IntExprTerm`] at concrete coordinates
/// (front-indexed; `Coord{axis_from_end}` reads
/// `coords[rank-1-axis_from_end]`). Symbolic vars refuse loudly.
pub fn eval_term(expr: &IntExprTerm, coords: &[usize]) -> Result<i64> {
    let rank = coords.len();
    Ok(match expr {
        IntExprTerm::Lit(v) => *v,
        IntExprTerm::Var(name) => bail!("test equality: symbolic dim `{name}` cannot evaluate"),
        IntExprTerm::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| anyhow!("coordinate axis {axis_from_end} out of rank {rank}"))?;
            coords[rank - 1 - axis] as i64
        }
        IntExprTerm::Add(a, b) => eval_term(a, coords)? + eval_term(b, coords)?,
        IntExprTerm::Mul(a, b) => eval_term(a, coords)? * eval_term(b, coords)?,
        IntExprTerm::TruncDiv(a, b) => {
            let (a, b) = (eval_term(a, coords)?, eval_term(b, coords)?);
            ensure!(b != 0, "division by zero in layout expression");
            // Rust's `/` on i64 IS truncation toward zero.
            a / b
        }
        IntExprTerm::TruncRem(a, b) => {
            let (a, b) = (eval_term(a, coords)?, eval_term(b, coords)?);
            ensure!(b != 0, "remainder by zero in layout expression");
            a % b
        }
        IntExprTerm::CeilDiv(a, b) => {
            let (a, b) = (eval_term(a, coords)?, eval_term(b, coords)?);
            ensure!(
                b > 0,
                "ceil-div by non-positive divisor in layout expression"
            );
            a.div_euclid(b) + if a.rem_euclid(b) != 0 { 1 } else { 0 }
        }
        IntExprTerm::Min(a, b) => eval_term(a, coords)?.min(eval_term(b, coords)?),
        IntExprTerm::Max(a, b) => eval_term(a, coords)?.max(eval_term(b, coords)?),
        IntExprTerm::LessThanCast(a, b) => (eval_term(a, coords)? < eval_term(b, coords)?) as i64,
    })
}

fn literal_extents(shape: &ShapeTerm) -> Result<Vec<usize>> {
    shape
        .0
        .iter()
        .map(|e| match e {
            IntExprTerm::Lit(v) => usize::try_from(*v)
                .ok()
                .filter(|&d| d > 0)
                .ok_or_else(|| anyhow!("non-positive layout extent {v}")),
            other => Err(anyhow!("symbolic layout extent {other:?}")),
        })
        .collect()
}

/// THE OPTION-B READER: element `coords` of a value interpreted under
/// its returned layout, down to the flat ELEMENT index into the backing
/// buffer's storage — evaluating the mirror layout's own expressions.
/// Fail-closed: symbolic extents, foreign-rank coordinates, and
/// out-of-domain coordinates all bail loudly.
pub fn element_index(layout: &MirrorLayout, coords: &[usize]) -> Result<usize> {
    let domain = |shape: &ShapeTerm| -> Result<()> {
        let extents = literal_extents(shape)?;
        ensure!(
            coords.len() == extents.len(),
            "{} coordinates for a rank-{} layout",
            coords.len(),
            extents.len()
        );
        for (axis, (&c, &d)) in coords.iter().zip(&extents).enumerate() {
            ensure!(c < d, "coordinate {c} out of extent {d} (axis {axis})");
        }
        Ok(())
    };
    let flat = match layout {
        MirrorLayout::RightMajor(rm) => {
            domain(&rm.shape)?;
            let extents = literal_extents(&rm.shape)?;
            coords
                .iter()
                .zip(&extents)
                .fold(0usize, |acc, (&c, &d)| acc * d + c) as i64
        }
        MirrorLayout::LeftMajor(lm) => {
            domain(&lm.shape)?;
            let extents = literal_extents(&lm.shape)?;
            let mut stride = 1usize;
            let mut acc = 0usize;
            for (&c, &d) in coords.iter().zip(&extents) {
                acc += c * stride;
                stride *= d;
            }
            acc as i64
        }
        MirrorLayout::Strided(st) => {
            domain(&st.shape)?;
            st.chain
                .iter()
                .map(|s| eval_term(s, coords))
                .sum::<Result<i64>>()?
        }
        MirrorLayout::ElementOffset(eo) => {
            domain(&eo.shape)?;
            eval_term(&eo.offset, coords)?
        }
        MirrorLayout::BitOffset(bo) => {
            domain(&bo.shape)?;
            let bits = eval_term(&bo.offset, coords)?;
            let width = bo.width.0;
            ensure!(width > 0, "non-positive bit width {width}");
            ensure!(
                bits % width == 0,
                "bit offset {bits} is not element-aligned to width {width}"
            );
            bits / width
        }
    };
    usize::try_from(flat).map_err(|_| anyhow!("negative element index {flat}"))
}

/// Read every element of a value (row-major coordinate order over
/// `value_dims`) out of `backing` through `layout` — the "ask for it in
/// dense" comparison helper for f32 buffers. Each index is bounds-checked
/// against the BACKING length (the buffer is the only reach authority —
/// offset-form layouts disclose none).
pub fn dense_f32(backing: &[f32], layout: &MirrorLayout, value_dims: &[usize]) -> Result<Vec<f32>> {
    let numel: usize = value_dims.iter().product();
    let rank = value_dims.len();
    let mut coords = vec![0usize; rank];
    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let flat = element_index(layout, &coords)?;
        ensure!(
            flat < backing.len(),
            "element index {flat} exceeds the backing buffer ({} elements)",
            backing.len()
        );
        out.push(backing[flat]);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < value_dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Ok(out)
}

/// Assert two (buffer, layout) pairs denote the SAME tensor of
/// `value_dims` extents, element by element, within `tolerance`.
pub fn assert_same_f32(
    a: (&[f32], &MirrorLayout),
    b: (&[f32], &MirrorLayout),
    value_dims: &[usize],
    tolerance: f32,
) {
    let left = dense_f32(a.0, a.1, value_dims).expect("left pair reads dense");
    let right = dense_f32(b.0, b.1, value_dims).expect("right pair reads dense");
    for (n, (l, r)) in left.iter().zip(&right).enumerate() {
        assert!(
            (l - r).abs() <= tolerance,
            "element {n} differs: {l} vs {r} (tolerance {tolerance})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::layouts::{
        BitOffsetExpressionLayout, BitWidthTerm, ElementOffsetExpressionLayout, MirrorLayout,
        RightMajorContiguousElementLayout, ShapeTerm, StridedElementLayout,
    };

    fn lit(v: i64) -> IntExprTerm {
        IntExprTerm::Lit(v)
    }
    fn coord(axis_from_end: i64) -> IntExprTerm {
        IntExprTerm::Coord { axis_from_end }
    }
    fn mul(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Mul(Box::new(a), Box::new(b))
    }
    fn add(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Add(Box::new(a), Box::new(b))
    }
    fn shape(dims: &[i64]) -> ShapeTerm {
        ShapeTerm(dims.iter().map(|&d| lit(d)).collect())
    }

    /// THE REPRESENTATIVE RESTRUCTURED COMPARISON (Option B): the
    /// device-suite `walked_dense` pattern — fetch bytes + layout,
    /// interpret, compare against a dense demand — rebuilt CPU-side on
    /// (buffer, layout) pairs. A transpose view's strided layout over
    /// the parent's bytes must equal the dense transpose of the same
    /// data under the row-major layout.
    #[test]
    fn transpose_view_pair_equals_the_dense_demand() {
        // Parent x: [2,3] row-major bytes.
        let x: Vec<f32> = (0..6).map(|n| n as f32 * 10.0).collect();
        // The view: [3,2], element (i,j) = x[j,i] at parent flat j*3+i.
        let view_layout = MirrorLayout::Strided(StridedElementLayout {
            shape: shape(&[3, 2]),
            chain: vec![mul(coord(0), lit(3)), coord(1)],
            width: BitWidthTerm(32),
        });
        // The dense demand: the same tensor materialized row-major.
        let dense: Vec<f32> = vec![x[0], x[3], x[1], x[4], x[2], x[5]];
        let dense_layout = MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: shape(&[3, 2]),
            width: BitWidthTerm(32),
        });
        assert_same_f32((&x, &view_layout), (&dense, &dense_layout), &[3, 2], 0.0);
    }

    /// The offset-expression forms read identically to their spelled
    /// function — including the bit form's width division — and a
    /// mid-element bit offset refuses.
    #[test]
    fn offset_forms_read_and_bit_alignment_refuses() {
        let eo = MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
            offset: add(mul(coord(1), lit(3)), coord(0)),
            shape: shape(&[2, 3]),
            width: BitWidthTerm(32),
        });
        assert_eq!(element_index(&eo, &[1, 2]).unwrap(), 5);
        let bo = MirrorLayout::BitOffset(BitOffsetExpressionLayout {
            offset: mul(add(mul(coord(1), lit(3)), coord(0)), lit(32)),
            shape: shape(&[2, 3]),
            width: BitWidthTerm(32),
        });
        assert_eq!(element_index(&bo, &[1, 2]).unwrap(), 5);
        let misaligned = MirrorLayout::BitOffset(BitOffsetExpressionLayout {
            offset: add(mul(coord(0), lit(32)), lit(8)),
            shape: shape(&[4]),
            width: BitWidthTerm(32),
        });
        assert!(element_index(&misaligned, &[1]).is_err());
    }

    /// Out-of-domain coordinates and symbolic extents refuse loudly —
    /// but retains the honesty limit this reader inherits from Option B:
    /// the DOMAIN is checked (coords vs the layout's shape); the RANGE
    /// is not a layout fact for offset forms, so `dense_f32`'s
    /// backing-length check is the only reach fence.
    #[test]
    fn domain_violations_refuse() {
        let rm = MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: shape(&[2, 3]),
            width: BitWidthTerm(32),
        });
        assert!(element_index(&rm, &[2, 0]).is_err());
        assert!(element_index(&rm, &[0]).is_err());
        let symbolic = MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Var("n".into()), lit(3)]),
            width: BitWidthTerm(32),
        });
        assert!(element_index(&symbolic, &[0, 0]).is_err());
        // An offset form pointing past the data: element_index itself
        // cannot know — dense_f32's backing check catches it.
        let escaping = MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
            offset: add(coord(0), lit(100)),
            shape: shape(&[2]),
            width: BitWidthTerm(32),
        });
        assert_eq!(element_index(&escaping, &[1]).unwrap(), 101);
        let backing = vec![0.0f32; 4];
        assert!(dense_f32(&backing, &escaping, &[2]).is_err());
    }
}
