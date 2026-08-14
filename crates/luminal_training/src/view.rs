//! Mapping gradients backward through `ShapeTracker` views.
//!
//! In luminal's HLIR, movement (permute/expand/reshape/offset-free slice) is
//! not an op — each consuming op stores a `ShapeTracker` describing how it
//! reads its input's physical buffer. The gradient of "reading through a view"
//! is the adjoint of that view: broadcast dims sum out, permutes invert,
//! reshapes reinterpret, and anything else scatter-adds through the view's
//! index expression.

use itertools::Itertools;
use luminal::prelude::*;

/// How a (fake-dim-free) view relates to the producer's contiguous output.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ViewClass {
    /// The view is exactly the producer's contiguous layout.
    Identity,
    /// Contiguous view over the same buffer with different dims (merge/split/flatten).
    Reshape,
    /// Pure permutation of the producer's axes. `axes[j]` is the view axis
    /// holding producer axis `j`, i.e. `g.permute(axes)` lands in producer space.
    Permute(Vec<usize>),
    /// Anything else (offset-free slices, overlapping/strided reads): needs a
    /// scatter-add through the index expression.
    General,
}

fn expr_eq(a: Expression, b: Expression) -> bool {
    if let (Some(x), Some(y)) = (a.as_num(), b.as_num()) {
        return x == y;
    }
    a.simplify() == b.simplify()
}

/// Contiguity check by simplified-expression comparison. `ShapeTracker::
/// is_contiguous` compares strides structurally, so semantically-equal
/// expressions built along different paths (e.g. `split_dims`' simplified
/// `z*20` vs row-major construction's nested `(z*2)*10`) fail it and would
/// spuriously demote an identity view to the O(N·M) scatter-add fallback.
pub(crate) fn is_contiguous_view(view: &ShapeTracker) -> bool {
    let expected = ShapeTracker::new(&view.dims.to_vec()[..]);
    view.strides
        .iter()
        .zip(expected.strides.iter())
        .all(|(a, b)| expr_eq(*a, *b))
}

/// Classify how `view` (with no stride-0 dims) reads the buffer a producer
/// wrote contiguously with `producer_dims`.
pub(crate) fn classify_view(view: &ShapeTracker, producer_dims: &[Expression]) -> ViewClass {
    let prod = ShapeTracker::new(producer_dims);
    if view.len() == prod.len()
        && (0..view.len()).all(|i| {
            expr_eq(view.dims[i], prod.dims[i]) && expr_eq(view.strides[i], prod.strides[i])
        })
    {
        return ViewClass::Identity;
    }
    if is_contiguous_view(view) && expr_eq(view.n_elements(), prod.n_elements()) {
        return ViewClass::Reshape;
    }
    if view.len() == prod.len() {
        // For each producer axis j, find an unused view axis with the same
        // (dim, stride) pair. Strides are structurally shared expressions
        // (views start from the producer's contiguous tracker and get
        // permuted), so equality is exact for pure permutations.
        let mut used = vec![false; view.len()];
        let mut axes = Vec::with_capacity(view.len());
        'outer: for j in 0..prod.len() {
            for (i, u) in used.iter_mut().enumerate() {
                if !*u
                    && expr_eq(view.dims[i], prod.dims[j])
                    && expr_eq(view.strides[i], prod.strides[j])
                {
                    *u = true;
                    axes.push(i);
                    continue 'outer;
                }
            }
            return ViewClass::General;
        }
        return ViewClass::Permute(axes);
    }
    ViewClass::General
}

/// Copy a possibly-strided view into a fresh contiguous buffer in logical
/// order (the same gather-an-identity-iota trick `GraphTensor::output` uses).
pub(crate) fn materialize(g: GraphTensor) -> GraphTensor {
    if is_contiguous_view(&g.shape) {
        return g;
    }
    let dims = g.dims();
    let total = dims.iter().copied().product::<Expression>();
    let idx = g.graph().iota('z', total);
    let mut m = g.gather(idx);
    m.shape = ShapeTracker::new(&dims[..]).with_element_bits(g.dtype.bits());
    m
}

/// View `g` as `dims` (same element count). Free when `g` is already
/// contiguous; otherwise materializes first.
pub(crate) fn reinterpret(g: GraphTensor, dims: &[Expression]) -> GraphTensor {
    let mut m = materialize(g);
    m.shape = ShapeTracker::new(dims).with_element_bits(m.dtype.bits());
    m
}

/// Map a gradient `g`, laid out in the logical space of `view` (a consumer's
/// input `ShapeTracker`), back into the producer's contiguous output space
/// (`producer_dims`). This is the adjoint of "read the producer through
/// `view`": broadcast (stride-0) dims sum out, permutations invert as free
/// views, reshapes reinterpret, and everything else scatter-adds via the
/// view's index expression.
pub fn unview(g: GraphTensor, view: ShapeTracker, producer_dims: &[Expression]) -> GraphTensor {
    assert_eq!(
        g.dims(),
        view.dims.to_vec(),
        "unview: gradient dims must match the view's logical dims"
    );
    // 1) Sum out broadcast dims. Reading a broadcast value N times means the
    // upstream gradient is the sum of the N downstream gradients.
    let fake_axes = (0..view.len())
        .filter(|i| view.strides[*i] == Expression::from(0))
        .collect_vec();
    let (g, view) = if fake_axes.is_empty() {
        (g, view)
    } else {
        let mut v = view;
        for ax in fake_axes.iter().rev() {
            v.remove_dim(*ax);
        }
        (g.sum(fake_axes), v)
    };
    match classify_view(&view, producer_dims) {
        ViewClass::Identity => g,
        ViewClass::Reshape => reinterpret(g, producer_dims),
        ViewClass::Permute(axes) => g.permute(axes),
        ViewClass::General => scatter_add_through_view(g, &view, producer_dims),
    }
}

/// Upper bound on how many index-expression evaluations the static-map
/// analysis will do at graph-build time. Beyond this, both adjoint
/// strategies are hopeless anyway; fall back to the one-hot.
const MAX_STATIC_MAP: usize = 1 << 24;

/// Residue classes are tracked in a u64 bitmask per target element.
const MAX_CLASSES: usize = 64;

/// Build the exact adjoint of reading through a static index map:
/// `out[j] = Σ_{z : f(z) = j} g[z]`, for `f` over the domain `0..m` into
/// `[0, l)`, with `g` the flattened (m,) gradient.
///
/// The map is evaluated once at graph-build time. If it's injective, one
/// overwrite-`Scatter` into zeros is exact — writes never collide. If not,
/// the domain is split into `b` residue classes (`z mod b`) such that `f`
/// restricted to each class *is* injective — the structured non-injective
/// maps that appear in practice separate this way (`unfold`'s overlapping
/// windows by kernel position, pad's clamped edges by parity) — and the
/// adjoint is the sum of one Scatter per class: O(m + b·l) instead of the
/// O(m·l) one-hot. Returns None when `f` isn't static or no `b` up to
/// [`MAX_CLASSES`] separates it; callers fall back to the one-hot.
pub(crate) fn static_scatter_add(
    g_flat: GraphTensor,
    idx_flat: GraphTensor,
    f: Expression,
    m: usize,
    l: usize,
) -> Option<GraphTensor> {
    if m > MAX_STATIC_MAP || !f.dyn_vars().iter().all(|c| c.is_reserved()) {
        return None;
    }
    let mut vals = Vec::with_capacity(m);
    for z in 0..m {
        let v = f.exec_single_var_checked(z)?;
        if v >= l {
            return None;
        }
        vals.push(v);
    }
    let b = (1..=MAX_CLASSES).find(|b| m.is_multiple_of(*b) && classes_injective(&vals, *b, l))?;
    let cx = g_flat.graph();
    let mut acc: Option<GraphTensor> = None;
    for k in 0..b {
        // Class k covers domain positions z = z'·b + k: column k of the
        // (m/b, b)-reshaped gradient and index tensors. Slicing the existing
        // index tensor (rather than building per-class iota expressions)
        // keeps the e-graph small — composed index expressions are large and
        // expensive for egglog to chew on.
        let (g_k, idx_k) = if b == 1 {
            (g_flat, idx_flat)
        } else {
            let gc = reinterpret(g_flat, &[(m / b).into(), b.into()]);
            let ic = reinterpret(idx_flat, &[(m / b).into(), b.into()]);
            (
                gc.slice((0.., k..k + 1)).squeeze(1), // (m/b,)
                ic.slice((0.., k..k + 1)).squeeze(1), // (m/b,) Int
            )
        };
        let dest = zeros_flat(cx, l.into(), g_flat.dtype);
        let scattered = g_k.scatter(idx_k, dest); // (l,)
        acc = Some(match acc {
            Some(a) => a + scattered,
            None => scattered,
        });
    }
    acc
}

/// Is `vals` (the evaluated index map) injective within every `z mod b`
/// residue class?
fn classes_injective(vals: &[usize], b: usize, l: usize) -> bool {
    let mut masks = vec![0u64; l];
    for (z, &v) in vals.iter().enumerate() {
        let bit = 1u64 << (z % b);
        if masks[v] & bit != 0 {
            return false;
        }
        masks[v] |= bit;
    }
    true
}

/// A flat (n,) tensor of zeros, expressed as a stride-0 view of a scalar
/// constant (used as the Scatter destination in scatter adjoints).
pub(crate) fn zeros_flat(cx: &mut Graph, n: Expression, dtype: DType) -> GraphTensor {
    let mut z = cx.constant_float(0.0);
    if dtype != DType::F32 {
        z = z.cast(dtype);
    }
    GraphTensor::from_id(
        z.id,
        ShapeTracker::fake(&[n][..]).with_element_bits(dtype.bits()),
        z.graph_ref,
        dtype,
    )
}

/// General fallback: `grad_producer[j] = Σ_{i : index_expr(i) = j} g[i]`.
///
/// Prefers the exact scatter decomposition from [`static_scatter_add`];
/// otherwise builds a one-hot (N_physical × M_logical) mask contracted
/// against the flattened gradient: always correct, O(N·M) work.
fn scatter_add_through_view(
    g: GraphTensor,
    view: &ShapeTracker,
    producer_dims: &[Expression],
) -> GraphTensor {
    let cx = g.graph();
    let m = view.n_elements().simplify();
    let n = producer_dims
        .iter()
        .copied()
        .product::<Expression>()
        .simplify();
    let g_flat = reinterpret(g, &[m]);
    // Physical index for each logical position of the view.
    let phys_idx = cx.iota(view.index_expression(), m); // (M,) Int
    if let (Some(ms), Some(ns)) = (m.as_num(), n.as_num()) {
        let e = view.index_expression();
        if let Some(flat) = static_scatter_add(g_flat, phys_idx, e, ms as usize, ns as usize) {
            return reinterpret(flat, producer_dims);
        }
    }
    let rows = cx.iota('z', n); // (N,) Int
    let onehot = rows
        .expand_dim(1, m)
        .eq(phys_idx.expand_dim(0, n))
        .cast(g_flat.dtype); // (N, M)
    let flat = (onehot * g_flat.expand_dim(0, n)).sum(1); // (N,)
    reinterpret(flat, producer_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(v: &[usize]) -> Vec<Expression> {
        v.iter().map(|d| Expression::from(*d)).collect()
    }

    #[test]
    fn classify_identity() {
        let prod = dims(&[2, 3]);
        let view = ShapeTracker::new(&prod[..]);
        assert_eq!(classify_view(&view, &prod), ViewClass::Identity);
    }

    #[test]
    fn classify_reshape() {
        let prod = dims(&[2, 3]);
        let mut view = ShapeTracker::new(&prod[..]);
        view.merge_dims(0, 1); // (6,)
        assert_eq!(classify_view(&view, &prod), ViewClass::Reshape);
    }

    #[test]
    fn classify_permute() {
        let prod = dims(&[2, 3]);
        let mut view = ShapeTracker::new(&prod[..]);
        view.permute(&[1, 0]); // (3, 2) transposed view
        // Inverting the transpose is the transpose again.
        assert_eq!(classify_view(&view, &prod), ViewClass::Permute(vec![1, 0]));
    }

    #[test]
    fn classify_permute_3d() {
        let prod = dims(&[2, 3, 4]);
        let mut view = ShapeTracker::new(&prod[..]);
        view.permute(&[2, 0, 1]); // view axis i holds producer axis [2,0,1][i]
        // Producer axis 0 lives at view axis 1, axis 1 at view axis 2, axis 2 at view axis 0.
        assert_eq!(
            classify_view(&view, &prod),
            ViewClass::Permute(vec![1, 2, 0])
        );
    }

    #[test]
    fn classify_slice_is_general() {
        let prod = dims(&[4, 5]);
        let mut view = ShapeTracker::new(&prod[..]);
        // Offset-free slice: shrink dim 1 but keep the stride of the wider buffer.
        view.dims[1] = 3.into();
        assert_eq!(classify_view(&view, &prod), ViewClass::General);
    }

    #[test]
    fn classify_permuted_reshape_is_general() {
        let prod = dims(&[2, 3]);
        let mut view = ShapeTracker::new(&prod[..]);
        view.permute(&[1, 0]);
        view.merge_dims(0, 1); // merged transposed view: non-contiguous flat read
        assert_eq!(classify_view(&view, &prod), ViewClass::General);
    }
}
