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

/// Upper bound on how many index-expression evaluations the static
/// injectivity check will do at graph-build time. Beyond this, both adjoint
/// strategies are hopeless anyway; fall back to the one-hot.
const MAX_STATIC_INJECTIVITY_CHECK: usize = 1 << 24;

/// Statically decide whether the composed index map `z ↦ chain(z)` over the
/// domain `0..m` is injective into `[0, l)`. Evaluated exactly, element by
/// element, at graph-build time — possible only when the domain size and
/// every expression in the chain are static.
///
/// When a read map is injective, no two logical positions share a physical
/// source, so its exact adjoint (a scatter-ADD) coincides with the
/// overwrite-`Scatter` op HLIR already has: writes never collide. That turns
/// the O(N·M) one-hot adjoint into an O(N+M) scatter.
pub(crate) fn is_static_injective(chain: &[Expression], m: usize, l: usize) -> bool {
    if m > MAX_STATIC_INJECTIVITY_CHECK {
        return false;
    }
    if chain
        .iter()
        .any(|e| !e.dyn_vars().iter().all(|c| *c == 'z'))
    {
        return false;
    }
    let mut seen = vec![false; l];
    for i in 0..m {
        let mut v = i;
        for e in chain {
            match e.exec_single_var_checked(v) {
                Some(x) => v = x,
                None => return false,
            }
        }
        if v >= l || seen[v] {
            return false;
        }
        seen[v] = true;
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
/// When the view's index map is statically injective, this is one `Scatter`
/// of the flattened gradient into zeros (exact — writes never collide).
/// Otherwise it's built as a one-hot (N_physical × M_logical) mask contracted
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
    if let (Some(ms), Some(ns)) = (m.as_num(), n.as_num()) {
        let e = view.index_expression();
        if is_static_injective(&[e], ms as usize, ns as usize) {
            let idx = cx.iota(e, m); // (M,) Int: logical -> physical
            let dest = zeros_flat(cx, n, g_flat.dtype);
            return reinterpret(g_flat.scatter(idx, dest), producer_dims);
        }
    }
    // Physical index for each logical position of the view.
    let phys_idx = cx.iota(view.index_expression(), m); // (M,) Int
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
