//! Reverse-mode autograd over luminal HLIR graphs.
//!
//! [`backward`] walks the HLIR graph in reverse topological order from a
//! scalar loss, emitting ordinary HLIR ops for each gradient. The backward
//! graph lives in the same [`Graph`] as the forward pass, so `cx.compile`
//! optimizes forward and backward together (shared subexpressions unify in
//! the e-graph, fusion sees across the boundary, and no backend needs to
//! know training exists).

use itertools::Itertools;
use luminal::hlir as hl;
use luminal::prelude::petgraph::algo::toposort;
use luminal::prelude::*;

use crate::view::{reinterpret, static_scatter_add, unview};

/// Extension trait adding reverse-mode autograd to [`Graph`].
pub trait Backward {
    /// Append the backward graph for `loss` (a scalar) and return one
    /// gradient tensor per entry of `params`, aligned by index. Parameters
    /// that don't influence the loss get a zero gradient. Call before
    /// `compile`, and mark the returned tensors `.output()` (or feed them to
    /// an optimizer) like any other tensor you want out of the graph.
    fn backward(&mut self, loss: GraphTensor, params: &[GraphTensor]) -> Vec<GraphTensor>;
}

impl Backward for Graph {
    fn backward(&mut self, loss: GraphTensor, params: &[GraphTensor]) -> Vec<GraphTensor> {
        backward(self, loss, params)
    }
}

fn is_float(d: DType) -> bool {
    matches!(d, DType::F32 | DType::F64 | DType::F16 | DType::Bf16)
}

/// See [`Backward::backward`].
pub fn backward(cx: &mut Graph, loss: GraphTensor, params: &[GraphTensor]) -> Vec<GraphTensor> {
    let topo = toposort(&**cx, None).expect("HLIR graph must be acyclic");

    // --- Forward analysis: canonical output dims + dtype per node ----------
    // Every op writes its output contiguously; "canonical dims" is that
    // contiguous shape. Views live on the consumer side (in each op's stored
    // input ShapeTrackers), so gradients are always accumulated in canonical
    // space and mapped through views by `unview`.
    let mut dims: FxHashMap<NodeIndex, Vec<Expression>> = FxHashMap::default();
    let mut dtypes: FxHashMap<NodeIndex, DType> = FxHashMap::default();
    for p in params {
        assert!(
            p.shape.is_contiguous(),
            "parameters passed to backward must be contiguous tensors (param {:?} has view {})",
            p.id,
            p.shape
        );
        dims.insert(p.id, p.dims());
        dtypes.insert(p.id, p.dtype);
    }
    for &n in &topo {
        let srcs = cx.get_sources(n);
        let (d, t) = derive_node_meta(cx, n, &srcs, &dims, &dtypes);
        if let Some(d) = d {
            dims.entry(n).or_insert(d);
        }
        if let Some(t) = t {
            dtypes.entry(n).or_insert(t);
        }
    }

    // --- Which nodes need gradients ----------------------------------------
    let mut rg: FxHashSet<NodeIndex> = params.iter().map(|p| p.id).collect();
    for &n in &topo {
        if rg.contains(&n) {
            continue;
        }
        let srcs = cx.get_sources(n);
        if srcs
            .iter()
            .enumerate()
            .any(|(slot, s)| rg.contains(s) && grad_flows(cx, n, slot, &srcs, &dtypes))
        {
            rg.insert(n);
        }
    }

    // --- Seed: d(loss)/d(loss) = 1 -----------------------------------------
    let loss_dims = dims
        .get(&loss.id)
        .cloned()
        .expect("loss node has no derivable shape");
    let loss_elements = loss_dims.iter().copied().product::<Expression>().simplify();
    assert_eq!(
        loss_elements.as_num(),
        Some(1),
        "loss must be a scalar (got shape ({}))",
        loss_dims.iter().map(|d| format!("{d}")).join(", ")
    );
    let loss_dtype = dtypes.get(&loss.id).copied().unwrap_or(DType::F32);
    let mut seed = cx.constant_float(1.0);
    if loss_dtype != DType::F32 {
        seed = seed.cast(loss_dtype);
    }
    // View the scalar 1 as the loss's canonical dims (all size-1 axes).
    let seed = GraphTensor::from_id(
        seed.id,
        ShapeTracker::fake(&loss_dims[..]).with_element_bits(loss_dtype.bits()),
        seed.graph_ref,
        loss_dtype,
    );

    // --- Reverse accumulation ----------------------------------------------
    let mut grads: FxHashMap<NodeIndex, GraphTensor> = FxHashMap::default();
    grads.insert(loss.id, seed);
    for &n in topo.iter().rev() {
        let Some(g) = grads.get(&n).copied() else {
            continue;
        };
        let srcs = cx.get_sources(n);
        if srcs.is_empty() {
            continue; // Input / Constant / Iota leaves
        }
        for (slot, local, view) in vjp(cx, n, g, &srcs, &dims, &dtypes, &rg) {
            let src = srcs[slot];
            let pdims = dims.get(&src).unwrap_or_else(|| {
                panic!(
                    "luminal_training: no derivable output shape for node {src:?} \
                     ({:?}) while backpropagating through node {n:?}",
                    cx.node_weight(src).unwrap()
                )
            });
            let gp = unview(local, view, pdims);
            match grads.get(&src).copied() {
                Some(existing) => {
                    grads.insert(src, existing + gp);
                }
                None => {
                    grads.insert(src, gp);
                }
            }
        }
    }

    // --- Collect ------------------------------------------------------------
    params
        .iter()
        .map(|p| {
            grads.get(&p.id).copied().unwrap_or_else(|| {
                // Parameter doesn't influence the loss: zero gradient.
                let mut z = cx.constant_float(0.0);
                if p.dtype != DType::F32 {
                    z = z.cast(p.dtype);
                }
                GraphTensor::from_id(
                    z.id,
                    ShapeTracker::fake(&p.dims()[..]).with_element_bits(p.dtype.bits()),
                    z.graph_ref,
                    p.dtype,
                )
            })
        })
        .collect()
}

/// Canonical output dims + dtype for a node, derived from the op's stored
/// shape metadata (ops write contiguous outputs, so this is exact).
fn derive_node_meta(
    cx: &Graph,
    n: NodeIndex,
    srcs: &[NodeIndex],
    dims: &FxHashMap<NodeIndex, Vec<Expression>>,
    dtypes: &FxHashMap<NodeIndex, DType>,
) -> (Option<Vec<Expression>>, Option<DType>) {
    if let Some(op) = cx.try_get_op::<hl::Input>(n) {
        // Dims of an Input aren't recorded on the node; params supply theirs.
        return (None, Some(op.dtype));
    }
    if cx.try_get_op::<hl::Constant>(n).is_some() {
        return (Some(vec![]), Some(DType::F32));
    }
    if let Some(op) = cx.try_get_op::<hl::Iota>(n) {
        return (Some(vec![op.1]), Some(DType::Int));
    }
    if let Some(op) = cx.try_get_op::<hl::Cast>(n) {
        // Cast converts the physical buffer; the view passes through, so its
        // canonical layout is its input's.
        return (dims.get(&srcs[0]).cloned(), Some(op.1));
    }
    macro_rules! unary {
        ($t:ty) => {
            if let Some(op) = cx.try_get_op::<$t>(n) {
                return (
                    Some(op.input_shape.dims.to_vec()),
                    dtypes.get(&srcs[0]).copied(),
                );
            }
        };
    }
    unary!(hl::Exp2);
    unary!(hl::Log2);
    unary!(hl::Sin);
    unary!(hl::Recip);
    unary!(hl::Sqrt);
    macro_rules! binary {
        ($t:ty) => {
            if let Some(op) = cx.try_get_op::<$t>(n) {
                return (
                    Some(op.input_shapes[0].dims.to_vec()),
                    dtypes.get(&srcs[0]).copied(),
                );
            }
        };
    }
    binary!(hl::Add);
    binary!(hl::Mul);
    binary!(hl::Mod);
    if let Some(op) = cx.try_get_op::<hl::LessThan>(n) {
        return (Some(op.input_shapes[0].dims.to_vec()), Some(DType::Bool));
    }
    macro_rules! reduce {
        ($t:ty) => {
            if let Some(op) = cx.try_get_op::<$t>(n) {
                let mut d = op.input_shape.dims.to_vec();
                d.remove(op.dim);
                return (Some(d), dtypes.get(&srcs[0]).copied());
            }
        };
    }
    reduce!(hl::SumReduce);
    reduce!(hl::MaxReduce);
    if let Some(op) = cx.try_get_op::<hl::Gather>(n) {
        return (
            Some(op.input_shapes[0].dims.to_vec()),
            dtypes.get(&srcs[1]).copied(),
        );
    }
    if let Some(op) = cx.try_get_op::<hl::Scatter>(n) {
        return (Some(op.dest_shape.clone()), dtypes.get(&srcs[0]).copied());
    }
    // Output, loop markers, custom ops: no canonical shape needed.
    (None, None)
}

/// Whether a mathematically nonzero gradient can flow from node `n`'s output
/// to its input at `slot`. Ops we can't differentiate yet (custom ops)
/// return `true` so the vjp errors loudly instead of silently producing
/// zero gradients.
fn grad_flows(
    cx: &Graph,
    n: NodeIndex,
    slot: usize,
    srcs: &[NodeIndex],
    dtypes: &FxHashMap<NodeIndex, DType>,
) -> bool {
    if cx.try_get_op::<hl::Add>(n).is_some()
        || cx.try_get_op::<hl::Mul>(n).is_some()
        || cx.try_get_op::<hl::Exp2>(n).is_some()
        || cx.try_get_op::<hl::Log2>(n).is_some()
        || cx.try_get_op::<hl::Sin>(n).is_some()
        || cx.try_get_op::<hl::Recip>(n).is_some()
        || cx.try_get_op::<hl::Sqrt>(n).is_some()
        || cx.try_get_op::<hl::SumReduce>(n).is_some()
        || cx.try_get_op::<hl::MaxReduce>(n).is_some()
    {
        return true;
    }
    if cx.try_get_op::<hl::Mod>(n).is_some() {
        return true;
    }
    if cx.try_get_op::<hl::Scatter>(n).is_some() {
        return slot != 1; // dest and src; indices are integers
    }
    if let Some(op) = cx.try_get_op::<hl::Cast>(n) {
        return is_float(op.1) && dtypes.get(&srcs[0]).copied().map(is_float).unwrap_or(false);
    }
    if cx.try_get_op::<hl::Gather>(n).is_some() {
        return slot == 1; // data only; indices are integers
    }
    if cx.try_get_op::<hl::LessThan>(n).is_some()
        || cx.try_get_op::<hl::Iota>(n).is_some()
        || cx.try_get_op::<hl::Constant>(n).is_some()
        || cx.try_get_op::<hl::Input>(n).is_some()
        || cx.try_get_op::<hl::Output>(n).is_some()
    {
        return false;
    }
    // CustomOpKind, loop markers, unknown ops: claim flow so the vjp raises
    // an unsupported-op error rather than dropping gradients.
    true
}

/// Per-op vector-Jacobian products. Given the output-space gradient `g` of
/// node `n`, returns `(input_slot, local_gradient, input_view)` triples for
/// every slot whose source requires a gradient. `local_gradient` lives in
/// `n`'s output space; the caller maps it through `input_view` with `unview`.
fn vjp(
    cx: &mut Graph,
    n: NodeIndex,
    g: GraphTensor,
    srcs: &[NodeIndex],
    dims: &FxHashMap<NodeIndex, Vec<Expression>>,
    dtypes: &FxHashMap<NodeIndex, DType>,
    rg: &FxHashSet<NodeIndex>,
) -> Vec<(usize, GraphTensor, ShapeTracker)> {
    let gref = g.graph_ref;
    let src_dtype =
        |slot: usize| -> DType { dtypes.get(&srcs[slot]).copied().unwrap_or(DType::F32) };
    // The forward node's own (contiguous) output, reusable in vjps to avoid
    // recomputing e.g. exp2(x).
    let out_tensor = |out_dims: &[Expression]| -> GraphTensor {
        GraphTensor::from_id(
            n,
            ShapeTracker::new(out_dims).with_element_bits(g.dtype.bits()),
            gref,
            g.dtype,
        )
    };
    let src_tensor = |slot: usize, view: ShapeTracker| -> GraphTensor {
        GraphTensor::from_id(srcs[slot], view, gref, src_dtype(slot))
    };
    let mut out = vec![];

    if let Some(op) = cx.try_get_op::<hl::Add>(n) {
        let shapes = op.input_shapes.clone();
        for slot in 0..srcs.len() {
            if rg.contains(&srcs[slot]) {
                out.push((slot, g, shapes[slot]));
            }
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Mul>(n) {
        let shapes = op.input_shapes.clone();
        for slot in 0..2 {
            if rg.contains(&srcs[slot]) {
                let other = src_tensor(1 - slot, shapes[1 - slot]);
                out.push((slot, g * other, shapes[slot]));
            }
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Mod>(n) {
        let shapes = op.input_shapes.clone();
        if rg.contains(&srcs[0]) {
            // d(a mod b)/da = 1 almost everywhere.
            out.push((0, g, shapes[0]));
        }
        if rg.contains(&srcs[1]) {
            // a mod b = a − b·trunc(a/b), so d/db = −trunc(a/b) almost
            // everywhere. Recover trunc(a/b) from the forward output as
            // (a − out)/b — exact under whatever rounding convention the
            // kernel used, with no floor op needed.
            let a = src_tensor(0, shapes[0]);
            let b = src_tensor(1, shapes[1]);
            let o = out_tensor(&shapes[0].dims);
            out.push((1, -(g * (a - o) * b.reciprocal()), shapes[1]));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Exp2>(n) {
        let shape = op.input_shape;
        if rg.contains(&srcs[0]) {
            // d/dx 2^x = 2^x · ln2
            let o = out_tensor(&shape.dims);
            out.push((0, g * o * std::f32::consts::LN_2, shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Log2>(n) {
        let shape = op.input_shape;
        if rg.contains(&srcs[0]) {
            // d/dx log2(x) = 1 / (x · ln2)
            let x = src_tensor(0, shape);
            out.push((
                0,
                g * x.reciprocal() * (1.0 / std::f32::consts::LN_2),
                shape,
            ));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Sin>(n) {
        let shape = op.input_shape;
        if rg.contains(&srcs[0]) {
            let x = src_tensor(0, shape);
            out.push((0, g * x.cos(), shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Recip>(n) {
        let shape = op.input_shape;
        if rg.contains(&srcs[0]) {
            // d/dx (1/x) = -1/x² = -out²
            let o = out_tensor(&shape.dims);
            out.push((0, -(g * o * o), shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Sqrt>(n) {
        let shape = op.input_shape;
        if rg.contains(&srcs[0]) {
            // d/dx √x = 1/(2√x) = 1/(2·out)
            let o = out_tensor(&shape.dims);
            out.push((0, g * o.reciprocal() * 0.5, shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Cast>(n) {
        let _ = op;
        if rg.contains(&srcs[0]) {
            let sdt = src_dtype(0);
            let local = if g.dtype != sdt { g.cast(sdt) } else { g };
            // Cast passes views through untouched, so its implicit input view
            // is the identity over the source's canonical layout.
            let pdims = dims
                .get(&srcs[0])
                .expect("Cast input has no derivable shape")
                .clone();
            out.push((0, local, ShapeTracker::new(&pdims[..])));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::SumReduce>(n) {
        let (dim, shape) = (op.dim, op.input_shape);
        if rg.contains(&srcs[0]) {
            // Broadcast the gradient back along the reduced axis (free view).
            out.push((0, g.expand_dim(dim, shape.dims[dim]), shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::MaxReduce>(n) {
        let (dim, shape) = (op.dim, op.input_shape);
        if rg.contains(&srcs[0]) {
            // Indicator of the max: g flows to every element equal to the
            // max (ties each receive the full gradient).
            let mut out_dims = shape.dims.to_vec();
            out_dims.remove(dim);
            let o = out_tensor(&out_dims).expand_dim(dim, shape.dims[dim]);
            let x = src_tensor(0, shape);
            let mask = x.eq(o).cast(g.dtype);
            out.push((0, g.expand_dim(dim, shape.dims[dim]) * mask, shape));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Gather>(n) {
        let shapes = op.input_shapes.clone();
        // If the indices come from an Iota, the full index map (iota
        // expression composed with the index view's own read pattern) is
        // statically known — Copy it out so the op borrow ends here.
        let index_iota = cx.try_get_op::<hl::Iota>(srcs[0]).map(|i| i.0);
        if rg.contains(&srcs[1]) {
            let (index_view, data_view) = (shapes[0], shapes[1]);
            // grad_data[l] = Σ_{i : idx[i] = l} g[i] over the data view's
            // logical space, then unview maps logical → physical.
            let m = index_view.n_elements().simplify();
            let l = data_view.n_elements().simplify();
            let g_flat = reinterpret(g, &[m]);
            let idx_flat = reinterpret(src_tensor(0, index_view), &[m]);
            // Exact scatter decomposition of the adjoint when the full index
            // map (iota expression composed with the index view's own read
            // pattern) is static: one Scatter if injective, a small sum of
            // per-residue-class Scatters for structured overlaps like unfold.
            let scatter_local = match (m.as_num(), l.as_num(), index_iota) {
                (Some(ms), Some(ls), Some(iota_expr)) => {
                    let f = iota_expr.substitute('z', index_view.index_expression());
                    static_scatter_add(g_flat, idx_flat, f.simplify(), ms as usize, ls as usize)
                }
                _ => None,
            };
            // Otherwise (runtime indices, dynamic shapes, or inseparable
            // duplicates): one-hot contraction — always correct, O(L·M).
            let local_flat = scatter_local.unwrap_or_else(|| {
                let rows = cx.iota('z', l); // (L,) Int
                let onehot = rows
                    .expand_dim(1, m)
                    .eq(idx_flat.expand_dim(0, l))
                    .cast(g_flat.dtype); // (L, M)
                (onehot * g_flat.expand_dim(0, l)).sum(1) // (L,)
            });
            let local = reinterpret(local_flat, &data_view.dims);
            out.push((1, local, data_view));
        }
        return out;
    }
    if let Some(op) = cx.try_get_op::<hl::Scatter>(n) {
        // out = copy(dest); out[idx[i]] = src[i], last write wins, in the
        // logical (flattened) space of the dest view. `g` lives in that same
        // space (canonical dims = dest_shape).
        let dest_dims = op.dest_shape.clone();
        let index_dims = op.index_shape.clone();
        let dest_view = ShapeTracker::new_strided(&op.dest_shape[..], &op.dest_strides[..]);
        let index_view = ShapeTracker::new_strided(&op.index_shape[..], &op.index_strides[..]);
        let src_view = ShapeTracker::new_strided(&op.index_shape[..], &op.src_strides[..]);
        let n_flat = dest_view.n_elements().simplify();
        let m_flat = index_view.n_elements().simplify();
        let g_flat = reinterpret(g, &[n_flat]);
        let idx_flat = reinterpret(src_tensor(1, index_view), &[m_flat]);
        let mut half = cx.constant_float(0.5);
        if g.dtype != DType::F32 {
            half = half.cast(g.dtype);
        }
        if rg.contains(&srcs[0]) {
            // Gradient passes through dest exactly where no write landed:
            // grad_dest[j] = g[j] · [count of i with idx[i] = j is zero].
            let rows = cx.iota('z', n_flat); // (N,) Int
            let counts = (rows
                .expand_dim(1, m_flat)
                .eq(idx_flat.expand_dim(0, n_flat))
                .cast(g.dtype))
            .sum(1); // (N,)
            let not_covered = counts.lt(half.expand_dim(0, n_flat)).cast(g.dtype);
            let local = reinterpret(g_flat * not_covered, &dest_dims);
            out.push((0, local, dest_view));
        }
        if rg.contains(&srcs[2]) {
            // grad_src[i] = g[idx[i]] for the write that won each location,
            // 0 for overwritten writes. Write i wins iff no later write i' > i
            // hits the same location, so mask by the count of later duplicates.
            let gathered = g_flat.gather(idx_flat); // (M,)
            let ri = cx.iota('z', m_flat).expand_dim(1, m_flat); // [i, i'] = i
            let ci = cx.iota('z', m_flat).expand_dim(0, m_flat); // [i, i'] = i'
            let later = ri.lt(ci).cast(g.dtype); // (M, M): i' after i
            let same_slot = idx_flat
                .expand_dim(1, m_flat)
                .eq(idx_flat.expand_dim(0, m_flat))
                .cast(g.dtype); // (M, M)
            let later_dups = (same_slot * later).sum(1); // (M,)
            let winner = later_dups.lt(half.expand_dim(0, m_flat)).cast(g.dtype);
            let local = reinterpret(gathered * winner, &index_dims);
            out.push((2, local, src_view));
        }
        return out;
    }
    if cx.try_get_op::<hl::LessThan>(n).is_some() {
        return out; // zero gradient almost everywhere
    }
    panic!(
        "luminal_training: cannot differentiate through op {:?} (node {n:?})",
        cx.node_weight(n).unwrap()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grad_shapes_match_params() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3));
        let b = cx.tensor((2, 3));
        let loss = (a * b).sum((0, 1));
        let grads = cx.backward(loss, &[a, b]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].dims(), a.dims());
        assert_eq!(grads[1].dims(), b.dims());
    }

    #[test]
    fn unused_param_gets_zero_grad() {
        let mut cx = Graph::new();
        let a = cx.tensor(3);
        let unused = cx.tensor((4, 2));
        let loss = a.sum(0);
        let grads = cx.backward(loss, &[a, unused]);
        assert_eq!(grads[1].dims(), unused.dims());
    }

    #[test]
    #[should_panic(expected = "loss must be a scalar")]
    fn non_scalar_loss_panics() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3));
        let loss = a.sum(1); // shape (2,) — not scalar
        cx.backward(loss, &[a]);
    }

    #[test]
    fn matmul_grad_shapes() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3));
        let w = cx.tensor((3, 4));
        let loss = x.matmul(w).sum((0, 1));
        let grads = cx.backward(loss, &[x, w]);
        assert_eq!(grads[0].dims(), x.dims());
        assert_eq!(grads[1].dims(), w.dims());
    }
}
