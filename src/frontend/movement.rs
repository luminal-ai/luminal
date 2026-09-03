use itertools::Itertools;

use crate::graph::{MapEntry, Movement, movement_entries};
use crate::prelude::*;

/// ONE-APPLY view composer for macro interiors (Austin's ratified rule
/// 2026-08-26: "one user call = one view node; macro interiors also mint
/// ONE apply per logical construct, never per-axis loops"). A chain of
/// movement steps is composed at CONSTRUCTION TIME on the MapEntry tree
/// — nothing is recorded until `finish()`, which mints exactly one
/// `LogicalIndexMapApply` (or returns the tensor untouched when the
/// composed map is the identity). Composition of maps that are ALREADY
/// recorded stays egglog's job (fold-1); this builder only ever builds
/// the single map for the single call.
#[must_use]
pub struct ViewChain {
    tensor: GraphTensor,
    /// Per PARENT axis, an entry over the current (virtual) out space.
    entries: Vec<MapEntry>,
    /// The current virtual out dims.
    dims: Vec<IntExpr>,
}

impl ViewChain {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> Vec<IntExpr> {
        self.dims.clone()
    }

    /// Compose one movement step (its map stated from its own
    /// parameters, exactly as `apply_movement` would record it).
    fn step(mut self, movement: Movement) -> Self {
        match movement_entries(movement, &self.dims) {
            Ok((step_entries, new_dims)) => {
                let cur_rank = self.dims.len();
                for entry in &mut self.entries {
                    *entry = entry.substitute(&step_entries, cur_rank);
                }
                self.dims = new_dims;
            }
            Err(reason) => {
                self.tensor.graph().logical.poison(reason);
            }
        }
        self
    }

    pub fn permute(self, axes: impl ToAxes) -> Self {
        self.step(Movement::Permute(axes.to_axes()))
    }

    pub fn expand_dim(self, axis: usize, size: impl Into<IntExpr>) -> Self {
        self.step(Movement::ExpandDim {
            axis,
            size: size.into(),
        })
    }

    pub fn unsqueeze(self, axis: usize) -> Self {
        self.expand_dim(axis, 1)
    }

    /// Same contract rails as `GraphTensor::squeeze`: static non-1
    /// extents panic loudly; symbolic extents record the extent==1
    /// post-saturation contract.
    pub fn squeeze(self, axis: usize) -> Self {
        let extent = self.dims[axis];
        match extent.to_usize() {
            Some(1) => {}
            Some(n) => panic!("Only dimensions of size 1 can be squeezed! (got {n})"),
            None => {
                let at = self.tensor.id.index();
                self.tensor
                    .graph()
                    .logical
                    .require_extent_eq_one(at, &extent, "squeeze");
            }
        }
        self.step(Movement::RemoveDim { axis })
    }

    /// Same divisibility contract as `GraphTensor::split_dims`
    /// (decision-only simplify — never a recorded spelling).
    pub fn split_dims(self, axis: usize, new_dim_size: impl Into<IntExpr>) -> Self {
        let new_dim_size = new_dim_size.into();
        assert!(
            new_dim_size.as_num().is_none_or(|n| n > 0),
            "split_dims inner dimension must be positive, got {new_dim_size}"
        );
        let old_dim = self.dims[axis];
        let outer_dim = (old_dim / new_dim_size).simplify();
        assert!(
            (outer_dim * new_dim_size)
                .simplify()
                .egglog_equal(old_dim.simplify()),
            "split_dims requires the old dimension ({old_dim}) to be exactly divisible by the inner dimension ({new_dim_size})"
        );
        self.step(Movement::SplitDims {
            axis,
            inner: new_dim_size,
        })
    }

    pub fn merge_dims(self, axis1: usize, axis2: usize) -> Self {
        assert!(axis1 < axis2, "axis1 must be less than axis2");
        self.step(Movement::MergeDims { axis1, axis2 })
    }

    /// PyTorch-style broadcast (size-1 axes grow to the target) as a
    /// chain step.
    pub fn expand(mut self, new_shape: impl ToShape) -> Self {
        let target = new_shape.to_shape();
        assert_eq!(target.len(), self.rank(), "expand rank mismatch");
        if self.dims == target {
            return self;
        }
        let rank = target.len();
        let step_entries: Vec<MapEntry> = (0..rank)
            .map(|p| {
                if self.dims[p] == target[p] {
                    MapEntry::Coord {
                        from_end: rank - 1 - p,
                        extent: target[p],
                    }
                } else {
                    assert_eq!(
                        self.dims[p],
                        IntExpr::from(1),
                        "expand: axis {p} is {}, only size-1 axes broadcast",
                        self.dims[p]
                    );
                    MapEntry::Lit(0.into())
                }
            })
            .collect();
        let cur_rank = self.dims.len();
        for entry in &mut self.entries {
            *entry = entry.substitute(&step_entries, cur_rank);
        }
        self.dims = target;
        self
    }

    /// Mint the ONE apply — or nothing at all if the composed map is
    /// the identity.
    pub fn finish(self) -> GraphTensor {
        let source_dims = self.tensor.dims();
        let rank = source_dims.len();
        let identity = self.dims == source_dims
            && self.entries.len() == rank
            && self.entries.iter().enumerate().all(|(p, entry)| {
                matches!(entry, MapEntry::Coord { from_end, .. } if *from_end == rank - 1 - p)
            });
        if identity {
            return self.tensor;
        }
        self.tensor.record_view_map(self.entries, self.dims)
    }
}

impl GraphTensor {
    /// Open a ONE-APPLY view chain (see [`ViewChain`]).
    pub fn view(self) -> ViewChain {
        let dims = self.dims();
        let rank = dims.len();
        ViewChain {
            tensor: self,
            entries: (0..rank)
                .map(|p| MapEntry::Coord {
                    from_end: rank - 1 - p,
                    extent: dims[p],
                })
                .collect(),
            dims,
        }
    }
}

impl GraphTensor {
    /// Swap dimensions of the tensor
    pub fn permute(self, axes: impl ToAxes) -> GraphTensor {
        let axes = axes.to_axes();
        assert!(
            axes.len() == self.rank(),
            "Permute axes ({}) doesn't match shape axes ({})",
            axes.len(),
            self.rank()
        );
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::Permute(axes),
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// Swap 2 dimensions. This is a view-only operation and does not materialize a new tensor
    pub fn transpose(self, dim0: usize, dim1: usize) -> GraphTensor {
        let num_dims = self.rank();
        assert!(
            dim0 < num_dims && dim1 < num_dims,
            "transpose dimensions ({dim0}, {dim1}) out of bounds for tensor with {num_dims} dimensions"
        );
        let mut perm_axes: Vec<usize> = (0..num_dims).collect();
        perm_axes.swap(dim0, dim1);
        self.permute(perm_axes)
    }

    /// Transpose a 2D tensor
    pub fn t(self) -> GraphTensor {
        assert_eq!(self.rank(), 2, ".t() supports only 2D tensors");
        self.transpose(0, 1)
    }

    /// Broadcast tensor along a new dimension
    pub fn expand_dim(self, axis: usize, size: impl Into<IntExpr>) -> GraphTensor {
        let size = size.into();
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::ExpandDim { axis, size },
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// Broadcast tensor along new dimensions on the right-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    ///
    /// ONE apply (ruling 2026-08-26): a rank-0 constant into shape S
    /// records an empty entry list tagged with the scalar source shape.
    pub fn expand_rhs(self, shape: impl ToShape) -> GraphTensor {
        let orig_dims = self.rank();
        let mut chain = self.view();
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            chain = chain.expand_dim(orig_dims + i, s);
        }
        chain.finish()
    }

    /// Tile a tensor along its existing dimensions without materializing a new buffer.
    pub fn repeat(self, repeats: impl ToShape) -> GraphTensor {
        let repeats = repeats.to_shape();
        assert_eq!(
            repeats.len(),
            self.rank(),
            "Repeat shape ({}) doesn't match tensor dimensions ({})",
            repeats.len(),
            self.rank()
        );
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::Repeat(repeats),
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// Record ONE view apply from an explicit entry list (fold-2
    /// removal, Austin's ruling 2026-08-26): the map is stated directly
    /// at the source of truth — no intermediate movement scaffolding.
    /// Entries are listed per PARENT axis (outermost inward), each an
    /// expression in the OUT space's coordinates (from-end convention).
    pub(crate) fn record_view_map(
        mut self,
        entries: Vec<crate::graph::MapEntry>,
        out_dims: Vec<IntExpr>,
    ) -> GraphTensor {
        let operand = (self.id, self.dims());
        let value = self
            .graph()
            .logical
            .view_op(&operand, &entries, out_dims.clone(), self.dtype);
        // Poisoned fallback: keep the stated out shape so downstream
        // reads stay panic-free; when recorded, with_logical re-derives
        // the dims from the recorder (R-D).
        self.dims = out_dims.into_iter().collect();
        self.with_logical(value)
    }

    /// Broadcast tensor along new dimensions on the left-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    /// PyTorch-style expand: every size-1 axis whose target size differs
    /// broadcasts to it. Non-1 axes must already match. Records ONE
    /// direct apply (fold-2 removal): broadcast axes read Lit 0, kept
    /// axes read their like-positioned out coordinate.
    pub fn expand(self, new_shape: impl ToShape) -> GraphTensor {
        let target = new_shape.to_shape();
        assert_eq!(target.len(), self.rank(), "expand rank mismatch");
        let current = self.dims();
        let rank = target.len();
        if current == target {
            return self;
        }
        let entries: Vec<crate::graph::MapEntry> = (0..rank)
            .map(|p| {
                if current[p] == target[p] {
                    crate::graph::MapEntry::Coord {
                        from_end: rank - 1 - p,
                        extent: target[p],
                    }
                } else {
                    assert_eq!(
                        current[p],
                        IntExpr::from(1),
                        "expand: axis {p} is {}, only size-1 axes broadcast",
                        current[p]
                    );
                    crate::graph::MapEntry::Lit(0.into())
                }
            })
            .collect();
        self.record_view_map(entries, target)
    }

    /// ONE apply (ruling 2026-08-26) — see `expand_rhs`.
    pub fn expand_lhs(self, shape: impl ToShape) -> GraphTensor {
        let mut chain = self.view();
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            chain = chain.expand_dim(i, s);
        }
        chain.finish()
    }

    /// ONE apply (ruling 2026-08-26) — see `expand_rhs`.
    pub fn expand_to_shape_on_axes(self, shape: impl ToShape, axes: impl ToAxes) -> GraphTensor {
        let shape = shape.to_shape();
        let axes = axes.to_axes();
        assert_eq!(shape.len(), self.rank() + axes.len());
        let mut chain = self.view();
        for axis in axes.into_iter().sorted() {
            chain = chain.expand_dim(axis, shape[axis]);
        }
        chain.finish()
    }

    /// Merge two dimensions together
    pub fn merge_dims(self, axis1: usize, axis2: usize) -> GraphTensor {
        assert!(axis1 < axis2, "axis1 must be less than axis2");
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::MergeDims { axis1, axis2 },
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// Flatten all dimensions into a single 1D tensor — ONE apply
    /// (ruling 2026-08-26): the full merge arithmetic is composed at
    /// map construction inside this one call.
    pub fn flatten(self) -> GraphTensor {
        let mut chain = self.view();
        while chain.rank() > 1 {
            chain = chain.merge_dims(0, 1);
        }
        chain.finish()
    }

    //// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(self, axis: usize, new_dim_size: impl Into<IntExpr>) -> GraphTensor {
        let new_dim_size = new_dim_size.into();
        assert!(
            new_dim_size.as_num().is_none_or(|n| n > 0),
            "split_dims inner dimension must be positive, got {new_dim_size}"
        );
        let old_dim = self.dims[axis];
        // Divisibility DECISION: simplify here is decision-only (never
        // stored, never rendered into the model).
        let outer_dim = (old_dim / new_dim_size).simplify();
        assert!(
            (outer_dim * new_dim_size)
                .simplify()
                .egglog_equal(old_dim.simplify()),
            "split_dims requires the old dimension ({old_dim}) to be exactly divisible by the inner dimension ({new_dim_size})"
        );
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::SplitDims {
                axis,
                inner: new_dim_size,
            },
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// add a new dimension of size 1 at the specified place
    pub fn unsqueeze(self, dim: usize) -> GraphTensor {
        assert!(self.rank() < 10, "Shape is maxed out at 10 dimensions");
        self.expand_dim(dim, 1)
    }

    /// remove a dimension of size 1
    pub fn squeeze(self, axis: usize) -> GraphTensor {
        let extent = self.dims()[axis];
        // DECISION site: `to_usize` EVALUATES the RPN (exec with no
        // vars), so any STATIC spelling — simplified or raw — lands in
        // the Some arms; only genuinely symbolic extents take the
        // recorded post-saturation contract below.
        match extent.to_usize() {
            Some(1) => {}
            Some(n) => panic!("Only dimensions of size 1 can be squeezed! (got {n})"),
            // A SYMBOLIC extent: squeeze is a CONTRACT that this axis
            // is 1 (a data-dependent rank is unrepresentable). Recorded
            // unconditionally; a post-saturation invariant refuses any
            // binding/bucket that admits values other than 1 (ruling
            // 2026-08-13, option 3 — bucket the dim to [1,1] to pass).
            None => {
                let at = self.id.index();
                self.graph()
                    .logical
                    .require_extent_eq_one(at, &extent, "squeeze");
            }
        }
        let current_dims = self.dims();
        let value = self.graph().logical.apply_movement(
            &(self.id, current_dims),
            crate::graph::Movement::RemoveDim { axis },
        );
        // R-D: dims derive from the recorded value (with_logical).
        self.with_logical(value)
    }

    /// Gather elements along an axis using per-element indices (ONNX GatherElements semantics).
    ///
    /// `output[i0,..,ik] = self[i0,..,i_{axis-1}, indices[i0,..,ik], i_{axis+1},..,ik]`
    ///
    /// indices must have the same rank as self and the same shape as the output.
    pub fn gather_elements(self, indexes: GraphTensor, axis: usize) -> GraphTensor {
        let dims = self.dims();
        let rank = dims.len();
        let out_shape: Vec<IntExpr> = indexes.dims();

        // Row-major strides: stride[i] = prod(dims[i+1..]) — SYMBOLIC
        // IntExpr products (ruling 2026-08-13: the frontend computes
        // nothing eagerly that the expression language can carry; static
        // dims still fold to literals via to_usize so the recorded text
        // is unchanged for concrete shapes).
        let strides: Vec<IntExpr> = (0..rank)
            .map(|i| {
                let product = dims[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, d| acc * *d);
                product.to_usize().map(IntExpr::from).unwrap_or(product)
            })
            .collect();

        // Normalize negative indices for axis dim (symbolic-capable).
        let axis_dim = dims[axis];
        // Int-native normalization (2026-08-11): the comparison and the
        // adjustment stay in i32 end to end — the old f32 detour ended
        // in a cast back to Int, which the cast policy now refuses.
        assert_eq!(indexes.dtype, DType::Int, "index tensor must be Int");
        let zero = indexes.graph().constant(0).expand_rhs(indexes.dims());
        let adj = indexes
            .graph()
            .constant(axis_dim)
            .expand_rhs(indexes.dims());
        let is_neg = indexes.lt(zero).cast(DType::Int);
        // Plain Int arithmetic is proof-gated (non-wrapping ruling): this
        // graph implements only when the caller DECLARES the index
        // tensor's value range at binding time (`bind_value_range` —
        // gather semantics already require indices in [-d, d), so the
        // declaration states what the data must satisfy anyway).
        let idx_normalized = indexes + is_neg * adj;

        // Non-axis flat contribution as a coordinate function — no
        // flat-index div/mod chain ever enters the model (P1, 2026-08-07).
        let non_axis_flat = self.graph().iota(out_shape, |c| {
            (0..rank)
                .filter(|d| *d != axis)
                .fold(IntExpr::from(0), |acc, d| acc + c[d] * strides[d])
        });

        // Axis contribution from the runtime index values
        let stride_tensor = self
            .graph()
            .constant(strides[axis])
            .expand_rhs(idx_normalized.dims());
        let flat_idx = non_axis_flat + idx_normalized * stride_tensor;

        self.gather1d(flat_idx)
    }

    /// Scatter updates into a copy of self at positions specified by per-element indices along an axis.
    ///
    /// ONNX ScatterElements semantics:
    /// `output[i0,..,i_{a-1}, indices[i0,..,ik], i_{a+1},..,ik] = updates[i0,..,ik]`
    ///
    /// indices and updates must have the same shape.
    /// Overlapping writes: last write wins.
    pub fn scatter_elements(
        self,
        indices: GraphTensor,
        updates: GraphTensor,
        axis: usize,
    ) -> GraphTensor {
        let data_dims = self.dims();
        let rank = data_dims.len();
        let idx_shape: Vec<IntExpr> = indices.dims();

        // Row-major strides for data — symbolic IntExpr products
        // (see gather_elements; static dims fold to literals).
        let strides: Vec<IntExpr> = (0..rank)
            .map(|i| {
                let product = data_dims[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, d| acc * *d);
                product.to_usize().map(IntExpr::from).unwrap_or(product)
            })
            .collect();

        // Normalize negative indices for axis dim (symbolic-capable).
        let axis_dim = data_dims[axis];
        // Int-native normalization (2026-08-11) — see gather_elements.
        assert_eq!(indices.dtype, DType::Int, "index tensor must be Int");
        let zero = indices.graph().constant(0).expand_rhs(indices.dims());
        let adj = indices
            .graph()
            .constant(axis_dim)
            .expand_rhs(indices.dims());
        let is_neg = indices.lt(zero).cast(DType::Int);
        // Proof-gated plain arithmetic — see gather_elements: the caller
        // declares the index range at binding time.
        let idx_normalized = indices + is_neg * adj;

        // Non-axis flat contribution as a coordinate function (P1).
        let non_axis_flat = self.graph().iota(idx_shape.clone(), |c| {
            (0..rank)
                .filter(|d| *d != axis)
                .fold(IntExpr::from(0), |acc, d| acc + c[d] * strides[d])
        });

        // Axis contribution from the runtime index values
        let stride_tensor = self
            .graph()
            .constant(strides[axis])
            .expand_rhs(idx_normalized.dims());
        let flat_dest = non_axis_flat + idx_normalized * stride_tensor;

        // Flatten to 1D using materialize + reshape
        let flat_dest_1d = flat_dest.flatten();
        let flat_updates = updates.flatten();
        let flat_data = self.flatten();

        // dest.flat[indexes[i]] = src[i], then rebuild data's shape
        let output_flat = flat_updates.scatter1d(flat_dest_1d, flat_data);
        output_flat.unflatten_to(&data_dims)
    }

    /// Scatter updates into a copy of self using multi-dimensional index vectors (ONNX ScatterND semantics).
    ///
    /// `indices` has shape [S0, ..., Sq-2, K] where K <= rank(data).
    /// `updates` has shape [S0, ..., Sq-2, D_K, ..., D_{r-1}].
    /// For each batch element (s0, ..., sq-2):
    ///   multi_idx = indices[s0, ..., sq-2, :]
    ///   output[multi_idx[0], ..., multi_idx[K-1], :, ..] = updates[s0, ..., sq-2, :, ..]
    pub fn scatter_nd(self, indices: GraphTensor, updates: GraphTensor) -> GraphTensor {
        // Was a convenience cast; a float index tensor would now record
        // the refused f32 -> Int cast, so demand Int at the door.
        assert_eq!(indices.dtype, DType::Int, "scatter_nd indices must be Int");
        let data_dims = self.dims();
        let data_rank = data_dims.len();
        let idx_dims = indices.dims();
        let idx_rank = idx_dims.len();

        // K is STRUCTURAL — it fixes how many slice extractions the
        // recorder emits (a rank, not an extent) — so it alone must be
        // concrete. Every other quantity stays a symbolic IntExpr
        // (ruling 2026-08-13: nothing eager in the frontend that the
        // expression language can carry; static dims fold to literals).
        let k = idx_dims[idx_rank - 1]
            .to_usize()
            .expect("scatter_nd: K (last indices dim) is structural and must be concrete");
        assert!(k <= data_rank, "scatter_nd: K must be <= data rank");

        let fold_product = |dims: &[IntExpr]| {
            let product = dims.iter().fold(IntExpr::from(1), |acc, d| acc * *d);
            product.to_usize().map(IntExpr::from).unwrap_or(product)
        };
        // Batch numel = product of indices shape without the last dim.
        let batch_numel = fold_product(&idx_dims[..idx_rank - 1]);
        // Trailing shape = data dims [K..].
        let trailing_shape: Vec<IntExpr> = data_dims[k..].to_vec();
        let trailing_numel = fold_product(&trailing_shape);

        // Row-major strides for data — symbolic products.
        let data_strides: Vec<IntExpr> = (0..data_rank)
            .map(|i| fold_product(&data_dims[i + 1..]))
            .collect();

        // Flatten batch dims of indices to [batch_numel, K] — ONE apply
        // (ruling 2026-08-26).
        let indices_flat = {
            let mut chain = indices.view();
            while chain.rank() > 2 {
                chain = chain.merge_dims(0, 1);
            }
            chain.finish()
        };
        // indices_flat: [batch_numel, K] or [K] if idx_rank == 1

        // For each k_dim, extract the slice and multiply by stride
        let mut flat_base: Option<GraphTensor> = None;
        for (k_dim, stride) in data_strides.iter().copied().enumerate().take(k) {
            let idx_k = indices_flat.slice_along(k_dim..k_dim + 1, indices_flat.dims().len() - 1);
            let idx_k = idx_k.squeeze(idx_k.dims().len() - 1);

            let stride_tensor = self.graph().constant(stride).expand_rhs(idx_k.dims());
            // Proof-gated plain arithmetic — see gather_elements: the
            // caller declares the index range at binding time.
            let contribution = idx_k * stride_tensor;

            flat_base = Some(match flat_base {
                Some(fb) => fb + contribution,
                None => contribution,
            });
        }
        let flat_base = flat_base.unwrap();

        let mut full_flat_dest =
            if trailing_shape.is_empty() || trailing_numel.to_usize() == Some(1) {
                flat_base
            } else {
                // The trailing offset is a pure COORDINATE FUNCTION over the
                // trailing space (P1, 2026-08-07: no flat div/mod chain) —
                // ONE iota + two broadcast applies replace the per-dim
                // arange/expand scaffolding (ruling 2026-08-26).
                let trailing_strides: Vec<IntExpr> = data_strides[k..].to_vec();
                let trailing_offset = self.graph().iota(trailing_shape.clone(), move |c| {
                    (0..c.len()).fold(IntExpr::from(0), |acc, ti| {
                        acc + c[ti] * trailing_strides[ti]
                    })
                });
                flat_base.expand_rhs(trailing_shape.clone())
                    + trailing_offset.expand_lhs(vec![batch_numel])
            };

        full_flat_dest = full_flat_dest.flatten();

        // Flatten data out
        let flat_updates = updates.flatten();
        let flat_data = self.flatten();

        // dest.flat[indexes[i]] = src[i], then rebuild data's shape
        let output_flat = flat_updates.scatter1d(full_flat_dest, flat_data);
        output_flat.unflatten_to(&data_dims)
    }

    /// COORDINATE-FORM gather — THE primary (ruling 2026-07-31): one Int
    /// coordinate tensor per data axis, each over the OUTPUT shape:
    /// `out[c] = data[coords[0][c], ..., coords[r-1][c]]` — numpy fancy
    /// indexing / unpacked GatherND, mapping 1:1 onto LogicalGather. The
    /// HLIR side lowers transitionally to flat-index arithmetic + the flat
    /// Gather (dies with the HLIR pipeline at M3 Step 4); the recorder
    /// emits the coordinate form DIRECTLY.
    pub fn gather(self, coords: &[GraphTensor]) -> GraphTensor {
        assert_eq!(
            coords.len(),
            self.rank(),
            "gather: one coordinate tensor per data axis"
        );
        assert!(!coords.is_empty(), "gather: rank-0 data has no axes");
        let out_dims = coords[0].dims();
        for coord in coords {
            assert_eq!(coord.dtype, DType::Int, "gather coordinates must be Int");
            assert_eq!(
                coord.dims(),
                out_dims,
                "gather coordinates share the out shape"
            );
        }
        let dims = self.dims();
        let data_operand = (self.id, dims);
        let coord_operands: Vec<_> = coords
            .iter()
            .map(|coord| (coord.id, coord.dims()))
            .collect();
        let id = self
            .graph()
            .logical
            .record_gather(&data_operand, &coord_operands, out_dims.clone(), self.dtype)
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, out_dims, self.graph_ref, self.dtype)
    }

    /// COORDINATE-FORM scatter — THE primary (ruling 2026-07-31): self is
    /// the init/dest; `src` lands at the positions the coordinate tensors
    /// name (one per dest axis, each over src's shape). Copy-then-write
    /// value semantics; in-place is a binding + search decision.
    pub fn scatter(self, coords: &[GraphTensor], src: GraphTensor) -> GraphTensor {
        assert_eq!(
            coords.len(),
            self.rank(),
            "scatter: one coordinate tensor per dest axis"
        );
        assert!(!coords.is_empty(), "scatter: rank-0 dest has no axes");
        let index_dims = src.dims();
        for coord in coords {
            assert_eq!(coord.dtype, DType::Int, "scatter coordinates must be Int");
            assert_eq!(
                coord.dims(),
                index_dims,
                "scatter coordinates share src's shape"
            );
        }
        let dims = self.dims();
        let init_operand = (self.id, dims.clone());
        let src_operand = (src.id, index_dims);
        let coord_operands: Vec<_> = coords
            .iter()
            .map(|coord| (coord.id, coord.dims()))
            .collect();
        let id = self
            .graph()
            .logical
            .record_scatter(
                &init_operand,
                &coord_operands,
                &src_operand,
                dims.clone(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, dims, self.graph_ref, self.dtype)
    }

    /// Rebuild a multi-dim shape from a flat tensor with recorded splits
    /// (the inverse of `flatten`; there is no wholesale reshape in the
    /// recorded vocabulary).
    pub(crate) fn unflatten_to(self, dims: &[IntExpr]) -> GraphTensor {
        assert_eq!(self.rank(), 1, "unflatten_to starts from a flat tensor");
        // ONE apply (ruling 2026-08-26): the split arithmetic composes
        // at map construction inside this one call.
        let mut chain = self.view();
        for axis in 0..dims.len().saturating_sub(1) {
            // Frontend simplification restored (revert ruling 2026-08-27).
            let inner: IntExpr = dims[axis + 1..]
                .iter()
                .copied()
                .fold(IntExpr::from(1), |acc, d| acc * d)
                .simplify();
            chain = chain.split_dims(axis, inner);
        }
        chain.finish()
    }

    /// FLAT gather over flat data: out[c] = data.flat[indexes[c]], out
    /// shape = indexes' shape. Sugar over the coordinate form: flatten the
    /// data, then the index tensor IS the single coordinate tensor.
    pub fn gather1d(self, indexes: GraphTensor) -> GraphTensor {
        assert_eq!(
            indexes.dtype,
            DType::Int,
            "Gather indexes must have an integer dtype!"
        );
        self.flatten().gather(&[indexes])
    }

    /// Scatter self (src) into dest at flat 1D positions given by indexes.
    /// output = copy(dest); output.flat[indexes[i]] = src[i]
    /// Sugar over the coordinate form: flatten dest/src/indexes, scatter,
    /// then rebuild dest's shape with recorded splits.
    pub fn scatter1d(self, indexes: GraphTensor, dest: GraphTensor) -> GraphTensor {
        assert_eq!(
            indexes.dtype,
            DType::Int,
            "Scatter indexes must have an integer dtype!"
        );
        assert_eq!(
            indexes.dims(),
            self.dims(),
            "scatter1d: indexes and src (self) share a shape"
        );
        let out_dims = dest.dims();
        let flat = dest.flatten().scatter(&[indexes.flatten()], self.flatten());
        flat.unflatten_to(&out_dims)
    }

    /// Extracts sliding local windows from an input tensor.
    pub fn unfold(
        self,
        kernel: impl ToShape,
        strides: impl ToShape,
        dilation: impl ToShape,
    ) -> GraphTensor {
        let (kernel, strides, dilation) =
            (kernel.to_shape(), strides.to_shape(), dilation.to_shape());
        let (entries, final_shape) = self.unfold_map(&kernel, &strides, &dilation, self.id.index());
        let operand = (self.id, self.dims());
        let logical =
            self.graph()
                .logical
                .view_op(&operand, &entries, final_shape.clone(), self.dtype);
        // The id is minted by the recorder itself now (PR #423: the SSA
        // node IS the identity); poisoned recording keeps the source id.
        GraphTensor::from_id(self.id, final_shape, self.graph_ref, self.dtype).with_logical(logical)
    }

    /// [`unfold`](Self::unfold) opened as a [`ViewChain`], so a macro
    /// that immediately reshapes the windows (convolution's im2col)
    /// can compose the whole construct into ONE apply (ruling
    /// 2026-08-26).
    pub fn unfold_view(
        self,
        kernel: impl ToShape,
        strides: impl ToShape,
        dilation: impl ToShape,
    ) -> ViewChain {
        let (kernel, strides, dilation) =
            (kernel.to_shape(), strides.to_shape(), dilation.to_shape());
        let (entries, dims) = self.unfold_map(&kernel, &strides, &dilation, self.id.index());
        ViewChain {
            tensor: self,
            entries,
            dims,
        }
    }

    /// The unfold index map + out shape, from the unfold's own
    /// parameters; records the window contracts.
    fn unfold_map(
        self,
        kernel: &[IntExpr],
        strides: &[IntExpr],
        dilation: &[IntExpr],
        at: usize,
    ) -> (Vec<crate::graph::MapEntry>, Vec<IntExpr>) {
        assert_eq!(
            self.rank(),
            kernel.len(),
            "Kernel must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.rank(),
            strides.len(),
            "Strides must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.rank(),
            dilation.len(),
            "Dilation must be same number of dimensions as tensor!"
        );

        let dims = self.dims();
        let n = dims.len();

        // Per-dim window counts
        let mut win = Vec::with_capacity(n);
        for (((dim, k), s), d) in dims.iter().zip(kernel).zip(strides).zip(dilation) {
            let effective_window = *d * (*k - 1) + 1;
            win.push((*dim - effective_window).floor_div(s) + 1);
        }

        // [win..., kernel...] — construction-simplified like every
        // other recorder-authored expression (Austin's revert ruling
        // 2026-08-27: the frontend simplifying ruleset is restored as
        // the shield on recorded spellings; the deeper ring
        // boundary/fence question is deferred to the pending boundary
        // analysis). Historical note: during the brief R-C raw-spelling
        // window this site was the one measured exception — the raw
        // symbolic window count ((d - (dil*(k-1)+1)) / s) + 1 as a
        // recorded EXTENT made the reference schedule non-terminating
        // (>52 min vs 0.23 s; the nested-IntAdd assoc/subst family
        // detonation, see the rejoin-divergence dossier).
        let mut final_shape: Vec<IntExpr> = win.into_iter().map(|e| e.simplify()).collect();
        final_shape.extend(kernel.iter().copied());

        let window_counts = final_shape[..n].to_vec();
        // WINDOW CONTRACTS (ruling 2026-08-13, same rail as squeeze):
        // a symbolic window count must reach 1 — the kernel fits within
        // dim + padding, or the binding's bucket refuses with the named
        // door. Static counts are checked right here, loudly.
        for (axis, count) in window_counts.iter().enumerate() {
            match count.to_usize() {
                Some(0) => panic!("unfold axis {axis}: kernel does not fit (window count 0)"),
                Some(_) => {}
                None => {
                    self.graph().logical.require_extent_at_least(
                        at,
                        count,
                        1,
                        &format!(
                            "unfold window on axis {axis} (kernel must fit within dim + padding)"
                        ),
                    );
                }
            }
        }
        // Out shape [win..., k...] (rank 2n): parent axis p reads
        // win_p·stride_p + k_p·dilation_p — window arithmetic straight
        // from the unfold's own parameters.
        let out_rank = 2 * n;
        let entries: Vec<crate::graph::MapEntry> = (0..n)
            .map(|p| {
                crate::graph::MapEntry::Add(
                    Box::new(crate::graph::MapEntry::Mul(
                        Box::new(crate::graph::MapEntry::Coord {
                            from_end: out_rank - 1 - p,
                            extent: window_counts[p],
                        }),
                        strides[p],
                    )),
                    Box::new(crate::graph::MapEntry::Mul(
                        Box::new(crate::graph::MapEntry::Coord {
                            from_end: out_rank - 1 - (n + p),
                            extent: kernel[p],
                        }),
                        dilation[p],
                    )),
                )
            })
            .collect();
        (entries, final_shape)
    }

    /// Take a slice of a tensor along multiple dimensions.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10), DType::F32);
    /// let b = a.slice((2..4, 1..)); // 2x9 tensor
    /// assert_eq!(b.dims(), vec![IntExpr::from(2), IntExpr::from(9)]);
    /// ```
    pub fn slice(self, slice: impl ToSlice) -> GraphTensor {
        let mut ranges = slice.to_range_vec();
        ranges.extend(
            self.dims()
                .iter()
                .skip(ranges.len())
                .map(|d| (0.into(), *d)),
        ); // Make sure we have a range per dim
        if ranges.iter().any(|(st, _)| *st != 0) {
            // Start slices are VIEWS. The structure-preserving SliceView node
            // keeps the per-axis starts first-class (the logical-SSA seam);
            // its to_egglog lowers to the same iota+gather the frontend used
            // to build eagerly here, so the existing pipeline is unchanged.
            let mut new_dims = vec![];
            let mut starts = vec![];
            for (dim, (start, end)) in self.dims().into_iter().zip(ranges) {
                starts.push(start);
                new_dims.push(dim.min(end) - start);
            }
            // The seam node's own parameters ARE the view: parent axis p
            // reads out coordinate p plus its start.
            let rank = new_dims.len();
            let entries: Vec<crate::graph::MapEntry> = (0..rank)
                .map(|p| {
                    let coord = crate::graph::MapEntry::Coord {
                        from_end: rank - 1 - p,
                        extent: new_dims[p],
                    };
                    if starts[p] == IntExpr::from(0) {
                        coord
                    } else {
                        crate::graph::MapEntry::Add(
                            Box::new(coord),
                            Box::new(crate::graph::MapEntry::Lit(starts[p])),
                        )
                    }
                })
                .collect();
            let operand = (self.id, self.dims());
            let id = self
                .graph()
                .logical
                .view_op(&operand, &entries, new_dims.clone(), self.dtype)
                .unwrap_or_else(crate::graph::unrecorded_value);
            GraphTensor::from_id(id, new_dims, self.graph_ref, self.dtype)
        } else {
            // No start slices so no iota needed, just reduce the shape down
            let mut new_dims = self.dims();
            for (sh, (_, end)) in new_dims.iter_mut().zip(&ranges) {
                *sh = sh.min(*end);
            }
            let current_dims = self.dims();
            let value = self.graph().logical.apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::Shrink { new_dims },
            );
            // R-D: dims derive from the recorded value (with_logical).
            self.with_logical(value)
        }
    }

    /// Take a slice of a tensor along a dimension.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10), DType::F32);
    /// let b = a.slice_along(4.., 1); // 5x6 tensor
    /// assert_eq!(b.dims(), vec![IntExpr::from(5), IntExpr::from(6)]);
    /// ```
    pub fn slice_along(self, slice: impl SliceRange, axis: usize) -> GraphTensor {
        let mut s = vec![(IntExpr::from(0), IntExpr::from(i64::MAX)); axis + 1];
        s[axis] = slice.bounds();
        self.slice(s)
    }

    /// Pad out dimensions of a tensor with an element
    pub fn pad(self, padding: impl ToPad, elem: f32) -> GraphTensor {
        let mut padding = padding.to_pad_vec();
        padding.extend(vec![(0.into(), 0.into()); self.rank() - padding.len()]); // Make sure we have a padding per dim
        if padding.iter().all(|(s, e)| *s == 0 && *e == 0) {
            return self;
        }
        // Structure-preserving seam (see SliceView): the read half is a
        // total clamped VIEW, the mask half a per-axis-structured indicator
        // iota; both nodes lower to the legacy flat forms for the existing
        // pipeline via their to_egglog.
        let dims = self.dims();
        let befores: Vec<IntExpr> = padding.iter().map(|(s, _)| *s).collect();
        let afters: Vec<IntExpr> = padding.iter().map(|(_, e)| *e).collect();
        // Frontend simplification restored (revert ruling 2026-08-27).
        let out_dims: Vec<IntExpr> = dims
            .iter()
            .zip(&padding)
            .map(|(d, (s, e))| (*d + *s + *e).simplify())
            .collect();

        // Pad's read half recorded as the TOTAL clamped view — per parent
        // axis min(max(c - before, 0), dim - 1), clamp sides only where
        // padding exists.
        let clamped_id;
        {
            let rank = dims.len();
            let entries: Vec<crate::graph::MapEntry> = (0..rank)
                .map(|k| {
                    let coord = crate::graph::MapEntry::Coord {
                        from_end: rank - 1 - k,
                        extent: out_dims[k],
                    };
                    let mut entry = coord;
                    if befores[k] != IntExpr::from(0) {
                        entry = crate::graph::MapEntry::Max(
                            Box::new(crate::graph::MapEntry::Add(
                                Box::new(entry),
                                Box::new(crate::graph::MapEntry::Lit(
                                    (IntExpr::from(0) - befores[k]).simplify(),
                                )),
                            )),
                            Box::new(crate::graph::MapEntry::Lit(0.into())),
                        );
                    }
                    if afters[k] != IntExpr::from(0) {
                        entry = crate::graph::MapEntry::Min(
                            Box::new(entry),
                            Box::new(crate::graph::MapEntry::Lit((dims[k] - 1).simplify())),
                        );
                    }
                    entry
                })
                .collect();
            let operand = (self.id, dims.clone());
            clamped_id = self
                .graph()
                .logical
                .view_op(&operand, &entries, out_dims.clone(), self.dtype)
                .unwrap_or_else(crate::graph::unrecorded_value);
        }
        let clamped =
            GraphTensor::from_id(clamped_id, out_dims.clone(), self.graph_ref, self.dtype);

        let mask_id = self
            .graph()
            .logical
            .record_mask_iota(&befores, &afters, &dims)
            .unwrap_or_else(crate::graph::unrecorded_value);
        let mask =
            GraphTensor::from_id(mask_id, out_dims, self.graph_ref, DType::Int).cast(self.dtype);

        let masked = clamped * mask;
        if elem == 0.0 {
            masked
        } else {
            masked + ((1.0 - mask) * elem)
        }
    }

    /// Pad along an existing dimension
    pub fn pad_along(
        self,
        left: impl Into<IntExpr>,
        right: impl Into<IntExpr>,
        axis: usize,
        elem: f32,
    ) -> GraphTensor {
        let mut p = vec![(IntExpr::from(0), IntExpr::from(0)); axis + 1];
        p[axis] = (left.into(), right.into());
        self.pad(p, elem)
    }

    /// Concat along an existing dimension
    pub fn concat_along(self, rhs: GraphTensor, axis: usize) -> GraphTensor {
        // Pad and add
        self.pad_along(0, rhs.dims()[axis], axis, 0.)
            + rhs.pad_along(self.dims()[axis], 0, axis, 0.)
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::binary::tests::test_binary;
    use crate::frontend::unary::tests::test_unary;
    use crate::tests::assert_exact;
    use candle_core::{IndexOp, Tensor};
    use luminal::prelude::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_1d(len in 1usize..64, left in 0usize..6, right in 0usize..6) {
            // Zero padding early-returns self => a PURE-IDENTITY graph,
            // natively unsupported (pinned by stage4b_probes::
            // pinned_pure_identity_output; binding-level fix at 4d/M4).
            prop_assume!(left + right > 0);
            test_unary(
                len,
                |a| a.pad((left, right), 0.),
                |a| a.pad_with_zeros(0, left, right).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_2d(rows in 1usize..32, cols in 1usize..32, top in 0usize..6, bottom in 0usize..6, left in 0usize..6, right in 0usize..6) {
            test_unary(
                (rows, cols),
                |a| a.pad(((top, bottom), (left, right)), 0.),
                |a| {
                    a.pad_with_zeros(0, top, bottom)
                        .unwrap()
                        .pad_with_zeros(1, left, right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_slice_pad(
            rows in 3usize..32,
            cols in 3usize..32,
            start_row in 0usize..32,
            end_row in 1usize..32,
            start_col in 0usize..32,
            end_col in 1usize..32,
            pad_top in 0usize..6,
            pad_bottom in 0usize..6,
            pad_left in 0usize..6,
            pad_right in 0usize..6,
        ) {
            prop_assume!(start_row < end_row && end_row <= rows);
            prop_assume!(start_col < end_col && end_col <= cols);
            test_unary(
                (rows, cols),
                |a| a.slice((start_row..end_row, start_col..end_col)).pad(((pad_top, pad_bottom), (pad_left, pad_right)), 0.),
                |a| {
                    a.i((start_row..end_row, start_col..end_col))
                        .unwrap()
                        .pad_with_zeros(0, pad_top, pad_bottom)
                        .unwrap()
                        .pad_with_zeros(1, pad_left, pad_right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_transpose(rows in 1usize..32, cols in 1usize..32) {
            test_unary(
                (rows, cols),
                |a| a.transpose(0, 1) * 1.0,
                |a| a.transpose(0, 1).unwrap(),
            );
        }
    }

    #[test]
    fn test_unfold() {
        // Need all this code because candle doesnt do unfold
        #[allow(clippy::too_many_arguments)]
        pub fn unfold_nd_f32(
            x: &[f32],
            shape: &[usize],
            strides: &[usize],
            kernel: &[usize],
            step: &[usize],
            dilation: &[usize],
            pad_before: &[usize],
            pad_after: &[usize],
        ) -> Vec<f32> {
            let n = shape.len();
            assert!(n > 0);
            assert_eq!(strides.len(), n);
            assert_eq!(kernel.len(), n);
            assert_eq!(step.len(), n);
            assert_eq!(dilation.len(), n);
            assert_eq!(pad_before.len(), n);
            assert_eq!(pad_after.len(), n);

            for d in 0..n {
                assert!(kernel[d] > 0);
                assert!(step[d] > 0);
                assert!(dilation[d] > 0);
                assert!(shape[d] > 0);
            }

            // Effective kernel size per dim: (K-1)*d + 1
            let eff_kernel: Vec<usize> =
                (0..n).map(|d| (kernel[d] - 1) * dilation[d] + 1).collect();

            // Output spatial shape (number of windows) per dim
            let mut out_shape = vec![0usize; n];
            for d in 0..n {
                let padded = shape[d] + pad_before[d] + pad_after[d];
                if padded < eff_kernel[d] {
                    return Vec::new();
                }
                out_shape[d] = (padded - eff_kernel[d]) / step[d] + 1;
            }

            let windows = prod(&out_shape);
            let window_elems = prod(kernel);
            let mut out = vec![0.0f32; windows * window_elems];

            // Precompute helpers
            let k_mul = row_major_multipliers(kernel);

            // Current output window position (row-major)
            let mut out_pos = vec![0usize; n];

            for w in 0..windows {
                if w > 0 {
                    incr_row_major(&mut out_pos, &out_shape);
                }

                // Window start in padded coordinates
                let start_padded: Vec<usize> = (0..n).map(|d| out_pos[d] * step[d]).collect();

                let base_out = w * window_elems;

                // Iterate kernel elements (flattened)
                for ke in 0..window_elems {
                    let k_idx = unravel_row_major(ke, kernel, &k_mul);

                    let mut flat: isize = 0;
                    let mut in_bounds = true;

                    for d in 0..n {
                        let p = start_padded[d] + k_idx[d] * dilation[d];
                        let logical = p as isize - pad_before[d] as isize;

                        if logical < 0 || logical >= shape[d] as isize {
                            in_bounds = false;
                            break;
                        }
                        flat += logical * strides[d] as isize;
                    }

                    let out_idx = base_out + ke;
                    out[out_idx] = if in_bounds { x[flat as usize] } else { 0.0 };
                }
            }

            out
        }

        // -------- helpers --------

        fn prod(xs: &[usize]) -> usize {
            xs.iter().copied().product()
        }

        fn row_major_multipliers(shape: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut mul = vec![1usize; n];
            let mut acc = 1usize;
            for d in (0..n).rev() {
                mul[d] = acc;
                acc *= shape[d];
            }
            mul
        }

        fn unravel_row_major(mut idx: usize, shape: &[usize], mul: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut coords = vec![0usize; n];
            for d in 0..n {
                coords[d] = idx / mul[d];
                idx %= mul[d];
            }
            coords
        }

        fn incr_row_major(pos: &mut [usize], shape: &[usize]) {
            for d in (0..pos.len()).rev() {
                pos[d] += 1;
                if pos[d] < shape[d] {
                    return;
                }
                pos[d] = 0;
            }
        }

        test_unary(
            5,
            |a| a.unfold(3, 1, 1),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[3],
                        &[1],
                        &[1],
                        &[0],
                        &[0],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
        test_unary(
            (8, 10),
            |a| a.pad(((0, 2), (4, 4)), 0.).unfold((2, 3), (1, 2), (2, 1)),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[2, 3],
                        &[1, 2],
                        &[2, 1],
                        &[0, 4],
                        &[2, 3],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
    }

    #[test]
    fn test_unfold_floor_div_shape_for_odd_window_numerator() {
        let mut cx = Graph::new();
        let inp = cx.tensor((80, 3000), DType::F32);
        let out = inp.pad(((0, 0), (1, 1)), 0.).unfold((1, 3), (1, 2), (1, 1));
        assert_eq!(out.dims(), &[80, 1500, 1, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let mut cx = Graph::new();
        let inp = cx.tensor((2, 2, 3), DType::F32);
        let out1 = inp.unsqueeze(1);
        let out2 = inp.unsqueeze(3);
        assert_eq!(out1.dims(), &[2, 1, 2, 3]);
        assert_eq!(out2.dims(), &[2, 2, 3, 1]);
        test_unary(
            (1, 3),
            |a| a.squeeze(0).expand_dim(0, 2) * 1.,
            |a| a.broadcast_as((2, 3)).unwrap(),
        );
        // `* 1.0` materializes: a bare squeeze is a pure-VIEW output, and
        // view-only outputs share the input's buffer id — the 4d binding
        // gap (see stage4b_probes::pinned_pure_identity_output).
        test_unary(
            (2, 1, 3),
            |a| a.squeeze(1) * 1.0,
            |a| a.reshape((2, 3)).unwrap(),
        );
        // Bare squeeze — a pure-VIEW output, no materializing op. The
        // delivery-copy fix (2026-08-05) materializes it at the boundary.
        test_unary((2, 1, 3), |a| a.squeeze(1), |a| a.reshape((2, 3)).unwrap());
    }

    #[test]
    fn test_concat() {
        test_binary(
            17,
            32,
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_binary(
            (10, 4),
            (10, 6),
            |a, b| a.concat_along(b, 1),
            |a, b| Tensor::cat(&[a, b], 1).unwrap(),
        );
        test_binary(
            (4, 10),
            (6, 10),
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_unary(
            (4, 10),
            |a| a.concat_along(a, 0),
            |a| Tensor::cat(&[a.clone(), a], 0).unwrap(),
        );
    }

    // test_gather_and_scatter_inverse / test_scatter_basic /
    // test_scatter_into_nonzero_dest / test_scatter_all_positions:
    // B-TAIL-GATED (Step 4b). They exercised gather1d/scatter1d — the
    // flat pair the recorder still poisons — through the deleted their-
    // pipeline. Coordinate-form gather/scatter carry the native
    // differential coverage (reference); the flat sugar's tests
    // return with the B-tail recordings.

    #[test]
    fn test_repeat_is_view_only() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3), DType::F32);
        let repeated = a.repeat((2, 2));

        assert_ne!(
            repeated.id, a.id,
            "a logical view is its own SSA value even when it requires no materialization"
        );
        let graph = cx.logical.petgraph();
        assert!(matches!(
            graph[repeated.id].op,
            LogicalOp::IndexMapApply { .. }
        ));
        assert!(graph.find_edge(a.id, repeated.id).is_some());
        assert_eq!(
            repeated.dims(),
            vec![IntExpr::from(4usize), IntExpr::from(6usize)]
        );
    }

    #[test]
    fn test_repeat_runtime_values() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3), DType::F32);
        let repeated = (a.repeat((2, 2)) * 1.0).output();

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(a.id, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].into())],
        );

        assert_exact(
            rt.get_f32(repeated.id).unwrap(),
            &[
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 4.0, 5.0, 6.0, //
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, 4.0, 5.0, 6.0,
            ],
        );
    }

    //     // #[test]
    //     // fn test_cumsum() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.constant(1.).expand_dim(0, 3);
    //     //     let b = a.cumsum_last_dim().retrieve();
    //     //     let c = a
    //     //         .expand_dim(1, 3)
    //     //         .permute((1, 0))
    //     //         .cumsum_last_dim()
    //     //         .permute((1, 0))
    //     //         .retrieve();
    //     //     cx.execute();

    //     //     assert_exact(&b.data(), &[1., 2., 3.]);
    //     //     assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_pool_1d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set([1., 2., 3., 4., 5.]);
    //     //     let inp2 = cx
    //     //         .tensor((2, 5))
    //     //         .set([[15., 14., 13., 12., 11.], [1., 2., 3., 4., 5.]]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(3, 2, 1).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out4 = inp1.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out5 = inp1.pad(((1, 1),)).pool_last_dim(3, 1, 2).retrieve();
    //     //     // Stride 1 Batch 2
    //     //     let out6 = inp2.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 3
    //     //     let out7 = inp2.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out8 = inp2.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out9 = inp2.pad(((0, 0), (1, 1))).pool_last_dim(3, 1, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 2., 3.]);
    //     //     assert_exact(&out4.data(), &[1., 3., 5.]);
    //     //     assert_exact(&out5.data(), &[0., 2., 4., 1., 3., 5., 2., 4., 0.]);
    //     //     assert_exact(
    //     //         &out6.data(),
    //     //         &[
    //     //             15., 14., 13., 14., 13., 12., 13., 12., 11., 1., 2., 3., 2., 3., 4., 3., 4., 5.,
    //     //         ],
    //     //     );
    //     //     assert_exact(&out7.data(), &[15., 14., 13., 1., 2., 3.]);
    //     //     assert_exact(&out8.data(), &[15., 13., 11., 1., 3., 5.]);
    //     //     assert_exact(
    //     //         &out9.data(),
    //     //         &[
    //     //             0., 14., 12., 15., 13., 11., 14., 12., 0., 0., 2., 4., 1., 3., 5., 2., 4., 0.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dims() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
    //     //             14., 15., 14., 15., 16.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_2d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // 3x3 kernel
    //     //     let out1 = inp1
    //     //         // Pool first dim first by moving it to end
    //     //         .permute((1, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now move other dim to end
    //     //         .permute((1, 2, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now swap middle two dims
    //     //         .permute((0, 2, 1, 3))
    //     //         // Now merge both pooled dimensions
    //     //         .reshape((4, 3, 3))
    //     //         .retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
    //     //             7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
    //     //             14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dilation() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_rotate_half() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.tensor((3, 2));
    //     //     a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
    //     //     let x1 = a.slice((.., ..1)).contiguous();
    //     //     let x2 = a.slice((.., 1..)).contiguous();
    //     //     let c = (-x2).concat_along(x1, 1);
    //     //     c.retrieve();
    //     //     cx.execute();

    //     //     let d_dev = Cpu::default();
    //     //     let d_a = d_dev.tensor_from_vec(
    //     //         vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
    //     //         (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
    //     //     );
    //     //     let d_x1 = d_a.clone().slice((.., ..1));
    //     //     let d_x2 = d_a.slice((.., 1..));
    //     //     let d_c = (-d_x2, d_x1)
    //     //         .concat_along(dfdx::shapes::Axis::<1>)
    //     //         .realize::<Rank2<3, 2>>();

    //     //     assert_close(&c.data(), &d_c.as_vec());
    //     // }
}
