use itertools::Itertools;

use crate::prelude::*;

impl GraphTensor {
    /// Swap dimensions of the tensor
    pub fn permute(mut self, axes: impl ToAxes) -> GraphTensor {
        let axes = axes.to_axes();
        assert!(
            axes.len() == self.rank(),
            "Permute axes ({}) doesn't match shape axes ({})",
            axes.len(),
            self.rank()
        );
        let current_dims = self.dims();
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::Permute(axes.clone()),
            )
            .expect("logical movement insertion failed");
        self.dims = axes.iter().map(|i| self.dims[*i]).collect();
        self
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
    pub fn expand_dim(mut self, axis: usize, size: impl Into<IntExpr>) -> GraphTensor {
        let size = size.into();
        let current_dims = self.dims();
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::ExpandDim { axis, size },
            )
            .expect("logical movement insertion failed");
        self.dims.insert(axis, size);
        self
    }

    /// Broadcast tensor along new dimensions on the right-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    pub fn expand_rhs(mut self, shape: impl ToShape) -> GraphTensor {
        let orig_dims = self.rank();
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self = self.expand_dim(orig_dims + i, s);
        }
        self
    }

    /// Tile a tensor along its existing dimensions without materializing a new buffer.
    pub fn repeat(mut self, repeats: impl ToShape) -> GraphTensor {
        let repeats = repeats.to_shape();
        assert_eq!(
            repeats.len(),
            self.rank(),
            "Repeat shape ({}) doesn't match tensor dimensions ({})",
            repeats.len(),
            self.rank()
        );
        let current_dims = self.dims();
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::Repeat(repeats.clone()),
            )
            .expect("logical movement insertion failed");
        for (dim, repeat) in self.dims.iter_mut().zip(repeats) {
            if repeat == IntExpr::from(1) {
                continue;
            }
            *dim = (*dim * repeat).simplify();
        }
        self
    }

    /// Broadcast tensor along new dimensions on the left-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    /// PyTorch-style expand: every size-1 axis whose target size differs
    /// broadcasts to it (a squeeze + broadcast expand per axis — pure,
    /// recorder-covered). Non-1 axes must already match.
    pub fn expand(mut self, new_shape: impl ToShape) -> GraphTensor {
        let target = new_shape.to_shape();
        assert_eq!(target.len(), self.rank(), "expand rank mismatch");
        for (axis, target_dim) in target.into_iter().enumerate() {
            let current = self.dims()[axis];
            if current == target_dim {
                continue;
            }
            assert_eq!(
                current,
                IntExpr::from(1),
                "expand: axis {axis} is {current}, only size-1 axes broadcast"
            );
            self = self.squeeze(axis).expand_dim(axis, target_dim);
        }
        self
    }

    pub fn expand_lhs(mut self, shape: impl ToShape) -> GraphTensor {
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self = self.expand_dim(i, s);
        }
        self
    }

    pub fn expand_to_shape_on_axes(
        mut self,
        shape: impl ToShape,
        axes: impl ToAxes,
    ) -> GraphTensor {
        let shape = shape.to_shape();
        let axes = axes.to_axes();
        assert_eq!(shape.len(), self.rank() + axes.len());
        for axis in axes.into_iter().sorted() {
            self = self.expand_dim(axis, shape[axis]);
        }
        self
    }

    /// Merge two dimensions together
    pub fn merge_dims(mut self, axis1: usize, axis2: usize) -> GraphTensor {
        assert!(axis1 < axis2, "axis1 must be less than axis2");
        let current_dims = self.dims();
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::MergeDims { axis1, axis2 },
            )
            .expect("logical movement insertion failed");
        // Move axis2 next to axis1 if not adjacent, then fold it in.
        if axis2 != axis1 + 1 {
            let dim = self.dims.remove(axis2);
            self.dims.insert(axis1 + 1, dim);
        }
        let inner = self.dims.remove(axis1 + 1);
        self.dims[axis1] = (self.dims[axis1] * inner).simplify();
        self
    }

    /// Flatten all dimensions into a single 1D tensor.
    pub fn flatten(mut self) -> GraphTensor {
        while self.rank() > 1 {
            self = self.merge_dims(0, 1);
        }
        self
    }

    //// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(mut self, axis: usize, new_dim_size: impl Into<IntExpr>) -> GraphTensor {
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
        let current_dims = self.dims();
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::SplitDims {
                    axis,
                    inner: new_dim_size,
                },
            )
            .expect("logical movement insertion failed");
        self.dims[axis] = outer_dim;
        self.dims.insert(axis + 1, new_dim_size);
        self
    }

    /// add a new dimension of size 1 at the specified place
    pub fn unsqueeze(self, dim: usize) -> GraphTensor {
        assert!(self.rank() < 10, "Shape is maxed out at 10 dimensions");
        self.expand_dim(dim, 1)
    }

    /// remove a dimension of size 1
    pub fn squeeze(mut self, axis: usize) -> GraphTensor {
        let extent = self.dims()[axis];
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
        self.id = self
            .graph()
            .logical
            .apply_movement(
                &(self.id, current_dims),
                crate::graph::Movement::RemoveDim { axis },
            )
            .expect("logical movement insertion failed");
        self.dims.remove(axis);
        self
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

        // Flatten batch dims of indices to [batch_numel, K] with recorded merges
        let mut indices_flat = indices;
        while indices_flat.rank() > 2 {
            indices_flat = indices_flat.merge_dims(0, 1);
        }
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
                // Expand flat_base to [batch_numel, trailing_numel]
                let mut base_expanded = flat_base.expand_dim(1, trailing_numel);

                let trailing_rank = trailing_shape.len();
                for (ti, d) in (k..data_rank).enumerate() {
                    let ar = self.graph().arange(data_dims[d]);
                    let mut ar_shaped = ar;
                    for _ in ti + 1..trailing_rank {
                        let n = ar_shaped.dims().len();
                        ar_shaped = ar_shaped.expand_dim(n, 1);
                    }
                    for _ in 0..ti {
                        ar_shaped = ar_shaped.expand_dim(0, 1);
                    }
                    ar_shaped = ar_shaped.expand(trailing_shape.clone());
                    // Flatten trailing dims with recorded merges, then broadcast
                    let ar_flat = ar_shaped.flatten().expand_dim(0, batch_numel);

                    let stride_tensor = self
                        .graph()
                        .constant(data_strides[d])
                        .expand_rhs(ar_flat.dims());
                    base_expanded += ar_flat * stride_tensor;
                }
                base_expanded
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
            .expect("logical gather insertion failed");
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
            .expect("logical scatter insertion failed");
        GraphTensor::from_id(id, dims, self.graph_ref, self.dtype)
    }

    /// Rebuild a multi-dim shape from a flat tensor with recorded splits
    /// (the inverse of `flatten`; there is no wholesale reshape in the
    /// recorded vocabulary).
    pub(crate) fn unflatten_to(mut self, dims: &[IntExpr]) -> GraphTensor {
        assert_eq!(self.rank(), 1, "unflatten_to starts from a flat tensor");
        for axis in 0..dims.len().saturating_sub(1) {
            let inner: IntExpr = dims[axis + 1..]
                .iter()
                .copied()
                .fold(IntExpr::from(1), |acc, d| acc * d)
                .simplify();
            self = self.split_dims(axis, inner);
        }
        self
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

        // Compute input strides (row-major contiguous)
        let dims = self.dims();
        let n = dims.len();
        let mut in_strides = vec![IntExpr::from(1); n];
        let mut acc = IntExpr::from(1);
        for (dim, in_stride) in dims.iter().zip(&mut in_strides).rev() {
            *in_stride = acc;
            acc *= dim;
        }

        // Per-dim window counts
        let mut win = Vec::with_capacity(n);
        for (((dim, k), s), d) in dims.iter().zip(&kernel).zip(&strides).zip(&dilation) {
            let effective_window = *d * (*k - 1) + 1;
            win.push((*dim - effective_window).floor_div(s) + 1);
        }

        // [win..., kernel...]
        let mut final_shape: Vec<IntExpr> = win.into_iter().map(|e| e.simplify()).collect();
        final_shape.extend(kernel.iter().copied());

        // Structure-preserving seam (see SliceView): the per-axis window
        // structure stays first-class; UnfoldView::to_egglog reproduces the
        // legacy flat iota+gather lowering for the existing pipeline.
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
                    let at = self.id.index();
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
        let operand = (self.id, self.dims());
        let id = self
            .graph()
            .logical
            .view_op(&operand, &entries, final_shape.clone(), self.dtype)
            .expect("logical view insertion failed");
        GraphTensor::from_id(id, final_shape, self.graph_ref, self.dtype)
    }

    /// Take a slice of a tensor along multiple dimensions.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
    /// let b = a.slice((2..4, 1..)); // 2x9 tensor
    /// assert_eq!(b.dims(), vec![IntExpr::from(2), IntExpr::from(9)]);
    /// ```
    pub fn slice(mut self, slice: impl ToSlice) -> GraphTensor {
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
                .expect("logical view insertion failed");
            GraphTensor::from_id(id, new_dims, self.graph_ref, self.dtype)
        } else {
            // No start slices so no iota needed, just reduce the shape down
            let mut new_dims = self.dims();
            for (sh, (_, end)) in new_dims.iter_mut().zip(&ranges) {
                *sh = sh.min(*end);
            }
            let current_dims = self.dims();
            self.id = self
                .graph()
                .logical
                .apply_movement(
                    &(self.id, current_dims),
                    crate::graph::Movement::Shrink {
                        new_dims: new_dims.clone(),
                    },
                )
                .expect("logical movement insertion failed");
            for (sh, new_dim) in self.dims.iter_mut().zip(new_dims) {
                *sh = new_dim;
            }
            self
        }
    }

    /// Take a slice of a tensor along a dimension.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
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
                .expect("logical view insertion failed");
        }
        let clamped =
            GraphTensor::from_id(clamped_id, out_dims.clone(), self.graph_ref, self.dtype);

        let mask_id = self
            .graph()
            .logical
            .record_mask_iota(&befores, &afters, &dims)
            .expect("logical mask iota insertion failed");
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
    use crate::{
        frontend::{binary::tests::test_binary, unary::tests::test_unary},
        prelude::*,
        tests::assert_exact,
    };
    use candle_core::{IndexOp, Tensor};
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
        let inp = cx.tensor((80, 3000));
        let out = inp.pad(((0, 0), (1, 1)), 0.).unfold((1, 3), (1, 2), (1, 1));
        assert_eq!(out.dims(), &[80, 1500, 1, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let mut cx = Graph::new();
        let inp = cx.tensor((2, 2, 3));
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
        let a = cx.tensor((2, 3));
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
        let a = cx.tensor((2, 3));
        let repeated = (a.repeat((2, 2)) * 1.0).output();

        let rt = crate::test_support::run_reference(
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
}
