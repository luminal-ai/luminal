//! Movement-adjacent lowerings (port batch 2): flip, diagonal,
//! diagonal_scatter, unfold, narrow/unbind/split, repeat_interleave,
//! constant_pad_nd, and the order-statistics family
//! (topk/sort/argsort/cummax/cummin/median).
//!
//! This branch proof-gates plain Int tensor arithmetic, so every ordering
//! here is built from IntExpr coordinate functions fed to coordinate-form
//! gathers; runtime index tensors only ever appear as gather coordinates.
#![allow(dead_code)]

use anyhow::{Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{broadcast_binary, normalize_dim, normalize_slice_bound};
use crate::pt2_schema::Node;

/// Flip dimensions must be in range and unique.
fn normalize_flip_dims(dims: &[i64], rank: usize) -> Result<Vec<usize>> {
    let mut normalized = Vec::with_capacity(dims.len());
    for &dim in dims {
        anyhow::ensure!(
            dim >= -(rank as i64) && dim < rank as i64,
            "flip dimension {dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(dim, rank);
        anyhow::ensure!(
            !normalized.contains(&dim),
            "flip dimensions must be unique, got {dims:?}"
        );
        normalized.push(dim);
    }
    Ok(normalized)
}

/// `diagonal`/`diagonal_scatter` dimension pair.
fn normalize_diagonal_dims(dim1: i64, dim2: i64, rank: usize) -> Result<(usize, usize)> {
    anyhow::ensure!(
        rank >= 2,
        "diagonal expects an input with at least two dimensions"
    );
    for dim in [dim1, dim2] {
        anyhow::ensure!(
            dim >= -(rank as i64) && dim < rank as i64,
            "diagonal dimension {dim} out of range for rank {rank}"
        );
    }
    let dims = (normalize_dim(dim1, rank), normalize_dim(dim2, rank));
    anyhow::ensure!(dims.0 != dims.1, "diagonal dimensions must be different");
    Ok(dims)
}

/// Float dtypes that can hold NaN (the only dtypes whose comparisons are
/// not total, so the only ones needing explicit extremum ordering).
fn dtype_can_nan(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F16 | DType::Bf16 | DType::F32 | DType::F64 | DType::TF32
    )
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Shared helpers
    // ---------------------------------------------------------------

    /// A rank-0 constant of the requested storage dtype (Int paths avoid the
    /// refused float->int cast).
    fn cast_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::I64 => self.cx.constant_i64(value as i64),
            DType::Int => self.cx.constant_i32(value as i64),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        }
    }

    /// Broadcast `mask`, `a`, `b`; select `a` where the mask is true.
    ///
    /// Selection goes through a packed gather, never arithmetic masking:
    /// `0 * inf` and `0 * NaN` are NaN, so `a*m + b*(1-m)` corrupts IEEE
    /// values in exactly the NaN/Inf cases this file must order. The packed
    /// layout places `b` at flat `2k` and `a` at `2k+1`; the runtime mask is
    /// used as the low gather coordinate and is never an arithmetic operand.
    /// (Same construction as the parked translator's complex select.)
    fn order_select(&mut self, mask: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = broadcast_binary(a, b);
        let (a, mask) = broadcast_binary(a, mask);
        let shape = a.dims();
        let rank = shape.len();
        let strides: Vec<IntExpr> = (0..rank)
            .map(|d| {
                shape[d + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, size| acc * *size)
            })
            .collect();
        let flat = |c: &[IntExpr]| -> IntExpr {
            (0..rank).fold(IntExpr::from(0), |acc, d| acc + c[d] * strides[d])
        };
        let even = self.cx.iota(shape.clone(), |c| flat(c) * IntExpr::from(2));
        let odd = self.cx.iota(shape.clone(), |c| {
            flat(c) * IntExpr::from(2) + IntExpr::from(1)
        });
        let numel = shape.iter().fold(IntExpr::from(1), |acc, size| acc * *size);
        let zeros = self
            .cast_scalar(0.0, a.dtype)
            .expand_rhs(vec![numel, IntExpr::from(2)]);
        let packed = b.scatter1d(even, zeros);
        let packed = a.scatter1d(odd, packed);
        let flat_coord = self.cx.iota(shape, |c| flat(c));
        packed.gather(&[flat_coord, mask.cast(DType::Int)])
    }

    /// Coordinate-form gather along `axis`. `indices` may be full-rank (one
    /// index per data coordinate) or reduced-rank (the axis dropped, one
    /// index per output coordinate) — the latter is what single-slot
    /// selection needs.
    fn order_gather(
        &mut self,
        value: GraphTensor,
        indices: GraphTensor,
        axis: usize,
    ) -> GraphTensor {
        let out_shape = indices.dims();
        let data_rank = value.rank();
        let index_rank = indices.rank();
        assert!(
            index_rank == data_rank || index_rank + 1 == data_rank,
            "gather indices rank {index_rank} must be {data_rank} or {}",
            data_rank - 1
        );
        let mut coords = Vec::with_capacity(data_rank);
        let mut out_axis = 0usize;
        for axis_index in 0..data_rank {
            if axis_index == axis {
                coords.push(indices);
            } else {
                let position = if index_rank == data_rank {
                    axis_index
                } else {
                    out_axis
                };
                out_axis += 1;
                coords.push(self.cx.iota(out_shape.clone(), move |c| c[position]));
            }
        }
        value.gather(&coords)
    }

    /// Shift a tensor right (towards larger indices) by `offset` along
    /// `axis`, clamping the prefix lanes to index 0, plus the validity mask
    /// for those lanes. Built from coordinate iotas so no Int tensor
    /// arithmetic is recorded.
    fn order_shift(
        &mut self,
        value: GraphTensor,
        axis: usize,
        offset: usize,
    ) -> (GraphTensor, GraphTensor) {
        let dims = value.dims();
        let rank = dims.len();
        let mut coords = Vec::with_capacity(rank);
        for axis_index in 0..rank {
            if axis_index == axis {
                let off = IntExpr::from(offset);
                coords.push(self.cx.iota(dims.clone(), move |c| {
                    (c[axis] - off).max(IntExpr::from(0i32))
                }));
            } else {
                coords.push(self.cx.iota(dims.clone(), move |c| c[axis_index]));
            }
        }
        let shifted = value.gather(&coords);
        let positions = self.cx.iota(dims.clone(), move |c| c[axis]);
        let off = self.cx.constant_i32(offset).expand_rhs(dims);
        let valid = positions.ge(off);
        (shifted, valid)
    }

    /// A sort key that the frontend's `stable_argsort` accepts. That helper
    /// hardcodes `* 1.0`/`+ 0.0`, which refuses an Int source, so integers
    /// sort through F64 (exact to 2^53 — stated assumption).
    fn order_sort_key(&mut self, value: GraphTensor) -> GraphTensor {
        match value.dtype {
            DType::Bool => value.cast(DType::F32),
            DType::Int
            | DType::I64
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::I4
            | DType::U4 => value.cast(DType::F64),
            _ => value,
        }
    }

    /// Bind a node's declared outputs by name (handles the single-entry
    /// `as_tensors` tuple and the multi-entry forms).
    fn movement_bind_outputs(&mut self, node: &Node, values: Vec<GraphTensor>) -> Result<()> {
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            names.len() == values.len(),
            "`{}` declared {} outputs but produced {}",
            node.target,
            names.len(),
            values.len()
        );
        for (name, value) in names.into_iter().zip(values) {
            self.values.insert(name, value);
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Movement
    // ---------------------------------------------------------------

    pub(super) fn translate_flip(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "flip on a rank-0 tensor is not ported");
        let axes = normalize_flip_dims(&self.get_ints_arg(node, 1)?, rank)?;
        if axes.is_empty() {
            return Ok(x);
        }
        let shape = x.dims();
        let strides: Vec<IntExpr> = (0..rank)
            .map(|d| {
                shape[d + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, size| acc * *size)
            })
            .collect();
        // Reversed coordinate iota, flattened to a single gather index.
        let index = self.cx.iota(shape.clone(), |c| {
            (0..rank).fold(IntExpr::from(0), |acc, d| {
                let coord = if axes.contains(&d) {
                    shape[d] - 1 - c[d]
                } else {
                    c[d]
                };
                acc + coord * strides[d]
            })
        });
        Ok(x.gather1d(index))
    }

    pub(super) fn translate_diagonal(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        let offset = self.get_int_arg(node, 1).unwrap_or(0);
        let (dim1, dim2) = normalize_diagonal_dims(
            self.get_int_arg(node, 2).unwrap_or(0),
            self.get_int_arg(node, 3).unwrap_or(1),
            rank,
        )?;
        let out_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            out_shape.len() + 1 == rank,
            "diagonal output rank {} does not match input rank {rank}",
            out_shape.len()
        );
        let last = out_shape.len() - 1;
        let mut next = 0usize;
        let mut coords = Vec::with_capacity(rank);
        for axis in 0..rank {
            if axis == dim1 {
                // Negative offsets start along dim1.
                coords.push(self.cx.iota(out_shape.clone(), move |c| {
                    c[last] + IntExpr::from((-offset).max(0))
                }));
            } else if axis == dim2 {
                coords.push(self.cx.iota(out_shape.clone(), move |c| {
                    c[last] + IntExpr::from(offset.max(0))
                }));
            } else {
                let position = next;
                next += 1;
                coords.push(self.cx.iota(out_shape.clone(), move |c| c[position]));
            }
        }
        Ok(x.gather(&coords))
    }

    pub(super) fn translate_diagonal_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?;
        let offset = self.get_int_arg(node, 2).unwrap_or(0);
        let (dim1, dim2) = normalize_diagonal_dims(
            self.get_int_arg(node, 3).unwrap_or(0),
            self.get_int_arg(node, 4).unwrap_or(1),
            destination.rank(),
        )?;
        anyhow::ensure!(
            destination.dtype == source.dtype,
            "diagonal_scatter requires matching dtypes, got {:?} and {:?}",
            destination.dtype,
            source.dtype
        );
        let src_shape = source.dims();
        anyhow::ensure!(
            src_shape.len() + 1 == destination.rank(),
            "diagonal_scatter source rank {} does not match destination rank {}",
            src_shape.len(),
            destination.rank()
        );
        let last = src_shape.len() - 1;
        let mut next = 0usize;
        let mut coords = Vec::with_capacity(destination.rank());
        for axis in 0..destination.rank() {
            if axis == dim1 {
                coords.push(self.cx.iota(src_shape.clone(), move |c| {
                    c[last] + IntExpr::from((-offset).max(0))
                }));
            } else if axis == dim2 {
                coords.push(self.cx.iota(src_shape.clone(), move |c| {
                    c[last] + IntExpr::from(offset.max(0))
                }));
            } else {
                let position = next;
                next += 1;
                coords.push(self.cx.iota(src_shape.clone(), move |c| c[position]));
            }
        }
        Ok(destination.scatter(&coords, source))
    }

    pub(super) fn translate_unfold(&mut self, node: &Node) -> Result<GraphTensor> {
        // `F.unfold` exports as `aten.im2col.default` on this torch version;
        // `Tensor.unfold` stays `aten.unfold.default`. Both are ported here
        // because they share the same "sliding windows" lowering.
        if node.target.ends_with("im2col.default") {
            return self.translate_im2col(node);
        }
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "unfold on a rank-0 tensor is not ported");
        let raw_dim = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "unfold dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let size = self.get_int_arg(node, 2)?;
        let step = self.get_int_arg(node, 3)?;
        anyhow::ensure!(size >= 0, "unfold size must be nonnegative, got {size}");
        anyhow::ensure!(step > 0, "unfold step must be positive, got {step}");
        // The declared output shape carries the window count; author the read
        // map directly instead of relying on the symbolic window contract.
        let out_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            out_shape.len() == rank + 1,
            "unfold output rank {} does not match input rank {rank}",
            out_shape.len()
        );
        let kernel_axis = rank;
        let mut coords = Vec::with_capacity(rank);
        for axis in 0..rank {
            coords.push(self.cx.iota(out_shape.clone(), move |c| {
                if axis == dim {
                    c[axis] * IntExpr::from(step) + c[kernel_axis]
                } else {
                    c[axis]
                }
            }));
        }
        Ok(x.gather(&coords))
    }

    /// `aten.im2col.default(self, kernel_size, dilation, padding, stride)`:
    /// the `(N, C, *spatial)` patch extraction behind `F.unfold`, producing
    /// `[N, C * prod(kernel), prod(out_spatial)]`.
    ///
    /// The frontend's full-rank [`GraphTensor::unfold`] yields
    /// `[win_N, win_C, win_spatial..., k_N, k_C, k_spatial...]`. Permuting the
    /// kernel dims next to the channel dims and merging gives PyTorch's
    /// channel-major `(C, k...)` flattening and `(out_spatial...)` L order,
    /// and the declared output shape pins the final reshape.
    fn translate_im2col(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(
            rank == 4,
            "im2col expects a rank-4 (N, C, H, W) input, got rank {rank}"
        );
        let kernel = self.get_ints_arg(node, 1)?;
        let dilation = self.get_ints_arg(node, 2)?;
        let padding = self.get_ints_arg(node, 3)?;
        let stride = self.get_ints_arg(node, 4)?;
        anyhow::ensure!(
            kernel.len() == 2 && dilation.len() == 2 && padding.len() == 2 && stride.len() == 2,
            "im2col expects two-element kernel/dilation/padding/stride lists"
        );
        anyhow::ensure!(
            kernel.iter().all(|&k| k > 0),
            "im2col kernel must be positive, got {kernel:?}"
        );
        anyhow::ensure!(
            dilation.iter().all(|&d| d > 0),
            "im2col dilation must be positive, got {dilation:?}"
        );
        anyhow::ensure!(
            padding.iter().all(|&p| p >= 0),
            "im2col padding must be nonnegative, got {padding:?}"
        );
        anyhow::ensure!(
            stride.iter().all(|&s| s > 0),
            "im2col stride must be positive, got {stride:?}"
        );

        // ATen lists padding per spatial dim, applied symmetrically; the
        // `pad` pairs are in dim order (N, C, H, W).
        let padded = x.pad(
            vec![
                (IntExpr::from(0), IntExpr::from(0)),
                (IntExpr::from(0), IntExpr::from(0)),
                (IntExpr::from(padding[0]), IntExpr::from(padding[0])),
                (IntExpr::from(padding[1]), IntExpr::from(padding[1])),
            ],
            0.0,
        );

        let kernel_full = vec![1usize, 1, kernel[0] as usize, kernel[1] as usize];
        let stride_full = vec![1usize, 1, stride[0] as usize, stride[1] as usize];
        let dilation_full = vec![1usize, 1, dilation[0] as usize, dilation[1] as usize];
        let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
        // Window-major output: [N, C, out_h, out_w, 1, 1, k_h, k_w]. The two
        // kernel axes of the untouched N/C dims are singleton.
        let squeezed = unfolded.squeeze(5).squeeze(4);
        // [N, C, out_h, out_w, k_h, k_w] -> [N, C, k_h, k_w, out_h, out_w],
        // then merge to PyTorch's channel-major (C, k...) / (out...) order.
        let patches = squeezed
            .permute(vec![0usize, 1, 4, 5, 2, 3])
            .merge_dims(1, 2)
            .merge_dims(1, 2)
            .merge_dims(2, 3);
        let out_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            out_shape.len() == 3,
            "im2col output rank {} does not match the expected 3",
            out_shape.len()
        );
        anyhow::ensure!(
            patches.dims() == out_shape,
            "im2col produced {:?} but the declared output is {:?}",
            patches.dims(),
            out_shape
        );
        Ok(patches)
    }

    pub(super) fn translate_narrow_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "narrow_copy on a rank-0 tensor is not ported");
        let raw_dim = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "narrow_copy dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let start = self.get_int_arg(node, 2)?;
        let length = self.get_int_arg(node, 3)?;
        anyhow::ensure!(
            length >= 0,
            "narrow_copy length must be nonnegative, got {length}"
        );
        let start = normalize_slice_bound(IntExpr::from(start), x.dims()[dim]);
        let end = start + IntExpr::from(length);
        Ok(x.slice_along(start..end, dim))
    }

    pub(super) fn translate_unbind_copy(&mut self, node: &Node) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "unbind_copy on a rank-0 tensor is not ported");
        let raw_dim = self.get_int_arg(node, 1).unwrap_or(0);
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "unbind_copy dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let names = Self::tensor_output_names(node);
        let axis_size = x.dims()[dim]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("unbind_copy requires a concrete unbound dimension"))?;
        anyhow::ensure!(
            names.len() == axis_size,
            "unbind_copy declared {} outputs for an axis of size {axis_size}",
            names.len()
        );
        for (index, name) in names.into_iter().enumerate() {
            let selected = x.slice_along(index..index + 1, dim).squeeze(dim);
            self.values.insert(name, selected);
        }
        Ok(())
    }

    pub(super) fn translate_split_with_sizes(&mut self, node: &Node) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(
            rank > 0,
            "split_with_sizes on a rank-0 tensor is not ported"
        );
        let sizes = self.get_ints_arg(node, 1)?;
        let raw_dim = self.get_int_arg(node, 2).unwrap_or(0);
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "split_with_sizes dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            names.len() == sizes.len(),
            "split_with_sizes declared {} outputs for {} sizes",
            names.len(),
            sizes.len()
        );
        let mut start = 0i64;
        for (name, size) in names.into_iter().zip(sizes) {
            anyhow::ensure!(
                size >= 0,
                "split_with_sizes sizes must be nonnegative, got {size}"
            );
            let end = start + size;
            let chunk = x.slice_along(IntExpr::from(start)..IntExpr::from(end), dim);
            self.values.insert(name, chunk);
            start = end;
        }
        Ok(())
    }

    pub(super) fn translate_repeat_interleave(&mut self, node: &Node) -> Result<GraphTensor> {
        if node.target.ends_with("self_int") {
            return self.repeat_interleave_self_int(node);
        }
        anyhow::ensure!(
            !node.target.ends_with("self_Tensor") && !node.target.ends_with("self.Tensor"),
            "{} is not ported (value-dependent repeat pattern)",
            node.target
        );
        // `aten.repeat_interleave.Tensor(repeats, output_size?)` produces the
        // repeated index tensor directly. Interval for input i is
        // [ends[i-1], ends[i]); counting ends <= position gives i.
        let repeats = self.operand(&node.inputs[0])?.cast(DType::Int).flatten();
        let out_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            out_shape.len() == 1,
            "repeat_interleave.Tensor expects a rank-1 output, got {out_shape:?}"
        );
        let out_len = out_shape[0];
        let count = repeats.dims()[0];
        if count.to_usize() == Some(0) {
            return Ok(self.cast_scalar(0.0, DType::I64).expand_rhs(vec![out_len]));
        }
        // F32 carries the inclusive prefix sums exactly (counts < 2^24).
        let ends = repeats.cast(DType::F32).cumsum(0);
        let ends = ends.expand_dim(0, out_len);
        let positions = self
            .cx
            .arange(out_len)
            .cast(DType::F32)
            .expand_dim(1, count);
        let counts = ends.le(positions).cast(DType::Int).sum(1);
        Ok(counts.cast(DType::I64))
    }

    fn repeat_interleave_self_int(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let repeats = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            repeats > 0,
            "repeat_interleave requires positive repeats, got {repeats}"
        );
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_int()));
        let (base, dim) = match raw_dim {
            Some(dim) => {
                let rank = x.rank();
                anyhow::ensure!(
                    rank > 0,
                    "repeat_interleave on a rank-0 tensor is not ported"
                );
                anyhow::ensure!(
                    dim >= -(rank as i64) && dim < rank as i64,
                    "repeat_interleave dimension {dim} out of range for rank {rank}"
                );
                (x, normalize_dim(dim, rank))
            }
            None => (x.flatten(), 0),
        };
        let out_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            out_shape.len() == base.rank(),
            "repeat_interleave output rank {} does not match input rank {}",
            out_shape.len(),
            base.rank()
        );
        let mut coords = Vec::with_capacity(out_shape.len());
        for axis in 0..out_shape.len() {
            coords.push(self.cx.iota(out_shape.clone(), move |c| {
                if axis == dim {
                    c[axis].floor_div(IntExpr::from(repeats))
                } else {
                    c[axis]
                }
            }));
        }
        Ok(base.gather(&coords))
    }

    pub(super) fn translate_constant_pad_nd(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(x.dtype);
        anyhow::ensure!(
            dtype == x.dtype,
            "constant_pad_nd changed dtype from {:?} to {dtype:?}",
            x.dtype
        );
        let raw = self.get_ints_arg(node, 1)?;
        let rank = x.rank();
        // ATen lists the last dimension first: [last_left, last_right, ...].
        let mut pairs: Vec<(IntExpr, IntExpr)> = Vec::new();
        let mut index = 0usize;
        while index < raw.len() {
            let left = raw[index];
            let right = raw.get(index + 1).copied().unwrap_or(0);
            pairs.push((IntExpr::from(left), IntExpr::from(right)));
            index += 2;
        }
        anyhow::ensure!(
            pairs.len() <= rank,
            "constant_pad_nd received {} padded dimensions for rank-{rank} input",
            pairs.len()
        );
        pairs.reverse();
        let mut padding = vec![(IntExpr::from(0), IntExpr::from(0)); rank - pairs.len()];
        padding.extend(pairs);
        // `aten.pad.default` carries (self, pad, mode, value?) while
        // `constant_pad_nd` carries (self, pad, value). Only constant mode
        // is this lowering; other modes bail by name.
        if let Some(input) = node.inputs.get(2)
            && let crate::pt2_schema::Argument::Other(value) = &input.arg
            && let Some(mode) = value.as_str()
        {
            anyhow::ensure!(mode == "constant", "pad mode {mode:?} is not ported");
        }
        let value = self
            .named_float_arg(node, "value")
            .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_float()))
            .or_else(|| node.inputs.get(3).and_then(|input| input.arg.as_float()))
            .unwrap_or(0.0);
        let fill = self.cast_scalar(value, x.dtype);
        Ok(x.pad_with(padding, fill))
    }

    // ---------------------------------------------------------------
    // Order statistics
    // ---------------------------------------------------------------

    pub(super) fn translate_topk(&mut self, node: &Node) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "topk on a rank-0 tensor is not ported");
        let k = match self
            .named_int_arg(node, "k")
            .or_else(|| node.inputs.get(1).and_then(|input| input.arg.as_int()))
        {
            Some(k) => k,
            None => bail!("topk with a tensor `k` (topk.values) is not ported"),
        };
        anyhow::ensure!(k >= 0, "topk k must be nonnegative, got {k}");
        let k = k as usize;
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_int()))
            .unwrap_or(-1);
        let dim = normalize_dim(raw_dim, rank);
        let largest = self
            .named_bool_arg(node, "largest")
            .or_else(|| node.inputs.get(3).and_then(|input| input.arg.as_bool()))
            .unwrap_or(true);
        // `sorted=false` permits any order, and a stable descending/ascending
        // sort is a valid order.
        let key = self.order_sort_key(x);
        let order = key.stable_argsort(dim, largest);
        let indices = order.slice_along(0..k, dim);
        let values = self.order_gather(x, indices, dim);
        let indices = indices.cast(DType::I64);
        self.movement_bind_outputs(node, vec![values, indices])
    }

    pub(super) fn translate_sort(&mut self, node: &Node, stable: bool) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "sort on a rank-0 tensor is not ported");
        // `sort.stable` inserts a keyword-only `stable` before dim.
        let dim_slot = if stable { 2 } else { 1 };
        let descending_slot = if stable { 3 } else { 2 };
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| {
                node.inputs
                    .get(dim_slot)
                    .and_then(|input| input.arg.as_int())
            })
            .unwrap_or(-1);
        let dim = normalize_dim(raw_dim, rank);
        let descending = self
            .named_bool_arg(node, "descending")
            .or_else(|| {
                node.inputs
                    .get(descending_slot)
                    .and_then(|input| input.arg.as_bool())
            })
            .unwrap_or(false);
        // `stable_argsort` is always stable, which `stable=false` permits.
        let key = self.order_sort_key(x);
        let order = key.stable_argsort(dim, descending);
        let values = self.order_gather(x, order, dim);
        let indices = order.cast(DType::I64);
        self.movement_bind_outputs(node, vec![values, indices])
    }

    pub(super) fn translate_argsort(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "argsort on a rank-0 tensor is not ported");
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| node.inputs.get(1).and_then(|input| input.arg.as_int()))
            .unwrap_or(-1);
        let dim = normalize_dim(raw_dim, rank);
        let descending = self
            .named_bool_arg(node, "descending")
            .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_bool()))
            .unwrap_or(false);
        Ok(self
            .order_sort_key(x)
            .stable_argsort(dim, descending)
            .cast(DType::I64))
    }

    pub(super) fn translate_cumextremum(&mut self, node: &Node, max: bool) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let name = if max { "cummax" } else { "cummin" };
        anyhow::ensure!(x.dtype != DType::Bool, "{name} on Bool is not ported");
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "{name} on a rank-0 tensor is not ported");
        let axis = normalize_dim(self.get_int_arg(node, 1)?, rank);
        let value_dims = x.dims();
        let length = value_dims[axis]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("{name} requires a concrete scan dimension"))?;
        let float = dtype_can_nan(x.dtype);
        let mut values = x;
        let mut indices = self.axis_positions(&value_dims, axis);
        let mut offset = 1usize;
        while offset < length {
            let (left_values, valid) = self.order_shift(values, axis, offset);
            let left_indices = self.order_shift(indices, axis, offset).0;
            let ordered_left_wins = if max {
                values.lt(left_values)
            } else {
                left_values.lt(values)
            };
            // PyTorch keeps the later index on ties; a NaN on the left only
            // beats an ordered value, so a prior NaN propagates while a later
            // NaN still takes over.
            let left_wins = if float {
                let left_nan = self.is_nan(left_values);
                let right_nan = self.is_nan(values);
                let right_nan_not = self.bool_not(right_nan);
                let left_nan_only = self.bool_and(left_nan, right_nan_not);
                self.bool_or(ordered_left_wins, left_nan_only)
            } else {
                ordered_left_wins
            };
            let take_left = self.bool_and(left_wins, valid);
            values = self.order_select(take_left, left_values, values);
            indices = self.order_select(take_left, left_indices, indices);
            offset *= 2;
        }
        let indices = indices.cast(DType::I64);
        self.movement_bind_outputs(node, vec![values, indices])
    }

    /// `(first NaN position, any-NaN)` along `axis`, both over the reduced
    /// shape. Descending stable sort of the 0/1 mask puts 1s first; slot 0
    /// is the earliest NaN.
    fn order_first_nan(&mut self, value: GraphTensor, axis: usize) -> (GraphTensor, GraphTensor) {
        let nan = self.is_nan(value);
        let order = nan.cast(DType::F32).stable_argsort(axis, true);
        let first = order.slice_along(0..1, axis).squeeze(axis);
        let count = nan.cast(DType::Int).sum(axis);
        let zero = self.cx.constant_i32(0).expand_rhs(count.dims());
        let has_nan = count.gt(zero);
        (first, has_nan)
    }

    pub(super) fn translate_median(&mut self, node: &Node, nan: bool) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            !names.is_empty(),
            "`{}` is missing its output name",
            node.target
        );
        let dim_variant = node.target.ends_with(".dim");

        if x.rank() == 0 {
            self.values.insert(names[0].clone(), x);
            if dim_variant {
                anyhow::ensure!(
                    names.len() >= 2,
                    "`{}` is missing its index output",
                    node.target
                );
                let zero = self.cast_scalar(0.0, DType::I64);
                self.values.insert(names[1].clone(), zero);
            }
            return Ok(());
        }

        let keepdim = if dim_variant {
            self.named_bool_arg(node, "keepdim")
                .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_bool()))
                .unwrap_or(false)
        } else {
            false
        };
        let (base, axis) = if dim_variant {
            (x, normalize_dim(self.get_int_arg(node, 1)?, x.rank()))
        } else {
            (x.flatten(), 0)
        };
        let axis_len = base.dims()[axis];
        anyhow::ensure!(
            axis_len.to_usize() != Some(0),
            "median over an empty axis is not ported"
        );
        let reduced_shape: Vec<IntExpr> = base
            .dims()
            .into_iter()
            .enumerate()
            .filter_map(|(dim, size)| (dim != axis).then_some(size))
            .collect();
        let float = dtype_can_nan(base.dtype);

        let (median_index, has_nan) = if nan && float {
            // (is_nan, value) ascending with NaNs last: sort by value with
            // NaN mapped to +inf, then stably reorder by the NaN mask. This
            // stays exact when real +inf values are present, unlike the
            // single-key +inf substitution.
            let nan_mask = self.is_nan(base);
            let inf = self.constant_like(base, f64::INFINITY);
            let key = self.order_select(nan_mask, inf, base);
            let value_order = key.stable_argsort(axis, false);
            let nan_order = nan_mask.cast(DType::F32).stable_argsort(axis, false);
            let sort_order = self.order_gather(value_order, nan_order, axis);
            let nan_count = nan_mask.cast(DType::Int).sum(axis);
            let length = self
                .cx
                .constant_i32(axis_len)
                .cast(DType::F32)
                .expand_rhs(reduced_shape.clone());
            let valid = length - nan_count.cast(DType::F32);
            let zero = self.cx.constant_f32(0.0).expand_rhs(reduced_shape.clone());
            // Torch's nanmedian takes the lower of the two middle valid
            // values: slot (valid_count - 1) / 2.
            let median_pos = ((valid - 1.0).maximum(zero) * 0.5)
                .floor()
                .trunc_cast(DType::Int);
            let median_index = self.order_gather(sort_order, median_pos, axis);
            (median_index, None)
        } else {
            let key = self.order_sort_key(base);
            let sort_order = key.stable_argsort(axis, false);
            let median_pos = self
                .cx
                .constant_i32((axis_len - IntExpr::from(1)).floor_div(IntExpr::from(2)))
                .expand_rhs(reduced_shape.clone());
            let median_index = self.order_gather(sort_order, median_pos, axis);
            let has_nan = float.then(|| {
                let nan_mask = self.is_nan(base);
                let count = nan_mask.cast(DType::Int).sum(axis);
                let zero = self.cx.constant_i32(0).expand_rhs(count.dims());
                count.gt(zero)
            });
            (median_index, has_nan)
        };

        let mut value = self.order_gather(base, median_index, axis);
        let mut index = median_index;
        if let Some(has_nan) = has_nan {
            // A NaN anywhere makes the median NaN (torch.median), reported at
            // the first NaN's position.
            let nan_value = self
                .floating_scalar(f64::NAN, base.dtype)
                .expand_rhs(value.dims());
            value = self.order_select(has_nan, nan_value, value);
            let (first_nan, _) = self.order_first_nan(base, axis);
            index = self.order_select(has_nan, first_nan, index);
        }
        let value = if keepdim {
            value.expand_dim(axis, 1usize)
        } else {
            value
        };
        self.values.insert(names[0].clone(), value);
        if dim_variant {
            anyhow::ensure!(
                names.len() >= 2,
                "`{}` is missing its index output",
                node.target
            );
            let index = if keepdim {
                index.expand_dim(axis, 1usize)
            } else {
                index
            };
            self.values.insert(names[1].clone(), index.cast(DType::I64));
        }
        Ok(())
    }
}
