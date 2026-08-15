use anyhow::{Context, Result, bail};
use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_expr::{ExprBounds, canonical_equal_expr, sym_char_ranges};
use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

use super::movement_dynamic::{logical_flat_indices, row_major_strides};

const SCATTER_INPUT_ARG: usize = 0;
const SCATTER_DIM_ARG: usize = 1;
const SCATTER_INDEX_ARG: usize = 2;
const SCATTER_VALUE_ARG: usize = 3;

pub(crate) fn normalize_flip_dims(dims: &[i64], rank: usize) -> Result<Vec<usize>> {
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

pub(crate) fn normalize_diagonal_dims(dim1: i64, dim2: i64, rank: usize) -> Result<(usize, usize)> {
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

pub(crate) fn flip_indices(input: GraphTensor, dims: &[usize]) -> GraphTensor {
    let shape = input.dims();
    let strides = row_major_strides(&shape);
    let contributions: Vec<Expression> = strides
        .iter()
        .enumerate()
        .map(|(axis, &stride)| {
            let coordinate = if dims.contains(&axis) {
                shape[axis] - 1 - Expression::from('z')
            } else {
                Expression::from('z')
            };
            coordinate * stride
        })
        .collect();
    logical_flat_indices(input.graph(), &shape, &contributions, 0.into())
}

pub(crate) fn diagonal_indices(
    input: GraphTensor,
    output_shape: &[Expression],
    offset: i64,
    dim1: usize,
    dim2: usize,
) -> Result<GraphTensor> {
    let input_shape = input.dims();
    anyhow::ensure!(
        output_shape.len() + 1 == input_shape.len(),
        "diagonal output rank {} does not match input rank {}",
        output_shape.len(),
        input_shape.len()
    );
    let strides = row_major_strides(&input_shape);
    let mut contributions: Vec<Expression> = (0..input_shape.len())
        .filter(|&axis| axis != dim1 && axis != dim2)
        .map(|axis| Expression::from('z') * strides[axis])
        .collect();
    contributions.push(Expression::from('z') * (strides[dim1] + strides[dim2]));

    // Positive offsets start along dim2; negative offsets start along dim1.
    // Express negation symbolically so the full signed ATen offset range does
    // not overflow Rust while translating an empty, out-of-bounds diagonal.
    let base = if offset >= 0 {
        Expression::from(offset) * strides[dim2]
    } else {
        Expression::from(offset) * Expression::from(-1) * strides[dim1]
    };
    Ok(logical_flat_indices(
        input.graph(),
        output_shape,
        &contributions,
        base,
    ))
}

pub(crate) fn diagonal_scatter_tensor(
    destination: GraphTensor,
    source: GraphTensor,
    offset: i64,
    dim1: usize,
    dim2: usize,
) -> Result<GraphTensor> {
    anyhow::ensure!(
        destination.dtype == source.dtype,
        "diagonal_scatter requires matching dtypes, got {:?} and {:?}",
        destination.dtype,
        source.dtype
    );
    let indices = diagonal_indices(destination, &source.dims(), offset, dim1, dim2)?;
    Ok(source.scatter(indices, destination))
}

pub(crate) fn index_select_tensor(
    input: GraphTensor,
    index: GraphTensor,
    dim: usize,
    output_shape: &[Expression],
) -> Result<GraphTensor> {
    if input.shape.is_empty() {
        anyhow::ensure!(
            output_shape.is_empty(),
            "scalar index_select must produce a scalar output"
        );
        return Ok(input);
    }
    anyhow::ensure!(
        index.shape.len() <= 1,
        "index_select index must be rank 0 or 1, got rank {}",
        index.shape.len()
    );
    Ok(super::movement_dynamic::pt2_index_select(
        input,
        index,
        dim,
        output_shape,
    ))
}

pub(crate) fn unfold_tensor(
    input: GraphTensor,
    dim: usize,
    size: i64,
    step: i64,
    output_shape: &[Expression],
) -> Result<GraphTensor> {
    anyhow::ensure!(size >= 0, "unfold size must be nonnegative, got {size}");
    anyhow::ensure!(step > 0, "unfold step must be positive, got {step}");
    if input.shape.is_empty() {
        anyhow::ensure!(
            output_shape.len() == 1,
            "scalar unfold must produce one output dimension"
        );
        anyhow::ensure!(size <= 1, "scalar unfold size cannot exceed 1");
        return Ok(input.expand_rhs(output_shape.to_vec()));
    }
    if let Some(dim_size) = input.shape.dims[dim].to_usize() {
        anyhow::ensure!(
            size as usize <= dim_size,
            "unfold size {size} exceeds dimension {dim} of length {dim_size}"
        );
    }

    let rank = input.shape.len();
    let mut kernel = vec![1usize; rank];
    let mut strides = vec![1usize; rank];
    kernel[dim] = size as usize;
    strides[dim] = step as usize;
    let mut result = input.unfold(kernel, strides, vec![1usize; rank]);

    // Core unfold returns [window_dims..., kernel_dims...]. Every kernel
    // dimension except the selected one is size 1, so remove those in reverse
    // order; the selected kernel dimension naturally becomes PyTorch's final
    // output dimension.
    for kernel_dim in (0..rank).rev() {
        if kernel_dim != dim {
            result = result.squeeze(rank + kernel_dim);
        }
    }
    anyhow::ensure!(
        result.shape.len() == output_shape.len(),
        "unfold produced rank {}, expected {}",
        result.shape.len(),
        output_shape.len()
    );
    for (actual, expected) in result.shape.dims.iter_mut().zip(output_shape) {
        *actual = *expected;
    }
    Ok(result)
}

fn normalize_concat_dims(
    lhs: &mut GraphTensor,
    rhs: &mut GraphTensor,
    skip_dim: Option<usize>,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) {
    for i in 0..lhs.shape.len() {
        if Some(i) == skip_dim {
            continue;
        }
        let lhs_dim = lhs.shape.dims[i];
        let rhs_dim = rhs.shape.dims[i];
        if let Some(canonical) = canonical_equal_expr(lhs_dim, rhs_dim, sym_ranges) {
            lhs.shape.dims[i] = canonical;
            rhs.shape.dims[i] = canonical;
        }
    }
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_reshape(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;

        let shape = if let Ok(target_shape) = self.get_ints_arg(node, 1) {
            resolve_neg1_dim(&target_shape, &a.shape.dims)
        } else {
            let exprs = self.get_exprs_arg(node, 1)?;
            resolve_neg1_dim_exprs(&exprs, &a.shape.dims)
        };

        let has_broadcast = a
            .shape
            .dims
            .iter()
            .zip(a.shape.strides.iter())
            .any(|(d, s)| s.to_usize() == Some(0) && d.to_usize() != Some(1));

        let a = if has_broadcast || !a.shape.is_contiguous() {
            a + 0.0
        } else {
            a
        };

        let new_shape = ShapeTracker::new(shape);
        Ok(GraphTensor {
            id: a.id,
            graph_ref: a.graph_ref,
            shape: new_shape,
            dtype: a.dtype,
        })
    }

    /// `aten.repeat`: tile the tensor `repeats[d]` times along each dim.
    /// Leading entries beyond the input rank prepend new dims (torch
    /// semantics); the tiling itself delegates to the core `repeat` view.
    pub(crate) fn translate_repeat(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut t = self.get_input_tensor(node, 0)?;
        let repeats = self.get_ints_arg(node, 1)?;
        anyhow::ensure!(
            repeats.len() >= t.shape.len(),
            "repeat expects at least as many repeats ({}) as dims ({})",
            repeats.len(),
            t.shape.len()
        );
        anyhow::ensure!(
            repeats.iter().all(|&r| r >= 1),
            "repeat counts must be >= 1, got {repeats:?}"
        );
        for _ in 0..(repeats.len() - t.shape.len()) {
            t = t.unsqueeze(0);
        }
        Ok(t.repeat(repeats.iter().map(|&r| r as usize).collect::<Vec<_>>()))
    }

    pub(crate) fn translate_upsample_nearest2d(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let input_dimensions = input.dims();

        anyhow::ensure!(
            input_dimensions.len() == 4,
            "upsample_nearest2d expects a 4D (N, C, H, W) input, got {}D",
            input_dimensions.len()
        );

        let input_height = input_dimensions[2]
            .to_usize()
            .context("upsample_nearest2d requires a static input height")?;

        let input_width = input_dimensions[3]
            .to_usize()
            .context("upsample_nearest2d requires a static input width")?;

        let output_dimensions = self.output_meta_shape(node)?;

        anyhow::ensure!(
            output_dimensions.len() == 4,
            "upsample_nearest2d expects a 4D output, got {}D",
            output_dimensions.len()
        );

        let output_height = output_dimensions[2]
            .to_usize()
            .context("upsample_nearest2d requires a static output height")?;

        let output_width = output_dimensions[3]
            .to_usize()
            .context("upsample_nearest2d requires a static output width")?;

        anyhow::ensure!(
            input_height != 0 && input_width != 0 && output_height != 0 && output_width != 0,
            "upsample_nearest2d requires non-zero spatial dims \
               (in {input_height}x{input_width} -> out {output_height}x{output_width})"
        );

        // Optional explicit scale_factors (arg 2): ATen's general branch
        // indexes by floor(j / s) when scales are provided, which differs
        // from floor(j * in / out) when in * s is non-integral.
        let scales: Option<(f64, f64)> = node.inputs.get(2).and_then(|i| match &i.arg {
            Argument::Other(v) => {
                let a = v.get("as_floats")?.as_array()?;
                if a.len() == 2 {
                    Some((a[0].as_f64()?, a[1].as_f64()?))
                } else {
                    None
                }
            }
            _ => None,
        });

        let result =
            self.upsample_nearest_axis(input, 2, input_height, output_height, scales.map(|s| s.0))?;
        let result =
            self.upsample_nearest_axis(result, 3, input_width, output_width, scales.map(|s| s.1))?;
        Ok(result)
    }

    /// Nearest-neighbor resample of one axis. `out == in` and `out == 2*in`
    /// are ATen kernel fast paths that ignore the scale, and integer scales
    /// matching out/in are pure shape movement (`expand_dim` + `merge_dims`).
    /// Everything else gathers with `src = min(floor(j * scale_inv), in-1)`
    /// where scale_inv = 1/s when scales were provided else in/out — the
    /// float chain deliberately mirrors ATen's float32 index math.
    fn upsample_nearest_axis(
        &mut self,
        t: GraphTensor,
        axis: usize,
        in_dim: usize,
        out_dim: usize,
        scale: Option<f64>,
    ) -> Result<GraphTensor> {
        if out_dim.is_multiple_of(in_dim) {
            let k = out_dim / in_dim;
            let scale_matches = scale.is_none_or(|s| (s - k as f64).abs() < 1e-9);
            if k <= 2 || scale_matches {
                if k == 1 {
                    return Ok(t);
                }
                return Ok(t.expand_dim(axis + 1, k).merge_dims(axis, axis + 1));
            }
        }

        let scale_inv = scale.map_or(in_dim as f64 / out_dim as f64, |s| 1.0 / s) as f32;
        let mut idx = (self.graph.arange(out_dim).cast(DType::F32) * scale_inv)
            .minimum_f32((in_dim - 1) as f32)
            .cast(DType::Int);
        for (d, &dim) in t.dims().iter().enumerate() {
            if d != axis {
                idx = idx.expand_dim(d, dim);
            }
        }
        Ok(super::movement_dynamic::pt2_gather_elements(t, idx, axis))
    }

    pub(crate) fn translate_permute(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dims = self.get_ints_arg(node, 1)?;
        let axes: Vec<usize> = dims
            .iter()
            .map(|&d| normalize_dim(d, a.shape.len()))
            .collect();
        Ok(a.permute(axes))
    }

    pub(crate) fn translate_flip(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let dims = normalize_flip_dims(&self.get_ints_arg(node, 1)?, input.shape.len())?;
        Ok(input.gather(flip_indices(input, &dims)))
    }

    pub(crate) fn translate_diagonal(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let offset = self.get_int_arg(node, 1).unwrap_or(0);
        let (dim1, dim2) = normalize_diagonal_dims(
            self.get_int_arg(node, 2).unwrap_or(0),
            self.get_int_arg(node, 3).unwrap_or(1),
            input.shape.len(),
        )?;
        let output_shape = self.output_meta_shape(node)?;
        Ok(input.gather(diagonal_indices(input, &output_shape, offset, dim1, dim2)?))
    }

    pub(crate) fn translate_diagonal_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.get_input_tensor(node, 0)?;
        let source = self.get_input_tensor(node, 1)?;
        let offset = self.get_int_arg(node, 2).unwrap_or(0);
        let (dim1, dim2) = normalize_diagonal_dims(
            self.get_int_arg(node, 3).unwrap_or(0),
            self.get_int_arg(node, 4).unwrap_or(1),
            destination.shape.len(),
        )?;
        diagonal_scatter_tensor(destination, source, offset, dim1, dim2)
    }

    pub(crate) fn translate_index_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let raw_dim = self.get_int_arg(node, 1)?;
        let dim = if input.shape.is_empty() {
            anyhow::ensure!(
                raw_dim == 0 || raw_dim == -1,
                "index_select dimension {raw_dim} out of range for a scalar"
            );
            0
        } else {
            anyhow::ensure!(
                raw_dim >= -(input.shape.len() as i64) && raw_dim < input.shape.len() as i64,
                "index_select dimension {raw_dim} out of range for rank {}",
                input.shape.len()
            );
            normalize_dim(raw_dim, input.shape.len())
        };
        let index = self.get_input_tensor(node, 2)?;
        index_select_tensor(input, index, dim, &self.output_meta_shape(node)?)
    }

    pub(crate) fn translate_unfold(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let raw_dim = self.get_int_arg(node, 1)?;
        let dim = if input.shape.is_empty() {
            anyhow::ensure!(
                raw_dim == 0 || raw_dim == -1,
                "unfold dimension {raw_dim} out of range for a scalar"
            );
            0
        } else {
            anyhow::ensure!(
                raw_dim >= -(input.shape.len() as i64) && raw_dim < input.shape.len() as i64,
                "unfold dimension {raw_dim} out of range for rank {}",
                input.shape.len()
            );
            normalize_dim(raw_dim, input.shape.len())
        };
        unfold_tensor(
            input,
            dim,
            self.get_int_arg(node, 2)?,
            self.get_int_arg(node, 3)?,
            &self.output_meta_shape(node)?,
        )
    }

    pub(crate) fn translate_expand(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut a = self.get_input_tensor(node, 0)?;
        let neg1_expr = Expression::from(-1i32);
        // torch's expand PREPENDS new dims when the target rank exceeds the
        // source rank, so `-1`/existing sizes resolve RIGHT-aligned against
        // the source shape (`class_embedding.expand(B, 1, -1)`: 1-D -> 3-D).
        // Unsqueeze leading dims first so the ShapeTracker expand sees
        // matching ranks; left-aligned indexing walks off the source shape.
        let raw: Vec<Expression> = if let Ok(sizes) = self.get_ints_arg(node, 1) {
            sizes
                .iter()
                .map(|&s| {
                    if s == -1 {
                        neg1_expr
                    } else {
                        Expression::from(s as usize)
                    }
                })
                .collect()
        } else {
            self.get_exprs_arg(node, 1)?
        };
        anyhow::ensure!(
            raw.len() >= a.shape.len(),
            "expand: target rank {} below source rank {}",
            raw.len(),
            a.shape.len()
        );
        let offset = raw.len() - a.shape.len();
        for _ in 0..offset {
            a = a.unsqueeze(0);
        }
        let target_shape: Vec<Expression> = raw
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                if e == neg1_expr {
                    anyhow::ensure!(
                        i >= offset,
                        "expand: -1 is only valid for existing (right-aligned) dims"
                    );
                    Ok(a.shape.dims[i])
                } else {
                    Ok(e)
                }
            })
            .collect::<Result<_>>()?;
        a.shape.expand(target_shape);
        Ok(a)
    }

    pub(crate) fn translate_slice(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1).unwrap_or(0);
        let dim = normalize_dim(dim, a.shape.len());

        let start: Expression = if node.inputs.len() > 2 {
            self.get_expr_arg(node, 2)
                .unwrap_or_else(|_| Expression::from(0usize))
        } else {
            Expression::from(0usize)
        };
        let start = normalize_slice_bound(start, a.shape.dims[dim]);

        if node.inputs.len() <= 3 {
            return Ok(a);
        }

        let end_is_sentinel = self
            .get_int_arg(node, 3)
            .map(|e| e == i64::MAX)
            .unwrap_or(false);

        if end_is_sentinel {
            return Ok(if start.to_usize() == Some(0) {
                a
            } else {
                a.slice_along(start.., dim)
            });
        }

        let end: Expression = self.get_expr_arg(node, 3)?;
        let end = normalize_slice_bound(end, a.shape.dims[dim]);

        if let Some(s) = start.to_usize()
            && let Some(e) = end.to_usize()
        {
            return Ok(a.slice_along(s..e, dim));
        }

        Ok(a.slice_along(start..end, dim))
    }

    /// `aten.select.int(self, dim, index)` — select element `index` along
    /// `dim`, dropping that dim. Output rank = input rank − 1, so a 1-D input
    /// produces a rank-0 scalar. Both `dim` and `index` may be negative and
    /// are normalized against the input shape.
    ///
    /// Slice and squeeze establish the view; gather materializes it without
    /// arithmetic, preserving signed zero, infinity, and NaN exactly.
    pub(crate) fn translate_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let index_raw = self.get_int_arg(node, 2)?;

        // Normalize a possibly-negative index. PyTorch accepts indices in
        // [-size, size); negative wraps from the end.
        let index = if index_raw < 0 {
            let axis_size = a.shape.dims[dim].to_usize().ok_or_else(|| {
                anyhow::anyhow!(
                    "select.int: dim {} must be concrete to normalize a negative index",
                    dim
                )
            })?;
            let normalized = axis_size as i64 + index_raw;
            if normalized < 0 {
                bail!(
                    "select.int: index {} out of range for dim {} of size {}",
                    index_raw,
                    dim,
                    axis_size
                );
            }
            normalized as usize
        } else {
            index_raw as usize
        };

        let selected = a.slice_along(index..index + 1, dim).squeeze(dim);
        let indexes = selected
            .graph()
            .iota(Expression::from('z'), selected.dims());
        Ok(selected.gather(indexes))
    }

    pub(crate) fn translate_cat(&mut self, node: &Node) -> Result<GraphTensor> {
        let tensors: Vec<GraphTensor> = if let Some(names) = node.inputs[0].arg.as_tensors() {
            names
                .iter()
                .map(|n| self.get_tensor(&n.name))
                .collect::<Result<_>>()?
        } else {
            let mut ts = Vec::new();
            for input in &node.inputs {
                if let Some(name) = input.arg.as_tensor_name()
                    && let Ok(t) = self.get_tensor(name)
                {
                    ts.push(t);
                }
            }
            ts
        };

        if tensors.is_empty() {
            bail!("cat: no tensor inputs found");
        }

        let dim = node
            .inputs
            .iter()
            .find(|i| i.arg.as_int().is_some() && i.name != "tensors")
            .and_then(|i| i.arg.as_int())
            .unwrap_or(0);

        let tensors: Vec<GraphTensor> = tensors
            .into_iter()
            .filter(|t| !t.shape.dims.iter().any(|d| d.to_usize() == Some(0)))
            .collect();

        if tensors.is_empty() {
            bail!("cat: all tensor inputs are empty");
        }

        let dim = normalize_dim(dim, tensors[0].shape.len());
        let mut result = tensors[0];
        let sym_ranges = sym_char_ranges(&self.sym_map);
        for t in &tensors[1..] {
            let mut next = *t;
            normalize_concat_dims(&mut result, &mut next, Some(dim), &sym_ranges);

            let lhs_axis = result.dims()[dim];
            let rhs_axis = next.dims()[dim];
            let mut lhs_padded = result.pad_along(0, rhs_axis, dim, 0.);
            let mut rhs_padded = next.pad_along(lhs_axis, 0, dim, 0.);
            normalize_concat_dims(&mut lhs_padded, &mut rhs_padded, None, &sym_ranges);
            result = lhs_padded + rhs_padded;
        }
        Ok(result)
    }

    pub(crate) fn translate_embedding(&mut self, node: &Node) -> Result<GraphTensor> {
        let weight = self.get_input_tensor(node, 0)?;
        let indices = self.get_input_tensor(node, 1)?;

        let hidden_dim = weight.shape.dims[1];
        let seq_shape = indices.shape.dims;

        let indices_int = indices.cast(DType::Int);
        let ids_expanded = (indices_int * hidden_dim).expand_dim(seq_shape.len(), hidden_dim);

        let arange = self.graph.arange(hidden_dim);
        let mut arange_expanded = arange;
        for d in seq_shape.iter().rev() {
            arange_expanded = arange_expanded.expand_dim(0, *d);
        }

        Ok(weight.gather(ids_expanded + arange_expanded))
    }

    pub(crate) fn translate_index_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let source = self.get_input_tensor(node, 0)?;

        // Handle indices as_tensors (all non-None) or as individual args with None entries
        let index_names: Vec<crate::pt2_schema::TensorName>;
        let mut first_non_none_dim = 0usize;

        if let Some(names) = node.inputs[1].arg.as_tensors() {
            index_names = names.to_vec();
        } else {
            let indices_arg = &node.inputs[1].arg;

            // Check if it's a single tensor (1D indexing)
            if let Some(name) = indices_arg.as_tensor_name() {
                index_names = vec![crate::pt2_schema::TensorName {
                    name: name.to_string(),
                }];
            } else if let Some(opt_tensors) = indices_arg.as_optional_tensors() {
                // Optional tensors list: [None, tensor, None, ...] for selective dim indexing
                use crate::pt2_schema::OptionalTensorEntry;
                let mut found_tensors: Vec<crate::pt2_schema::TensorName> = Vec::new();
                for (i, entry) in opt_tensors.iter().enumerate() {
                    if let OptionalTensorEntry::Tensor(t) = entry {
                        if found_tensors.is_empty() {
                            first_non_none_dim = i;
                        }
                        found_tensors.push(t.as_tensor.clone());
                    }
                }
                if found_tensors.is_empty() {
                    bail!("index.Tensor: no index tensors in optional_tensors list");
                }
                index_names = found_tensors;
                // Simple case: single non-None index on a specific dim → gather_elements
                if first_non_none_dim > 0 && index_names.len() == 1 {
                    let idx = self.get_tensor(&index_names[0].name)?.cast(DType::Int);
                    // gather_elements requires indices to have the same rank as data.
                    // PyTorch fancy indexing gives 1D indices that broadcast across other dims.
                    // Add unit leading dims to match rank, then broadcast to output shape.
                    let src_dims = source.shape.dims;
                    let src_rank = src_dims.len();
                    let mut expanded = idx;
                    for _ in 0..(src_rank - expanded.shape.len()) {
                        expanded = expanded.expand_dim(0, Expression::from(1usize));
                    }
                    // Build target shape: source dims everywhere except the indexed dim
                    let idx_dim_size = expanded.shape.dims[first_non_none_dim];
                    let mut target: Vec<Expression> = src_dims.to_vec();
                    target[first_non_none_dim] = idx_dim_size;
                    expanded.shape.expand(target);
                    return Ok(super::movement_dynamic::pt2_gather_elements(
                        source,
                        expanded,
                        first_non_none_dim,
                    ));
                }
            } else {
                bail!(
                    "index.Tensor: unsupported indices format: {:?}",
                    indices_arg
                );
            }
        }

        let index_names = &index_names;

        let src_shape = source.shape.dims;
        let n_indexed = index_names.len();

        let mut strides: Vec<Expression> = vec![Expression::from(1usize); n_indexed];
        for i in (0..n_indexed - 1).rev() {
            strides[i] = strides[i + 1] * src_shape[i + 1];
        }

        let mut flat_idx: Option<GraphTensor> = None;
        for (dim_idx, idx_name) in index_names.iter().enumerate() {
            let idx_tensor = self.get_tensor(&idx_name.name)?;

            // Normalize negative indices for this dimension. Stay in Int —
            // multiplying an Int tensor by an Expression broadcasts the axis
            // size, so we avoid three Cast nodes (Int→F32 for indices, F32→Int
            // for the result, Bool→F32 for the negative mask) per indexed dim.
            let axis_size = src_shape[dim_idx];
            let idx_int = idx_tensor.cast(DType::Int);
            let zero = self.graph.constant(0).expand_rhs(idx_int.shape);
            let is_negative = idx_int.lt(zero).cast(DType::Int);
            let idx_int = idx_int + is_negative * axis_size;

            let stride = &strides[dim_idx];
            let weighted = if stride.to_usize() == Some(1) {
                idx_int
            } else {
                idx_int * *stride
            };

            flat_idx = Some(match flat_idx {
                Some(acc) => {
                    let (acc_b, w_b) = broadcast_binary(acc, weighted);
                    acc_b + w_b
                }
                None => weighted,
            });
        }

        let mut indexed_size = Expression::from(1usize);
        for i in 0..n_indexed {
            indexed_size *= src_shape[i];
        }
        let remaining_dims: Vec<Expression> = src_shape[n_indexed..].to_vec();

        let mut flat_shape = vec![indexed_size];
        flat_shape.extend_from_slice(&remaining_dims);
        let flat_source = reshape_tensor(source, flat_shape);

        let flat_idx = flat_idx.context("index.Tensor: no indices")?;

        if remaining_dims.is_empty() {
            Ok(flat_source.gather(flat_idx))
        } else {
            let mut remaining_size = Expression::from(1usize);
            for d in &remaining_dims {
                remaining_size *= *d;
            }

            let idx_shape = flat_idx.shape.dims;
            let mut expanded_idx = flat_idx * remaining_size;

            expanded_idx = expanded_idx.expand_dim(idx_shape.len(), remaining_size);

            let arange = self.graph.arange(remaining_size);
            let mut arange_expanded = arange;
            for d in idx_shape.iter().rev() {
                arange_expanded = arange_expanded.expand_dim(0, *d);
            }

            let final_idx = expanded_idx + arange_expanded;
            let total_elements = indexed_size * remaining_size;
            let fully_flat = reshape_tensor(flat_source, vec![total_elements]);
            let gathered = fully_flat.gather(final_idx);

            let mut result_shape: Vec<Expression> = idx_shape.to_vec();
            result_shape.extend_from_slice(&remaining_dims);
            Ok(reshape_tensor(gathered, result_shape))
        }
    }

    pub(crate) fn translate_gather(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?;

        // PyTorch eager allows torch.gather(rank-1, 0, rank-0) and returns
        // a rank-0 scalar — the only rank-mismatch case eager permits. Our
        // gather_elements requires the index rank to match the source rank,
        // so unsqueeze the rank-0 index to (1,), gather, then squeeze back.
        let promoted_rank0 = indices.shape.is_empty() && a.shape.len() == 1;
        let indices = if promoted_rank0 {
            indices.unsqueeze(0)
        } else {
            indices
        };

        // Normalize negative indices: -1 → last, -2 → second-to-last, etc.
        // Stay in Int the whole way — multiplying an Int tensor by an
        // Expression broadcasts the axis size and avoids three Cast nodes
        // (Int→F32 for indices, F32→Int for the result, plus a Bool→F32 for
        // the negative mask) that the previous F32-routed path emitted.
        let axis_dim = a.shape.dims[dim];
        let indices_int = indices.cast(DType::Int);
        let zero = self.graph.constant(0).expand_rhs(indices_int.shape);
        let is_negative = indices_int.lt(zero).cast(DType::Int);
        let normalized = indices_int + is_negative * axis_dim;

        let result = super::movement_dynamic::pt2_gather_elements(a, normalized, dim);
        Ok(if promoted_rank0 {
            result.squeeze(0)
        } else {
            result
        })
    }

    pub(crate) fn translate_scatter_src(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?;
        let src = self.get_input_tensor(node, 3)?;
        Ok(super::movement_dynamic::pt2_scatter_elements(
            a,
            indices.cast(DType::Int),
            src,
            dim,
        ))
    }

    pub(crate) fn translate_scatter_value(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, SCATTER_INPUT_ARG)?;
        let dim = self.get_int_arg(node, SCATTER_DIM_ARG)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, SCATTER_INDEX_ARG)?;
        let value_arg = &node
            .inputs
            .get(SCATTER_VALUE_ARG)
            .context("scatter.value missing value input")?
            .arg;
        let value = if let Some(b) = value_arg.as_bool() {
            self.graph.constant(if b { 1 } else { 0 }).cast(a.dtype)
        } else if let Some(i) = value_arg.as_int() {
            self.graph.constant(i).cast(a.dtype)
        } else if let Some(f) = value_arg.as_float() {
            self.graph.constant_float(f as f32).cast(a.dtype)
        } else {
            bail!("scatter.value: unsupported scalar argument {:?}", value_arg);
        }
        .expand_rhs(indices.shape);
        Ok(super::movement_dynamic::pt2_scatter_elements(
            a,
            indices.cast(DType::Int),
            value,
            dim,
        ))
    }

    pub(crate) fn translate_index_put(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        if let Some(entries) = node.inputs[1].arg.as_optional_tensors() {
            let mut axis_and_name = None;
            for (dim, entry) in entries.iter().enumerate() {
                if let OptionalTensorEntry::Tensor(t) = entry {
                    if axis_and_name.is_some() {
                        bail!(
                            "index_put: multiple tensor indices not supported: {:?}",
                            node.inputs[1].arg
                        );
                    }
                    axis_and_name = Some((dim, t.as_tensor.name.clone()));
                }
            }
            let Some((axis, idx_name)) = axis_and_name else {
                bail!(
                    "index_put: optional_tensors indices contain no tensor: {:?}",
                    node.inputs[1].arg
                );
            };
            let values = self.get_input_tensor(node, 2)?;
            let idx = self.get_tensor(&idx_name)?.cast(DType::Int);
            if idx.shape.len() != 1 {
                bail!(
                    "index_put: only a 1-D tensor index is supported, got rank {}",
                    idx.shape.len()
                );
            }
            let val_dims = values.dims();
            if val_dims.len() != a.dims().len() {
                bail!(
                    "index_put: values rank {} != data rank {} (broadcasting not supported)",
                    val_dims.len(),
                    a.dims().len()
                );
            }
            if axis >= val_dims.len() {
                bail!(
                    "index_put: index dim {axis} out of range for values rank {}",
                    val_dims.len()
                );
            }
            let mut idx_full = idx;
            for (dim, &size) in val_dims.iter().enumerate().take(axis) {
                idx_full = idx_full.expand_dim(dim, size);
            }
            for (dim, &size) in val_dims.iter().enumerate().skip(axis + 1) {
                idx_full = idx_full.expand_dim(dim, size);
            }
            let values = if values.dtype == a.dtype {
                values
            } else {
                values.cast(a.dtype)
            };
            let result = super::movement_dynamic::pt2_scatter_elements(a, idx_full, values, axis);
            return Ok(result);
        }
        let index_names = if let Some(names) = node.inputs[1].arg.as_tensors() {
            names
                .iter()
                .map(|name| name.name.clone())
                .collect::<Vec<_>>()
        } else if let Some(name) = node.inputs[1].arg.as_tensor_name() {
            vec![name.to_string()]
        } else {
            bail!("index_put: indices not tensor(s): {:?}", node.inputs[1].arg);
        };
        let values = self.get_input_tensor(node, 2)?;

        if index_names.len() == 1 {
            let idx_tensor = self.get_tensor(&index_names[0])?;

            // Boolean-mask index_put: when the only index is a Bool tensor whose
            // shape matches the data tensor, PyTorch semantics are
            //   data[mask] = value   ↔   where(mask, value, data)
            // NOT a scatter into positions. Casting the Bool mask to Int and
            // feeding it to scatter_nd would reinterpret True/False as row
            // indices 1/0 and silently corrupt the data. Reproducer:
            //   x = arange(16).reshape(4, 4); mask = zeros(4, 4, dtype=bool)
            //   y = x.clone(); y[mask] = 99   # eager: y == x (no-op)
            // Pre-fix the compiled graph wrote 99 to row 0; this branch
            // ensures the bool-mask path lowers to a where-blend instead.
            if idx_tensor.dtype == DType::Bool && idx_tensor.shape.dims == a.shape.dims {
                // Broadcast the (often scalar) value tensor to match data shape,
                // then blend by mask. Cast mask to data's dtype for the
                // arithmetic so this works for both integer and float data.
                let mask_f = idx_tensor.cast(a.dtype);
                let values_b = values.cast(a.dtype).expand_rhs(a.shape);
                // where(mask, value, a) as `a + mask*(value - a)`. Saves a mul
                // and the `1.0` constant compared to the `a*(1 - m) + v*m`
                // form; works for any numeric dtype without a dedicated cond.
                return Ok(a + mask_f * (values_b - a));
            }

            // Integer-index scatter: index_put with indices=[idx_tensor] writes
            // into dim 0 of `a` at every position named in idx_tensor (flattened),
            // broadcasting values across the trailing dims of `a`. idx_tensor can
            // be ANY shape — its whole shape is "batch dims" in scatter_nd terms,
            // and K is always 1 (number of dims we're indexing into). Always pad
            // a trailing size-1 dim so the rank-1 and rank-N cases share a path.
            let indices = idx_tensor.cast(DType::Int);
            let new_last = indices.shape.len();
            let indices = indices.expand_dim(new_last, Expression::from(1usize));
            Ok(super::movement_dynamic::pt2_scatter_nd(a, indices, values))
        } else {
            bail!("index_put with multiple index tensors not yet supported");
        }
    }

    pub(crate) fn translate_split_with_sizes(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let sizes = self.get_ints_arg(node, 1)?;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(0)
        } else {
            0
        };
        let dim = normalize_dim(dim, a.shape.len());

        let output_names: Vec<String> = node
            .outputs
            .first()
            .and_then(|o| o.as_tensors.as_ref())
            .map(|ts| ts.iter().map(|t| t.name.clone()).collect())
            .unwrap_or_else(|| {
                node.outputs
                    .iter()
                    .filter_map(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
                    .collect()
            });

        let mut offset = 0usize;
        let mut first_chunk = None;
        for (i, &size) in sizes.iter().enumerate() {
            let size = size as usize;
            let chunk = a.slice_along(offset..offset + size, dim);
            if let Some(name) = output_names.get(i) {
                self.tensors.insert(name.clone(), chunk);
            }
            if i == 0 {
                first_chunk = Some(chunk);
            }
            offset += size;
        }

        first_chunk.ok_or_else(|| anyhow::anyhow!("split_with_sizes: empty sizes list"))
    }
}
