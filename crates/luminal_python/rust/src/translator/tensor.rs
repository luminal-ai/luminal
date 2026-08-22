use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

const FULL_SHAPE_ARG: usize = 0;
const FULL_VALUE_ARG: usize = 1;

const FULL_LIKE_VALUE_ARG: usize = 1;

const CONSTANT_PAD_INPUT_ARG: usize = 0;
const CONSTANT_PAD_PADDING_ARG: usize = 1;

const TOPK_INPUT_ARG: usize = 0;
const TOPK_K_ARG: usize = 1;
const TOPK_DIM_ARG: usize = 2;

const SORT_INPUT_ARG: usize = 0;

const WHERE_COND_ARG: usize = 0;
const WHERE_X_ARG: usize = 1;
const WHERE_OTHER_ARG: usize = 2;

const TRIANGULAR_INPUT_ARG: usize = 0;
const TRIANGULAR_DIAGONAL_ARG: usize = 1;

#[derive(Clone, Copy, Debug)]
enum ArangeScalar {
    Int(i64),
    Float(f64),
    Expr(Expression),
}

/// A PyTorch `Scalar` constructor argument before it is converted to the
/// constructor's recorded output dtype.
#[derive(Clone, Debug)]
pub(crate) enum ConstructorScalar {
    Value(String),
    Bool(bool),
    Int(i64),
    Float(f64),
    Complex(f64, f64),
}

impl ConstructorScalar {
    pub(crate) fn is_literal_zero(&self) -> bool {
        match self {
            Self::Bool(value) => !value,
            Self::Int(value) => *value == 0,
            Self::Float(value) => *value == 0.0,
            Self::Complex(real, imag) => *real == 0.0 && *imag == 0.0,
            Self::Value(_) => false,
        }
    }

    pub(crate) fn is_literal_one(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Int(value) => *value == 1,
            Self::Float(value) => *value == 1.0,
            Self::Complex(real, imag) => *real == 1.0 && *imag == 0.0,
            Self::Value(_) => false,
        }
    }
}

/// Copy `source` into the logical shape and dtype of `destination`.
///
/// ATen permits the source to broadcast but never changes the destination's
/// shape. Using the destination in `broadcast_binary` gives us its exact
/// symbolic dimensions while rejecting the inverse case where broadcasting
/// would have to grow a destination dimension.
pub(crate) fn copy_tensor(
    destination: GraphTensor,
    source: GraphTensor,
    dtype: DType,
) -> Result<GraphTensor> {
    let destination_shape = destination.dims();
    let (broadcast_destination, mut source) = broadcast_binary(destination, source.cast(dtype));
    anyhow::ensure!(
        broadcast_destination.dims() == destination_shape,
        "copy source shape {:?} cannot broadcast into destination shape {:?}",
        source.dims(),
        destination_shape
    );
    // Canonicalize dimensions to the destination spelling. Equal symbolic
    // dimensions can arrive under different PT2 expressions, and the copied
    // value inherits the destination's shape contract.
    for (actual, expected) in source.shape.dims.iter_mut().zip(destination_shape) {
        *actual = expected;
    }
    Ok(source)
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_trilinear(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut inputs = [
            self.get_input_tensor(node, 0)?,
            self.get_input_tensor(node, 1)?,
            self.get_input_tensor(node, 2)?,
        ];
        for (input, argument) in inputs.iter_mut().zip(3..6) {
            for raw_dim in self.get_ints_arg(node, argument)? {
                let dim = normalize_dim(raw_dim, input.shape.len() + 1);
                *input = input.unsqueeze(dim);
            }
        }
        let (left, middle) = broadcast_binary(inputs[0], inputs[1]);
        let (product, right) = broadcast_binary(left * middle, inputs[2]);
        let rank = product.shape.len();
        let dimensions = self
            .get_ints_arg(node, 6)?
            .into_iter()
            .map(|dim| normalize_dim(dim, rank))
            .collect::<Vec<_>>();
        Ok((product * right).sum(dimensions))
    }

    pub(crate) fn constructor_scalar_arg(
        &self,
        node: &Node,
        index: usize,
    ) -> Result<ConstructorScalar> {
        let arg = &node
            .inputs
            .get(index)
            .with_context(|| format!("{} is missing scalar input {index}", node.target))?
            .arg;
        if let Some(name) = arg.as_value_name() {
            return Ok(ConstructorScalar::Value(name.to_string()));
        }
        if let Some((real, imag)) = arg.as_complex() {
            return Ok(ConstructorScalar::Complex(real, imag));
        }
        if let Some(value) = arg.as_float() {
            return Ok(ConstructorScalar::Float(value));
        }
        if let Some(value) = arg.as_int() {
            return Ok(ConstructorScalar::Int(value));
        }
        if let Some(value) = arg.as_bool() {
            return Ok(ConstructorScalar::Bool(value));
        }
        anyhow::bail!("{} has unsupported scalar argument {arg:?}", node.target)
    }

    pub(crate) fn typed_scalar_constant(
        &mut self,
        value: &ConstructorScalar,
        dtype: DType,
    ) -> Result<GraphTensor> {
        let tensor = match value {
            ConstructorScalar::Value(name) => {
                anyhow::bail!("runtime scalar {name} is not a literal constant")
            }
            ConstructorScalar::Bool(value) => self.graph.constant(i64::from(*value)).cast(dtype),
            ConstructorScalar::Int(value) => match dtype {
                DType::F64 => self.graph.constant_float64(*value as f64),
                DType::F32 | DType::F16 | DType::Bf16 => {
                    self.graph.constant_float(*value as f32).cast(dtype)
                }
                DType::Int | DType::I64 | DType::I16 | DType::I8 | DType::U8 | DType::Bool => {
                    self.graph.constant_i64(*value).cast(dtype)
                }
                other => anyhow::bail!("unsupported constructor dtype {other:?}"),
            },
            ConstructorScalar::Float(value) => match dtype {
                DType::F64 => self.graph.constant_float64(*value),
                DType::F32 | DType::F16 | DType::Bf16 => {
                    self.graph.constant_float(*value as f32).cast(dtype)
                }
                DType::Int | DType::I64 | DType::I16 | DType::I8 | DType::U8 | DType::Bool => {
                    self.graph.constant_float64(*value).cast(dtype)
                }
                other => anyhow::bail!("unsupported constructor dtype {other:?}"),
            },
            ConstructorScalar::Complex(real, imag) => {
                anyhow::ensure!(
                    *imag == 0.0,
                    "complex value with nonzero imaginary component cannot construct {dtype:?}"
                );
                return self.typed_scalar_constant(&ConstructorScalar::Float(*real), dtype);
            }
        };
        Ok(tensor)
    }

    pub(crate) fn real_constructor_scalar(
        &mut self,
        node: &Node,
        index: usize,
        dtype: DType,
    ) -> Result<GraphTensor> {
        let value = self.constructor_scalar_arg(node, index)?;
        if let ConstructorScalar::Value(name) = &value {
            anyhow::ensure!(
                !self.complex_tensors.contains_key(name),
                "complex runtime scalar {name} cannot construct a real tensor"
            );
            return Ok(reshape_tensor(self.get_tensor(name)?, vec![]).cast(dtype));
        }
        self.typed_scalar_constant(&value, dtype)
    }

    fn scale_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: GraphTensor,
    ) -> Result<GraphTensor> {
        let Some(index) = node.inputs.iter().position(|input| input.name == name) else {
            return Ok(value);
        };
        let scalar = self.constructor_scalar_arg(node, index)?;
        if scalar.is_literal_one() {
            return Ok(value);
        }
        if scalar.is_literal_zero() {
            return Ok(self
                .typed_scalar_constant(&ConstructorScalar::Int(0), value.dtype)?
                .expand_rhs(value.shape));
        }
        let scalar = self.real_constructor_scalar(node, index, value.dtype)?;
        Ok(value * scalar.expand_rhs(value.shape))
    }

    pub(crate) fn translate_addmv(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let compute_dtype = match output_dtype {
            DType::F16 | DType::Bf16 => DType::F32,
            dtype => dtype,
        };
        let input = self.get_input_tensor(node, 0)?.cast(compute_dtype);
        let matrix = self.get_input_tensor(node, 1)?.cast(compute_dtype);
        let vector = self.get_input_tensor(node, 2)?.cast(compute_dtype);
        anyhow::ensure!(matrix.shape.len() == 2, "addmv matrix must be rank 2");
        anyhow::ensure!(vector.shape.len() == 1, "addmv vector must be rank 1");

        let product = matrix.matmul(vector.unsqueeze(1)).squeeze(1);
        let input = self.scale_by_named_scalar(node, "beta", input)?;
        let product = self.scale_by_named_scalar(node, "alpha", product)?;
        let (input, product) = broadcast_binary(input, product);
        Ok((input + product).cast(output_dtype))
    }

    pub(crate) fn translate_addbmm(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let compute_dtype = match output_dtype {
            DType::F16 | DType::Bf16 => DType::F32,
            dtype => dtype,
        };
        let input = self.get_input_tensor(node, 0)?.cast(compute_dtype);
        let batch1 = self.get_input_tensor(node, 1)?.cast(compute_dtype);
        let batch2 = self.get_input_tensor(node, 2)?.cast(compute_dtype);
        anyhow::ensure!(batch1.shape.len() == 3, "addbmm batch1 must be rank 3");
        anyhow::ensure!(batch2.shape.len() == 3, "addbmm batch2 must be rank 3");

        // CPU ATen evaluates addbmm as sequential fused addmm updates. A
        // single bmm+sum changes the F32 rounding order, and for BF16 it can
        // move by whole representable values because every update is rounded
        // back to BF16. Preserve the observable order when the batch length is
        // concrete; symbolic batches use the algebraically equivalent fallback.
        if matches!(output_dtype, DType::Bf16 | DType::F32)
            && let Some(batch_count) = batch1.shape.dims[0].to_usize()
        {
            let mut result = self.scale_by_named_scalar(node, "beta", input)?;
            for batch in 0..batch_count {
                let lhs = batch1.slice_along(batch..batch + 1, 0).squeeze(0);
                let rhs = batch2.slice_along(batch..batch + 1, 0).squeeze(0);
                let product = self.scale_by_named_scalar(node, "alpha", lhs.matmul(rhs))?;
                let (result_broadcast, product) =
                    broadcast_binary(result.cast(compute_dtype), product);
                result = (result_broadcast + product).cast(output_dtype);
            }
            return Ok(result.cast(output_dtype));
        }

        let product = batch1.matmul(batch2).sum(0);
        let input = self.scale_by_named_scalar(node, "beta", input)?;
        let product = self.scale_by_named_scalar(node, "alpha", product)?;
        let (input, product) = broadcast_binary(input, product);
        Ok((input + product).cast(output_dtype))
    }

    pub(crate) fn translate_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        copy_tensor(
            self.get_input_tensor(node, 0)?,
            self.get_input_tensor(node, 1)?,
            dtype,
        )
    }

    pub(crate) fn constant_pad_spec(
        &self,
        node: &Node,
        rank: usize,
    ) -> Result<Vec<(Expression, Expression)>> {
        let raw = self.get_exprs_arg(node, CONSTANT_PAD_PADDING_ARG)?;
        anyhow::ensure!(
            raw.len().is_multiple_of(2),
            "constant_pad_nd requires left/right pairs, got {} values",
            raw.len()
        );
        let padded_dims = raw.len() / 2;
        anyhow::ensure!(
            padded_dims <= rank,
            "constant_pad_nd received {padded_dims} padded dimensions for rank-{rank} input"
        );

        // ATen lists the last dimension first: [last_left, last_right,
        // penultimate_left, ...]. GraphTensor::pad expects tensor-axis order.
        let mut pairs = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0], pair[1]))
            .collect::<Vec<_>>();
        pairs.reverse();
        let mut padding = vec![(0.into(), 0.into()); rank - padded_dims];
        padding.extend(pairs);
        Ok(padding)
    }

    pub(crate) fn translate_arange(&mut self, node: &Node) -> Result<GraphTensor> {
        // PT2's start_step overload names these arguments even when the
        // original call used a keyword (`torch.arange(0, end=3)`). Resolve by
        // schema name rather than collecting every decodable positional arg:
        // the old filter_map silently discarded float/bool values and shifted
        // the remaining operands into the wrong start/end/step slots.
        let start = self.arange_scalar_arg(node, "start")?;
        let step = node
            .inputs
            .iter()
            .find(|input| input.name == "step")
            .map(|input| self.decode_arange_scalar(&input.arg))
            .transpose()?
            .unwrap_or(ArangeScalar::Int(1));

        // Export has already applied PyTorch's dtype-sensitive ceiling and
        // endpoint rules. Its tensor metadata is therefore authoritative for
        // the number of values, including fractional/negative steps and empty
        // ranges. Recomputing `(end-start)/step` in Expression arithmetic used
        // truncating division and disagreed with shapes such as arange(-1,2,2).
        let output_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            output_shape.len() == 1,
            "arange: expected rank-1 output metadata, got {output_shape:?}"
        );
        let output_dtype = self.output_meta_dtype(node)?;
        let indices = self.graph.arange(output_shape[0]).cast(output_dtype);
        let shape = indices.shape;
        let step = self
            .arange_scalar_constant(step, output_dtype)
            .expand_rhs(shape);
        let start = self
            .arange_scalar_constant(start, output_dtype)
            .expand_rhs(shape);
        Ok(indices * step + start)
    }

    fn arange_scalar_arg(&self, node: &Node, name: &str) -> Result<ArangeScalar> {
        let input = node
            .inputs
            .iter()
            .find(|input| input.name == name)
            .with_context(|| format!("arange: missing `{name}` argument"))?;
        self.decode_arange_scalar(&input.arg)
    }

    fn decode_arange_scalar(&self, arg: &Argument) -> Result<ArangeScalar> {
        if let Some(value) = arg.as_int() {
            return Ok(ArangeScalar::Int(value));
        }
        if let Some(value) = arg.as_float() {
            return Ok(ArangeScalar::Float(value));
        }
        if let Some(value) = arg.as_bool() {
            return Ok(ArangeScalar::Int(i64::from(value)));
        }
        if let Some(value) = self.resolve_arg_as_expression(arg) {
            return Ok(ArangeScalar::Expr(value));
        }
        anyhow::bail!("arange: unsupported scalar argument {arg:?}")
    }

    fn arange_scalar_constant(&mut self, value: ArangeScalar, dtype: DType) -> GraphTensor {
        match value {
            ArangeScalar::Int(value) => match dtype {
                DType::F64 => self.graph.constant_float64(value as f64),
                DType::F32 | DType::F16 | DType::Bf16 => {
                    self.graph.constant_float(value as f32).cast(dtype)
                }
                _ => self.graph.constant(value).cast(dtype),
            },
            ArangeScalar::Float(value) => match dtype {
                DType::F64 => self.graph.constant_float64(value),
                DType::Int | DType::I64 => self.graph.constant_float64(value).cast(dtype),
                _ => self.graph.constant_float(value as f32).cast(dtype),
            },
            ArangeScalar::Expr(value) => self.graph.constant(value).cast(dtype),
        }
    }

    pub(crate) fn translate_full(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.output_meta_shape(node)?;
        let dtype = self.output_meta_dtype(node)?;
        let value = self.real_constructor_scalar(node, FULL_VALUE_ARG, dtype)?;
        Ok(if shape.is_empty() {
            value
        } else {
            value.expand_rhs(shape)
        })
    }

    /// Lower `aten.histc.default` for the integer-bincount case.
    ///
    /// Qwen3-MoE's expert-balance layer calls
    /// `torch.histc(expert_ids.int(), bins=K, min=0, max=K-1)` to count how
    /// many tokens were routed to each expert. With those args every
    /// integer value `i ∈ [0, K-1]` maps to exactly bin `i`, and the result
    /// is equivalent to `torch.bincount`. We implement that case as a
    /// broadcast equality + sum:
    ///
    ///   counts[b] = sum_i (input[i] == b + min)   for b in [0, bins)
    ///
    /// More general histc bin widths (`bins != max - min + 1`, or
    /// non-integer values that span fractional bins) are not supported
    /// today — the equality path would silently drop them. We bail rather
    /// than produce wrong counts.
    pub(crate) fn translate_histc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let bins_i64: i64 = self
            .get_int_arg(node, 1)
            .context("histc: missing `bins` arg (#1)")?;
        // `min`/`max` are float kwargs (default 0.0 each, which means
        // "auto-pick from input"); for the qwen3-moe call they're always
        // integers passed as floats.
        let min = self.get_float_arg(node, 2).unwrap_or(0.0);
        let max = self.get_float_arg(node, 3).unwrap_or(0.0);

        anyhow::ensure!(
            input.shape.len() == 1,
            "histc: only 1D input is supported, got {}D",
            input.shape.len()
        );
        anyhow::ensure!(
            bins_i64 > 0,
            "histc: bins must be positive, got {}",
            bins_i64
        );
        // Bincount-equivalent case: one integer value per bin.
        anyhow::ensure!(
            (max - min - (bins_i64 - 1) as f64).abs() < 1e-6,
            "histc: only the bincount-equivalent case (bins == max - min + 1) is \
             supported; got bins={}, min={}, max={}. Other cases would need a \
             general bin-width / right-edge-inclusion implementation.",
            bins_i64,
            min,
            max,
        );

        let bins_u = bins_i64 as usize;
        let n = input.shape.dims[0];

        // arange(bins) [bins] → cast to input dtype, optionally shift by min,
        // broadcast to [bins, N], compare for equality with input broadcast.
        let mut bins_arange = self.graph.arange(Expression::from(bins_u));
        if min != 0.0 {
            // `min` is non-zero (uncommon in the qwen3-moe path but legal)
            // — shift the comparison values to start at min.
            let min_i = min as i64;
            let shift = self
                .graph
                .constant_float(min_i as f32)
                .cast(bins_arange.dtype)
                .expand_rhs(bins_arange.shape);
            bins_arange += shift;
        }
        let bins_expanded = bins_arange.cast(input.dtype).expand_dim(1, n);
        let input_expanded = input.expand_dim(0, Expression::from(bins_u));
        let matches = input_expanded.eq(bins_expanded); // Bool [bins, N]

        let out_dtype = self.output_meta_dtype(node)?;
        Ok(matches.cast(out_dtype).sum(1))
    }

    /// Lower bucketize/searchsorted by comparing each query with every entry
    /// in its sorted row and reducing the boolean insertion predicate. This is
    /// O(N) per query, but it is exact, fixed-shape, and uses only existing
    /// broadcast/comparison/reduction primitives.
    pub(crate) fn translate_searchsorted(&mut self, node: &Node) -> Result<GraphTensor> {
        let bucketize = node.target == "torch.ops.aten.bucketize.Tensor";
        let mut sorted = self.get_input_tensor(node, usize::from(bucketize))?;
        let query_index = usize::from(!bucketize);
        let query = if let Some(name) = node.inputs[query_index].arg.as_tensor_name() {
            self.get_tensor(name)?
        } else {
            let scalar = self.constructor_scalar_arg(node, query_index)?;
            self.typed_scalar_constant(&scalar, sorted.dtype)?
        };

        anyhow::ensure!(
            !sorted.shape.is_empty(),
            "searchsorted requires a sorted sequence with rank at least one"
        );
        let sorted_axis = sorted.shape.len() - 1;
        if let Some(sorter_name) = node
            .inputs
            .iter()
            .find(|input| input.name == "sorter")
            .and_then(|input| input.arg.as_tensor_name())
        {
            let sorter = self.get_tensor(sorter_name)?;
            sorted = super::movement_dynamic::pt2_gather_elements(sorted, sorter, sorted_axis);
        }

        let (sorted, query) = ensure_same_dtype(sorted, query);
        let row_length = sorted.dims()[sorted_axis];
        let query_rank = query.shape.len();
        let (boundaries, queries) = if sorted.shape.len() == 1 {
            let mut boundaries = sorted;
            for (axis, size) in query.dims().into_iter().enumerate() {
                boundaries = boundaries.expand_dim(axis, size);
            }
            (boundaries, query.expand_dim(query_rank, row_length))
        } else {
            anyhow::ensure!(
                query_rank == sorted.shape.len(),
                "batched searchsorted requires query and sequence ranks to match"
            );
            for axis in 0..sorted_axis {
                anyhow::ensure!(
                    query.dims()[axis] == sorted.dims()[axis],
                    "batched searchsorted prefix dimensions must match"
                );
            }
            let query_width = query.dims()[sorted_axis];
            (
                sorted.expand_dim(sorted_axis, query_width),
                query.expand_dim(query_rank, row_length),
            )
        };

        let right = node
            .inputs
            .iter()
            .position(|input| input.name == "right")
            .and_then(|index| self.get_bool_arg(node, index).ok())
            .unwrap_or(false)
            || node.inputs.iter().any(|input| {
                input.name == "side"
                    && matches!(&input.arg, Argument::Other(value) if value.as_str() == Some("right") || value.get("as_string").and_then(|v| v.as_str()) == Some("right"))
            });
        let before = if right {
            boundaries.le(queries)
        } else {
            boundaries.lt(queries)
        };
        let mut result = before.cast(DType::I64).sum(query_rank);
        if matches!(
            query.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            let nan = self.is_nan(query);
            let end = self
                .graph
                .constant(row_length)
                .cast(DType::I64)
                .expand_rhs(result.shape);
            result = self.select(nan, end, result);
        }
        Ok(result.cast(self.output_meta_dtype(node)?))
    }

    pub(crate) fn translate_triangular_indices(&mut self, node: &Node) -> Result<GraphTensor> {
        let rows = self.get_int_arg(node, 0)?;
        let columns = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            rows >= 0 && columns >= 0,
            "triangular index dimensions must be nonnegative"
        );
        let offset = self.get_int_arg(node, 2).unwrap_or(0);
        let rows = Expression::from(rows);
        let columns = Expression::from(columns);
        let row = self
            .graph
            .arange(rows)
            .cast(DType::I64)
            .expand_dim(1, columns);
        let column = self
            .graph
            .arange(columns)
            .cast(DType::I64)
            .expand_dim(0, rows);
        let shifted_row = row + offset;
        let truth = if node.target == "torch.ops.aten.tril_indices.default" {
            column.le(shifted_row)
        } else {
            column.ge(shifted_row)
        };
        let output_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            output_shape.len() == 2,
            "triangular indices must have rank two"
        );
        let coordinates =
            super::movement::nonzero_static_from_truth(self, truth, output_shape[1], 0);
        Ok(coordinates
            .permute(&[1, 0])
            .cast(self.output_meta_dtype(node)?))
    }

    /// Lower `aten.empty.memory_format` and `aten.empty_permuted.default`.
    ///
    /// Both allocate an uninitialised tensor; the caller is responsible for
    /// writing into it. We materialise zeros instead — luminal has no
    /// "uninitialised" notion, and PyTorch's contract on `empty` outputs is
    /// undefined for any read prior to a write, so a zero-fill is sound.
    /// `aten.empty_permuted` additionally takes a `physical_layout` arg
    /// (the storage permutation); for a zero-filled tensor that's a no-op.
    pub(crate) fn translate_empty(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape_arg = if node.target == "torch.ops.aten.new_empty_strided.default" {
            1
        } else {
            FULL_SHAPE_ARG
        };
        let shape = self.get_exprs_arg(node, shape_arg)?;
        let dtype = self.output_meta_dtype(node)?;
        let zero = self.graph.constant_float(0.0).cast(dtype);
        Ok(if shape.is_empty() {
            zero
        } else {
            zero.expand_rhs(shape)
        })
    }

    pub(crate) fn translate_full_like(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.output_meta_shape(node)?;
        let dtype = self.output_meta_dtype(node)?;
        let value = self.real_constructor_scalar(node, FULL_LIKE_VALUE_ARG, dtype)?;
        Ok(if shape.is_empty() {
            value
        } else {
            value.expand_rhs(shape)
        })
    }

    pub(crate) fn translate_constant_pad_nd(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, CONSTANT_PAD_INPUT_ARG)?;
        let output_dtype = self.output_meta_dtype(node)?;
        anyhow::ensure!(
            input.dtype == output_dtype,
            "constant_pad_nd changed dtype from {:?} to {output_dtype:?}",
            input.dtype
        );
        let padding = self.constant_pad_spec(node, input.shape.len())?;
        let value_index = node.inputs.iter().position(|input| input.name == "value");
        let fill = match value_index {
            Some(index) => self.real_constructor_scalar(node, index, output_dtype)?,
            None => self.typed_scalar_constant(&ConstructorScalar::Int(0), output_dtype)?,
        };
        Ok(input.pad_with(padding, fill))
    }

    /// Translate `aten._grouped_mm.default(input, weight, offs)` → `Tensor[S, N]`.
    ///
    /// Grouped matmul: `input` is `[S, K]` (tokens sorted by expert), `weight` is
    /// `[G, K, N]` (per-expert weights), `offs` is `[G]` cumulative token counts.
    /// Output `[S, N]` where token m (in group g s.t. `offs[g-1] <= m < offs[g]`)
    /// is multiplied by `weight[g]`.
    ///
    /// Implementation: for each token m we (a) compute its expert id from offs,
    /// (b) gather only that expert's `[K, N]` slice from weight, and (c) do a
    /// single per-token matmul. The gather pattern mirrors the rust qwen3_moe
    /// example's `gather_experts`, which the GLUMoE host-op fusion in
    /// `luminal_cuda_lite` is designed to recognise.
    ///
    /// Why not the straightforward `[G, S, K] @ [G, K, N] → [G, S, N]` + mask:
    /// it forces a full F32 cast of the entire `[G, K, N]` weight tensor as
    /// search-time intermediate, which OOMs on real MoE checkpoints
    /// (Qwen3-30B-A3B: 1.5 GB / layer × 48 layers for gate-up alone). Gathering
    /// first keeps the F32 cast on `[S, K, N]` instead — for prefill (S = top_k)
    /// that is a 16× shrink (G=128, top_k=8).
    ///
    /// `offs` flows through as a runtime tensor — the routing decision is computed
    /// at execution time by the gate network and the same compiled graph handles
    /// any routing pattern without recompilation.
    pub(crate) fn translate_grouped_mm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let weight = self.get_input_tensor(node, 1)?;
        let offs = self.get_input_tensor(node, 2)?;
        let out_dtype = self.output_meta_dtype(node)?;

        anyhow::ensure!(
            input.shape.len() == 2,
            "_grouped_mm: input must be 2D, got {}D",
            input.shape.len()
        );
        anyhow::ensure!(
            weight.shape.len() == 3,
            "_grouped_mm: weight must be 3D, got {}D",
            weight.shape.len()
        );
        anyhow::ensure!(
            offs.shape.len() == 1,
            "_grouped_mm: offs must be 1D, got {}D",
            offs.shape.len()
        );

        let s = input.shape.dims[0];
        let g = weight.shape.dims[0];
        let k = weight.shape.dims[1];
        let n = weight.shape.dims[2];

        // expert_id[m] = number of g s.t. m >= offs[g], clamped to [0, G-1].
        // Same value as HF MoE's `expert_ids.clamp(0, num_experts-1)` for
        // invalid expert IDs from EP, AND protects search-time profiling:
        // dummy-1 input bytes give offs=[1,…,1], which pushes the raw count
        // to G for any token with index ≥ 1 and would OOB the weight gather.
        //
        // Stay in Int throughout — arange / offs are already Int, ge → Bool
        // → cast(Int), sum stays Int, and the binary `minimum` handles the
        // clamp without an F32 round-trip.
        let _ = g
            .to_usize()
            .context("_grouped_mm: G (num_experts) must be concrete")?;
        let s_arange = self.graph.arange(s); // Int [S]
        let ge_int = s_arange
            .expand_dim(0, g)
            .ge(offs.expand_dim(1, s)) // Bool [G, S]
            .cast(DType::Int); // Int [G, S]
        let raw = ge_int.sum(0); // Int [S], values in [0, G]
        let cap = self.graph.constant(g - 1).expand_dim(0, s); // Int [S], all G-1
        let expert_id = raw.minimum(cap); // Int [S]

        // Flat gather index into weight (treated as a length-G*K*N 1D buffer):
        //   flat[m, k_, n_] = expert_id[m] * (K*N) + k_ * N + n_
        // Encoded as `Mul(expert_id, Iota(io_const)) + Iota(MIter, K*N)` so the
        // resulting Gather matches the GLUMoE / gather-experts egglog patterns.
        let io = k * n;
        let base = expert_id * io;
        let within = self.graph.iota(Expression::from('z'), (k, n));
        let exp_base = base.expand_dim(1, k).expand_dim(2, n);
        let exp_within = within.expand_dim(0, s);
        let flat_idx = exp_base + exp_within;

        // Gather → [S, K, N], then normalize both operands to the op's declared
        // output dtype before matmul. On real Qwen3-MoE bf16 checkpoints the FX
        // graph inserts casts on the activation path, and relying on the input
        // tensor's translated dtype can leave us with mixed F32/Bf16 operands
        // by the time matmul expands into elementwise Mul. Using the PT2 output
        // metadata keeps the matmul dtype aligned with the exported contract
        // without upcasting the full expert weight bank.
        let weight_gathered = weight.gather(flat_idx).cast(out_dtype);
        let input = input.cast(out_dtype);

        // Per-token matmul: [S, 1, K] @ [S, K, N] → [S, 1, N] → [S, N].
        // Operands stay in their native dtype — no F32 cast on the gathered
        // weight or the input. The earlier cast(F32) was a holdover from the
        // broadcast-and-mask version (which had to use F32 because of the
        // cast(F32) on the mask). Gather-then-matmul has no such requirement,
        // and casting `[S, K, N]` to F32 doubled the gather scratch (~100 MB
        // to ~200 MB per layer for Qwen3-30B-A3B prefill). Matmul rewrites
        // (cuBLASLt etc.) handle bf16 input with F32 accumulator internally.
        let result = input.unsqueeze(1).matmul(weight_gathered).squeeze(1);

        Ok(result.cast(out_dtype))
    }

    /// Build the where-formula graph: `cond * x + (1 - cond) * y`, computed
    /// in F32, cast back to `out_dtype`. Shared between `translate_where`,
    /// `translate_where_scalar_other`, and `translate_masked_fill_scalar` so
    /// they all go through one well-tested code path.
    pub(crate) fn where_formula(
        &mut self,
        cond: GraphTensor,
        x: GraphTensor,
        y: GraphTensor,
        out_dtype: DType,
    ) -> GraphTensor {
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let (cond_bc, y_b) = broadcast_binary(cond_b, y);
        let (x_bc, y_bc) = broadcast_binary(x_b, y_b);
        // Lower as `y + c*(x - y)` rather than `c*x + (1-c)*y`: 3 ops vs 4 ops
        // plus the explicit `1.0` constant. Mathematically identical for
        // c ∈ {0, 1} and produces the same F32 output type.
        let c = cond_bc.cast(DType::F32);
        let x_f = x_bc.cast(DType::F32);
        let y_f = y_bc.cast(DType::F32);
        // Cast back: an F32 result downstream-interpreted as bf16 walks the
        // buffer at half-stride, returning every-other-element zeros.
        (y_f + c * (x_f - y_f)).cast(out_dtype)
    }

    pub(crate) fn translate_where(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let y = self.get_input_tensor(node, 2)?;
        let (x, y) = ensure_same_dtype(x, y);
        let out_dtype = x.dtype;
        Ok(self.where_formula(cond, x, y, out_dtype))
    }

    pub(crate) fn translate_where_scalar_other(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, WHERE_COND_ARG)?;
        let x = self.get_input_tensor(node, WHERE_X_ARG)?;
        let other_val = self.get_float_arg(node, WHERE_OTHER_ARG)? as f32;
        let out_dtype = x.dtype;
        // Build a tensor for the scalar `other` matching `x`'s shape so we
        // can route through the shared where_formula helper.
        let other = self.graph.constant_float(other_val).expand_rhs(x.shape);
        Ok(self.where_formula(cond, x, other, out_dtype))
    }

    pub(crate) fn translate_tril(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, false)
    }

    pub(crate) fn translate_triu(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, true)
    }

    fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, TRIANGULAR_INPUT_ARG)?;
        let diagonal = if node.inputs.len() > TRIANGULAR_DIAGONAL_ARG {
            self.get_int_arg(node, TRIANGULAR_DIAGONAL_ARG).unwrap_or(0) as i32
        } else {
            0
        };
        let dims = a.shape.dims;
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        let (r_val, c_val) = match (rows.to_usize(), cols.to_usize()) {
            (Some(r), Some(c)) => (r, c),
            _ => anyhow::bail!("tril/triu requires concrete matrix dimensions"),
        };
        let size = r_val.max(c_val);
        let mask = if upper {
            self.graph.triu(size, diagonal)
        } else {
            self.graph.tril(size, diagonal)
        }
        .cast(DType::F32);
        let mask = if rows != cols {
            mask.slice_along(0..r_val, 0).slice_along(0..c_val, 1)
        } else {
            mask
        };
        let mut mask_expanded = mask;
        for i in (0..dims.len() - 2).rev() {
            mask_expanded = mask_expanded.expand_dim(0, dims[i]);
        }
        Ok(a * mask_expanded)
    }

    pub(crate) fn translate_topk(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, TOPK_INPUT_ARG)?;
        let k = self.get_int_arg(node, TOPK_K_ARG)? as usize;
        let dim = if node.inputs.len() > TOPK_DIM_ARG {
            self.get_int_arg(node, TOPK_DIM_ARG).unwrap_or(-1)
        } else {
            -1
        };
        let dim = normalize_dim(dim, a.shape.len());

        // Determine output names
        let tuple_outputs = node.outputs.first().and_then(|o| o.as_tensors.as_ref());
        let values_name = if let Some(ts) = tuple_outputs {
            ts.first().map(|t| t.name.clone())
        } else {
            node.outputs
                .first()
                .and_then(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
        };
        let indices_name = if let Some(ts) = tuple_outputs {
            ts.get(1).map(|t| t.name.clone())
        } else if node.outputs.len() > 1 {
            node.outputs[1].as_tensor.as_ref().map(|t| t.name.clone())
        } else {
            None
        };

        // Build top-k outputs from a full stable argsort. Slice the indices
        // before gathering values so the gather shape matches the requested
        // top-k output rather than the full sort width. Cast to I64 so the
        // emitted indices match PyTorch's `torch.topk` semantics (indices
        // are int64); `gather_elements` accepts any int dtype on its index
        // operand, so a single I64 tensor serves both consumers.
        let full_argsort = a.stable_argsort(dim, true);
        let topk_indices = (full_argsort.slice_along(..k, dim) * 1.0).cast(DType::I64);

        // Only build the outputs that are consumed.
        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = super::movement_dynamic::pt2_gather_elements(a, topk_indices, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            self.tensors.insert(idx_name, topk_indices);
        }

        Ok(())
    }

    pub(crate) fn translate_sort(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, SORT_INPUT_ARG)?;
        // `sort.stable` inserts a keyword-only `stable` argument before dim,
        // so resolve these by schema name rather than by overload-dependent
        // position. `stable_argsort` is stable regardless; `stable=None` and
        // `stable=false` permit stability but do not require instability.
        let dim = node
            .inputs
            .iter()
            .position(|input| input.name == "dim")
            .and_then(|index| self.get_int_arg(node, index).ok())
            .unwrap_or(-1);
        let descending = node
            .inputs
            .iter()
            .position(|input| input.name == "descending")
            .and_then(|index| self.get_bool_arg(node, index).ok())
            .unwrap_or(false);
        let dim = normalize_dim(dim, a.shape.len());

        // Determine output names (sort returns (values, indices))
        let tuple_outputs = node
            .outputs
            .first()
            .and_then(|output| output.as_tensors.as_ref());
        let values_name = if let Some(outputs) = tuple_outputs {
            outputs.first().map(|tensor| tensor.name.clone())
        } else {
            node.outputs
                .first()
                .and_then(|output| output.as_tensor.as_ref().map(|tensor| tensor.name.clone()))
        };
        let indices_name = if let Some(outputs) = tuple_outputs {
            outputs.get(1).map(|tensor| tensor.name.clone())
        } else if node.outputs.len() > 1 {
            node.outputs[1]
                .as_tensor
                .as_ref()
                .map(|tensor| tensor.name.clone())
        } else {
            None
        };

        let sort_key = if a.dtype == DType::Bool {
            a.cast(DType::F32)
        } else {
            a
        };
        let full_argsort = sort_key.stable_argsort(dim, descending);

        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = super::movement_dynamic::pt2_gather_elements(a, full_argsort, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            // `torch.sort` returns int64 indices; cast at the PT2 boundary.
            let indices = (full_argsort * 1.0).cast(DType::I64);
            self.tensors.insert(idx_name, indices);
        }

        Ok(())
    }

    pub(crate) fn translate_wrap_set_grad(&mut self, node: &Node) -> Result<()> {
        let subgraph = node.inputs[1]
            .arg
            .as_graph()
            .context("wrap_with_set_grad: missing subgraph")?
            .clone();

        let sg_inputs = &subgraph.graph.inputs;
        let forwarded_args = &node.inputs[2..];
        for (sg_input, fwd_arg) in sg_inputs.iter().zip(forwarded_args) {
            if let Some(sg_name) = sg_input.as_tensor.as_ref()
                && let Some(main_name) = fwd_arg.arg.as_tensor_name()
            {
                let tensor = self.get_tensor(main_name)?;
                self.tensors.insert(sg_name.name.clone(), tensor);
            }
        }

        for (k, v) in &subgraph.graph.tensor_values {
            self.extra_tensor_values.insert(k.clone(), v.clone());
        }

        let sg_nodes = subgraph.graph.nodes.clone();
        for (i, sg_node) in sg_nodes.iter().enumerate() {
            self.translate_node(sg_node)
                .with_context(|| format!("Subgraph node {i}: {}", sg_node.target))?;
        }

        for (main_out, sg_out) in node.outputs.iter().zip(subgraph.graph.outputs.iter()) {
            if let (Some(main_name), Some(sg_name)) =
                (main_out.as_tensor.as_ref(), sg_out.as_tensor.as_ref())
                && main_name.name != sg_name.name
            {
                let tensor = self.get_tensor(&sg_name.name)?;
                self.tensors.insert(main_name.name.clone(), tensor);
            }
        }

        Ok(())
    }
}
