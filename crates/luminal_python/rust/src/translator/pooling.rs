use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::dim_arith::product_of_dims;
use crate::pt2_schema::{Argument, Node};
use crate::pt2_util::reshape_tensor;

use super::Translator;

fn expand_spatial(values: &[i64], spatial_rank: usize, default: i64) -> Result<Vec<usize>> {
    let values = if values.is_empty() {
        vec![default; spatial_rank]
    } else if values.len() == 1 {
        vec![values[0]; spatial_rank]
    } else {
        values.to_vec()
    };
    anyhow::ensure!(
        values.len() == spatial_rank,
        "expected {spatial_rank} spatial values"
    );
    values
        .into_iter()
        .map(|value| usize::try_from(value).context("spatial values must be nonnegative"))
        .collect()
}

impl<'a> Translator<'a> {
    fn pool_ints(&self, node: &Node, name: &str, index: usize) -> Result<Vec<i64>> {
        let index = Self::named_input_index(node, name).unwrap_or(index);
        self.get_ints_arg(node, index)
    }

    #[allow(clippy::too_many_arguments)]
    fn pool_windows(
        &mut self,
        input: GraphTensor,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        ceil_mode: bool,
        output_shape: &[Expression],
        fill: GraphTensor,
    ) -> GraphTensor {
        let rank = input.shape.len();
        let spatial_rank = kernel.len();
        let first_spatial = rank - spatial_rank;
        let mut pad = vec![(Expression::from(0), Expression::from(0)); rank];
        for spatial in 0..spatial_rank {
            let right_extra = if ceil_mode { stride[spatial] - 1 } else { 0 };
            pad[first_spatial + spatial] = (
                Expression::from(padding[spatial]),
                Expression::from(padding[spatial] + right_extra),
            );
        }
        let padded = input.pad_with(pad, fill);

        let mut full_kernel = vec![Expression::from(1); rank];
        let mut full_stride = vec![Expression::from(1); rank];
        let mut full_dilation = vec![Expression::from(1); rank];
        for spatial in 0..spatial_rank {
            full_kernel[first_spatial + spatial] = Expression::from(kernel[spatial]);
            full_stride[first_spatial + spatial] = Expression::from(stride[spatial]);
            full_dilation[first_spatial + spatial] = Expression::from(dilation[spatial]);
        }
        let mut windows = padded.unfold(full_kernel, full_stride, full_dilation);
        for spatial in 0..spatial_rank {
            let axis = first_spatial + spatial;
            windows = windows.slice_along(Expression::from(0)..output_shape[axis], axis);
        }
        // Remove the size-one kernel axes belonging to batch/channel axes.
        for axis in (0..first_spatial).rev() {
            windows = windows.squeeze(rank + axis);
        }
        windows
    }

    pub(crate) fn translate_avg_pool(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(
            input.shape.len() == spatial_rank + 1 || input.shape.len() == spatial_rank + 2,
            "avg_pool{spatial_rank}d input rank is invalid"
        );
        let kernel_raw = self.pool_ints(node, "kernel_size", 1)?;
        let kernel = expand_spatial(&kernel_raw, spatial_rank, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&value| value > 0),
            "pool kernel must be positive"
        );
        let stride_raw = self.pool_ints(node, "stride", 2).unwrap_or_default();
        let stride = if stride_raw.is_empty() {
            kernel.clone()
        } else {
            expand_spatial(&stride_raw, spatial_rank, 0)?
        };
        anyhow::ensure!(
            stride.iter().all(|&value| value > 0),
            "pool stride must be positive"
        );
        let padding = expand_spatial(
            &self.pool_ints(node, "padding", 3).unwrap_or_default(),
            spatial_rank,
            0,
        )?;
        let ceil_mode = self.named_bool_arg(node, "ceil_mode").unwrap_or(false);
        let count_include_pad = self
            .named_bool_arg(node, "count_include_pad")
            .unwrap_or(true);
        let divisor_override = self.named_int_arg(node, "divisor_override");
        let output_shape = self.output_meta_shape(node)?;
        let rank = input.shape.len();
        let unit_dilation = vec![1; spatial_rank];
        let zero = self.graph.constant_float(0.0).cast(input.dtype);
        let windows = self.pool_windows(
            input,
            &kernel,
            &stride,
            &padding,
            &unit_dilation,
            ceil_mode,
            &output_shape,
            zero,
        );
        let kernel_axes = (rank..rank + spatial_rank).collect::<Vec<_>>();
        let sum = windows.sum(kernel_axes.clone());

        let divisor = if let Some(divisor) = divisor_override {
            anyhow::ensure!(divisor != 0, "pool divisor_override cannot be zero");
            self.graph
                .constant(divisor)
                .cast(input.dtype)
                .expand_rhs(sum.shape)
        } else if count_include_pad {
            self.graph
                .constant(product_of_dims(
                    kernel.iter().copied().map(Expression::from),
                ))
                .cast(input.dtype)
                .expand_rhs(sum.shape)
        } else {
            let one = self.graph.constant_float(1.0).cast(input.dtype);
            let ones = one.expand_rhs(input.dims());
            let zero = self.graph.constant_float(0.0).cast(input.dtype);
            self.pool_windows(
                ones,
                &kernel,
                &stride,
                &padding,
                &unit_dilation,
                ceil_mode,
                &output_shape,
                zero,
            )
            .sum(kernel_axes)
        };
        Ok(sum / divisor)
    }

    fn adaptive_pool_candidates(
        &mut self,
        input: GraphTensor,
        output_shape: &[Expression],
        spatial_rank: usize,
    ) -> (GraphTensor, GraphTensor, usize) {
        let rank = input.shape.len();
        let prefix_rank = rank - spatial_rank;
        let input_shape = input.dims();
        let input_spatial = &input_shape[prefix_rank..];
        let output_spatial = &output_shape[prefix_rank..];
        let mut full_shape = input_shape[..prefix_rank].to_vec();
        full_shape.extend_from_slice(output_spatial);
        full_shape.extend_from_slice(input_spatial);
        let mut expanded = input;
        for (spatial, size) in output_spatial.iter().copied().enumerate() {
            expanded = expanded.expand_dim(prefix_rank + spatial, size);
        }
        let mut membership = self.graph.iota(1, full_shape.clone()).cast(DType::Bool);
        for spatial in 0..spatial_rank {
            let output_position = self.axis_positions(&full_shape, prefix_rank + spatial);
            let input_position =
                self.axis_positions(&full_shape, prefix_rank + spatial_rank + spatial);
            let input_size = self
                .graph
                .constant(input_spatial[spatial])
                .cast(DType::F64)
                .expand_rhs(output_position.shape);
            let output_size = self
                .graph
                .constant(output_spatial[spatial])
                .cast(DType::F64)
                .expand_rhs(output_position.shape);
            let start =
                (output_position.cast(DType::F64) * input_size / output_size).cast(DType::Int);
            let end = (((output_position + 1) * input_spatial[spatial]
                + (output_spatial[spatial] - 1))
                .cast(DType::F64)
                / output_size)
                .cast(DType::Int);
            membership = self.bool_and(
                membership,
                self.bool_and(input_position.ge(start), input_position.lt(end)),
            );
        }
        (expanded, membership, prefix_rank)
    }

    pub(crate) fn translate_adaptive_avg_pool(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.shape.len();
        anyhow::ensure!(
            rank == spatial_rank + 1 || rank == spatial_rank + 2,
            "adaptive_avg_pool{spatial_rank}d input rank is invalid"
        );
        let output_shape = self.output_meta_shape(node)?;
        let (expanded, membership, prefix_rank) =
            self.adaptive_pool_candidates(input, &output_shape, spatial_rank);
        let zero = self.full_tensor(expanded.dims(), input.dtype, 0.0);
        let selected = self.select(membership, expanded, zero);
        let reduction_axes =
            (prefix_rank + spatial_rank..prefix_rank + 2 * spatial_rank).collect::<Vec<_>>();
        let sum = selected.sum(reduction_axes.clone());
        let count = membership.cast(input.dtype).sum(reduction_axes);
        Ok(sum / count)
    }

    pub(crate) fn translate_adaptive_max_pool(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.shape.len();
        anyhow::ensure!(
            rank == spatial_rank + 1 || rank == spatial_rank + 2,
            "adaptive max pool input rank is invalid"
        );
        let input_shape = input.dims();
        let output_shape = self.output_meta_shape(node)?;
        let (expanded, membership, prefix_rank) =
            self.adaptive_pool_candidates(input, &output_shape, spatial_rank);
        let input_spatial = &input_shape[prefix_rank..];
        let output_spatial = &output_shape[prefix_rank..];
        let spatial_numel = product_of_dims(input_spatial.iter().copied());
        let logical_indices = self
            .graph
            .iota(Expression::from('z') % spatial_numel, input_shape.clone())
            .cast(DType::I64);
        let mut expanded_indices = logical_indices;
        for (spatial, size) in output_spatial.iter().copied().enumerate() {
            expanded_indices = expanded_indices.expand_dim(prefix_rank + spatial, size);
        }

        let lowest = self.lowest_scalar(input.dtype).expand_rhs(expanded.shape);
        let candidates = self.select(membership, expanded, lowest);
        let outputs = self.select_pool_max(candidates, expanded_indices, rank);
        self.store_tensor_outputs(node, &outputs)
    }

    fn lowest_scalar(&mut self, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.graph.constant_float64(f64::NEG_INFINITY),
            DType::F16 | DType::Bf16 | DType::F32 => {
                self.graph.constant_float(f32::NEG_INFINITY).cast(dtype)
            }
            DType::I64 => self.graph.constant(i64::MIN).cast(dtype),
            DType::Int => self.graph.constant(i32::MIN as i64).cast(dtype),
            DType::I16 => self.graph.constant(i16::MIN as i64).cast(dtype),
            DType::I8 | DType::I4 => self.graph.constant(i8::MIN as i64).cast(dtype),
            DType::U16 | DType::U8 | DType::U4 | DType::Bool => self.graph.constant(0).cast(dtype),
            _ => self.graph.constant_float(f32::NEG_INFINITY).cast(dtype),
        }
    }

    fn select_pool_max(
        &mut self,
        mut candidates: GraphTensor,
        mut indices: GraphTensor,
        output_rank: usize,
    ) -> [GraphTensor; 2] {
        while candidates.shape.len() > output_rank + 1 {
            let last = candidates.shape.len() - 1;
            candidates = candidates.merge_dims(last - 1, last);
            indices = indices.merge_dims(last - 1, last);
        }
        let key = if candidates.dtype == DType::Bool {
            candidates.cast(DType::F32)
        } else {
            candidates
        };
        let selected = key
            .stable_argsort(output_rank, true)
            .slice_along(0..1, output_rank);
        let values =
            super::movement_dynamic::pt2_gather_elements(candidates, selected, output_rank)
                .squeeze(output_rank);
        let indices = super::movement_dynamic::pt2_gather_elements(indices, selected, output_rank)
            .squeeze(output_rank)
            .cast(DType::I64);
        [values, indices]
    }

    pub(crate) fn translate_max_pool(&mut self, node: &Node, spatial_rank: usize) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.shape.len();
        anyhow::ensure!(
            rank == spatial_rank + 1 || rank == spatial_rank + 2,
            "max pool requires channel-first input"
        );
        let kernel = expand_spatial(&self.pool_ints(node, "kernel_size", 1)?, spatial_rank, 0)?;
        let stride_raw = self.pool_ints(node, "stride", 2).unwrap_or_default();
        let stride = if stride_raw.is_empty() {
            kernel.clone()
        } else {
            expand_spatial(&stride_raw, spatial_rank, 0)?
        };
        let padding = expand_spatial(
            &self.pool_ints(node, "padding", 3).unwrap_or_default(),
            spatial_rank,
            0,
        )?;
        let dilation = expand_spatial(
            &self.pool_ints(node, "dilation", 4).unwrap_or_default(),
            spatial_rank,
            1,
        )?;
        let ceil_mode = self.named_bool_arg(node, "ceil_mode").unwrap_or(false);
        let output_shape = self.output_meta_shape(node)?;
        let lowest = self.lowest_scalar(input.dtype);
        let windows = self.pool_windows(
            input,
            &kernel,
            &stride,
            &padding,
            &dilation,
            ceil_mode,
            &output_shape,
            lowest,
        );

        let prefix_rank = rank - spatial_rank;
        let spatial_numel = product_of_dims(input.dims()[prefix_rank..].iter().copied());
        let logical_indices = self
            .graph
            .iota(Expression::from('z') % spatial_numel, input.dims())
            .cast(DType::I64);
        let zero_index = self.graph.constant(0i64).cast(DType::I64);
        let index_windows = self.pool_windows(
            logical_indices,
            &kernel,
            &stride,
            &padding,
            &dilation,
            ceil_mode,
            &output_shape,
            zero_index,
        );
        let outputs = self.select_pool_max(windows, index_windows, rank);
        self.store_tensor_outputs(node, &outputs)
    }

    pub(crate) fn translate_max_pool_backward(&mut self, node: &Node) -> Result<GraphTensor> {
        let updates = self.get_input_tensor(node, 0)?;
        let input = self.get_input_tensor(node, 1)?;
        let indices = self.get_input_tensor(node, 7)?.cast(DType::Int);
        let input_shape = input.dims();
        let prefix_rank = input_shape.len() - 2;
        let strides = super::movement_dynamic::row_major_strides(&input_shape);
        let output_shape = updates.dims();
        let contributions = output_shape
            .iter()
            .enumerate()
            .map(|(axis, _)| {
                if axis < prefix_rank {
                    Expression::from('z') * strides[axis]
                } else {
                    Expression::from(0)
                }
            })
            .collect::<Vec<_>>();
        let base = super::movement_dynamic::logical_flat_indices(
            &mut self.graph,
            &output_shape,
            &contributions,
            0.into(),
        );
        let destinations = (base + indices).flatten();
        let zero = self
            .full_tensor(input_shape.clone(), input.dtype, 0.0)
            .flatten();
        let output = super::movement_dynamic::pt2_scatter_elements_reduce(
            zero,
            destinations,
            updates.flatten(),
            0,
            super::movement_dynamic::ScatterReduction::Add,
        )?;
        Ok(reshape_tensor(output, input_shape))
    }

    pub(crate) fn translate_fractional_max_pool(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let random_samples = self.get_input_tensor(node, 3)?;
        let rank = input.shape.len();
        anyhow::ensure!(
            rank == spatial_rank + 1 || rank == spatial_rank + 2,
            "fractional max pool input rank is invalid"
        );
        anyhow::ensure!(
            random_samples.shape.len() == 3
                && random_samples.dims()[2].to_usize() == Some(spatial_rank),
            "fractional max pool random_samples must have shape [N, C, spatial_rank]"
        );
        let kernel = expand_spatial(&self.pool_ints(node, "kernel_size", 1)?, spatial_rank, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&size| size > 0),
            "fractional max pool kernel must be positive"
        );
        let output_shape = self.output_meta_shape(node)?;
        let prefix_rank = rank - spatial_rank;
        let input_shape = input.dims();
        let output_spatial = &output_shape[prefix_rank..];
        let input_spatial = &input_shape[prefix_rank..];
        let mut window_shape = output_shape.clone();
        window_shape.extend(kernel.iter().copied().map(Expression::from));

        let input_strides = super::movement_dynamic::row_major_strides(&input_shape);
        let mut flat_indices = self.axis_positions(&window_shape, 0) * input_strides[0];
        for (prefix, stride) in input_strides
            .iter()
            .copied()
            .enumerate()
            .take(prefix_rank)
            .skip(1)
        {
            flat_indices += self.axis_positions(&window_shape, prefix) * stride;
        }
        let mut logical_spatial_indices = self
            .graph
            .constant(0)
            .cast(DType::Int)
            .expand_rhs(window_shape.clone());

        for spatial in 0..spatial_rank {
            let sample_lane = if spatial_rank == 2 {
                spatial_rank - 1 - spatial
            } else {
                spatial
            };
            let mut sample = random_samples
                .slice_along(sample_lane..sample_lane + 1, 2)
                .squeeze(2);
            if prefix_rank == 1 {
                sample = sample.squeeze(0);
            }
            for output_size in output_spatial.iter().copied() {
                sample = sample.expand_dim(sample.shape.len(), output_size);
            }
            let output_axis = prefix_rank + spatial;
            let output_position = self.axis_positions(&output_shape, output_axis);
            let numerator = self
                .graph
                .constant(input_spatial[spatial] - kernel[spatial])
                .cast(sample.dtype)
                .expand_rhs(sample.shape);
            let denominator = self
                .graph
                .constant(output_spatial[spatial] - 1)
                .cast(sample.dtype)
                .expand_rhs(sample.shape);
            let output_is_one = self.is_zero(denominator);
            let one = self.constant_like(denominator, 1.0);
            let safe_denominator = self.select(output_is_one, one, denominator);
            let alpha = numerator / safe_denominator;
            let ordinary_start = self
                .floor_tensor((output_position.cast(sample.dtype) + sample) * alpha)
                - self.floor_tensor(sample * alpha);
            let terminal_position = self
                .graph
                .constant(output_spatial[spatial] - 1)
                .cast(DType::Int)
                .expand_rhs(output_position.shape);
            let final_position = output_position.eq(terminal_position);
            let terminal_start = self
                .graph
                .constant(input_spatial[spatial] - kernel[spatial])
                .cast(sample.dtype)
                .expand_rhs(ordinary_start.shape);
            let final_or_ordinary = self.select(final_position, terminal_start, ordinary_start);
            let start = self.select(output_is_one, terminal_start, final_or_ordinary);
            let mut coordinate = start;
            for size in kernel.iter().copied() {
                coordinate = coordinate.expand_dim(coordinate.shape.len(), size);
            }
            let kernel_axis = rank + spatial;
            coordinate = coordinate
                + self
                    .axis_positions(&window_shape, kernel_axis)
                    .cast(coordinate.dtype);
            let coordinate = coordinate.cast(DType::Int);
            flat_indices += coordinate * input_strides[prefix_rank + spatial];
            let spatial_stride = product_of_dims(input_spatial[spatial + 1..].iter().copied());
            logical_spatial_indices += coordinate * spatial_stride;
        }

        let candidates =
            reshape_tensor(input.flatten().gather(flat_indices.flatten()), window_shape);
        let outputs = self.select_pool_max(candidates, logical_spatial_indices, rank);
        self.store_tensor_outputs(node, &outputs)
    }

    fn bilinear_axis(
        &mut self,
        input: GraphTensor,
        axis: usize,
        input_size: usize,
        output_size: usize,
        align_corners: bool,
        explicit_scale: Option<f64>,
    ) -> GraphTensor {
        if input_size == output_size {
            return input;
        }
        let positions = self.graph.arange(output_size).cast(DType::F32);
        let source = if align_corners {
            let scale = if output_size > 1 {
                (input_size - 1) as f32 / (output_size - 1) as f32
            } else {
                0.0
            };
            positions * scale
        } else {
            let inverse =
                explicit_scale.map_or(input_size as f64 / output_size as f64, |s| 1.0 / s) as f32;
            ((positions + 0.5) * inverse - 0.5).maximum_f32(0.0)
        };
        let lower = source.cast(DType::Int);
        let upper = (lower + 1).minimum(
            self.graph
                .constant((input_size - 1) as i64)
                .cast(DType::Int)
                .expand_rhs(lower.shape),
        );
        let weight = source - lower.cast(DType::F32);
        let mut lower = lower;
        let mut upper = upper;
        let mut weight = weight.cast(input.dtype);
        for (dim, size) in input.dims().into_iter().enumerate() {
            if dim != axis {
                lower = lower.expand_dim(dim, size);
                upper = upper.expand_dim(dim, size);
                weight = weight.expand_dim(dim, size);
            }
        }
        let left = super::movement_dynamic::pt2_gather_elements(input, lower, axis);
        let right = super::movement_dynamic::pt2_gather_elements(input, upper, axis);
        left + (right - left) * weight
    }

    fn static_resize_dimensions(
        input: GraphTensor,
        output_shape: &[Expression],
        operation: &str,
    ) -> Result<[usize; 4]> {
        let [
            Some(input_height),
            Some(input_width),
            Some(output_height),
            Some(output_width),
        ] = [
            input.dims()[2],
            input.dims()[3],
            output_shape[2],
            output_shape[3],
        ]
        .map(|size| size.to_usize())
        else {
            anyhow::bail!("{operation} dimensions must be static");
        };
        let dimensions = [input_height, input_width, output_height, output_width];
        anyhow::ensure!(
            dimensions.iter().all(|&size| size > 0),
            "{operation} dimensions must be nonzero"
        );
        Ok(dimensions)
    }

    pub(crate) fn translate_upsample_bilinear2d(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(input.shape.len() == 4, "bilinear2d requires NCHW input");
        let output_shape = self.output_meta_shape(node)?;
        let [input_height, input_width, output_height, output_width] =
            Self::static_resize_dimensions(input, &output_shape, "bilinear2d")?;
        let align_corners = self.get_bool_arg(node, 2)?;
        let scales = node.inputs.get(3).and_then(|input| match &input.arg {
            Argument::Other(value) => {
                let values = value.get("as_floats")?.as_array()?;
                if values.len() == 2 {
                    Some((values[0].as_f64()?, values[1].as_f64()?))
                } else {
                    None
                }
            }
            _ => None,
        });
        let height = self.bilinear_axis(
            input,
            2,
            input_height,
            output_height,
            align_corners,
            scales.map(|value| value.0),
        );
        Ok(self.bilinear_axis(
            height,
            3,
            input_width,
            output_width,
            align_corners,
            scales.map(|value| value.1),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn antialias_bilinear_axis(
        &mut self,
        input: GraphTensor,
        axis: usize,
        input_size: usize,
        output_size: usize,
        align_corners: bool,
        explicit_scale: Option<f64>,
        quantized_u8: bool,
    ) -> GraphTensor {
        if input_size == output_size {
            return input;
        }
        let scale_value = if align_corners {
            if output_size > 1 {
                (input_size - 1) as f64 / (output_size - 1) as f64
            } else {
                0.0
            }
        } else {
            explicit_scale.map_or(input_size as f64 / output_size as f64, |scale| {
                scale.recip()
            })
        };
        let support_value = scale_value.max(1.0);

        // ATen's antialias path uses a normalized triangle filter. Construct
        // the complete [output, input] weight matrix: both dimensions are
        // static metadata here, and the tensor values never affect extents.
        let weight_dtype = if quantized_u8 {
            DType::F64
        } else {
            input.dtype
        };
        let output_positions = self
            .graph
            .arange(output_size)
            .cast(weight_dtype)
            .expand_dim(1, input_size);
        let input_positions = self
            .graph
            .arange(input_size)
            .cast(weight_dtype)
            .expand_dim(0, output_size);
        let half = self.constant_like(output_positions, 0.5);
        let scale = self.constant_like(output_positions, scale_value);
        let source = (output_positions + half) * scale - half;
        let distance = self.real_abs(input_positions - source);
        let one = self.constant_like(distance, 1.0);
        let support = self.constant_like(distance, support_value);
        let unbounded = one - distance / support;
        let zero = self.constant_like(unbounded, 0.0);
        let positive = unbounded.gt(zero);
        let mut weights = self.select(positive, unbounded, zero);
        let normalization = weights.sum(1).expand_dim(1, input_size);
        weights /= normalization;

        let weights_precision = if quantized_u8 {
            let mut maximum = 0.0_f64;
            for output_index in 0..output_size {
                let center = scale_value * (output_index as f64 + 0.5);
                let row = (0..input_size)
                    .map(|input_index| {
                        (1.0 - ((input_index as f64 + 0.5 - center) / support_value).abs()).max(0.0)
                    })
                    .collect::<Vec<_>>();
                let total = row.iter().sum::<f64>();
                maximum = maximum.max(
                    row.into_iter()
                        .map(|weight| weight / total)
                        .fold(0.0_f64, f64::max),
                );
            }
            let mut precision = 0_u32;
            while precision < 22 {
                let next = (0.5 + maximum * ((1_u64 << (precision + 1)) as f64)) as i64;
                if next >= 1_i64 << 15 {
                    break;
                }
                precision += 1;
            }
            let multiplier = self.constant_like(weights, (1_u64 << precision) as f64);
            let half = self.constant_like(weights, 0.5);
            weights = self
                .floor_tensor(weights * multiplier + half)
                .cast(DType::I64);
            Some(precision)
        } else {
            None
        };

        let mut candidates = if quantized_u8 {
            input.cast(DType::I64).expand_dim(axis, output_size)
        } else {
            input.expand_dim(axis, output_size)
        };
        for dim in 0..axis {
            weights = weights.expand_dim(dim, input.dims()[dim]);
        }
        for dim in axis + 1..input.shape.len() {
            weights = weights.expand_dim(dim + 1, input.dims()[dim]);
        }
        candidates *= weights;
        let result = candidates.sum(axis + 1);
        if let Some(precision) = weights_precision {
            let result = result.cast(DType::F64);
            let bias = self.constant_like(result, (1_u64 << (precision - 1)) as f64);
            let divisor = self.constant_like(result, (1_u64 << precision) as f64);
            let rounded = self.floor_tensor((result + bias) / divisor);
            let zero = self.constant_like(rounded, 0.0);
            let maximum = self.constant_like(rounded, u8::MAX as f64);
            let lower = self.select(rounded.lt(zero), zero, rounded);
            self.select(lower.gt(maximum), maximum, lower)
                .cast(DType::U8)
        } else {
            result
        }
    }

    pub(crate) fn translate_upsample_bilinear2d_aa(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(
            input.shape.len() == 4,
            "antialiased bilinear2d requires NCHW input"
        );
        let output_shape = self.output_meta_shape(node)?;
        let [input_height, input_width, output_height, output_width] =
            Self::static_resize_dimensions(input, &output_shape, "antialiased bilinear2d")?;
        let align_corners = self.get_bool_arg(node, 2)?;
        let scale_height = node.inputs.get(3).and_then(|input| input.arg.as_float());
        let scale_width = node.inputs.get(4).and_then(|input| input.arg.as_float());
        let quantized_u8 = input.dtype == DType::U8;
        let compute = if quantized_u8 || input.dtype == DType::F64 {
            input
        } else {
            input.cast(DType::F32)
        };
        // The CPU uint8 kernel quantizes after each separable pass, width first.
        let width = self.antialias_bilinear_axis(
            compute,
            3,
            input_width,
            output_width,
            align_corners,
            scale_width,
            quantized_u8,
        );
        let output = self.antialias_bilinear_axis(
            width,
            2,
            input_height,
            output_height,
            align_corners,
            scale_height,
            quantized_u8,
        );
        Ok(if quantized_u8 {
            output
        } else {
            output.cast(input.dtype)
        })
    }

    fn store_zero_for_output(&mut self, name: &str) -> Result<()> {
        let meta = self
            .tensor_meta(name)
            .context("batch norm output metadata is missing")?;
        let shape = self.tensor_meta_to_shape(meta)?;
        let dtype = crate::pt2_util::torch_dtype_int_to_luminal(meta.dtype);
        let zero = self.full_tensor(shape, dtype, 0.0);
        self.tensors.insert(name.to_string(), zero);
        Ok(())
    }

    pub(crate) fn translate_batch_norm_functional(&mut self, node: &Node) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(
            input.shape.len() >= 2,
            "batch norm input rank must be at least two"
        );
        let output_names = Self::tensor_output_names(node);
        anyhow::ensure!(!output_names.is_empty(), "batch norm has no tensor outputs");
        let no_training =
            node.target == "torch.ops.aten._native_batch_norm_legit_no_training.default";
        let functional = matches!(
            node.target.as_str(),
            "torch.ops.aten._native_batch_norm_legit_functional.default"
                | "torch.ops.aten._batch_norm_with_update_functional.default"
        );
        let training = if node.target == "torch.ops.aten._batch_norm_with_update_functional.default"
        {
            true
        } else if no_training {
            false
        } else {
            self.named_bool_arg(node, "training").unwrap_or(true)
        };
        anyhow::ensure!(
            training || functional || no_training,
            "batch norm without running stats requires training=true"
        );
        let momentum = self.named_float_arg(node, "momentum").unwrap_or(0.1);
        let eps = self.named_float_arg(node, "eps").unwrap_or(1e-5);

        let compute_dtype = if matches!(input.dtype, DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            input.dtype
        };
        let compute = input.cast(compute_dtype);
        let axes = (0..input.shape.len())
            .filter(|&axis| axis != 1)
            .collect::<Vec<_>>();
        let (batch_mean, batch_var) = if training {
            let batch_mean = compute.mean(axes.clone());
            let expanded_mean = batch_mean.expand_to_shape_on_axes(compute.shape, axes.clone());
            let centered = compute - expanded_mean;
            let batch_var = centered.square().mean(axes.clone());
            (Some(batch_mean), Some(batch_var))
        } else {
            // Inference uses only the stored running statistics. Besides being
            // dead work, adding batch reductions here makes every eval-mode
            // BatchNorm layer unnecessarily enlarge the compiler search space.
            (None, None)
        };

        let running_mean = self.named_tensor_arg(node, "running_mean")?;
        let running_var = self.named_tensor_arg(node, "running_var")?;
        let (mean, variance) = if training {
            (
                batch_mean.context("training batch norm requires batch mean")?,
                batch_var.context("training batch norm requires batch variance")?,
            )
        } else {
            (
                running_mean
                    .context("inference batch norm requires running_mean")?
                    .cast(compute_dtype),
                running_var
                    .context("inference batch norm requires running_var")?
                    .cast(compute_dtype),
            )
        };
        let invstd = (variance + self.constant_like(variance, eps))
            .sqrt()
            .reciprocal();
        let mean_expanded = mean.expand_to_shape_on_axes(compute.shape, axes.clone());
        let invstd_expanded = invstd.expand_to_shape_on_axes(compute.shape, axes.clone());
        let mut output = (compute - mean_expanded) * invstd_expanded;
        if let Some(weight) = self.named_tensor_arg(node, "weight")? {
            let weight = weight
                .cast(compute_dtype)
                .expand_to_shape_on_axes(output.shape, axes.clone());
            output *= weight;
        }
        if let Some(bias) = self.named_tensor_arg(node, "bias")? {
            let bias = bias
                .cast(compute_dtype)
                .expand_to_shape_on_axes(output.shape, axes.clone());
            output += bias;
        }
        self.tensors
            .insert(output_names[0].clone(), output.cast(input.dtype));

        for (index, statistic) in [(1, batch_mean), (2, Some(invstd))] {
            if output_names.len() > index {
                if training {
                    self.tensors.insert(
                        output_names[index].clone(),
                        statistic.context("training batch norm statistic is missing")?,
                    );
                } else {
                    self.store_zero_for_output(&output_names[index])?;
                }
            }
        }

        let running_start = output_names.len().saturating_sub(2);
        if functional && output_names.len() >= 5 {
            for name in &output_names[3..running_start] {
                self.store_zero_for_output(name)?;
            }
            let running_mean =
                running_mean.context("functional batch norm requires running_mean")?;
            let running_var = running_var.context("functional batch norm requires running_var")?;
            let (mean_out, var_out) = if training {
                let batch_mean = batch_mean.context("training batch norm requires batch mean")?;
                let batch_var = batch_var.context("training batch norm requires batch variance")?;
                let mean_out = running_mean * (1.0 - momentum as f32)
                    + batch_mean.cast(running_mean.dtype) * momentum as f32;
                let count = product_of_dims(axes.iter().map(|&axis| input.dims()[axis]));
                let count_tensor = self
                    .graph
                    .constant(count)
                    .cast(batch_var.dtype)
                    .expand_rhs(batch_var.shape);
                let denominator = self
                    .graph
                    .constant(count - 1)
                    .cast(batch_var.dtype)
                    .expand_rhs(batch_var.shape);
                let unbiased = batch_var * count_tensor / denominator;
                let var_out = running_var * (1.0 - momentum as f32)
                    + unbiased.cast(running_var.dtype) * momentum as f32;
                (mean_out, var_out)
            } else {
                (running_mean, running_var)
            };
            self.tensors
                .insert(output_names[running_start].clone(), mean_out);
            self.tensors
                .insert(output_names[running_start + 1].clone(), var_out);
        }
        Ok(())
    }
}
