//! Pooling and normalization composites (port batch 4).
//!
//! All lowerings compose the recorder primitives (`slice`, `pad`, `unfold`,
//! reductions); none needs a backend-specific kernel.

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::broadcast_binary;
use crate::pt2_schema::Node;

fn expand_spatial(values: &[i64], spatial: usize, default: i64) -> Result<Vec<usize>> {
    let values = if values.is_empty() {
        vec![default; spatial]
    } else if values.len() == 1 {
        vec![values[0]; spatial]
    } else {
        values.to_vec()
    };
    anyhow::ensure!(values.len() == spatial, "expected {spatial} spatial values");
    values
        .into_iter()
        .map(|value| usize::try_from(value).context("spatial values must be nonnegative"))
        .collect()
}

impl Translator<'_> {
    /// Read a pooling arg by name, falling back to its positional slot.
    fn pool_ints(&self, node: &Node, name: &str, index: usize) -> Result<Vec<i64>> {
        let index = node
            .inputs
            .iter()
            .position(|input| input.name == name)
            .unwrap_or(index);
        self.get_ints_arg(node, index)
    }

    /// Sliding windows over the trailing `spatial` dims, shaped
    /// `[prefix..., out_spatial..., kernel...]`. `fill` supplies the
    /// out-of-bounds value (zero for avg, lowest for max). The window
    /// count is clamped to the declared output shape, which is what
    /// makes `ceil_mode`'s partial trailing windows land correctly.
    #[allow(clippy::too_many_arguments)]
    fn pool_windows(
        &mut self,
        input: GraphTensor,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        ceil_mode: bool,
        output_shape: &[IntExpr],
        fill: GraphTensor,
    ) -> GraphTensor {
        let rank = input.rank();
        let spatial_rank = kernel.len();
        let first_spatial = rank - spatial_rank;
        let mut pad = vec![(IntExpr::from(0), IntExpr::from(0)); rank];
        for spatial in 0..spatial_rank {
            let right_extra = if ceil_mode { stride[spatial] - 1 } else { 0 };
            pad[first_spatial + spatial] = (
                IntExpr::from(padding[spatial]),
                IntExpr::from(padding[spatial] + right_extra),
            );
        }
        let padded = input.pad_with(pad, fill);

        let mut full_kernel = vec![IntExpr::from(1); rank];
        let mut full_stride = vec![IntExpr::from(1); rank];
        let mut full_dilation = vec![IntExpr::from(1); rank];
        for spatial in 0..spatial_rank {
            let axis = first_spatial + spatial;
            full_kernel[axis] = IntExpr::from(kernel[spatial]);
            full_stride[axis] = IntExpr::from(stride[spatial]);
            full_dilation[axis] = IntExpr::from(dilation[spatial]);
        }
        let mut windows = padded.unfold(full_kernel, full_stride, full_dilation);
        for spatial in 0..spatial_rank {
            let axis = first_spatial + spatial;
            windows = windows.slice_along(IntExpr::from(0)..output_shape[axis], axis);
        }
        // Remove the size-one kernel axes belonging to batch/channel axes.
        for axis in (0..first_spatial).rev() {
            windows = windows.squeeze(rank + axis);
        }
        windows
    }

    /// `where(condition, a, b)` keeping the mask in `a`'s dtype (the
    /// shared `util::select` projects through F32, which mismatches
    /// F64/F16 values).
    fn pool_select(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = broadcast_binary(a, condition);
        let (a, b) = broadcast_binary(a, b);
        a.cond(condition, b)
    }

    fn pool_zero(&mut self, dtype: DType) -> GraphTensor {
        self.full_tensor(Vec::new(), dtype, 0.0)
    }

    /// The max-pool padding value: negative infinity for floats, false
    /// for Bool. Integer inputs bail — the stable-argsort selection key
    /// is built with a float multiplier.
    fn pool_lowest(&mut self, dtype: DType) -> Result<GraphTensor> {
        Ok(match dtype {
            DType::F64 => self.cx.constant_f64(f64::NEG_INFINITY),
            DType::F32 | DType::F16 | DType::Bf16 | DType::TF32 => {
                self.cx.constant_f32(f32::NEG_INFINITY).cast(dtype)
            }
            DType::Bool => self.cx.constant_f32(0.0).cast(DType::Bool),
            other => bail!(
                "max pooling on {other:?} inputs is not ported: the stable-argsort \
                 selection needs a float key"
            ),
        })
    }

    /// Max over the trailing axis, selecting via one stable-argsort pass.
    /// `candidates`/`indices` carry `[prefix..., out_spatial..., flatten]`
    /// and the result is the top-1 slice along the flattened axis. The
    /// index gathered from `indices` is PyTorch's flattened-within-plane
    /// index (row-major over the spatial dims).
    fn select_pool_max(
        &mut self,
        mut candidates: GraphTensor,
        mut indices: GraphTensor,
        output_rank: usize,
    ) -> Result<[GraphTensor; 2]> {
        while candidates.rank() > output_rank + 1 {
            let last = candidates.rank();
            candidates = candidates.merge_dims(last - 2, last - 1);
            indices = indices.merge_dims(last - 2, last - 1);
        }
        let key = match candidates.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::Bf16 | DType::TF32 => candidates,
            DType::Bool => candidates.cast(DType::F32),
            other => bail!(
                "max pooling on {other:?} inputs is not ported: the stable-argsort \
                 selection needs a float key"
            ),
        };
        let axis = output_rank;
        let selected = key
            .stable_argsort(axis, true)
            .slice_along(0..1, axis)
            .squeeze(axis);
        // Coordinate gather; nothing here mints flat Int arithmetic.
        let out_shape = selected.dims();
        let mut coordinates: Vec<GraphTensor> = (0..axis)
            .map(|index| self.axis_positions(&out_shape, index))
            .collect();
        coordinates.push(selected);
        let values = candidates.gather(&coordinates);
        let picked = indices.gather(&coordinates);
        let picked = if picked.dtype == DType::I64 {
            picked
        } else {
            picked.cast(DType::I64)
        };
        Ok([values, picked])
    }

    /// The adaptive pooling start/end tables: `start = floor(i*in/out)`,
    /// `end = ceil((i+1)*in/out)` from ATen's adaptive-pool kernel. Done
    /// in F32 and truncated, so no Int arithmetic enters the model (the
    /// reference executor has no F64 elementwise arms).
    fn adaptive_pool_candidates(
        &mut self,
        input: GraphTensor,
        output_shape: &[IntExpr],
        spatial: usize,
    ) -> Result<(GraphTensor, GraphTensor, usize)> {
        let rank = input.rank();
        let prefix_rank = rank - spatial;
        let input_shape = input.dims();
        let input_spatial = &input_shape[prefix_rank..];
        let output_spatial = &output_shape[prefix_rank..];
        let mut full_shape = input_shape[..prefix_rank].to_vec();
        full_shape.extend_from_slice(output_spatial);
        full_shape.extend_from_slice(input_spatial);
        let mut expanded = input;
        for (index, size) in output_spatial.iter().copied().enumerate() {
            expanded = expanded.expand_dim(prefix_rank + index, size);
        }
        let mut membership: Option<GraphTensor> = None;
        for index in 0..spatial {
            let output_position = self.axis_positions(&full_shape, prefix_rank + index);
            let input_position = self.axis_positions(&full_shape, prefix_rank + spatial + index);
            let input_size = self
                .cx
                .constant_i32(input_spatial[index])
                .cast(DType::F32)
                .expand_rhs(output_position.dims());
            let output_size = self
                .cx
                .constant_i32(output_spatial[index])
                .cast(DType::F32)
                .expand_rhs(output_position.dims());
            let output_position_f = output_position.cast(DType::F32);
            let start = (output_position_f * input_size / output_size).trunc_cast(DType::Int);
            let end = ((output_position_f + 1.0f32) * input_size / output_size)
                .ceil()
                .trunc_cast(DType::Int);
            let in_window = self.bool_and(input_position.ge(start), input_position.lt(end));
            membership = Some(match membership {
                Some(previous) => self.bool_and(previous, in_window),
                None => in_window,
            });
        }
        let Some(membership) = membership else {
            bail!("adaptive pooling requires at least one spatial dimension");
        };
        Ok((expanded, membership, prefix_rank))
    }

    fn spatial_flat_indices(
        &mut self,
        input_shape: &[IntExpr],
        prefix_rank: usize,
        spatial: usize,
    ) -> GraphTensor {
        let strides: Vec<IntExpr> = (0..spatial)
            .map(|index| {
                input_shape[prefix_rank + index + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, dim| acc * *dim)
            })
            .collect();
        self.cx
            .iota(input_shape.to_vec(), |coords| {
                (0..spatial).fold(IntExpr::from(0), |acc, index| {
                    acc + coords[prefix_rank + index] * strides[index]
                })
            })
            .cast(DType::I64)
    }

    pub(super) fn translate_avg_pool(
        &mut self,
        node: &Node,
        spatial: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.rank();
        anyhow::ensure!(
            rank == spatial + 1 || rank == spatial + 2,
            "avg_pool{spatial}d input rank is invalid"
        );
        let kernel = expand_spatial(&self.pool_ints(node, "kernel_size", 1)?, spatial, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&value| value > 0),
            "pool kernel must be positive"
        );
        let stride_raw = self.pool_ints(node, "stride", 2).unwrap_or_default();
        let stride = if stride_raw.is_empty() {
            kernel.clone()
        } else {
            expand_spatial(&stride_raw, spatial, 0)?
        };
        anyhow::ensure!(
            stride.iter().all(|&value| value > 0),
            "pool stride must be positive"
        );
        let padding = expand_spatial(
            &self.pool_ints(node, "padding", 3).unwrap_or_default(),
            spatial,
            0,
        )?;
        let ceil_mode = self.named_bool_arg(node, "ceil_mode").unwrap_or(false);
        let count_include_pad = self
            .named_bool_arg(node, "count_include_pad")
            .unwrap_or(true);
        let divisor_override = self.named_int_arg(node, "divisor_override");
        let output_shape = self.output_meta_shape(node)?;

        let in_dtype = input.dtype;
        // ATen accumulates average pools in the opmath float type.
        let compute_dtype = if matches!(in_dtype, DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            in_dtype
        };
        let compute = input.cast(compute_dtype);
        let unit_dilation = vec![1; spatial];
        let zero = self.pool_zero(compute_dtype);
        let windows = self.pool_windows(
            compute,
            &kernel,
            &stride,
            &padding,
            &unit_dilation,
            ceil_mode,
            &output_shape,
            zero,
        );
        let kernel_axes = (rank..rank + spatial).collect::<Vec<_>>();
        let sum = windows.sum(kernel_axes.clone());

        let divisor = if let Some(divisor) = divisor_override {
            anyhow::ensure!(divisor != 0, "pool divisor_override cannot be zero");
            self.full_tensor(sum.dims(), compute_dtype, divisor as f64)
        } else if count_include_pad {
            let kernel_product: usize = kernel.iter().product();
            self.full_tensor(sum.dims(), compute_dtype, kernel_product as f64)
        } else {
            let ones = self.full_tensor(compute.dims(), compute_dtype, 1.0);
            let zero = self.pool_zero(compute_dtype);
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
        Ok((sum / divisor).cast(in_dtype))
    }

    pub(super) fn translate_adaptive_avg_pool(
        &mut self,
        node: &Node,
        spatial: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.rank();
        anyhow::ensure!(
            rank == spatial + 1 || rank == spatial + 2,
            "adaptive_avg_pool{spatial}d input rank is invalid"
        );
        let output_shape = self.output_meta_shape(node)?;
        let in_dtype = input.dtype;
        let compute_dtype = if matches!(in_dtype, DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            in_dtype
        };
        let compute = input.cast(compute_dtype);
        let (expanded, membership, prefix_rank) =
            self.adaptive_pool_candidates(compute, &output_shape, spatial)?;
        let zero = self.full_tensor(expanded.dims(), compute_dtype, 0.0);
        let selected = self.pool_select(membership, expanded, zero);
        let reduction_axes = (prefix_rank + spatial..prefix_rank + 2 * spatial).collect::<Vec<_>>();
        let sum = selected.sum(reduction_axes.clone());
        let count = membership.cast(compute_dtype).sum(reduction_axes);
        Ok((sum / count).cast(in_dtype))
    }

    /// Binds (values, indices) for the max-pool family.
    pub(super) fn translate_max_pool(&mut self, node: &Node, spatial: usize) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.rank();
        anyhow::ensure!(
            rank == spatial + 1 || rank == spatial + 2,
            "max pool requires channel-first input"
        );
        let kernel = expand_spatial(&self.pool_ints(node, "kernel_size", 1)?, spatial, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&value| value > 0),
            "pool kernel must be positive"
        );
        let stride_raw = self.pool_ints(node, "stride", 2).unwrap_or_default();
        let stride = if stride_raw.is_empty() {
            kernel.clone()
        } else {
            expand_spatial(&stride_raw, spatial, 0)?
        };
        anyhow::ensure!(
            stride.iter().all(|&value| value > 0),
            "pool stride must be positive"
        );
        let padding = expand_spatial(
            &self.pool_ints(node, "padding", 3).unwrap_or_default(),
            spatial,
            0,
        )?;
        let dilation = expand_spatial(
            &self.pool_ints(node, "dilation", 4).unwrap_or_default(),
            spatial,
            1,
        )?;
        let ceil_mode = self.named_bool_arg(node, "ceil_mode").unwrap_or(false);
        let output_shape = self.output_meta_shape(node)?;

        let lowest = self.pool_lowest(input.dtype)?;
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

        let prefix_rank = rank - spatial;
        // Values-only max pools (`max_pool2d.default`) never need the
        // flattened logical indices or the argsort-based selection: a max
        // reduction over the trailing kernel axes of the windows is exact
        // and keeps the search small. The window axes were already sliced
        // to the declared output shape, so the reduction lands there.
        if node.outputs.len() <= 1 {
            let kernel_axes = (rank..rank + spatial).collect::<Vec<_>>();
            let values = windows.max(kernel_axes);
            let values = if values.dims() == output_shape {
                values
            } else {
                super::util::reshape_tensor(values, &output_shape)
            };
            return self.bind_outputs(node, vec![values]);
        }
        let input_shape = input.dims();
        let logical_indices = self.spatial_flat_indices(&input_shape, prefix_rank, spatial);
        let zero_index = self.cx.constant_i64(0);
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
        let [values, indices] = self.select_pool_max(windows, index_windows, rank)?;
        if node.outputs.len() > 1 {
            self.bind_outputs(node, vec![values, indices])
        } else {
            self.bind_outputs(node, vec![values])
        }
    }

    pub(super) fn translate_adaptive_max_pool(
        &mut self,
        node: &Node,
        spatial: usize,
    ) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let rank = input.rank();
        anyhow::ensure!(
            rank == spatial + 1 || rank == spatial + 2,
            "adaptive max pool input rank is invalid"
        );
        let input_shape = input.dims();
        let output_shape = self.output_meta_shape(node)?;
        let (expanded, membership, prefix_rank) =
            self.adaptive_pool_candidates(input, &output_shape, spatial)?;
        let output_spatial = output_shape[prefix_rank..].to_vec();
        let logical_indices = self.spatial_flat_indices(&input_shape, prefix_rank, spatial);
        let mut expanded_indices = logical_indices;
        for (index, size) in output_spatial.iter().copied().enumerate() {
            expanded_indices = expanded_indices.expand_dim(prefix_rank + index, size);
        }
        let lowest = self.pool_lowest(input.dtype)?.expand_rhs(expanded.dims());
        let candidates = self.pool_select(membership, expanded, lowest);
        let [values, indices] = self.select_pool_max(candidates, expanded_indices, rank)?;
        if node.outputs.len() > 1 {
            self.bind_outputs(node, vec![values, indices])
        } else {
            self.bind_outputs(node, vec![values])
        }
    }

    /// A tensor operand by input name (`None` for absent / non-tensor).
    fn named_tensor_arg(&mut self, node: &Node, name: &str) -> Result<Option<GraphTensor>> {
        let Some(index) = node.inputs.iter().position(|input| input.name == name) else {
            return Ok(None);
        };
        self.optional_tensor_operand(&node.inputs[index])
    }

    /// A zero tensor in the dtype/shape the named output declares.
    fn zero_for_output(&mut self, name: &str) -> Result<GraphTensor> {
        let meta = self
            .tensor_meta(name)
            .context("output metadata is missing")?
            .clone();
        let shape = self.tensor_meta_to_shape(&meta)?;
        let dtype = super::dtype_of(meta.dtype)?;
        Ok(self.full_tensor(shape, dtype, 0.0))
    }

    /// Bind every declared tensor output by name (multi-output nodes may
    /// group theirs under `as_tensors`, which `bind_outputs` cannot name).
    fn bind_declared_outputs(&mut self, node: &Node, values: &[GraphTensor]) -> Result<()> {
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            names.len() == values.len(),
            "`{}` declares {} tensor outputs but the lowering produced {}",
            node.target,
            names.len(),
            values.len()
        );
        for (name, value) in names.into_iter().zip(values.iter().copied()) {
            self.values.insert(name, value);
        }
        Ok(())
    }

    /// `_native_batch_norm_legit*` family: binds whatever outputs the node
    /// declares (out[, save_mean, save_var, .., running_mean, running_var]).
    pub(super) fn translate_batch_norm_functional(&mut self, node: &Node) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(
            input.rank() >= 2,
            "batch norm input rank must be at least two"
        );
        let output_names = Self::tensor_output_names(node);
        anyhow::ensure!(!output_names.is_empty(), "batch norm has no tensor outputs");
        let target = node.target.as_str();
        let no_training = target.ends_with("_native_batch_norm_legit_no_training.default");
        let functional = target.ends_with("_native_batch_norm_legit_functional.default")
            || target.ends_with("_batch_norm_with_update_functional.default");
        let training = if target.ends_with("_batch_norm_with_update_functional.default") {
            true
        } else if no_training {
            false
        } else {
            self.named_bool_arg(node, "training").unwrap_or(true)
        };
        // Inference is legal whenever stored running statistics exist
        // (`batch_norm.default` training=false); only the stat-less native
        // spellings truly require training.
        let running_mean = self.named_tensor_arg(node, "running_mean")?;
        let running_var = self.named_tensor_arg(node, "running_var")?;
        anyhow::ensure!(
            training
                || functional
                || no_training
                || (running_mean.is_some() && running_var.is_some()),
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
        let axes = (0..input.rank())
            .filter(|&axis| axis != 1)
            .collect::<Vec<_>>();
        let (batch_mean, batch_var) = if training {
            let batch_mean = compute.mean(axes.clone());
            let expanded_mean = batch_mean.expand_to_shape_on_axes(compute.dims(), axes.clone());
            let centered = compute - expanded_mean;
            let batch_var = centered.square().mean(axes.clone());
            (Some(batch_mean), Some(batch_var))
        } else {
            // Inference uses only the stored running statistics.
            (None, None)
        };

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
        let eps_tensor = self.constant_like(variance, eps);
        let invstd = (variance + eps_tensor).sqrt().reciprocal();
        let mean_expanded = mean.expand_to_shape_on_axes(compute.dims(), axes.clone());
        let invstd_expanded = invstd.expand_to_shape_on_axes(compute.dims(), axes.clone());
        let mut output = (compute - mean_expanded) * invstd_expanded;
        if let Some(weight) = self.named_tensor_arg(node, "weight")? {
            let weight = weight
                .cast(compute_dtype)
                .expand_to_shape_on_axes(output.dims(), axes.clone());
            output *= weight;
        }
        if let Some(bias) = self.named_tensor_arg(node, "bias")? {
            let bias = bias
                .cast(compute_dtype)
                .expand_to_shape_on_axes(output.dims(), axes.clone());
            output += bias;
        }

        let mut bound = vec![output.cast(input.dtype)];
        for (index, statistic) in [(1usize, batch_mean), (2usize, Some(invstd))] {
            if output_names.len() > index {
                bound.push(if training {
                    statistic.context("training batch norm statistic is missing")?
                } else {
                    self.zero_for_output(&output_names[index])?
                });
            }
        }

        let running_start = output_names.len().saturating_sub(2);
        if functional && output_names.len() >= 5 {
            for name in &output_names[3..running_start] {
                bound.push(self.zero_for_output(name)?);
            }
            let running_mean =
                running_mean.context("functional batch norm requires running_mean")?;
            let running_var = running_var.context("functional batch norm requires running_var")?;
            let (mean_out, var_out) = if training {
                let batch_mean = batch_mean.context("training batch norm requires batch mean")?;
                let batch_var = batch_var.context("training batch norm requires batch variance")?;
                let mean_out = running_mean * (1.0 - momentum as f32)
                    + batch_mean.cast(running_mean.dtype) * momentum as f32;
                let count: IntExpr = axes.iter().map(|&axis| input.dims()[axis]).product();
                let count_tensor = self
                    .cx
                    .constant_i32(count)
                    .cast(batch_var.dtype)
                    .expand_rhs(batch_var.dims());
                let denominator = self
                    .cx
                    .constant_i32(count - 1)
                    .cast(batch_var.dtype)
                    .expand_rhs(batch_var.dims());
                let unbiased = batch_var * count_tensor / denominator;
                let var_out = running_var * (1.0 - momentum as f32)
                    + unbiased.cast(running_var.dtype) * momentum as f32;
                (mean_out, var_out)
            } else {
                (running_mean, running_var)
            };
            bound.push(mean_out);
            bound.push(var_out);
        }
        self.bind_declared_outputs(node, &bound)
    }

    /// `aten._fused_rms_norm`: frontend `std_norm` + optional affine,
    /// computed in F32. Only `out` is returned; the `rstd` second output
    /// is left to the caller (the wired dispatch binds a single value).
    pub(super) fn translate_fused_rms_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let normalized_shape = self
            .get_ints_arg(node, 1)
            .unwrap_or_else(|_| vec![input.rank() as i64]);
        let rank = input.rank();
        let num_norm_dims = normalized_shape.len();
        anyhow::ensure!(
            num_norm_dims <= rank,
            "rms_norm normalized_shape rank {num_norm_dims} exceeds input rank {rank}"
        );
        let axes: Vec<usize> = (rank - num_norm_dims..rank).collect();
        let eps = self.get_float_arg(node, 3).unwrap_or(f32::EPSILON as f64) as f32;
        let out_dtype = input.dtype;
        let mut result = input.cast(DType::F32).std_norm(axes, eps);
        if let Some(weight) = self.named_tensor_arg(node, "weight")? {
            let weight = weight.cast(DType::F32);
            let (result_b, weight) = broadcast_binary(result, weight);
            result = result_b * weight;
        }
        Ok(result.cast(out_dtype))
    }

    /// `native_group_norm.default` -> (out, mean, rstd); the public
    /// `group_norm.default` spelling -> out, taking the same lowering with
    /// its own argument layout (input, num_groups, weight, bias, eps,
    /// cudnn). The static channel dim is split into (groups, group_size)
    /// and normalized over the group volume + spatial axes.
    pub(super) fn translate_group_norm(&mut self, node: &Node) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        // Native: (input, weight, bias, N, C, HxW, group, eps).
        // Public: (input, num_groups, weight, bias, eps, cudnn_enabled).
        let native = node.target.contains("native_group_norm");
        let num_groups = match self
            .named_int_arg(node, "num_groups")
            .or_else(|| self.named_int_arg(node, "group"))
        {
            Some(groups) => groups,
            None => self.get_int_arg(node, if native { 6 } else { 1 })?,
        };
        let num_groups =
            usize::try_from(num_groups).context("group_norm num_groups is negative")?;
        let eps = match self.named_float_arg(node, "eps") {
            Some(eps) => eps,
            None => self
                .get_float_arg(node, if native { 7 } else { 4 })
                .unwrap_or(1e-5),
        };

        let orig_dims = input.dims();
        let ndim = orig_dims.len();
        anyhow::ensure!(
            ndim >= 2,
            "group_norm expects input rank >= 2 (N, C, ...), got {ndim}"
        );
        let channels = orig_dims[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("group_norm requires a static channel dim"))?;
        anyhow::ensure!(
            num_groups != 0 && channels % num_groups == 0,
            "group_norm: num_channels ({channels}) must be a positive multiple of \
             num_groups ({num_groups})"
        );
        let group_size = channels / num_groups;

        let out_dtype = input.dtype;
        let compute_dtype = if matches!(out_dtype, DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            out_dtype
        };
        // (N, C, ...) -> (N, num_groups, group_size, ...): splitting the
        // STATIC channel dim keeps the group volume symbolic-safe. Only the
        // split itself (plus the final merge back) is a movement op — the
        // normalization is plain reductions/expands, because split/merge
        // chains rejoin in the e-graph and detonate the search.
        let mut t = input.cast(compute_dtype);
        t = t.split_dims(1, group_size);
        let group_axes: Vec<usize> = (2..t.rank()).collect();
        let mean = t.mean(group_axes.clone());
        let centered = t - mean.expand_to_shape_on_axes(t.dims(), group_axes.clone());
        let variance = centered.square().mean(group_axes.clone());
        let eps_tensor = self.constant_like(variance, eps);
        let rstd = (variance + eps_tensor).sqrt().reciprocal();
        t = centered * rstd.expand_to_shape_on_axes(t.dims(), group_axes);

        // Back to the original (N, C, ...spatial).
        t = t.merge_dims(1, 2);

        // Per-channel affine on axis 1.
        let non_channel_axes = (0..ndim).filter(|&axis| axis != 1).collect::<Vec<_>>();
        if let Some(weight) = self.named_tensor_arg(node, "weight")? {
            let weight = weight
                .cast(compute_dtype)
                .expand_to_shape_on_axes(t.dims(), non_channel_axes.clone());
            t *= weight;
        }
        if let Some(bias) = self.named_tensor_arg(node, "bias")? {
            let bias = bias
                .cast(compute_dtype)
                .expand_to_shape_on_axes(t.dims(), non_channel_axes);
            t += bias;
        }
        let out = t.cast(out_dtype);
        // Native binds (out, save_mean, save_rstd); the public spelling
        // declares only the normalized tensor, so the statistic casts are
        // minted only when a later output actually consumes them (dead
        // derived-shape casts detonate the e-graph search).
        let mut bound = vec![out];
        for (index, name) in Self::tensor_output_names(node)
            .into_iter()
            .enumerate()
            .skip(1)
        {
            bound.push(match index {
                1 => mean.cast(out_dtype),
                2 => rstd.cast(out_dtype),
                _ => bail!("group_norm has no statistic for output {name}"),
            });
        }
        self.bind_declared_outputs(node, &bound)
    }
}
