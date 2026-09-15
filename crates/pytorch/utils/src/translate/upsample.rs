//! Upsample / resize lowerings (port batch 6).
//!
//! `upsample_nearest2d.vec`, `upsample_bilinear2d.vec`, and
//! `_upsample_bilinear2d_aa.default` on NCHW inputs. Spatial dims must be
//! static; the nearest path is pure shape movement when the scale is an
//! integer factor, otherwise a coordinate gather.

use anyhow::{Context, Result};
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, Node, NodeInput};

/// A PT2 float-list argument (`{"as_floats": [...]}` or the symbolic
/// spelling). The schema has no dedicated Floats variant, so these arrive
/// as `Argument::Other`. Absent/unparseable lists yield `None` so callers
/// can fall back to the `in/out` scale.
fn float_list_arg(input: &NodeInput) -> Option<Vec<f64>> {
    let Argument::Other(value) = &input.arg else {
        return None;
    };
    for key in ["as_floats", "as_sym_floats"] {
        if let Some(array) = value.get(key).and_then(|v| v.as_array()) {
            let parsed: Option<Vec<f64>> = array
                .iter()
                .map(|entry| {
                    entry
                        .as_f64()
                        .or_else(|| entry.get("as_float").and_then(|v| v.as_f64()))
                })
                .collect();
            if let Some(values) = parsed {
                return Some(values);
            }
        }
    }
    None
}

/// Concrete `[input_height, input_width, output_height, output_width]`.
fn static_resize_dimensions(
    input: &GraphTensor,
    output_shape: &[IntExpr],
    operation: &str,
) -> Result<[usize; 4]> {
    anyhow::ensure!(input.rank() == 4, "{operation} requires a 4D NCHW input");
    anyhow::ensure!(output_shape.len() == 4, "{operation} requires a 4D output");
    let input_height = input.dims()[2]
        .to_usize()
        .with_context(|| format!("{operation} requires a static input height"))?;
    let input_width = input.dims()[3]
        .to_usize()
        .with_context(|| format!("{operation} requires a static input width"))?;
    let output_height = output_shape[2]
        .to_usize()
        .with_context(|| format!("{operation} requires a static output height"))?;
    let output_width = output_shape[3]
        .to_usize()
        .with_context(|| format!("{operation} requires a static output width"))?;
    anyhow::ensure!(
        input_height != 0 && input_width != 0 && output_height != 0 && output_width != 0,
        "{operation} requires non-zero spatial dims \
         (in {input_height}x{input_width} -> out {output_height}x{output_width})"
    );
    Ok([input_height, input_width, output_height, output_width])
}

impl Translator<'_> {
    /// Wrap negative gather indices into `[0, axis_dim)`. Mirrors the old
    /// translator's F32-based normalization (Int arithmetic is proof-gated);
    /// the upsample callers never emit negative indices, so this is a no-op
    /// numerically for them.
    fn upsample_normalize_index(&mut self, indices: GraphTensor, axis_dim: IntExpr) -> GraphTensor {
        let idx_f32 = indices.cast(DType::F32);
        let zero = self.cx.constant_f32(0.0).expand_rhs(idx_f32.dims());
        let adjustment = self
            .cx
            .constant_i32(axis_dim)
            .cast(DType::F32)
            .expand_rhs(idx_f32.dims());
        let is_negative = idx_f32.lt(zero).cast(DType::F32);
        (idx_f32 + is_negative * adjustment).trunc_cast(DType::Int)
    }

    /// GatherElements along `axis` in coordinate form: one Int coordinate
    /// tensor per data axis over the index shape. Non-axis coordinates are
    /// the output iota; the axis coordinate is the (normalized) index.
    fn upsample_gather_elements(
        &mut self,
        data: GraphTensor,
        indices: GraphTensor,
        axis: usize,
    ) -> GraphTensor {
        let out_dims = indices.dims();
        let mut coordinates = Vec::with_capacity(data.rank());
        for dim in 0..data.rank() {
            if dim == axis {
                coordinates.push(self.upsample_normalize_index(indices, data.dims()[dim]));
            } else {
                coordinates.push(self.axis_positions(&out_dims, dim));
            }
        }
        data.gather(&coordinates)
    }

    pub(super) fn translate_upsample_nearest2d(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        anyhow::ensure!(
            input.rank() == 4,
            "upsample_nearest2d expects a 4D (N, C, H, W) input, got {}D",
            input.rank()
        );

        let input_height = input.dims()[2]
            .to_usize()
            .context("upsample_nearest2d requires a static input height")?;
        let input_width = input.dims()[3]
            .to_usize()
            .context("upsample_nearest2d requires a static input width")?;

        let output_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            output_shape.len() == 4,
            "upsample_nearest2d expects a 4D output, got {}D",
            output_shape.len()
        );
        let output_height = output_shape[2]
            .to_usize()
            .context("upsample_nearest2d requires a static output height")?;
        let output_width = output_shape[3]
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
        let scales: Option<(f64, f64)> =
            node.inputs
                .get(2)
                .and_then(float_list_arg)
                .and_then(|values| match values.as_slice() {
                    [height, width] => Some((*height, *width)),
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
        let idx = (self.cx.arange(out_dim).cast(DType::F32) * scale_inv)
            .minimum_f32((in_dim - 1) as f32)
            .trunc_cast(DType::Int);
        let mut idx = idx;
        for (dim, &size) in t.dims().iter().enumerate() {
            if dim != axis {
                idx = idx.expand_dim(dim, size);
            }
        }
        Ok(self.upsample_gather_elements(t, idx, axis))
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
        let positions = self.cx.arange(output_size).cast(DType::F32);
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
        let lower = source.trunc_cast(DType::Int);
        // Clamp in F32 (Int `minimum` is proof-gated and unplannable for
        // caller data); `floor(source + 1)` equals `floor(source) + 1` for
        // `source >= 0`.
        let upper = (source + 1.0)
            .minimum_f32((input_size - 1) as f32)
            .trunc_cast(DType::Int);
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
        let left = self.upsample_gather_elements(input, lower, axis);
        let right = self.upsample_gather_elements(input, upper, axis);
        left + (right - left) * weight
    }

    pub(super) fn translate_upsample_bilinear2d(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        anyhow::ensure!(input.rank() == 4, "bilinear2d requires NCHW input");
        let output_shape = self.output_meta_shape(node)?;
        let [input_height, input_width, output_height, output_width] =
            static_resize_dimensions(&input, &output_shape, "bilinear2d")?;
        let align_corners = self.get_bool_arg(node, 2)?;
        let scales: Option<(f64, f64)> =
            node.inputs
                .get(3)
                .and_then(float_list_arg)
                .and_then(|values| match values.as_slice() {
                    [height, width] => Some((*height, *width)),
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
            .cx
            .arange(output_size)
            .cast(weight_dtype)
            .expand_dim(1, input_size);
        let input_positions = self
            .cx
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
        // `cond` keeps the weight dtype exact (util::select projects the
        // condition through F32, which would mix dtypes for F64/F16).
        let mut weights = unbounded.cond(positive, zero);
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
            weights = (weights * multiplier + half).floor().trunc_cast(DType::I64);
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
        for dim in axis + 1..input.rank() {
            weights = weights.expand_dim(dim + 1, input.dims()[dim]);
        }
        candidates *= weights;
        let result = candidates.sum(axis + 1);
        if let Some(precision) = weights_precision {
            let result = result.cast(DType::F64);
            let bias = self.constant_like(result, (1_u64 << (precision - 1)) as f64);
            let divisor = self.constant_like(result, (1_u64 << precision) as f64);
            let rounded = ((result + bias) / divisor).floor();
            let zero = self.constant_like(rounded, 0.0);
            let maximum = self.constant_like(rounded, u8::MAX as f64);
            let lower = zero.cond(rounded.lt(zero), rounded);
            maximum
                .cond(lower.gt(maximum), lower)
                .trunc_cast(DType::I64)
                .cast(DType::U8)
        } else {
            result
        }
    }

    pub(super) fn translate_upsample_bilinear2d_aa(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        anyhow::ensure!(
            input.rank() == 4,
            "antialiased bilinear2d requires NCHW input"
        );
        let output_shape = self.output_meta_shape(node)?;
        let [input_height, input_width, output_height, output_width] =
            static_resize_dimensions(&input, &output_shape, "antialiased bilinear2d")?;
        let align_corners = self.get_bool_arg(node, 2)?;
        // `.vec` carries one float list (arg 3); `.default` carries two
        // optional single floats (args 3 and 4).
        let (scale_height, scale_width) =
            match node.inputs.get(3).and_then(float_list_arg).as_deref() {
                Some([height, width]) => (Some(*height), Some(*width)),
                _ => (
                    node.inputs.get(3).and_then(|input| input.arg.as_float()),
                    node.inputs.get(4).and_then(|input| input.arg.as_float()),
                ),
            };
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, BoolArg, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntsArg,
        Node, NodeInput, Signature, TensorArg, TensorMeta, TensorName, TensorRef,
    };

    fn tensor_ref(name: &str) -> TensorRef {
        TensorRef {
            as_tensor: Some(TensorName {
                name: name.to_string(),
            }),
            as_tensors: None,
            as_sym_int: None,
            as_sym_float: None,
            as_sym_bool: None,
        }
    }

    fn tensor_arg(name: &str) -> Argument {
        Argument::Tensor(TensorArg {
            as_tensor: TensorName {
                name: name.to_string(),
            },
        })
    }

    fn input(name: &str, arg: Argument) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            arg,
            kind: 1,
        }
    }

    fn sizes(values: &[i64]) -> Vec<DimSize> {
        values
            .iter()
            .map(|value| DimSize::Int(DimInt { as_int: *value }))
            .collect()
    }

    /// Translate a single upsample node on `[1, 1, in_h, in_w]` producing
    /// `out_shape`, returning whether translation succeeded.
    fn translates(node: Node, dtype: u32, in_shape: &[i64], out_shape: &[i64]) -> bool {
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "x".to_string(),
            TensorMeta {
                dtype,
                sizes: sizes(in_shape),
            },
        );
        tensor_values.insert(
            "y".to_string(),
            TensorMeta {
                dtype,
                sizes: sizes(out_shape),
            },
        );
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: Graph {
                    inputs: vec![tensor_ref("x")],
                    outputs: vec![tensor_ref("y")],
                    nodes: vec![node],
                    tensor_values,
                    sym_int_values: HashMap::new(),
                },
                signature: Signature {
                    input_specs: Vec::new(),
                    output_specs: Vec::new(),
                },
            },
            range_constraints: HashMap::new(),
        };
        let parsed = ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        };
        crate::translate::translate(&parsed).is_ok()
    }

    fn upsample_node(target: &str, args: Vec<NodeInput>) -> Node {
        Node {
            target: target.to_string(),
            inputs: args,
            outputs: vec![tensor_ref("y")],
        }
    }

    fn ints(name: &str, values: &[i64]) -> NodeInput {
        input(
            name,
            Argument::Ints(IntsArg {
                as_ints: values.to_vec(),
            }),
        )
    }

    fn align_corners() -> NodeInput {
        input("align_corners", Argument::Bool(BoolArg { as_bool: false }))
    }

    #[test]
    fn nearest_integer_factor_and_gather_paths_record() {
        // Integer factor: pure shape movement.
        assert!(translates(
            upsample_node(
                "torch.ops.aten.upsample_nearest2d.vec",
                vec![
                    input("input", tensor_arg("x")),
                    ints("output_size", &[6, 6]),
                    input(
                        "scale_factors",
                        Argument::Other(serde_json::json!({"as_floats": [2.0, 2.0]})),
                    ),
                ],
            ),
            7,
            &[1, 1, 3, 3],
            &[1, 1, 6, 6],
        ));
        // Non-integral ratio: coordinate gather.
        assert!(translates(
            upsample_node(
                "torch.ops.aten.upsample_nearest2d.vec",
                vec![
                    input("input", tensor_arg("x")),
                    ints("output_size", &[5, 5]),
                ],
            ),
            7,
            &[1, 1, 3, 3],
            &[1, 1, 5, 5],
        ));
    }

    #[test]
    fn bilinear_and_antialias_record() {
        assert!(translates(
            upsample_node(
                "torch.ops.aten.upsample_bilinear2d.vec",
                vec![
                    input("input", tensor_arg("x")),
                    ints("output_size", &[6, 6]),
                    align_corners(),
                    input(
                        "scale_factors",
                        Argument::Other(serde_json::json!({"as_floats": [2.0, 2.0]})),
                    ),
                ],
            ),
            7,
            &[1, 1, 3, 3],
            &[1, 1, 6, 6],
        ));
        assert!(translates(
            upsample_node(
                "torch.ops.aten._upsample_bilinear2d_aa.default",
                vec![
                    input("input", tensor_arg("x")),
                    ints("output_size", &[6, 6]),
                    align_corners(),
                    input("scales_h", Argument::Float(FloatArg { as_float: 2.0 })),
                    input("scales_w", Argument::Float(FloatArg { as_float: 2.0 })),
                ],
            ),
            7,
            &[1, 1, 3, 3],
            &[1, 1, 6, 6],
        ));
        // The uint8 kernel quantizes the F64 weight matrix.
        assert!(translates(
            upsample_node(
                "torch.ops.aten._upsample_bilinear2d_aa.default",
                vec![
                    input("input", tensor_arg("x")),
                    ints("output_size", &[6, 6]),
                    align_corners(),
                    input("scales_h", Argument::Float(FloatArg { as_float: 1.5 })),
                    input("scales_w", Argument::Float(FloatArg { as_float: 1.5 })),
                ],
            ),
            1,
            &[1, 1, 4, 4],
            &[1, 1, 6, 6],
        ));
    }
}
