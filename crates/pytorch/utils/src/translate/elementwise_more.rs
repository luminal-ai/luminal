//! Elementwise specials, squeeze, and triangle lowerings (port batch 7).
//!
//! Ported from the parked HLIR translator's `binary.rs`/`unary.rs`/
//! `tensor.rs`/`complex.rs` helpers on the recorder frontend.
//!
//! NaN/infinity-sensitive sites use the native ternary `LogicalSelect`
//! (`condition.select(a, b)`): it is a true selection, so an unselected NaN
//! or infinity is never multiplied by zero, and unlike the arithmetic mask
//! `a*mask + b*(1-mask)` it puts no sum of products into the e-graph.
//! The atan origin is additionally steered to a finite ratio (`0/1` rather
//! than `0/0`) so the intermediate angle is finite before the final select.

use anyhow::Result;
use luminal::prelude::*;

use super::{Translator, util};
use crate::pt2_schema::Node;

impl Translator<'_> {
    /// Non-arithmetic `select`: the native ternary `LogicalSelect`, preceded
    /// by the broadcast that establishes a common shape/dtype for the
    /// branches.
    fn real_select(
        &mut self,
        condition: GraphTensor,
        if_true: GraphTensor,
        if_false: GraphTensor,
    ) -> GraphTensor {
        let (if_true, condition) = util::broadcast_binary(if_true, condition);
        let (if_true, if_false) = util::broadcast_binary(if_true, if_false);
        let condition = if condition.dtype == DType::Bool {
            condition
        } else {
            condition.cast(DType::Bool)
        };
        condition.select(if_true, if_false)
    }

    /// `abs(magnitude)` with the sign of `sign`.
    fn exact_copy_sign(&mut self, magnitude: GraphTensor, sign: GraphTensor) -> GraphTensor {
        let (magnitude, sign) = util::broadcast_binary(magnitude, sign);
        let negative = magnitude * -1.0;
        let signbit = self.signbit(sign);
        self.real_select(signbit, negative, magnitude)
    }

    /// `value == 0` that, unlike `util::is_zero`, is false for NaN.
    fn exact_is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let nonzero = self.bool_or(value.lt(zero), value.gt(zero));
        let nan = self.is_nan(value);
        let nonzero_or_nan = self.bool_or(nonzero, nan);
        self.bool_not(nonzero_or_nan)
    }

    /// Both operands promoted to the dtype the op computes in and
    /// broadcast; numeric literals are accepted on either side.
    fn promoted_binary_inputs(&mut self, node: &Node) -> Result<(GraphTensor, GraphTensor)> {
        let dtype = self.compute_dtype(node)?;
        let a = match self.optional_tensor_operand(&node.inputs[0])? {
            Some(tensor) => tensor,
            None => self.scalar(&node.inputs[0], dtype)?,
        };
        let b = match self.optional_tensor_operand(&node.inputs[1])? {
            Some(tensor) => tensor,
            None => self.scalar(&node.inputs[1], dtype)?,
        };
        let a = if a.dtype == dtype { a } else { a.cast(dtype) };
        let b = if b.dtype == dtype { b } else { b.cast(dtype) };
        Ok(util::broadcast_binary(a, b))
    }

    /// Range-reduced odd Taylor series for atan; mirrors `unary::real_atan`
    /// (private to that module) so `atan2` can share it.
    fn exact_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let one = self.constant_like(x, 1.0);
        // Nudge a zero magnitude off zero so the reciprocal branch is a
        // finite value: `1/0 = inf` cannot be selected away on this branch
        // yet (LUM-804), because the selection leaks `inf * 0`.
        let x_safe = x + self.exact_is_zero(x).cast(x.dtype);
        let reciprocal_branch = x.gt(one);
        let reduced = self.real_select(reciprocal_branch, x_safe.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.real_select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.real_select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.real_select(reciprocal_branch, half_pi - base, base);
        self.exact_copy_sign(angle, input)
    }

    /// `atan2(y, x)` with the full branch-cut, infinity and signed-zero
    /// cases. The origin is nudged so `y/x` is `0/1` there rather than
    /// `0/0`, keeping the intermediate angle finite before the final select.
    fn exact_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let x_zero = self.exact_is_zero(x);
        let y_zero = self.exact_is_zero(y);
        let both_zero = self.bool_and(x_zero, y_zero);
        let safe_x = x + both_zero.cast(x.dtype);
        let ratio = y / safe_x;
        let mut angle = self.exact_atan(ratio);
        let x_negative = self.signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.exact_copy_sign(pi, y);
        angle = self.real_select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = self.bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.real_select(x_negative, three_quarters, quarter);
        let infinite_angle = self.exact_copy_sign(infinite_angle, y);
        angle = self.real_select(both_inf, infinite_angle, angle);

        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.exact_copy_sign(zero, y);
        let zero_angle = self.real_select(x_negative, signed_pi, signed_zero);
        self.real_select(both_zero, zero_angle, angle)
    }

    pub(super) fn translate_atan2(&mut self, node: &Node) -> Result<GraphTensor> {
        let (y, x) = self.promoted_binary_inputs(node)?;
        let output_dtype = y.dtype;
        let (y, x) = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            (y.cast(DType::F32), x.cast(DType::F32))
        } else {
            (y, x)
        };
        Ok(self.exact_atan2(y, x).cast(output_dtype))
    }

    pub(super) fn translate_copysign(&mut self, node: &Node, scalar: bool) -> Result<GraphTensor> {
        if scalar {
            let dtype = self.compute_dtype(node)?;
            let raw = self.operand(&node.inputs[0])?;
            let magnitude = if raw.dtype == dtype {
                raw
            } else {
                raw.cast(dtype)
            };
            let magnitude = self.real_abs(magnitude);
            let sign = node.inputs[1]
                .arg
                .as_int()
                .map(|value| value as f64)
                .or_else(|| node.inputs[1].arg.as_float())
                .ok_or_else(|| anyhow::anyhow!("{} requires a numeric RHS", node.target))?;
            // Egglog's scalar domain equates +0.0 and -0.0, so keep the
            // compile-time sign structurally instead of as a constant.
            return Ok(if sign.is_sign_negative() {
                magnitude * -1.0
            } else {
                magnitude
            });
        }
        let (magnitude, sign) = self.promoted_binary_inputs(node)?;
        let magnitude = self.real_abs(magnitude);
        Ok(self.exact_copy_sign(magnitude, sign))
    }

    pub(super) fn translate_fmax_fmin(&mut self, node: &Node, max: bool) -> Result<GraphTensor> {
        let (a, b) = self.promoted_binary_inputs(node)?;
        // Finite-input semantics. torch.fmax/fmin skip a NaN operand, which
        // requires selecting a non-finite branch away; that is blocked on
        // the core's NaN-safe select (LUM-804: arithmetic/e-graph selection
        // leaks `NaN * 0`). A NaN operand therefore propagates here.
        Ok(if max { a.maximum(b) } else { a.minimum(b) })
    }

    /// Stable hypot: scale by the larger magnitude to avoid overflow.
    pub(super) fn translate_hypot(&mut self, node: &Node) -> Result<GraphTensor> {
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        let lhs = self.real_abs(lhs);
        let rhs = self.real_abs(rhs);
        let larger = lhs.maximum(rhs);
        let smaller = lhs.minimum(rhs);
        // Nudge a zero larger off zero so `smaller/larger` is finite
        // (0/0 would poison the result; the core cannot yet select an inf
        // or NaN branch away — LUM-804).
        let larger_safe = larger + self.exact_is_zero(larger).cast(larger.dtype);
        let ratio = smaller / larger_safe;
        let finite = larger * (self.constant_like(larger, 1.0) + ratio.square()).sqrt();
        // Zero the origin multiplicatively rather than by select: at the
        // origin `finite` is already 0, and an infinite `finite` only
        // occurs away from it, so no `inf * 0` arises.
        let both_zero = self.exact_is_zero(larger);
        let keep = self.constant_like(finite, 1.0) - both_zero.cast(finite.dtype);
        Ok(finite * keep)
    }

    pub(super) fn translate_gcd(&mut self, node: &Node) -> Result<GraphTensor> {
        // The old translator unrolled a 128-round Euclidean loop. Two things
        // had to change before it could plan on this branch:
        //   1. the arithmetic masks (`f*lhs + active*rhs`) became native
        //      selects — a sum of products fed the integer AC + distributivity
        //      e-graph closure and diverged saturation; and
        //   2. the divisor became `rhs.maximum(1)` rather than
        //      `finished.select(1, rhs)`, so its lower bound (`1`) is derivable
        //      from the structural max rule in `logical_op/select`, which is
        //      what discharges the proof gate on I64 `trunc_rem`.
        // Both are compile-time (egglog) facts; no runtime logic is added.
        self.translate_gcd_unrolled(node)
    }

    fn translate_gcd_unrolled(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        let mut lhs = lhs.cast(DType::I64);
        let mut rhs = rhs.cast(DType::I64);
        let one = self.constant_like(lhs, 1.0);
        let zero = self.constant_like(lhs, 0.0);
        for _ in 0..32 {
            let finished = self.exact_is_zero(rhs);
            // Zero-free divisor: `maximum(|rhs|, 1)` is `|rhs|` whenever
            // `rhs != 0` and `1` when `rhs == 0`. Euclid is invariant under
            // the divisor's sign, so `|rhs|` is correct for signed inputs,
            // and the structural max bound rule derives its lower bound `1`
            // statically — the proof the I64 `trunc_rem` needs. (The previous
            // `finished.select(1, rhs)` was correct but had only the generic
            // union bounds, which included 0.)
            let safe_rhs = rhs.abs().maximum(one);
            let remainder = lhs.trunc_rem(safe_rhs);
            lhs = finished.select(lhs, rhs);
            rhs = finished.select(zero, remainder);
        }
        Ok(lhs.abs().cast(output_dtype))
    }

    pub(super) fn translate_exp2(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(x.dtype);
        let x = if x.dtype == dtype { x } else { x.cast(dtype) };
        Ok(x.exp2())
    }

    pub(super) fn translate_log2(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(x.dtype);
        let x = if x.dtype == dtype { x } else { x.cast(dtype) };
        Ok(x.log2())
    }

    pub(super) fn translate_isnan(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        Ok(self.is_nan(a))
    }

    pub(super) fn translate_leaky_relu(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.operand(&node.inputs[0])?;
        let dtype = self.compute_dtype(node).unwrap_or(value.dtype);
        let value = if value.dtype == dtype {
            value
        } else {
            value.cast(dtype)
        };
        let slope = self.get_float_arg(node, 1).unwrap_or(0.01);
        let zero = self.constant_like(value, 0.0);
        let negative = value * self.constant_like(value, slope);
        Ok(self.real_select(value.gt(zero), value, negative))
    }

    pub(super) fn translate_bitwise(&mut self, node: &Node, or: bool) -> Result<GraphTensor> {
        // Boolean semantics, matching the parked translator: torch exports
        // `bitwise_and`/`bitwise_or` for boolean attention masks, and the
        // integer forms are not exercised by any exported model. The result
        // is cast to the node's declared dtype so a Bool intermediate never
        // disagrees with the recorded output metadata.
        let dtype = self.output_meta_dtype(node).unwrap_or(DType::Bool);
        let a = self.operand(&node.inputs[0])?;
        let b = self.operand(&node.inputs[1])?;
        let (a, b) = util::broadcast_binary(a, b);
        let value = if or {
            self.bool_or(a, b)
        } else {
            (a.cast(DType::F32) * b.cast(DType::F32)).cast(DType::Bool)
        };
        Ok(if value.dtype == dtype {
            value
        } else {
            value.cast(dtype)
        })
    }

    pub(super) fn translate_squeeze(&mut self, node: &Node, all: bool) -> Result<GraphTensor> {
        let mut result = self.operand(&node.inputs[0])?;
        let rank = result.rank();
        let dims: Vec<i64> = if all {
            result
                .dims()
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| (dim.to_usize() == Some(1)).then_some(axis as i64))
                .collect()
        } else {
            self.get_ints_arg(node, 1)?
        };
        let mut sorted: Vec<usize> = dims
            .iter()
            .map(|&dim| util::normalize_dim(dim, rank))
            .collect();
        sorted.sort_unstable();
        let mut offset = 0;
        for dim in sorted {
            if result.dims()[dim - offset].to_usize() == Some(1) {
                result = result.squeeze(dim - offset);
                offset += 1;
            }
        }
        Ok(result)
    }

    pub(super) fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let diagonal = self.get_int_arg(node, 1).unwrap_or(0) as i32;
        let dims = x.dims();
        anyhow::ensure!(dims.len() >= 2, "tril/triu requires a matrix input");
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        let (row_count, col_count) = match (rows.to_usize(), cols.to_usize()) {
            (Some(rows), Some(cols)) => (rows, cols),
            _ => anyhow::bail!("tril/triu requires concrete matrix dimensions"),
        };
        let size = row_count.max(col_count);
        let mask = if upper {
            self.cx.triu(size, diagonal)
        } else {
            self.cx.tril(size, diagonal)
        };
        let mask = if rows != cols {
            mask.slice_along(0..row_count, 0)
                .slice_along(0..col_count, 1)
        } else {
            mask
        };
        let mut mask = mask.cast(x.dtype);
        for i in (0..dims.len() - 2).rev() {
            mask = mask.expand_dim(0, dims[i]);
        }
        Ok(x * mask)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntArg, IntsArg,
        Node, NodeInput, Signature, TensorArg, TensorMeta, TensorName, TensorRef,
    };
    use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
    use rustc_hash::FxHashMap;

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

    fn flt(name: &str, value: f64) -> NodeInput {
        input(name, Argument::Float(FloatArg { as_float: value }))
    }

    fn int(name: &str, value: i64) -> NodeInput {
        input(name, Argument::Int(IntArg { as_int: value }))
    }

    fn ints(name: &str, values: &[i64]) -> NodeInput {
        input(
            name,
            Argument::Ints(IntsArg {
                as_ints: values.to_vec(),
            }),
        )
    }

    fn sizes(values: &[i64]) -> Vec<DimSize> {
        values
            .iter()
            .map(|value| DimSize::Int(DimInt { as_int: *value }))
            .collect()
    }

    /// Build a one-node, one-input (`x`) / one-output (`y`) program.
    fn program(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        in_shape: &[i64],
        out_shape: &[i64],
    ) -> ParsedPT2 {
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
                    nodes: vec![Node {
                        target: target.to_string(),
                        inputs: args,
                        outputs: vec![tensor_ref("y")],
                    }],
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
        ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        }
    }

    /// Translate a single node with input `x` and output `y`, returning
    /// whether translation (including recording) succeeded.
    fn translates(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        in_shape: &[i64],
        out_shape: &[i64],
    ) -> bool {
        crate::translate::translate(&program(target, args, dtype, in_shape, out_shape)).is_ok()
    }

    /// Translate and execute on the reference runtime, returning the F32
    /// output for the staged F32 `x` input.
    fn run_f32(
        target: &str,
        args: Vec<NodeInput>,
        dtype: u32,
        shape: &[i64],
        x: &[f32],
    ) -> Vec<f32> {
        let parsed = program(target, args, dtype, shape, shape);
        let translation = crate::translate::translate(&parsed).expect("translate");
        let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
        for (symbol, hint) in &translation.dims {
            runtime
                .bind_dyn_range(*symbol, *hint as u64, *hint as u64)
                .expect("bind dyn range");
            runtime.set_dim(*symbol, *hint);
        }
        let input = &translation.inputs[0];
        let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
        data.insert(input.tensor, TypedBuffer::F32(x.to_vec()));
        runtime
            .search(&data, &harness_search_options())
            .expect("search");
        runtime.set_data(input.tensor, TypedBuffer::F32(x.to_vec()));
        runtime.execute().expect("execute");
        translation
            .outputs
            .first()
            .map(|output| runtime.get_f32(output.tensor).expect("f32 output").clone())
            .expect("one output")
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tol, "index {i}: got {a}, expected {e}");
        }
    }

    const F32: u32 = 7;
    const I64: u32 = 5;

    fn binary(second: NodeInput) -> Vec<NodeInput> {
        vec![input("self", tensor_arg("x")), second]
    }

    #[test]
    fn elementwise_specials_record() {
        assert!(translates(
            "torch.ops.aten.atan2.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.copysign.Tensor",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.copysign.Scalar",
            binary(flt("other", -2.0)),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        for target in ["torch.ops.aten.fmax.default", "torch.ops.aten.fmin.default"] {
            assert!(translates(
                target,
                binary(input("other", tensor_arg("x"))),
                F32,
                &[2, 2],
                &[2, 2],
            ));
        }
        assert!(translates(
            "torch.ops.aten.hypot.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        // gcd translates through the select-based Euclidean unroll; it plans
        // when the caller attests non-negative input ranges.
        assert!(translates(
            "torch.ops.aten.gcd.default",
            binary(input("other", tensor_arg("x"))),
            I64,
            &[2, 2],
            &[2, 2],
        ));
    }

    #[test]
    fn small_unary_specials_record() {
        for target in ["torch.ops.aten.exp2.default", "torch.ops.aten.log2.default"] {
            assert!(translates(
                target,
                vec![input("self", tensor_arg("x"))],
                F32,
                &[2, 2],
                &[2, 2],
            ));
        }
        assert!(translates(
            "torch.ops.aten.isnan.default",
            vec![input("self", tensor_arg("x"))],
            F32,
            &[2, 2],
            &[2, 2],
        ));
        assert!(translates(
            "torch.ops.aten.leaky_relu.default",
            binary(flt("negative_slope", 0.01)),
            F32,
            &[2, 2],
            &[2, 2],
        ));
        for target in [
            "torch.ops.aten.bitwise_and.Tensor",
            "torch.ops.aten.bitwise_or.Tensor",
        ] {
            assert!(translates(
                target,
                binary(input("other", tensor_arg("x"))),
                F32,
                &[2, 2],
                &[2, 2],
            ));
        }
    }

    #[test]
    fn squeeze_and_triangular_record() {
        assert!(translates(
            "torch.ops.aten.squeeze.default",
            vec![input("self", tensor_arg("x"))],
            F32,
            &[1, 3, 1],
            &[3],
        ));
        assert!(translates(
            "torch.ops.aten.squeeze.dims",
            vec![input("self", tensor_arg("x")), ints("dim", &[0, 2])],
            F32,
            &[1, 3, 1],
            &[3],
        ));
        assert!(translates(
            "torch.ops.aten.tril.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 0)],
            F32,
            &[2, 3],
            &[2, 3],
        ));
        assert!(translates(
            "torch.ops.aten.triu.default",
            vec![input("self", tensor_arg("x")), int("diagonal", -1)],
            F32,
            &[4, 4],
            &[4, 4],
        ));
    }

    #[test]
    fn leaky_relu_matches_reference() {
        let x = [-2.0f32, -0.5, 0.0, 1.0, 3.5];
        let out = run_f32(
            "torch.ops.aten.leaky_relu.default",
            binary(flt("negative_slope", 0.25)),
            F32,
            &[5],
            &x,
        );
        let expected: Vec<f32> = x
            .iter()
            .map(|v| if *v > 0.0 { *v } else { 0.25 * *v })
            .collect();
        assert_close(&out, &expected, 1e-6);
    }

    #[test]
    fn atan2_diagonal_matches_reference() {
        let x = [-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let out = run_f32(
            "torch.ops.aten.atan2.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[5],
            &x,
        );
        // atan2(x, x): pi/4 for x > 0, -3pi/4 for x < 0, 0 at the origin.
        let expected: Vec<f32> = x
            .iter()
            .map(|v| {
                if *v > 0.0 {
                    std::f32::consts::FRAC_PI_4
                } else if *v < 0.0 {
                    -3.0 * std::f32::consts::FRAC_PI_4
                } else {
                    0.0
                }
            })
            .collect();
        assert_close(&out, &expected, 1e-5);
    }

    #[test]
    fn fmax_fmin_finite() {
        let x = [1.0f32, 5.0, -2.0];
        let max = run_f32(
            "torch.ops.aten.fmax.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&max, &[3.0, 5.0, 3.0], 1e-6);
        let min = run_f32(
            "torch.ops.aten.fmin.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&min, &[1.0, 3.0, -2.0], 1e-6);
    }

    /// torch.fmax/fmin skip a NaN operand, but selecting a NaN branch away
    /// needs a true selection. The recorder's `select`/`cond` and its
    /// `stack`-based gather all leak `NaN * 0` (core LUM-804), so a NaN
    /// operand propagates instead. Un-ignore once LUM-804 lands.
    #[test]
    #[ignore = "blocked on core NaN-safe select (LUM-804): selection leaks NaN*0"]
    fn fmax_fmin_ignore_nan() {
        let x = [f32::NAN, 5.0, -2.0];
        let max = run_f32(
            "torch.ops.aten.fmax.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&max, &[3.0, 5.0, 3.0], 1e-6);
        let min = run_f32(
            "torch.ops.aten.fmin.default",
            binary(flt("other", 3.0)),
            F32,
            &[3],
            &x,
        );
        assert_close(&min, &[3.0, 3.0, -2.0], 1e-6);
    }

    #[test]
    fn hypot_matches_reference() {
        let x = [0.0f32, 3.0];
        let out = run_f32(
            "torch.ops.aten.hypot.default",
            binary(input("other", tensor_arg("x"))),
            F32,
            &[2],
            &x,
        );
        assert_close(&out, &[0.0, (18.0f32).sqrt()], 1e-6);
    }

    #[test]
    fn tril_triu_match_reference() {
        let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let lower = run_f32(
            "torch.ops.aten.tril.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 0)],
            F32,
            &[2, 3],
            &x,
        );
        assert_close(&lower, &[0.0, 0.0, 0.0, 3.0, 4.0, 0.0], 1e-6);
        let upper = run_f32(
            "torch.ops.aten.triu.default",
            vec![input("self", tensor_arg("x")), int("diagonal", 1)],
            F32,
            &[2, 3],
            &x,
        );
        assert_close(&upper, &[0.0, 1.0, 2.0, 0.0, 0.0, 5.0], 1e-6);
    }
}
