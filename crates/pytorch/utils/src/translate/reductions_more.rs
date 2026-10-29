//! Reductions, linalg, copies, and empty constructors (port batch 8).
//!
//! Ports the parked translator's `any.*`, `var_mean.*`, the `addmm` family,
//! `copy.default`, the materialized `*_copy` views, and the uninitialized
//! `empty*` constructors onto the native recorder frontend.

use anyhow::Result;
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::{Argument, Node};

/// Which linear-algebra composite an `addmm`-family target is.
#[derive(Clone, Copy)]
#[allow(clippy::enum_variant_names)] // names carry the ATen spelling (addmm/addbmm/addmv)
pub(super) enum AddMmVariant {
    AddMm,
    AddBmm,
    AddMv,
}

/// Normalize a possibly negative axis list against `rank`.
fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&a| {
            let a = if a < 0 { a + rank as i64 } else { a };
            usize::try_from(a)
                .ok()
                .filter(|a| *a < rank)
                .ok_or_else(|| anyhow::anyhow!("axis {a} out of range for rank {rank}"))
        })
        .collect()
}

/// An explicit `None` argument (PT2 spells it `{"as_none": ...}`).
fn is_none_arg(arg: &Argument) -> bool {
    matches!(arg, Argument::Other(value) if value.is_null() || value.get("as_none").is_some())
}

/// Re-insert the reduced axes as size-1 extents (keepdim).
fn keep_reduced(mut t: GraphTensor, axes: &[usize]) -> GraphTensor {
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    for axis in sorted {
        t = t.expand_dim(axis, 1usize);
    }
    t
}

impl Translator<'_> {
    /// `any.default` / `any.dim` / `any.dims`: truth-value reduction to Bool.
    ///
    /// An explicit empty dim list (`any.dims(dim=[])`) is an elementwise bool
    /// cast, not a full reduction; a missing/default dim list reduces every
    /// axis. The sum is carried in F32 because Int `minimum`/`maximum` are
    /// proof-gated on this branch.
    pub(super) fn translate_any(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        let rank = input.rank();

        let (axes, keepdim) = if node.target.ends_with("any.default") {
            ((0..rank).collect::<Vec<_>>(), false)
        } else if node.target.ends_with("any.dim") {
            let dim = self.get_int_arg(node, 1)?;
            let keepdim = self
                .named_bool_arg(node, "keepdim")
                .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
                .unwrap_or(false);
            if rank == 0 {
                anyhow::ensure!(
                    matches!(dim, -1 | 0),
                    "any dimension {dim} out of range for a scalar"
                );
                (Vec::new(), keepdim)
            } else {
                anyhow::ensure!(
                    dim >= -(rank as i64) && dim < rank as i64,
                    "any dimension {dim} out of range for rank {rank}"
                );
                (vec![util::normalize_dim(dim, rank)], keepdim)
            }
        } else {
            // `any.dims`; an absent/unreadable list means "reduce everything".
            let keepdim = self
                .named_bool_arg(node, "keepdim")
                .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
                .unwrap_or(false);
            let axes = match self.get_ints_arg(node, 1) {
                Ok(dims) => {
                    let mut axes = Vec::with_capacity(dims.len());
                    for dim in dims {
                        anyhow::ensure!(
                            dim >= -(rank as i64) && dim < rank as i64,
                            "any dimension {dim} out of range for rank {rank}"
                        );
                        let axis = util::normalize_dim(dim, rank);
                        anyhow::ensure!(!axes.contains(&axis), "any dimensions must be unique");
                        axes.push(axis);
                    }
                    axes
                }
                Err(_) => (0..rank).collect(),
            };
            (axes, keepdim)
        };

        let zero = self.constant_like(input, 0.0);
        let truth = input.ne(zero);
        if axes.is_empty() {
            return Ok(truth);
        }

        let counts = truth.cast(DType::F32).sum(&axes);
        let zero = self.cx.constant_f32(0.0).expand_rhs(counts.dims());
        let result = counts.gt(zero);
        Ok(if keepdim {
            keep_reduced(result, &axes)
        } else {
            result
        })
    }

    /// `var.*` / `std.*`: the shared variance lowering, square-rooted for
    /// `std` (torch takes the root in the accumulate type, before rounding).
    pub(super) fn translate_var(&mut self, node: &Node, std: bool) -> Result<GraphTensor> {
        let (_, axes, variance) = self.lower_variance(node)?;
        let result = if std { variance.sqrt() } else { variance };
        Ok(if self.keepdim_flag(node, 3) {
            keep_reduced(result, &axes)
        } else {
            result
        })
    }

    /// `var_mean.default` / `var_mean.dim` / `var_mean.correction` -> (var, mean).
    pub(super) fn translate_var_mean(&mut self, node: &Node) -> Result<()> {
        let (input, axes, variance) = self.lower_variance(node)?;
        let mean = input.mean(&axes);
        let (variance, mean) = if self.keepdim_flag(node, 3) {
            (keep_reduced(variance, &axes), keep_reduced(mean, &axes))
        } else {
            (variance, mean)
        };
        self.bind_outputs(node, vec![variance, mean])
    }

    /// The ONE variance lowering behind `var.*`, `std.*` and `var_mean.*`
    /// (`std_mean` has no arm and reaches `var_mean` through its
    /// decomposition). Returns the operand at the compute dtype (opmath
    /// widens half precision to F32, torch's accumulate type), the reduced
    /// axes, and the variance over them. The correction stays real-valued
    /// all the way into core `var_correction`, which owns the divisor rule
    /// `max(0, N - correction)` and the NaN of an empty reduction.
    fn lower_variance(&mut self, node: &Node) -> Result<(GraphTensor, Vec<usize>, GraphTensor)> {
        let input = self.operand(&node.inputs[0])?;
        let axes = self.reduction_axes(node, 1, input.rank())?;
        let correction = self.variance_correction(node)?;
        let dtype = self.compute_dtype(node).unwrap_or(input.dtype);
        let input = input.cast(dtype);
        let variance = input.var_correction(&axes, correction);
        Ok((input, axes, variance))
    }

    /// `addmm` / `addbmm` / `addmv`:
    /// `beta * input + alpha * product`, with `product` the variant's matmul.
    pub(super) fn translate_addmm(
        &mut self,
        node: &Node,
        variant: AddMmVariant,
    ) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        let lhs = self.operand(&node.inputs[1])?;
        let rhs = self.operand(&node.inputs[2])?;
        let product = match variant {
            AddMmVariant::AddMm => lhs.matmul(rhs),
            AddMmVariant::AddBmm => {
                anyhow::ensure!(lhs.rank() == 3, "addbmm batch1 must be rank 3");
                anyhow::ensure!(rhs.rank() == 3, "addbmm batch2 must be rank 3");
                lhs.matmul(rhs).sum(vec![0])
            }
            AddMmVariant::AddMv => {
                anyhow::ensure!(lhs.rank() == 2, "addmv matrix must be rank 2");
                anyhow::ensure!(rhs.rank() == 1, "addmv vector must be rank 1");
                lhs.matmul(rhs.unsqueeze(1)).squeeze(1)
            }
        };
        let input = self.scale_by_named_scalar(node, "beta", input)?;
        let product = self.scale_by_named_scalar(node, "alpha", product)?;
        let (input, product) = util::broadcast_binary(input, product);
        Ok(input + product)
    }

    /// `copy.default(destination, source)`: source broadcast into the
    /// destination's shape and dtype, returning the destination-shaped value.
    pub(super) fn translate_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(destination.dtype);
        let destination_shape = destination.dims();
        let (broadcast_destination, source) =
            util::broadcast_binary(destination, source.cast(dtype));
        anyhow::ensure!(
            broadcast_destination.dims() == destination_shape,
            "copy source shape {:?} cannot broadcast into destination shape {:?}",
            source.dims(),
            destination_shape
        );
        Ok(source)
    }

    /// `view_copy.default`: the materialized form of `view` (value-identical
    /// on the stride-less recorder).
    pub(super) fn translate_view_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.translate_view(node)?;
        Ok(util::materialize_tensor(value))
    }

    /// `permute_copy.default`: the materialized form of `permute`.
    pub(super) fn translate_permute_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let axes = self.get_ints_arg(node, 1)?;
        let axes = normalize_axes(&axes, x.rank())?;
        Ok(util::materialize_tensor(x.permute(axes)))
    }

    /// `empty.memory_format` / `empty_permuted` / `empty_strided` /
    /// `new_empty_strided`: lower the uninitialized allocation to a
    /// zero-filled tensor of the declared shape and dtype. Reads before a
    /// write are UB in PyTorch, and all callers write first, so this is sound.
    pub(super) fn translate_empty(&mut self, node: &Node) -> Result<GraphTensor> {
        // `new_empty_strided(self, size, stride)` shifts the shape one slot;
        // every other variant takes `size` first, with any stride/layout
        // argument ignored for a zero fill.
        let shape_index = if node.target.ends_with("new_empty_strided.default") {
            1
        } else {
            0
        };
        let shape = match self.resolve_shape_arg(node, shape_index) {
            Some(shape) => shape,
            None => self
                .get_ints_arg(node, shape_index)?
                .into_iter()
                .map(IntExpr::from)
                .collect(),
        };
        let dtype = self.output_meta_dtype(node)?;
        Ok(self.full_tensor(shape, dtype, 0.0))
    }

    /// The reduced axes for a variance/reduction node. A missing or empty
    /// dim list means "reduce every axis". A rank-0 operand has no axis to
    /// reduce; torch accepts dim 0 or -1 for it.
    fn reduction_axes(&self, node: &Node, index: usize, rank: usize) -> Result<Vec<usize>> {
        let dims = match node.inputs.get(index) {
            Some(input) if input.arg.as_ints().is_some() || input.arg.as_sym_ints().is_some() => {
                self.get_ints_arg(node, index)?
            }
            _ => Vec::new(),
        };
        if rank == 0 {
            anyhow::ensure!(
                dims.iter().all(|&d| d == 0 || d == -1),
                "{}: dims {dims:?} out of range for a rank-0 operand",
                node.target
            );
            return Ok(Vec::new());
        }
        Ok(if dims.is_empty() {
            (0..rank).collect()
        } else {
            normalize_axes(&dims, rank)?
        })
    }

    /// The degrees-of-freedom correction, kept real-valued: torch's
    /// `Scalar? correction` may be any Python float or int (0.5 and -1 are
    /// legal), so truncating it to an unsigned integer changes the answer.
    /// A named `correction` wins; an absent or `None` one falls back to the
    /// `unbiased` bool of the `.default`/`.dim` overloads (true is Bessel's
    /// 1, false is 0), and then to torch's default of 1. A correction this
    /// arm cannot read as a literal number is refused, never defaulted.
    fn variance_correction(&self, node: &Node) -> Result<f64> {
        if let Some(input) = node.inputs.iter().find(|input| input.name == "correction") {
            if let Some(correction) = input.arg.as_float() {
                return Ok(correction);
            }
            if let Some(correction) = input.arg.as_int() {
                return Ok(correction as f64);
            }
            anyhow::ensure!(
                is_none_arg(&input.arg),
                "{}: correction {:?} is not a literal number",
                node.target,
                input.arg
            );
        }
        Ok(self
            .named_bool_arg(node, "unbiased")
            .map_or(1.0, |unbiased| if unbiased { 1.0 } else { 0.0 }))
    }

    fn keepdim_flag(&self, node: &Node, positional: usize) -> bool {
        self.named_bool_arg(node, "keepdim")
            .or_else(|| node.inputs.get(positional).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false)
    }

    /// Scale `value` by the addmm-family coefficient `name` (`beta` or
    /// `alpha`); an absent coefficient is 1.
    ///
    /// Invariant: torch treats a coefficient equal to 0 as dropping its
    /// term, not as a multiply. `beta == 0` ignores `input` and `alpha == 0`
    /// ignores the product, so NaN/inf in the dropped term never reach the
    /// result (`NaN * 0` would). A coefficient equal to 1 is no multiply.
    /// Both tests are on the literal's value, so every spelling of 0 and 1
    /// takes them: `False`/`True`, `0`/`1`, and `0.0`/`-0.0`/`1.0`. A
    /// symbolic coefficient is refused, because whether it is 0 is only
    /// known at run time.
    fn scale_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: GraphTensor,
    ) -> Result<GraphTensor> {
        let Some(scalar) = node.inputs.iter().find(|input| input.name == name) else {
            return Ok(value);
        };
        let Some(literal) = util::ScalarLiteral::of(&scalar.arg) else {
            anyhow::bail!(
                "{} coefficient {name:?} is not a literal real scalar (arg: {:?})",
                node.target,
                scalar.arg
            );
        };
        if literal.is_one() {
            return Ok(value);
        }
        if literal.is_zero() {
            return Ok(self.constant_like(value, 0.0));
        }
        // Preserve integer literals before any conversion to the compute dtype.
        // Going through f64 would discard low bits above 2^53.
        let factor = self.scalar(scalar, value.dtype)?.expand_rhs(value.dims());
        Ok(value * factor)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, BoolArg, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntArg,
        IntsArg, Node, NodeInput, Signature, TensorArg, TensorMeta, TensorName, TensorRef,
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

    fn ints(name: &str, values: &[i64]) -> NodeInput {
        input(
            name,
            Argument::Ints(IntsArg {
                as_ints: values.to_vec(),
            }),
        )
    }

    fn scalar_int(name: &str, value: i64) -> NodeInput {
        input(name, Argument::Int(IntArg { as_int: value }))
    }

    fn scalar_float(name: &str, value: f64) -> NodeInput {
        input(name, Argument::Float(FloatArg { as_float: value }))
    }

    fn scalar_bool(name: &str, value: bool) -> NodeInput {
        input(name, Argument::Bool(BoolArg { as_bool: value }))
    }

    /// Translate a single node over the declared tensors, returning whether
    /// translation succeeded. Every tensor operand must be listed in
    /// `inputs` (graph order) and have a `(dtype, shape)` entry in `tensors`.
    fn translates(
        node: Node,
        tensors: &[(&str, u32, Vec<i64>)],
        inputs: &[&str],
        outputs: &[&str],
    ) -> bool {
        let mut tensor_values = HashMap::new();
        for (name, dtype, shape) in tensors {
            tensor_values.insert(
                name.to_string(),
                TensorMeta {
                    dtype: *dtype,
                    sizes: sizes(shape),
                    strides: None,
                    storage_offset: None,
                    layout: None,
                    device: None,
                },
            );
        }
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: Graph {
                    inputs: inputs.iter().map(|name| tensor_ref(name)).collect(),
                    outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
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

    fn node(target: &str, inputs: Vec<NodeInput>, outputs: &[&str]) -> Node {
        Node {
            target: target.to_string(),
            inputs,
            outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
        }
    }

    #[test]
    fn any_variants_record() {
        // any.default reduces every axis to a scalar Bool.
        assert!(translates(
            node(
                "torch.ops.aten.any.default",
                vec![input("self", tensor_arg("x"))],
                &["y"],
            ),
            &[("x", 12, vec![2, 3]), ("y", 12, vec![])],
            &["x"],
            &["y"],
        ));
        // any.dim with keepdim.
        assert!(translates(
            node(
                "torch.ops.aten.any.dim",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_int("dim", 1),
                    scalar_bool("keepdim", true),
                ],
                &["y"],
            ),
            &[("x", 12, vec![2, 3]), ("y", 12, vec![2, 1])],
            &["x"],
            &["y"],
        ));
        // any.dims over a list.
        assert!(translates(
            node(
                "torch.ops.aten.any.dims",
                vec![
                    input("self", tensor_arg("x")),
                    ints("dim", &[0, 1]),
                    scalar_bool("keepdim", false),
                ],
                &["y"],
            ),
            &[("x", 7, vec![2, 3]), ("y", 12, vec![])],
            &["x"],
            &["y"],
        ));
        // any.dims(dim=[]) is an elementwise bool cast, preserving shape.
        assert!(translates(
            node(
                "torch.ops.aten.any.dims",
                vec![
                    input("self", tensor_arg("x")),
                    ints("dim", &[]),
                    scalar_bool("keepdim", false),
                ],
                &["y"],
            ),
            &[("x", 7, vec![2, 3]), ("y", 12, vec![2, 3])],
            &["x"],
            &["y"],
        ));
    }

    #[test]
    fn var_mean_records_two_outputs() {
        assert!(translates(
            node(
                "torch.ops.aten.var_mean.dim",
                vec![
                    input("self", tensor_arg("x")),
                    ints("dim", &[1]),
                    scalar_bool("unbiased", false),
                    scalar_bool("keepdim", true),
                ],
                &["var", "mean"],
            ),
            &[
                ("x", 7, vec![2, 3]),
                ("var", 7, vec![2, 1]),
                ("mean", 7, vec![2, 1]),
            ],
            &["x"],
            &["var", "mean"],
        ));
        // correction overload.
        assert!(translates(
            node(
                "torch.ops.aten.var_mean.correction",
                vec![
                    input("self", tensor_arg("x")),
                    ints("dim", &[1]),
                    scalar_float("correction", 0.0),
                    scalar_bool("keepdim", false),
                ],
                &["var", "mean"],
            ),
            &[
                ("x", 7, vec![2, 3]),
                ("var", 7, vec![2]),
                ("mean", 7, vec![2]),
            ],
            &["x"],
            &["var", "mean"],
        ));
    }

    #[test]
    fn addmm_family_records() {
        assert!(translates(
            node(
                "torch.ops.aten.addmm.default",
                vec![
                    input("input", tensor_arg("input")),
                    input("mat1", tensor_arg("mat1")),
                    input("mat2", tensor_arg("mat2")),
                    scalar_float("beta", 1.0),
                    scalar_float("alpha", 1.0),
                ],
                &["y"],
            ),
            &[
                ("input", 7, vec![2, 3]),
                ("mat1", 7, vec![2, 4]),
                ("mat2", 7, vec![4, 3]),
                ("y", 7, vec![2, 3]),
            ],
            &["input", "mat1", "mat2"],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.addbmm.default",
                vec![
                    input("input", tensor_arg("input")),
                    input("batch1", tensor_arg("batch1")),
                    input("batch2", tensor_arg("batch2")),
                    scalar_float("beta", 0.5),
                    scalar_float("alpha", 2.0),
                ],
                &["y"],
            ),
            &[
                ("input", 7, vec![2, 3]),
                ("batch1", 7, vec![5, 2, 4]),
                ("batch2", 7, vec![5, 4, 3]),
                ("y", 7, vec![2, 3]),
            ],
            &["input", "batch1", "batch2"],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.addmv.default",
                vec![
                    input("input", tensor_arg("input")),
                    input("mat", tensor_arg("mat")),
                    input("vec", tensor_arg("vec")),
                    scalar_float("beta", 1.0),
                    scalar_float("alpha", 1.0),
                ],
                &["y"],
            ),
            &[
                ("input", 7, vec![3]),
                ("mat", 7, vec![3, 2]),
                ("vec", 7, vec![2]),
                ("y", 7, vec![3]),
            ],
            &["input", "mat", "vec"],
            &["y"],
        ));
    }

    #[test]
    fn copy_and_materialized_views_record() {
        assert!(translates(
            node(
                "torch.ops.aten.copy.default",
                vec![
                    input("self", tensor_arg("destination")),
                    input("src", tensor_arg("source")),
                ],
                &["y"],
            ),
            &[
                ("destination", 7, vec![2, 3]),
                ("source", 7, vec![2, 3]),
                ("y", 7, vec![2, 3]),
            ],
            &["destination", "source"],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.view_copy.default",
                vec![input("self", tensor_arg("x")), ints("size", &[2, 3]),],
                &["y"],
            ),
            &[("x", 7, vec![6]), ("y", 7, vec![2, 3])],
            &["x"],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.permute_copy.default",
                vec![input("self", tensor_arg("x")), ints("dims", &[2, 0, 1]),],
                &["y"],
            ),
            &[("x", 7, vec![2, 3, 4]), ("y", 7, vec![4, 2, 3])],
            &["x"],
            &["y"],
        ));
    }

    #[test]
    fn empty_constructors_record() {
        assert!(translates(
            node(
                "torch.ops.aten.empty.memory_format",
                vec![ints("size", &[2, 3])],
                &["y"],
            ),
            &[("y", 7, vec![2, 3])],
            &[],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.empty_permuted.default",
                vec![ints("size", &[2, 3]), ints("physical_layout", &[0, 1])],
                &["y"],
            ),
            &[("y", 7, vec![2, 3])],
            &[],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.empty_strided.default",
                vec![ints("size", &[2, 3]), ints("stride", &[3, 1])],
                &["y"],
            ),
            &[("y", 7, vec![2, 3])],
            &[],
            &["y"],
        ));
        assert!(translates(
            node(
                "torch.ops.aten.new_empty_strided.default",
                vec![
                    input("self", tensor_arg("x")),
                    ints("size", &[2, 3]),
                    ints("stride", &[3, 1]),
                ],
                &["y"],
            ),
            &[("x", 7, vec![2, 3]), ("y", 7, vec![2, 3])],
            &["x"],
            &["y"],
        ));
    }
}

#[cfg(test)]
mod scalar_precision_tests {
    use crate::translate::{Translator, test_support::*};
    use luminal::prelude::*;
    use std::collections::HashMap;

    #[test]
    fn addmm_scaling_preserves_large_integer_literal_bits() {
        let parsed = parsed_nodes(Vec::new(), &[], &[], &[]);
        let mut translator = Translator {
            cx: Graph::new(),
            values: HashMap::new(),
            input_values: HashMap::new(),
            symbols: HashMap::new(),
            ranges: HashMap::new(),
            complex_tensors: HashMap::new(),
            parsed: &parsed,
            dims: HashMap::new(),
            opmath: None,
            storage_values: HashMap::new(),
            storage_geometries: HashMap::new(),
            widened: HashMap::new(),
        };
        let x = translator.cx.tensor(vec![1], DType::I64);
        let alpha = (1i64 << 54) + 3;
        let beta = (1i64 << 54) + 1;
        let node = node(
            "torch.ops.aten.addmm.default",
            vec![scalar_int("alpha", alpha), scalar_int("beta", beta)],
            &[],
        );
        let a = translator.scale_by_named_scalar(&node, "alpha", x).unwrap();
        let b = translator.scale_by_named_scalar(&node, "beta", x).unwrap();
        let rt = luminal_reference::harness::run_reference_with_ranges(
            &translator.cx,
            &[(
                x.id,
                luminal_reference::typed_buffer::TypedBuffer::I64(vec![1]),
            )],
            &[(x.id, 1, 1)],
        );
        assert_eq!(rt.get_i64(a.id).unwrap(), &[alpha]);
        assert_eq!(rt.get_i64(b.id).unwrap(), &[beta]);
    }
}

#[cfg(test)]
mod variance_tests {
    use crate::pt2_schema::{Argument, NodeInput, ScalarName, SymFloatArg};
    use crate::translate::test_support::*;

    const ROWS: [[f32; 4]; 3] = [
        [1.0, 2.0, 4.0, 8.0],
        [0.5, -1.0, 3.0, 2.5],
        [2.0, 2.0, 2.0, 5.0],
    ];

    /// Independent two-pass variance in f64 with torch's divisor rule.
    fn torch_variance(values: &[f32], correction: f64) -> f64 {
        let n = values.len() as f64;
        let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
        let squares: f64 = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum();
        let degrees = if n > correction && n > 0.0 {
            n - correction
        } else {
            0.0
        };
        squares / degrees
    }

    fn per_row(correction: f64) -> Vec<f64> {
        ROWS.iter()
            .map(|row| torch_variance(row, correction))
            .collect()
    }

    fn everything(correction: f64) -> Vec<f64> {
        let flat: Vec<f32> = ROWS.iter().flatten().copied().collect();
        vec![torch_variance(&flat, correction)]
    }

    /// Translate one variance node over the 3x4 `x` and run it on the
    /// reference runtime, returning every output.
    fn run(target: &str, args: Vec<NodeInput>, outputs: &[(&str, &[i64])]) -> Vec<Vec<f32>> {
        let mut inputs = vec![tensor_input("self", "x")];
        inputs.extend(args);
        let names: Vec<&str> = outputs.iter().map(|(name, _)| *name).collect();
        let mut tensors: Vec<(&str, u32, &[i64])> = vec![("x", FLOAT, &[3, 4])];
        tensors.extend(outputs.iter().map(|(name, shape)| (*name, FLOAT, *shape)));
        let t = translate_one(
            node(&format!("torch.ops.aten.{target}"), inputs, &names),
            &tensors,
            &["x"],
            &names,
        );
        let flat: Vec<f32> = ROWS.iter().flatten().copied().collect();
        let rt = luminal_reference::harness::run_reference(
            &t.graph,
            &[(t.inputs[0].tensor, flat.into())],
        );
        t.outputs
            .iter()
            .map(|output| rt.get_f32(output.tensor).unwrap().to_vec())
            .collect()
    }

    fn assert_matches(got: &[f32], expected: &[f64], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
            let expected = expected as f32;
            let same = (got.is_nan() && expected.is_nan())
                || got == expected
                || (got - expected).abs() <= 1e-5 * expected.abs().max(1.0);
            assert!(same, "{what}[{index}]: got {got}, expected {expected}");
        }
    }

    /// Every variance arm reads torch's correction as a real number: the
    /// `unbiased` bool of `.default`/`.dim`, a float or int `correction`
    /// (fractional, negative, past N), and the default of 1 for `None`.
    #[test]
    fn variance_arms_read_the_real_valued_correction() {
        let out = run(
            "var.default",
            vec![scalar_bool("unbiased", false)],
            &[("y", &[])],
        );
        assert_matches(&out[0], &everything(0.0), "var.default unbiased=False");

        let out = run(
            "var.dim",
            vec![
                ints("dim", &[1]),
                scalar_bool("unbiased", false),
                scalar_bool("keepdim", true),
            ],
            &[("y", &[3, 1])],
        );
        assert_matches(&out[0], &per_row(0.0), "var.dim unbiased=False");

        let out = run(
            "var.correction",
            vec![ints("dim", &[1]), scalar_float("correction", 0.5)],
            &[("y", &[3])],
        );
        assert_matches(&out[0], &per_row(0.5), "var.correction 0.5");

        let out = run(
            "std.correction",
            vec![ints("dim", &[-1]), scalar_int("correction", 5)],
            &[("y", &[3])],
        );
        let std: Vec<f64> = per_row(5.0).iter().map(|v| v.sqrt()).collect();
        assert!(std.iter().all(|v| *v == f64::INFINITY));
        assert_matches(&out[0], &std, "std.correction 5 > N");

        let out = run(
            "var_mean.correction",
            vec![
                ints("dim", &[1]),
                scalar_int("correction", -1),
                scalar_bool("keepdim", true),
            ],
            &[("var", &[3, 1]), ("mean", &[3, 1])],
        );
        assert_matches(&out[0], &per_row(-1.0), "var_mean.correction -1 variance");
        let means: Vec<f64> = ROWS
            .iter()
            .map(|row| row.iter().map(|&v| v as f64).sum::<f64>() / 4.0)
            .collect();
        assert_matches(&out[1], &means, "var_mean.correction -1 mean");

        let out = run(
            "var.correction",
            vec![none("dim"), none("correction")],
            &[("y", &[])],
        );
        assert_matches(&out[0], &everything(1.0), "var.correction None");
    }

    /// A correction that is not a literal number is refused by name rather
    /// than silently replaced by the default.
    #[test]
    fn variance_refuses_a_non_literal_correction() {
        let parsed = parsed_nodes(
            vec![node(
                "torch.ops.aten.var.correction",
                vec![
                    tensor_input("self", "x"),
                    ints("dim", &[1]),
                    input(
                        "correction",
                        Argument::SymFloat(SymFloatArg {
                            as_sym_float: ScalarName {
                                as_name: "c".to_string(),
                            },
                        }),
                    ),
                ],
                &["y"],
            )],
            &[("x", FLOAT, &[3, 4]), ("y", FLOAT, &[3])],
            &["x"],
            &["y"],
        );
        let error = crate::translate::translate(&parsed)
            .err()
            .expect("a symbolic correction must be refused");
        assert!(
            format!("{error:#}").contains("is not a literal number"),
            "{error:#}"
        );
    }
}

#[cfg(test)]
mod bool_coefficient_tests {
    use crate::pt2_schema::NodeInput;
    use crate::translate::test_support::*;
    use luminal_reference::harness::run_reference;

    const NAN: f32 = f32::NAN;
    const INF: f32 = f32::INFINITY;

    /// One addmm-family target over f32 operands, named as in its ATen
    /// schema, with finite data and a poisoned (NaN/inf) replacement for
    /// the `input` and for the left matrix operand.
    struct Case {
        target: &'static str,
        names: [&'static str; 3],
        shapes: [&'static [i64]; 3],
        output: &'static [i64],
        data: [Vec<f32>; 3],
        poisoned_input: Vec<f32>,
        poisoned_lhs: Vec<f32>,
        /// The variant's product of `data[1]` and `data[2]`.
        product: Vec<f32>,
    }

    impl Case {
        /// Translate and execute the node with `operands` and `coefficients`.
        fn run(&self, operands: [Vec<f32>; 3], coefficients: Vec<NodeInput>) -> Vec<f32> {
            let mut args: Vec<NodeInput> = self
                .names
                .iter()
                .map(|name| tensor_input(name, name))
                .collect();
            args.extend(coefficients);
            let mut tensors: Vec<(&str, u32, &[i64])> = self
                .names
                .iter()
                .zip(self.shapes)
                .map(|(name, shape)| (*name, FLOAT, shape))
                .collect();
            tensors.push(("out", FLOAT, self.output));
            let t = translate_one(
                node(self.target, args, &["out"]),
                &tensors,
                &self.names,
                &["out"],
            );
            let inputs = t
                .inputs
                .iter()
                .zip(operands)
                .map(|(input, data)| (input.tensor, data.into()))
                .collect::<Vec<_>>();
            let rt = run_reference(&t.graph, &inputs);
            rt.get_f32(t.outputs[0].tensor).unwrap().clone()
        }
    }

    fn cases() -> [Case; 3] {
        [
            Case {
                target: "torch.ops.aten.addmm.default",
                names: ["self", "mat1", "mat2"],
                shapes: [&[2, 2], &[2, 2], &[2, 2]],
                output: &[2, 2],
                // An identity mat1: the product is mat2.
                data: [
                    vec![1., 2., 3., 4.],
                    vec![1., 0., 0., 1.],
                    vec![5., 6., 7., 8.],
                ],
                poisoned_input: vec![NAN, INF, -INF, NAN],
                poisoned_lhs: vec![INF, NAN, -INF, INF],
                product: vec![5., 6., 7., 8.],
            },
            Case {
                target: "torch.ops.aten.addmv.default",
                names: ["self", "mat", "vec"],
                shapes: [&[2], &[2, 2], &[2]],
                output: &[2],
                data: [vec![1., 2.], vec![1., 0., 0., 1.], vec![5., 6.]],
                poisoned_input: vec![NAN, INF],
                poisoned_lhs: vec![INF, NAN, -INF, INF],
                product: vec![5., 6.],
            },
            Case {
                target: "torch.ops.aten.addbmm.default",
                names: ["self", "batch1", "batch2"],
                shapes: [&[2, 2], &[2, 2, 2], &[2, 2, 2]],
                output: &[2, 2],
                // Two identity batches: the product sums batch2's matrices.
                data: [
                    vec![1., 2., 3., 4.],
                    vec![1., 0., 0., 1., 1., 0., 0., 1.],
                    vec![5., 6., 7., 8., 1., 1., 1., 1.],
                ],
                poisoned_input: vec![NAN, INF, -INF, NAN],
                poisoned_lhs: vec![INF, NAN, -INF, INF, 1., 0., 0., 1.],
                product: vec![6., 7., 8., 9.],
            },
        ]
    }

    /// A Python-bool `beta`/`alpha` is the literal 0 or 1. `False` drops its
    /// term, so NaN/inf there never reach the result (`NaN * 0` would), and
    /// `True` keeps the term unscaled.
    #[test]
    fn bool_coefficients_drop_or_keep_their_term() {
        for case in cases() {
            let target = case.target;
            let [input, lhs, rhs] = case.data.clone();
            for coefficients in [
                vec![scalar_bool("beta", false)],
                vec![scalar_bool("beta", false), scalar_bool("alpha", true)],
            ] {
                let operands = [case.poisoned_input.clone(), lhs.clone(), rhs.clone()];
                let got = case.run(operands, coefficients);
                assert_eq!(got, case.product, "{target}: beta=False drops the input");
            }
            for coefficients in [
                vec![scalar_bool("alpha", false)],
                vec![scalar_bool("beta", true), scalar_bool("alpha", false)],
            ] {
                let operands = [input.clone(), case.poisoned_lhs.clone(), rhs.clone()];
                let got = case.run(operands, coefficients);
                assert_eq!(got, input, "{target}: alpha=False drops the product");
            }
            let sum: Vec<f32> = input
                .iter()
                .zip(&case.product)
                .map(|(x, y)| x + y)
                .collect();
            let got = case.run(
                [input, lhs, rhs],
                vec![scalar_bool("beta", true), scalar_bool("alpha", true)],
            );
            assert_eq!(got, sum, "{target}: True coefficients are 1");
        }
    }
}
