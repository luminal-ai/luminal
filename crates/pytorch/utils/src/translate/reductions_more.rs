//! Reductions, linalg, copies, and empty constructors (port batch 8).
//!
//! Ports the parked translator's `any.*`, `var_mean.*`, the `addmm` family,
//! `copy.default`, the materialized `*_copy` views, and the uninitialized
//! `empty*` constructors onto the native recorder frontend.

use anyhow::Result;
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::Node;

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

    /// `var_mean.default` / `var_mean.dim` / `var_mean.correction` -> (var, mean).
    pub(super) fn translate_var_mean(&mut self, node: &Node) -> Result<()> {
        let input = self.operand(&node.inputs[0])?;
        let axes = self.reduction_axes(node, 1, input.rank())?;
        let correction = self.variance_correction(node);
        let dtype = self.compute_dtype(node).unwrap_or(input.dtype);
        let input = input.cast(dtype);
        let variance = input.var_options(&axes, correction);
        let mean = input.mean(&axes);
        let keepdim = self.keepdim_flag(node, 3);
        let (variance, mean) = if keepdim {
            (keep_reduced(variance, &axes), keep_reduced(mean, &axes))
        } else {
            (variance, mean)
        };
        self.bind_outputs(node, vec![variance, mean])
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
                lhs.matmul(rhs).sum(0)
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
    /// dim list means "reduce every axis".
    fn reduction_axes(&self, node: &Node, index: usize, rank: usize) -> Result<Vec<usize>> {
        let dims = match node.inputs.get(index) {
            Some(input) if input.arg.as_ints().is_some() || input.arg.as_sym_ints().is_some() => {
                self.get_ints_arg(node, index)?
            }
            _ => Vec::new(),
        };
        Ok(if dims.is_empty() {
            (0..rank).collect()
        } else {
            dims.iter().map(|&d| util::normalize_dim(d, rank)).collect()
        })
    }

    /// `correction` (degrees-of-freedom offset) or the `unbiased` bool.
    fn variance_correction(&self, node: &Node) -> usize {
        if let Some(correction) = self.named_float_arg(node, "correction") {
            return correction.max(0.0) as usize;
        }
        self.named_bool_arg(node, "unbiased").map_or(1, usize::from)
    }

    fn keepdim_flag(&self, node: &Node, positional: usize) -> bool {
        self.named_bool_arg(node, "keepdim")
            .or_else(|| node.inputs.get(positional).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false)
    }

    /// Scale `value` by the named scalar `name`, defaulting to 1.0.
    fn scale_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: GraphTensor,
    ) -> Result<GraphTensor> {
        let Some(scalar) = self.named_float_arg(node, name) else {
            return Ok(value);
        };
        if scalar == 1.0 {
            return Ok(value);
        }
        if scalar == 0.0 {
            return Ok(self.constant_like(value, 0.0));
        }
        Ok(value * self.constant_like(value, scalar))
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
