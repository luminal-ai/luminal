//! Facts about the dtype pattern the translator records for torch's
//! opmath ops: half-precision math is performed in F32 and rounded once,
//! at the store, to the dtype the export declares.

use luminal::prelude::petgraph::Direction;
use luminal::prelude::petgraph::visit::EdgeRef;
use luminal::prelude::*;

use super::test_support::*;
use super::{Translation, exact_arithmetic_target, opmath_target};

// ---------------------------------------------------------------------
// Graph facts
// ---------------------------------------------------------------------

fn op_of(t: &Translation, id: NodeIndex) -> &LogicalOp {
    &t.graph.logical.petgraph()[id].op
}

fn dtype_of(t: &Translation, id: NodeIndex) -> DType {
    t.graph.logical.petgraph()[id].dtype
}

/// A node's operands in port order.
fn operands(t: &Translation, id: NodeIndex) -> Vec<NodeIndex> {
    let graph = t.graph.logical.petgraph();
    let mut edges: Vec<_> = graph
        .edges_directed(id, Direction::Incoming)
        .map(|edge| (edge.weight().0, edge.source()))
        .collect();
    edges.sort_by_key(|(port, _)| *port);
    edges.into_iter().map(|(_, source)| source).collect()
}

/// The value this one carries, seen through the movement ops (a broadcast
/// is geometry, not arithmetic).
fn through_views(t: &Translation, mut id: NodeIndex) -> NodeIndex {
    while matches!(op_of(t, id), LogicalOp::IndexMapApply { .. }) {
        id = operands(t, id)[0];
    }
    id
}

fn nodes(t: &Translation) -> Vec<NodeIndex> {
    t.graph.logical.petgraph().node_indices().collect()
}

/// Whether the graph records any node matching `predicate`.
fn any_node(t: &Translation, predicate: impl Fn(&LogicalOp, DType) -> bool) -> bool {
    nodes(t)
        .into_iter()
        .any(|id| predicate(op_of(t, id), dtype_of(t, id)))
}

fn is_op(op: &LogicalOp, wanted: &LogicalOp) -> bool {
    std::mem::discriminant(op) == std::mem::discriminant(wanted)
}

fn is_cast_to(op: &LogicalOp, dtype: DType) -> bool {
    matches!(op, LogicalOp::Cast(target) if *target == dtype)
}

fn is_input_named(op: &LogicalOp, name: &str) -> bool {
    matches!(op, LogicalOp::Input { label } if label == name)
}

/// The single node of `wanted`'s form in the graph.
fn only_node(t: &Translation, wanted: &LogicalOp) -> NodeIndex {
    let found: Vec<NodeIndex> = nodes(t)
        .into_iter()
        .filter(|id| is_op(op_of(t, *id), wanted))
        .collect();
    assert_eq!(found.len(), 1, "expected exactly one {wanted:?} node");
    found[0]
}

/// The nodes whose operand (through views) is the graph input `name`.
fn consumers_of_input(t: &Translation, name: &str) -> Vec<NodeIndex> {
    nodes(t)
        .into_iter()
        .filter(|id| {
            operands(t, *id)
                .into_iter()
                .any(|operand| is_input_named(op_of(t, through_views(t, operand)), name))
        })
        .collect()
}

/// Whether some `Cast` to `dtype` reads the graph input `name`.
fn input_is_cast_to(t: &Translation, name: &str, dtype: DType) -> bool {
    consumers_of_input(t, name)
        .into_iter()
        .any(|id| is_cast_to(op_of(t, id), dtype))
}

fn add_node(a: u32, b: u32, out: u32) -> Translation {
    translate_one(
        node(
            "torch.ops.aten.add.Tensor",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["y"],
        ),
        &[("a", a, &[4]), ("b", b, &[4]), ("y", out, &[4])],
        &["a", "b"],
        &["y"],
    )
}

// ---------------------------------------------------------------------
// The opmath pattern
// ---------------------------------------------------------------------

#[test]
fn half_add_is_an_f32_add_between_casts() {
    let t = add_node(BFLOAT16, BFLOAT16, BFLOAT16);
    let out = t.outputs[0].tensor;
    assert_eq!(t.outputs[0].dtype, DType::Bf16);
    assert!(
        is_cast_to(op_of(&t, out), DType::Bf16),
        "the store rounds once"
    );
    let add = through_views(&t, operands(&t, out)[0]);
    assert!(is_op(op_of(&t, add), &LogicalOp::Add));
    assert_eq!(
        dtype_of(&t, add),
        DType::F32,
        "the math is performed in F32"
    );
    for operand in operands(&t, add) {
        let widened = through_views(&t, operand);
        assert!(is_cast_to(op_of(&t, widened), DType::F32));
        let source = through_views(&t, operands(&t, widened)[0]);
        assert!(matches!(op_of(&t, source), LogicalOp::Input { .. }));
    }
    assert!(
        !any_node(&t, |op, dtype| is_op(op, &LogicalOp::Add)
            && dtype == DType::Bf16),
        "no half-typed add is recorded"
    );
}

#[test]
fn mixed_half_and_f32_add_promotes_to_f32_and_narrows_nothing() {
    let t = add_node(BFLOAT16, FLOAT, FLOAT);
    assert_eq!(t.outputs[0].dtype, DType::F32);
    assert!(is_op(op_of(&t, t.outputs[0].tensor), &LogicalOp::Add));
    assert!(input_is_cast_to(&t, "a", DType::F32));
    assert!(
        consumers_of_input(&t, "b")
            .into_iter()
            .all(|id| is_op(op_of(&t, id), &LogicalOp::Add))
    );
    assert!(!any_node(&t, |op, _| is_cast_to(op, DType::Bf16)));
}

#[test]
fn python_scalar_is_read_at_the_compute_dtype() {
    let t = translate_one(
        node(
            "torch.ops.aten.add.Tensor",
            vec![tensor_input("self", "a"), scalar_float("other", 0.1)],
            &["y"],
        ),
        &[("a", HALF, &[4]), ("y", HALF, &[4])],
        &["a"],
        &["y"],
    );
    let add = only_node(&t, &LogicalOp::Add);
    let constant = operands(&t, add)
        .into_iter()
        .map(|operand| through_views(&t, operand))
        .find(|id| matches!(op_of(&t, *id), LogicalOp::Constant(_)))
        .expect("the scalar feeds the add");
    assert_eq!(dtype_of(&t, constant), DType::F32);
    assert!(
        !nodes(&t).into_iter().any(|id| {
            is_cast_to(op_of(&t, id), DType::F16)
                && operands(&t, id)
                    .into_iter()
                    .any(|operand| through_views(&t, operand) == constant)
        }),
        "the literal is never rounded to half"
    );
}

#[test]
fn integer_tensor_operand_is_rounded_to_the_common_dtype_then_widened() {
    let t = translate_one(
        node(
            "torch.ops.aten.add.Tensor",
            vec![tensor_input("self", "a"), tensor_input("other", "i")],
            &["y"],
        ),
        &[("a", HALF, &[4]), ("i", INT, &[4]), ("y", HALF, &[4])],
        &["a", "i"],
        &["y"],
    );
    let rounded = consumers_of_input(&t, "i");
    assert_eq!(rounded.len(), 1);
    assert!(is_cast_to(op_of(&t, rounded[0]), DType::F16));
    let widened: Vec<NodeIndex> = nodes(&t)
        .into_iter()
        .filter(|id| {
            operands(&t, *id)
                .into_iter()
                .any(|operand| through_views(&t, operand) == rounded[0])
        })
        .collect();
    assert!(
        widened
            .iter()
            .all(|id| is_cast_to(op_of(&t, *id), DType::F32))
    );
}

#[test]
fn half_sum_reduces_in_f32_and_stores_half() {
    let t = translate_one(
        node(
            "torch.ops.aten.sum.dim_IntList",
            vec![
                tensor_input("self", "a"),
                ints("dim", &[-1]),
                scalar_bool("keepdim", false),
            ],
            &["y"],
        ),
        &[("a", BFLOAT16, &[4, 8]), ("y", BFLOAT16, &[4])],
        &["a"],
        &["y"],
    );
    let reduce = only_node(&t, &LogicalOp::ReduceSum { axis_from_end: 0 });
    assert_eq!(dtype_of(&t, reduce), DType::F32);
    assert!(is_cast_to(op_of(&t, t.outputs[0].tensor), DType::Bf16));
}

#[test]
fn sum_with_a_declared_f32_result_narrows_nothing() {
    let t = translate_one(
        node(
            "torch.ops.aten.sum.dim_IntList",
            vec![
                tensor_input("self", "a"),
                ints("dim", &[-1]),
                scalar_bool("keepdim", false),
                scalar_type("dtype", FLOAT),
            ],
            &["y"],
        ),
        &[("a", BFLOAT16, &[4, 8]), ("y", FLOAT, &[4])],
        &["a"],
        &["y"],
    );
    assert_eq!(t.outputs[0].dtype, DType::F32);
    assert!(!any_node(&t, |op, _| is_cast_to(op, DType::Bf16)));
}

#[test]
fn native_layer_norm_outputs_take_their_own_declared_dtypes() {
    let t = translate_one(
        node(
            "torch.ops.aten.native_layer_norm.default",
            vec![
                tensor_input("input", "x"),
                ints("normalized_shape", &[8]),
                none("weight"),
                none("bias"),
                scalar_float("eps", 1e-5),
            ],
            &["out", "mean", "rstd"],
        ),
        &[
            ("x", BFLOAT16, &[4, 8]),
            ("out", BFLOAT16, &[4, 8]),
            ("mean", FLOAT, &[4, 1]),
            ("rstd", FLOAT, &[4, 1]),
        ],
        &["x"],
        &["out", "mean", "rstd"],
    );
    let dtypes: Vec<DType> = t.outputs.iter().map(|output| output.dtype).collect();
    assert_eq!(dtypes, vec![DType::Bf16, DType::F32, DType::F32]);
    let statistic_ranks: Vec<usize> = t.outputs[1..]
        .iter()
        .map(|output| output.shape.len())
        .collect();
    assert_eq!(statistic_ranks, vec![2, 2], "the normalized axes are kept");
}

#[test]
fn batch_norm_parameters_are_read_wide_without_rounding() {
    let t = translate_one(
        node(
            "torch.ops.aten._native_batch_norm_legit_no_training.default",
            vec![
                tensor_input("input", "x"),
                tensor_input("weight", "w"),
                tensor_input("bias", "b"),
                tensor_input("running_mean", "rm"),
                tensor_input("running_var", "rv"),
                scalar_float("momentum", 0.1),
                scalar_float("eps", 1e-5),
            ],
            &["out"],
        ),
        &[
            ("x", HALF, &[2, 3, 4, 4]),
            ("w", FLOAT, &[3]),
            ("b", FLOAT, &[3]),
            ("rm", FLOAT, &[3]),
            ("rv", FLOAT, &[3]),
            ("out", HALF, &[2, 3, 4, 4]),
        ],
        &["x", "w", "b", "rm", "rv"],
        &["out"],
    );
    assert_eq!(t.outputs[0].dtype, DType::F16);
    assert!(input_is_cast_to(&t, "x", DType::F32));
    for parameter in ["w", "b", "rm", "rv"] {
        assert!(
            !input_is_cast_to(&t, parameter, DType::F16),
            "{parameter} is read at F32 unrounded"
        );
    }
}

#[test]
fn scalar_comparison_rounds_the_literal_to_the_operand_dtype() {
    let t = translate_one(
        node(
            "torch.ops.aten.gt.Scalar",
            vec![tensor_input("self", "a"), scalar_float("other", 0.1)],
            &["y"],
        ),
        &[("a", HALF, &[4]), ("y", BOOL, &[4])],
        &["a"],
        &["y"],
    );
    assert_eq!(t.outputs[0].dtype, DType::Bool);
    let compare = only_node(&t, &LogicalOp::LessThan);
    let sources: Vec<NodeIndex> = operands(&t, compare)
        .into_iter()
        .map(|operand| through_views(&t, operand))
        .collect();
    assert!(sources.iter().all(|id| dtype_of(&t, *id) == DType::F32));
    let constant = only_node(&t, &LogicalOp::Constant(0.0));
    let rounded: Vec<NodeIndex> = nodes(&t)
        .into_iter()
        .filter(|id| {
            operands(&t, *id)
                .into_iter()
                .any(|operand| through_views(&t, operand) == constant)
        })
        .collect();
    assert!(
        rounded
            .iter()
            .all(|id| is_cast_to(op_of(&t, *id), DType::F16)),
        "the literal is rounded to the operand dtype first"
    );
    assert!(
        sources
            .iter()
            .any(|id| is_cast_to(op_of(&t, *id), DType::F32))
    );
}

#[test]
fn integer_tensor_against_a_float_literal_compares_at_f32() {
    let t = translate_one(
        node(
            "torch.ops.aten.ge.Scalar",
            vec![tensor_input("self", "i"), scalar_float("other", 0.5)],
            &["y"],
        ),
        &[("i", INT, &[4]), ("y", BOOL, &[4])],
        &["i"],
        &["y"],
    );
    assert_eq!(t.outputs[0].dtype, DType::Bool);
    assert!(input_is_cast_to(&t, "i", DType::F32));
}

/// An exact op whose lowering computes (`neg` is a multiply by -1, `abs`
/// a multiply by the sign, a rounding is a math function) reads narrow
/// storage through an explicit widening cast and rounds back to the
/// declared dtype: the graph computes nothing in F16 or BF16.
#[test]
fn exact_arithmetic_widens_narrow_storage() {
    for target in [
        "abs.default",
        "neg.default",
        "floor.default",
        "ceil.default",
        "trunc.default",
        "round.default",
    ] {
        for (code, dtype) in [(HALF, DType::F16), (BFLOAT16, DType::Bf16)] {
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{target}"),
                    vec![tensor_input("self", "a")],
                    &["y"],
                ),
                &[("a", code, &[4]), ("y", code, &[4])],
                &["a"],
                &["y"],
            );
            assert!(
                input_is_cast_to(&t, "a", DType::F32),
                "{target} {dtype:?} widens its operand"
            );
            assert_eq!(t.outputs[0].dtype, dtype, "{target} rounds back");
            let logical = &t.graph.logical;
            assert!(
                logical
                    .narrow_float_arithmetic(logical.petgraph().node_indices())
                    .is_empty(),
                "{target} {dtype:?} computes in its storage dtype"
            );
        }
    }
}

#[test]
fn exact_ops_never_cast_their_operand() {
    let clamp = translate_one(
        node(
            "torch.ops.aten.clamp.default",
            vec![
                tensor_input("self", "a"),
                scalar_float("min", 0.1),
                scalar_float("max", 0.7),
            ],
            &["y"],
        ),
        &[("a", HALF, &[4]), ("y", HALF, &[4])],
        &["a"],
        &["y"],
    );
    assert!(
        consumers_of_input(&clamp, "a")
            .into_iter()
            .all(|id| !is_op(op_of(&clamp, id), &LogicalOp::Cast(DType::F32)))
    );

    let select = translate_one(
        node(
            "torch.ops.aten.select.int",
            vec![
                tensor_input("self", "a"),
                scalar_int("dim", 0),
                scalar_int("index", 1),
            ],
            &["y"],
        ),
        &[("a", HALF, &[4, 8]), ("y", HALF, &[8])],
        &["a"],
        &["y"],
    );
    assert!(!any_node(&select, |op, _| is_op(
        op,
        &LogicalOp::Cast(DType::F32)
    )));

    let view = translate_one(
        node(
            "torch.ops.aten.view.default",
            vec![tensor_input("self", "a"), ints("size", &[32])],
            &["y"],
        ),
        &[("a", HALF, &[4, 8]), ("y", HALF, &[32])],
        &["a"],
        &["y"],
    );
    assert!(!any_node(&view, |op, _| is_op(
        op,
        &LogicalOp::Cast(DType::F32)
    )));
}

#[test]
fn attention_reads_a_bool_mask_as_a_predicate() {
    let t = translate_one(
        node(
            "torch.ops.aten.scaled_dot_product_attention.default",
            vec![
                tensor_input("query", "q"),
                tensor_input("key", "k"),
                tensor_input("value", "v"),
                tensor_input("attn_mask", "m"),
                scalar_float("dropout_p", 0.0),
                scalar_bool("is_causal", false),
            ],
            &["out"],
        ),
        &[
            ("q", HALF, &[1, 1, 4, 8]),
            ("k", HALF, &[1, 1, 4, 8]),
            ("v", HALF, &[1, 1, 4, 8]),
            ("m", BOOL, &[4, 4]),
            ("out", HALF, &[1, 1, 4, 8]),
        ],
        &["q", "k", "v", "m"],
        &["out"],
    );
    assert_eq!(t.outputs[0].dtype, DType::F16);
    assert!(
        consumers_of_input(&t, "m").into_iter().any(|id| {
            is_op(op_of(&t, id), &LogicalOp::Select)
                && is_input_named(op_of(&t, through_views(&t, operands(&t, id)[0])), "m")
        }),
        "the keep-mask is read as a select condition"
    );
    for dtype in [DType::F16, DType::F32] {
        assert!(
            !input_is_cast_to(&t, "m", dtype),
            "the keep-mask is never an operand of the score arithmetic ({dtype:?})"
        );
    }
}

#[test]
fn opmath_class_membership() {
    for target in [
        "add.Tensor",
        "sum.dim_IntList",
        "exp.default",
        "mm.default",
        "lt.Scalar",
        "grid_sampler_2d.default",
        "native_layer_norm.default",
    ] {
        assert!(opmath_target(target), "{target} rounds in torch's opmath");
    }
    for target in [
        "view.default",
        "where.self",
        "neg.default",
        "clamp.default",
        "_to_copy.default",
        "_grouped_mm.default",
        "cumsum.default",
        "max.dim",
    ] {
        assert!(!opmath_target(target), "{target} is exact");
    }
    assert!(exact_arithmetic_target("neg.default"));
    assert!(!exact_arithmetic_target("clamp.default"));
}

#[test]
fn attention_keeps_compute_precision_until_the_operator_output() {
    for (code, dtype) in [
        (BFLOAT16, DType::Bf16),
        (HALF, DType::F16),
        (FLOAT, DType::F32),
    ] {
        for variant in [
            "scaled_dot_product_attention.default",
            "_scaled_dot_product_flash_attention.default",
            "_scaled_dot_product_flash_attention_for_cpu.default",
            "_scaled_dot_product_efficient_attention.default",
            "_scaled_dot_product_cudnn_attention.default",
            "_scaled_dot_product_fused_attention_overrideable.default",
        ] {
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{variant}"),
                    vec![
                        tensor_input("query", "q"),
                        tensor_input("key", "k"),
                        tensor_input("value", "v"),
                        scalar_float("dropout_p", 0.0),
                        scalar_bool("is_causal", false),
                        scalar_float("scale", 1.0),
                    ],
                    &["out"],
                ),
                &[
                    ("q", code, &[1, 1, 1, 2]),
                    ("k", code, &[1, 1, 2, 2]),
                    ("v", code, &[1, 1, 2, 2]),
                    ("out", code, &[1, 1, 1, 2]),
                ],
                &["q", "k", "v"],
                &["out"],
            );
            // ReferenceRuntime does not execute BF16. Assert the general
            // precision invariant on the recorded graph; downstream device
            // regressions compare the result with higher-precision arithmetic.
            let output = t.outputs[0].tensor;
            assert_eq!(t.outputs[0].dtype, dtype);
            let normalized = if dtype == DType::F32 {
                through_views(&t, output)
            } else {
                assert!(is_cast_to(op_of(&t, output), dtype));
                through_views(&t, operands(&t, output)[0])
            };
            assert!(is_op(op_of(&t, normalized), &LogicalOp::Mul));
            let divisor = operands(&t, normalized)
                .into_iter()
                .map(|id| through_views(&t, id))
                .find(|&id| is_op(op_of(&t, id), &LogicalOp::Recip))
                .expect("normalization follows the value contraction");
            // A row with no admissible key divides by 1 (RowStats); every
            // other row selects the reduced sum, the false branch.
            let guarded = through_views(&t, operands(&t, divisor)[0]);
            assert!(is_op(op_of(&t, guarded), &LogicalOp::Select));
            let denominator = through_views(&t, operands(&t, guarded)[2]);
            assert!(is_op(
                op_of(&t, denominator),
                &LogicalOp::ReduceSum { axis_from_end: 0 }
            ));
            let weights = through_views(&t, operands(&t, denominator)[0]);
            assert!(is_op(op_of(&t, weights), &LogicalOp::Exp2));
            assert_eq!(dtype_of(&t, weights), DType::F32);
            for id in nodes(&t) {
                if id == output || matches!(op_of(&t, id), LogicalOp::Input { .. }) {
                    continue;
                }
                // The empty-row predicate (a comparison of the row max) is
                // Bool by type; it selects values but is not one of them.
                if dtype_of(&t, id) == DType::Bool {
                    assert!(
                        matches!(
                            op_of(&t, id),
                            LogicalOp::LessThan | LogicalOp::IndexMapApply { .. }
                        ),
                        "{variant}: a Bool SDPA node is not a predicate: {:?}",
                        op_of(&t, id)
                    );
                    continue;
                }
                assert_eq!(
                    dtype_of(&t, id),
                    DType::F32,
                    "{variant}: an internal SDPA value was narrowed to {dtype:?}"
                );
            }
        }
    }
}

#[test]
fn float_to_narrow_storage_records_truncation_and_keeps_output_dtype() {
    use crate::dtype::TorchDType;
    for source in [FLOAT, HALF, BFLOAT16] {
        for (target, dtype) in [
            (TorchDType::Char, DType::I8),
            (TorchDType::Short, DType::I16),
            (TorchDType::Byte, DType::U8),
        ] {
            let t = translate_one(
                node(
                    "torch.ops.aten._to_copy.default",
                    vec![
                        tensor_input("self", "x"),
                        scalar_type("dtype", target.code()),
                    ],
                    &["out"],
                ),
                &[("x", source, &[8]), ("out", target.code(), &[8])],
                &["x"],
                &["out"],
            );
            assert_eq!(t.outputs[0].dtype, dtype);
            assert!(any_node(&t, |op, _| matches!(
                op,
                LogicalOp::TruncCast(DType::Int)
            )));
            assert!(any_node(
                &t,
                |op, _| matches!(op, LogicalOp::Cast(d) if *d == dtype)
            ));
        }
    }
}

#[test]
fn narrow_scalar_helpers_and_rounded_division_stay_integer() {
    use crate::dtype::TorchDType;
    for (code, dtype) in [
        (TorchDType::Char.code(), DType::I8),
        (TorchDType::Short.code(), DType::I16),
        (TorchDType::Byte.code(), DType::U8),
    ] {
        for (target, extra) in [
            ("relu.default", vec![]),
            ("sign.default", vec![]),
            ("bitwise_not.default", vec![]),
            (
                "clamp.default",
                vec![scalar_int("min", 0), scalar_int("max", 6)],
            ),
            (
                "hardtanh.default",
                vec![scalar_int("min_val", 0), scalar_int("max_val", 6)],
            ),
            ("floor_divide.default", vec![tensor_input("other", "b")]),
        ] {
            let mut inputs = vec![tensor_input("self", "a")];
            inputs.extend(extra);
            let t = translate_one(
                node(&format!("torch.ops.aten.{target}"), inputs, &["out"]),
                &[("a", code, &[8]), ("b", code, &[8]), ("out", code, &[8])],
                &["a", "b"],
                &["out"],
            );
            assert_eq!(t.outputs[0].dtype, dtype, "{target}");
            assert!(
                !any_node(&t, |op, _| matches!(op, LogicalOp::TruncCast(_))),
                "{target} must not convert integer values through floating point"
            );
            if target == "floor_divide.default" {
                assert!(any_node(&t, |op, dtype| matches!(op, LogicalOp::TruncDiv)
                    && dtype == DType::Int));
            }
        }
    }
}

#[test]
fn fp64_scalar_literals_do_not_require_runtime_numeric_conversion() {
    use crate::dtype::TorchDType;
    for literal in [
        scalar_int("s", (1_i64 << 40) + 1),
        scalar_bool("s", true),
        scalar_float("s", 1.25),
    ] {
        let t = translate_one(
            node(
                "torch.ops.aten.scalar_tensor.default",
                vec![literal, scalar_type("dtype", TorchDType::Double.code())],
                &["out"],
            ),
            &[("out", TorchDType::Double.code(), &[])],
            &[],
            &["out"],
        );
        assert_eq!(t.outputs[0].dtype, DType::F64);
        assert!(!any_node(&t, |op, _| matches!(
            op,
            LogicalOp::Cast(DType::F64)
        )));
    }
}

#[test]
fn uint64_rounded_division_does_not_pass_through_float32() {
    use crate::dtype::TorchDType;
    let code = TorchDType::Uint64.code();
    let t = translate_one(
        node(
            "torch.ops.aten.floor_divide.default",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["out"],
        ),
        &[("a", code, &[8]), ("b", code, &[8]), ("out", code, &[8])],
        &["a", "b"],
        &["out"],
    );
    assert_eq!(t.outputs[0].dtype, DType::U64);
    assert!(any_node(&t, |op, dtype| matches!(op, LogicalOp::TruncDiv)
        && dtype == DType::U64));
    assert!(!any_node(&t, |op, _| matches!(
        op,
        LogicalOp::TruncCast(_) | LogicalOp::Recip
    )));
}
