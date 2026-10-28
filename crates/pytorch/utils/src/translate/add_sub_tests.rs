//! `aten.add` / `aten.sub` / `aten.rsub`: `alpha` scales `other`, and
//! Bool addition is logical OR. Both were silently dropped once (the
//! alpha-aware lowering sat in a file that was never compiled in), so these
//! pin the recorded graph, not just its numerics. The arithmetic ops also
//! take a Python or symbolic scalar on either side.

use luminal::prelude::*;

use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

use super::Translation;
use super::test_support::*;
use crate::dtype::TorchDType;
use crate::pt2_schema::{Argument, NodeInput, RangeConstraint, SymIntArg, SymIntValue};

fn add(dtype: TorchDType, alpha: Option<NodeInput>, target: &str) -> Translation {
    let mut args = vec![tensor_input("self", "x"), tensor_input("other", "z")];
    args.extend(alpha);
    let code = dtype.code();
    translate_one(
        node(&format!("torch.ops.aten.{target}"), args, &["y"]),
        &[
            ("x", code, &[2, 3]),
            ("z", code, &[1, 3]),
            ("y", code, &[2, 3]),
        ],
        &["x", "z"],
        &["y"],
    )
}

fn count(t: &Translation, op: impl Fn(&LogicalOp) -> bool) -> usize {
    let graph = t.graph.logical.petgraph();
    graph.node_indices().filter(|&i| op(&graph[i].op)).count()
}

#[test]
fn bool_add_is_logical_or_never_an_add() {
    for alpha in [
        None,
        Some(scalar_bool("alpha", true)),
        Some(scalar_int("alpha", 2)),
        Some(scalar_int("alpha", -1)),
    ] {
        let t = add(TorchDType::Bool, alpha, "add.Tensor");
        assert_eq!(count(&t, |op| matches!(op, LogicalOp::Or)), 1);
        assert_eq!(count(&t, |op| matches!(op, LogicalOp::Add)), 0);
    }
}

#[test]
fn bool_add_with_zero_alpha_is_self() {
    let t = add(
        TorchDType::Bool,
        Some(scalar_bool("alpha", false)),
        "add.Tensor",
    );
    assert_eq!(
        count(&t, |op| matches!(op, LogicalOp::Or | LogicalOp::Add)),
        0
    );
}

#[test]
fn non_unit_alpha_scales_other() {
    // Relative to the same op without alpha: the frontend spells `a - b`
    // with its own multiply by -1, so the absolute count differs by op.
    let muls = |t: &Translation| count(t, |op| matches!(op, LogicalOp::Mul));
    for target in ["add.Tensor", "sub.Tensor"] {
        let plain = add(TorchDType::Double, None, target);
        let scaled = add(
            TorchDType::Double,
            Some(scalar_float("alpha", -3.125)),
            target,
        );
        assert_eq!(muls(&scaled), muls(&plain) + 1, "{target}");
    }
}

#[test]
fn omitted_or_unit_alpha_records_no_multiply() {
    for alpha in [
        None,
        Some(scalar_int("alpha", 1)),
        Some(scalar_float("alpha", 1.0)),
    ] {
        let t = add(TorchDType::Double, alpha, "add.Tensor");
        assert_eq!(count(&t, |op| matches!(op, LogicalOp::Mul)), 0);
        assert_eq!(count(&t, |op| matches!(op, LogicalOp::Add)), 1);
    }
}

// ---- scalar operands on either side, literal or symbolic ----
//
// Export records a Python number in either operand slot (`1 - x` is
// `rsub.Scalar(x, 1)` undecomposed and `sub.Tensor(1, x)` decomposed) and
// a dimension (`x.shape[0]`) as a `SymInt`. Both were refused once
// because the arithmetic lowerings read operand 0 as a tensor and operand
// 1 as a literal.

fn run_f32(t: &Translation, values: Vec<TypedBuffer>) -> Vec<f32> {
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(&values)
        .map(|(input, value)| (input.tensor, value.clone()))
        .collect();
    let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    for (input, value) in t.inputs.iter().zip(values) {
        runtime.set_data(input.tensor, value);
    }
    runtime.execute().unwrap();
    runtime.get_f32(t.outputs[0].tensor).unwrap().clone()
}

/// One F32 node over `x` (and `z` when an operand names it), both `[3]`.
fn f32_program(target: &str, args: Vec<NodeInput>) -> Translation {
    let inputs: Vec<&str> = ["x", "z"]
        .into_iter()
        .filter(|name| {
            args.iter()
                .any(|arg| arg.arg.as_tensor_name() == Some(*name))
        })
        .collect();
    translate_one(
        node(&format!("torch.ops.aten.{target}"), args, &["y"]),
        &[("x", FLOAT, &[3]), ("z", FLOAT, &[3]), ("y", FLOAT, &[3])],
        &inputs,
        &["y"],
    )
}

const X: [f32; 3] = [1.0, 2.0, 4.0];
const Z: [f32; 3] = [0.5, -3.0, 8.0];

#[test]
fn a_scalar_first_operand_is_subtracted_from_not_subtracted() {
    for (scalar, value) in [
        (scalar_float("self", 2.5), 2.5),
        (scalar_int("self", 1), 1.0),
    ] {
        let t = f32_program("sub.Tensor", vec![scalar, tensor_input("other", "x")]);
        let expected: Vec<f32> = X.iter().map(|x| value - x).collect();
        assert_eq!(run_f32(&t, vec![TypedBuffer::F32(X.to_vec())]), expected);
    }
}

#[test]
fn rsub_is_other_minus_alpha_times_self() {
    let x = || TypedBuffer::F32(X.to_vec());
    let t = f32_program(
        "rsub.Scalar",
        vec![tensor_input("self", "x"), scalar_int("other", 1)],
    );
    let expected: Vec<f32> = X.iter().map(|x| 1.0 - x).collect();
    assert_eq!(run_f32(&t, vec![x()]), expected);

    let t = f32_program(
        "rsub.Scalar",
        vec![
            tensor_input("self", "x"),
            scalar_float("other", 2.0),
            scalar_float("alpha", 3.0),
        ],
    );
    let expected: Vec<f32> = X.iter().map(|x| 2.0 - 3.0 * x).collect();
    assert_eq!(run_f32(&t, vec![x()]), expected);

    let t = f32_program(
        "rsub.Tensor",
        vec![
            tensor_input("self", "x"),
            tensor_input("other", "z"),
            scalar_int("alpha", 2),
        ],
    );
    let expected: Vec<f32> = X.iter().zip(Z).map(|(x, z)| z - 2.0 * x).collect();
    assert_eq!(
        run_f32(&t, vec![x(), TypedBuffer::F32(Z.to_vec())]),
        expected
    );
}

#[test]
fn a_scalar_first_operand_is_read_at_the_compute_dtype() {
    // F64 keeps the double: no F32 rounding of 0.1 is widened (the one
    // F32 constant is the frontend's exact `-1` of `a - b = a + -b`).
    let t = translate_one(
        node(
            "torch.ops.aten.sub.Tensor",
            vec![scalar_float("self", 0.1), tensor_input("other", "x")],
            &["y"],
        ),
        &[
            ("x", TorchDType::Double.code(), &[3]),
            ("y", TorchDType::Double.code(), &[3]),
        ],
        &["x"],
        &["y"],
    );
    assert_eq!(
        count(
            &t,
            |op| matches!(op, LogicalOp::ConstantF64(v) if *v == 0.1)
        ),
        1
    );
    assert_eq!(
        count(&t, |op| matches!(op, LogicalOp::Constant(v) if *v != -1.0)),
        0
    );

    // Half computes in F32 and rounds once, at the store.
    let t = translate_one(
        node(
            "torch.ops.aten.rsub.Scalar",
            vec![tensor_input("self", "x"), scalar_float("other", 1.0)],
            &["y"],
        ),
        &[("x", HALF, &[3]), ("y", HALF, &[3])],
        &["x"],
        &["y"],
    );
    let logical = &t.graph.logical;
    let output = t.outputs[0].tensor;
    assert_eq!(logical.value_dtype(output), DType::F16);
    assert_eq!(
        count(&t, |op| matches!(op, LogicalOp::Cast(DType::F16))),
        1,
        "only the store rounds to F16"
    );
}

#[test]
fn bool_rsub_is_refused_like_torch() {
    let error = crate::translate::translate(&parsed_nodes(
        vec![node(
            "torch.ops.aten.rsub.Scalar",
            vec![tensor_input("self", "x"), scalar_bool("other", true)],
            &["y"],
        )],
        &[("x", BOOL, &[3]), ("y", BOOL, &[3])],
        &["x"],
        &["y"],
    ))
    .err()
    .expect("Bool subtraction must be refused");
    assert!(
        format!("{error:#}").contains("subtraction of Bool"),
        "{error:#}"
    );
}

// ---- symbolic scalars ----

fn sym_int(name: &str, value: &str) -> NodeInput {
    input(
        name,
        Argument::SymInt(SymIntArg {
            as_sym_int: SymIntValue {
                as_name: value.to_string(),
            },
        }),
    )
}

/// `x: dtype[s0, 3]` with `n = s0` and `n1 = s0 + 1` as sym_int values,
/// `s0` in `[2, 64]`, and one node over it whose result is `out` of
/// `[s0, 3]`.
fn symbolic_program(
    target: &str,
    args: Vec<NodeInput>,
    dtype: u32,
    out: u32,
) -> anyhow::Result<Translation> {
    let symbol = |expr: &str, hint: i64| serde_json::json!({ "as_expr": { "expr_str": expr, "hint": { "as_int": hint } } });
    let s0 = "Symbol('s0', positive=True, integer=True)";
    let mut parsed = parsed_nodes(
        vec![node(&format!("torch.ops.aten.{target}"), args, &["y"])],
        &[
            ("x", dtype, &[4, 3]),
            ("z", dtype, &[4, 3]),
            ("y", out, &[4, 3]),
        ],
        &["x", "z"],
        &["y"],
    );
    let graph = &mut parsed.program.graph_module.graph;
    for name in ["x", "z", "y"] {
        graph.tensor_values.get_mut(name).unwrap().sizes[0] =
            serde_json::from_value(symbol(s0, 4)).unwrap();
    }
    graph.sym_int_values.insert("n".to_string(), symbol(s0, 4));
    graph.sym_int_values.insert(
        "n1".to_string(),
        symbol(&format!("Add({s0}, Integer(1))"), 5),
    );
    parsed.program.range_constraints.insert(
        "s0".to_string(),
        RangeConstraint {
            min_val: Some(2),
            max_val: Some(64),
        },
    );
    crate::translate::translate(&parsed)
}

/// The recorded iotas that read the dimension `s0`.
fn symbolic_iotas(t: &Translation) -> usize {
    count(
        t,
        |op| matches!(op, LogicalOp::Iota { value_expr } if value_expr.contains("\"s0\"")),
    )
}

#[test]
fn a_symbolic_scalar_is_its_dimension_expression_on_either_side() {
    let x = || tensor_input("self", "x");
    for (target, args) in [
        ("mul.Tensor", vec![x(), sym_int("other", "n")]),
        ("mul.Tensor", vec![x(), sym_int("other", "n1")]),
        ("mul.Scalar", vec![x(), sym_int("other", "n")]),
        ("add.Tensor", vec![x(), sym_int("other", "n")]),
        ("add.Scalar", vec![x(), sym_int("other", "n1")]),
        ("sub.Tensor", vec![x(), sym_int("other", "n")]),
        ("div.Tensor", vec![x(), sym_int("other", "n")]),
        ("div.Scalar", vec![x(), sym_int("other", "n")]),
        // `x.shape[0] - x`, undecomposed and decomposed.
        ("rsub.Scalar", vec![x(), sym_int("other", "n")]),
        (
            "sub.Tensor",
            vec![sym_int("self", "n"), tensor_input("other", "x")],
        ),
    ] {
        let t = symbolic_program(target, args, FLOAT, FLOAT)
            .unwrap_or_else(|error| panic!("{target}: {error:#}"));
        assert_eq!(symbolic_iotas(&t), 1, "{target}");
        assert_eq!(
            t.graph.logical.value_dtype(t.outputs[0].tensor),
            DType::F32,
            "{target}"
        );
    }
}

#[test]
fn a_symbolic_scalar_takes_the_declared_result_dtype() {
    let long = TorchDType::Long.code();
    let args = || vec![tensor_input("self", "x"), sym_int("other", "n")];
    // int64 * SymInt stays int64: the dimension is never read as a float.
    let t = symbolic_program("mul.Tensor", args(), long, long).unwrap();
    assert_eq!(symbolic_iotas(&t), 1);
    assert_eq!(t.graph.logical.value_dtype(t.outputs[0].tensor), DType::I64);
    assert_eq!(count(&t, |op| matches!(op, LogicalOp::Cast(DType::F32))), 0);
    // int64 / SymInt is true division: both sides meet at F32.
    let t = symbolic_program("div.Tensor", args(), long, FLOAT).unwrap();
    assert_eq!(symbolic_iotas(&t), 1);
    assert_eq!(t.graph.logical.value_dtype(t.outputs[0].tensor), DType::F32);
    assert_eq!(count(&t, |op| matches!(op, LogicalOp::Cast(DType::F32))), 2);
}

#[test]
fn a_symbolic_alpha_scales_other() {
    let args = |alpha| {
        let mut args = vec![tensor_input("self", "x"), tensor_input("other", "z")];
        args.extend(alpha);
        args
    };
    let plain = symbolic_program("add.Tensor", args(None), FLOAT, FLOAT).unwrap();
    let scaled = symbolic_program(
        "add.Tensor",
        args(Some(sym_int("alpha", "n"))),
        FLOAT,
        FLOAT,
    )
    .unwrap();
    let muls = |t: &Translation| count(t, |op| matches!(op, LogicalOp::Mul));
    assert_eq!(muls(&scaled), muls(&plain) + 1);
    assert_eq!(symbolic_iotas(&scaled), 1);

    // Whether a Bool add keeps `other` would be the alpha's runtime value.
    let error = symbolic_program("add.Tensor", args(Some(sym_int("alpha", "n"))), BOOL, BOOL)
        .err()
        .expect("a symbolic alpha on a Bool add must be refused");
    assert!(format!("{error:#}").contains("symbolic alpha"), "{error:#}");
}
