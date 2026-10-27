//! `aten.add` / `aten.sub`: `alpha` scales `other`, and Bool addition is
//! logical OR. Both were silently dropped once (the alpha-aware lowering
//! sat in a file that was never compiled in), so these pin the recorded
//! graph, not just its numerics.

use luminal::prelude::*;

use super::Translation;
use super::test_support::*;
use crate::dtype::TorchDType;
use crate::pt2_schema::NodeInput;

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
