use luminal::prelude::*;

use super::test_support::*;

fn gelu(approximate: &str) -> super::Translation {
    translate_one(
        node(
            "torch.ops.aten.gelu.default",
            vec![
                tensor_input("self", "input"),
                string("approximate", approximate),
            ],
            &["output"],
        ),
        &[("input", FLOAT, &[8]), ("output", FLOAT, &[8])],
        &["input"],
        &["output"],
    )
}

#[test]
fn tanh_gelu_preserves_the_authored_tanh_approximation() {
    let translation = gelu("tanh");
    assert!(
        translation
            .graph
            .logical
            .petgraph()
            .node_weights()
            .any(|node| node.op == LogicalOp::Tanh)
    );
}

#[test]
fn exact_gelu_does_not_silently_become_tanh_gelu() {
    let translation = gelu("none");
    assert!(
        !translation
            .graph
            .logical
            .petgraph()
            .node_weights()
            .any(|node| node.op == LogicalOp::Tanh)
    );
}
