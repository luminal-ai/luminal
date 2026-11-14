//! `luminal_collective.all_reduce`: the translator records it as the logical
//! op, verbatim (a bf16 all-reduce stays bf16).
use super::test_support::*;
use crate::pt2_schema::Argument;

const GROUP: &str = r#"{"groups":[[0,1]],"world":2}"#;

fn all_reduce_node(name: &str, group: Option<&str>, width: i64, sequence: i64) -> crate::pt2_schema::Node {
    let mut inputs = vec![tensor_input("input", "x")];
    inputs.push(match group {
        Some(group) => input(
            "group",
            Argument::Other(serde_json::json!({ "as_string": group })),
        ),
        None => none("group"),
    });
    inputs.push(scalar_int("width", width));
    inputs.push(scalar_int("sequence", sequence));
    node("torch.ops.luminal_collective.all_reduce.default", inputs, &[name])
}

#[test]
fn the_all_reduce_keeps_the_operand_shape_and_dtype_unwidened() {
    for dtype in [FLOAT, BFLOAT16] {
        let t = translate_one(
            all_reduce_node("out", Some(GROUP), 2, 0),
            &[("x", dtype, &[4, 16]), ("out", dtype, &[4, 16])],
            &["x"],
            &["out"],
        );
        let rendered = t.graph.logical.render_all().unwrap();
        assert!(
            rendered.contains(r#"(LogicalAllReduce v0 "{\"groups\":[[0,1]],\"world\":2}" 2 0)"#),
            "{rendered}"
        );
        // the sum is the op itself: no widening cast around it
        assert!(!rendered.contains("LogicalCast"), "{rendered}");
        let out = t.graph.logical.leaves()[0];
        let dims = t.graph.logical.petgraph()[out].dims.clone();
        assert_eq!(
            dims,
            vec![luminal::prelude::IntExpr::from(4usize), luminal::prelude::IntExpr::from(16usize)]
        );
    }
}

#[test]
fn a_missing_all_reduce_group_string_is_refused() {
    let parsed = parsed_nodes(
        vec![all_reduce_node("out", None, 2, 0)],
        &[("x", FLOAT, &[4]), ("out", FLOAT, &[4])],
        &["x"],
        &["out"],
    );
    let error = crate::translate::translate(&parsed).err().expect("refused");
    assert!(format!("{error:#}").contains("group must be a string"), "{error:#}");
}
