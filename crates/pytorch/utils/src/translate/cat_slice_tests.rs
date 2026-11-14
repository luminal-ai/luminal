//! A slice of a concatenation that reads one operand whole is that operand
//! (diffusers' `flip_sin_to_cos`: `cat([sin, cos])[:, half:]` is `cos`). The
//! concatenation is a select on a mask over clamped padding views; the slice
//! distributes over the select, sinks down the mask's truth test to the mask
//! iota, the bounds decide it over the slice's range, the select collapses,
//! and the slice of the padding view opens to the identity.
use super::test_support::*;
use crate::pt2_schema::{Argument, TensorName, TensorsArg};
use luminal::egglog_snippet::{assembled_program_for, new_egraph};

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

fn cat(names: &[&str], out: &str) -> crate::pt2_schema::Node {
    node(
        "torch.ops.aten.cat.default",
        vec![
            input(
                "tensors",
                Argument::Tensors(TensorsArg {
                    as_tensors: names
                        .iter()
                        .map(|name| TensorName {
                            name: name.to_string(),
                        })
                        .collect(),
                }),
            ),
            scalar_int("dim", -1),
        ],
        &[out],
    )
}

fn slice(x: &str, start: i64, end: i64, out: &str) -> crate::pt2_schema::Node {
    node(
        "torch.ops.aten.slice.Tensor",
        vec![
            tensor_input("self", x),
            scalar_int("dim", 1),
            scalar_int("start", start),
            scalar_int("end", end),
            scalar_int("step", 1),
        ],
        &[out],
    )
}

#[test]
fn a_slice_of_a_concatenation_that_reads_one_operand_is_that_operand() {
    let t = translate_nodes(
        vec![
            cat(&["sin", "cos"], "emb"),
            slice("emb", 128, i64::MAX, "high"),
            slice("emb", 0, 128, "low"),
        ],
        &[
            ("sin", FLOAT, &[2, 128]),
            ("cos", FLOAT, &[2, 128]),
            ("emb", FLOAT, &[2, 256]),
            ("high", FLOAT, &[2, 128]),
            ("low", FLOAT, &[2, 128]),
        ],
        &["sin", "cos"],
        &["high", "low"],
    );
    let logical = &t.graph.logical;
    let sin = logical.let_name(t.inputs[0].tensor);
    let cos = logical.let_name(t.inputs[1].tensor);
    let high = logical.let_name(t.outputs[0].tensor);
    let low = logical.let_name(t.outputs[1].tensor);
    let program = format!(
        "{}\n{}\n{SCHEDULE}\n(check (= {high} {cos}))\n(check (= {low} {sin}))\n",
        assembled_program_for(&[]),
        logical.render_all().unwrap()
    );
    new_egraph()
        .parse_and_run_program(None, &program)
        .expect("each slice reads one operand of the concatenation whole and is that operand");
}
