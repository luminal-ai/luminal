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

fn cat_along(names: &[&str], dim: i64, out: &str) -> crate::pt2_schema::Node {
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
            scalar_int("dim", dim),
        ],
        &[out],
    )
}

fn heads_first(x: &str, out: &str) -> crate::pt2_schema::Node {
    node(
        "torch.ops.aten.permute.default",
        vec![tensor_input("self", x), ints("dims", &[0, 2, 1, 3])],
        &[out],
    )
}

fn sdpa(q: &str, k: &str, v: &str, out: &str) -> crate::pt2_schema::Node {
    node(
        "torch.ops.aten.scaled_dot_product_attention.default",
        vec![
            tensor_input("query", q),
            tensor_input("key", k),
            tensor_input("value", v),
            scalar_float("dropout_p", 0.0),
            scalar_bool("is_causal", false),
        ],
        &[out],
    )
}

fn linear(x: &str, w: &str, out: &str) -> crate::pt2_schema::Node {
    node(
        "torch.ops.aten.linear.default",
        vec![tensor_input("input", x), tensor_input("weight", w)],
        &[out],
    )
}

/// An attention over the concatenation of two query streams, split back
/// into the streams (`split_with_sizes`) and projected, is one attention
/// per query stream over the concatenated keys (select/concat_sink.egg):
/// the translator's own spelling of cat, permute, the attention and the
/// split, in f32 and in bf16 (whose attention computes in f32 opmath).
#[test]
fn an_attention_split_by_query_rows_is_one_attention_per_stream() {
    const A: i64 = 4;
    const B: i64 = 6;
    const S: i64 = A + B;
    for dtype in [FLOAT, BFLOAT16] {
        let streams = |rows: i64| [1, rows, 2, 8];
        let joint = [1, S, 2, 8];
        let t = translate_nodes(
            vec![
                cat_along(&["qa", "qb"], 1, "q"),
                cat_along(&["ka", "kb"], 1, "k"),
                cat_along(&["va", "vb"], 1, "v"),
                heads_first("q", "qp"),
                heads_first("k", "kp"),
                heads_first("v", "vp"),
                sdpa("qp", "kp", "vp", "o"),
                heads_first("o", "ot"),
                node(
                    "torch.ops.aten.split_with_sizes.default",
                    vec![
                        tensor_input("self", "ot"),
                        ints("split_sizes", &[A, B]),
                        scalar_int("dim", 1),
                    ],
                    &["ot_a", "ot_b"],
                ),
                linear("ot_a", "w_a", "txt"),
                linear("ot_b", "w_b", "img"),
                heads_first("qa", "qap"),
                heads_first("qb", "qbp"),
                sdpa("qap", "kp", "vp", "oa"),
                sdpa("qbp", "kp", "vp", "ob"),
                heads_first("oa", "oat"),
                heads_first("ob", "obt"),
                linear("oat", "w_a", "txt_exp"),
                linear("obt", "w_b", "img_exp"),
            ],
            &[
                ("qa", dtype, &streams(A)),
                ("ka", dtype, &streams(A)),
                ("va", dtype, &streams(A)),
                ("qb", dtype, &streams(B)),
                ("kb", dtype, &streams(B)),
                ("vb", dtype, &streams(B)),
                ("w_a", dtype, &[8, 8]),
                ("w_b", dtype, &[8, 8]),
                ("q", dtype, &joint),
                ("k", dtype, &joint),
                ("v", dtype, &joint),
                ("qp", dtype, &[1, 2, S, 8]),
                ("kp", dtype, &[1, 2, S, 8]),
                ("vp", dtype, &[1, 2, S, 8]),
                ("o", dtype, &[1, 2, S, 8]),
                ("ot", dtype, &joint),
                ("ot_a", dtype, &streams(A)),
                ("ot_b", dtype, &streams(B)),
                ("txt", dtype, &streams(A)),
                ("img", dtype, &streams(B)),
                ("qap", dtype, &[1, 2, A, 8]),
                ("qbp", dtype, &[1, 2, B, 8]),
                ("oa", dtype, &[1, 2, A, 8]),
                ("ob", dtype, &[1, 2, B, 8]),
                ("oat", dtype, &streams(A)),
                ("obt", dtype, &streams(B)),
                ("txt_exp", dtype, &streams(A)),
                ("img_exp", dtype, &streams(B)),
            ],
            &["qa", "ka", "va", "qb", "kb", "vb", "w_a", "w_b"],
            &["txt", "img", "txt_exp", "img_exp"],
        );
        let logical = &t.graph.logical;
        let name = |i: usize| logical.let_name(t.outputs[i].tensor);
        let program = format!(
            "{}\n{}\n{SCHEDULE}\n(check (= {} {}))\n(check (= {} {}))\n",
            assembled_program_for(&[]),
            logical.render_all().unwrap(),
            name(0),
            name(2),
            name(1),
            name(3)
        );
        new_egraph()
            .parse_and_run_program(None, &program)
            .unwrap_or_else(|err| {
                panic!("dtype {dtype}: each stream's output is its own attention: {err}")
            });
    }
}
