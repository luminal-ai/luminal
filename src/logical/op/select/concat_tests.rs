//! Tests for the concatenation rules (concat_slices.egg): one per rule on
//! small graphs, and a saturation regression on a transformer-shaped graph,
//! a chain of blocks that each concatenate two streams on entry and slice
//! them apart on exit. The regression saturates the model alone and records
//! the e-graph's size against a budget, then adds the expected spelling and
//! checks it lands in the model's classes.

use crate::egglog_snippet::{assembled_program_for, new_egraph};
use crate::prelude::*;

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

const A: usize = 4;
const B: usize = 6;
const S: usize = A + B;
const H: usize = 2;
const D: usize = 8;
const HD: usize = H * D;
const M: usize = 32;

/// Interleaved rotary embedding over `[1, S, H, D]` with `(S, D/2)`
/// tables, as vllm-omni's `apply_rotary_emb_torch(interleaved=True)`
/// spells it: even/odd lanes, `stack((-x2, x1), -1)` flattened back, the
/// tables repeated per pair and broadcast over the heads.
fn rope(x: GraphTensor, cos: GraphTensor, sin: GraphTensor) -> GraphTensor {
    let pairs = x.split_dims(3, 2);
    let x1 = pairs.slice_along(0..1, 4);
    let x2 = pairs.slice_along(1..2, 4);
    let rot = (-x2).concat_along(x1, 4).merge_dims(3, 4);
    let table = |t: GraphTensor| {
        t.expand_dim(1, H)
            .expand_dim(3, 2)
            .merge_dims(2, 3)
            .expand_dim(0, 1)
    };
    x * table(cos) + rot * table(sin)
}

/// The per-head RMS norm over the last axis.
fn rms(x: GraphTensor) -> GraphTensor {
    x.std_norm(vec![3], 1e-6_f32)
}

fn constant_like(t: GraphTensor, value: f32) -> GraphTensor {
    t.graph().constant_f32(value).expand_rhs(t.dims())
}

/// Scaled dot-product attention as the PyTorch translator decomposes it
/// (`translate/attention.rs::translate_sdpa`, no mask): `[1, S, H, D]`
/// operands permuted to heads first, the result permuted back and its heads
/// flattened, `[1, S, H*D]`.
fn sdpa(q: GraphTensor, k: GraphTensor, v: GraphTensor) -> GraphTensor {
    let q = q.permute(vec![0, 2, 1, 3]);
    let k = k.permute(vec![0, 2, 1, 3]);
    let v = v.permute(vec![0, 2, 1, 3]);
    let scores = q.matmul(k.transpose(2, 3));
    let scores = scores * constant_like(scores, 1.0 / (D as f32).sqrt());
    let axis = 3;
    let row_max = scores.max(vec![axis]);
    let empty = row_max.eq(constant_like(row_max, f32::NEG_INFINITY));
    let row_max = empty.select(constant_like(row_max, 0.0), row_max);
    let maximum = row_max.expand_to_shape_on_axes(scores.dims(), vec![axis]);
    let weights = (scores - maximum).exp();
    let row_sum = weights.sum(vec![axis]);
    let row_sum = empty.select(constant_like(row_sum, 1.0), row_sum);
    let numerator = weights.matmul(v);
    let denominator = row_sum.expand_to_shape_on_axes(numerator.dims(), vec![axis]);
    (numerator / denominator)
        .permute(vec![0, 2, 1, 3])
        .merge_dims(2, 3)
}

fn linear(x: GraphTensor, w: GraphTensor) -> GraphTensor {
    x.matmul(w.permute(vec![1, 0]))
}

/// The e-graph's size after saturating `text`, and the graph.
fn saturate(text: &str) -> (egglog::EGraph, usize, f64) {
    let mut egraph = new_egraph();
    egraph
        .parse_and_run_program(None, &assembled_program_for(&[]))
        .unwrap();
    let start = std::time::Instant::now();
    egraph.parse_and_run_program(None, text).unwrap();
    egraph.parse_and_run_program(None, SCHEDULE).unwrap();
    let rows = egraph.num_tuples();
    (egraph, rows, start.elapsed().as_secs_f64())
}

/// The `let`s of `text` not already bound in `bound`.
fn new_lets(text: &str, bound: &str) -> String {
    let names: std::collections::HashSet<&str> = bound
        .lines()
        .filter_map(|line| line.strip_prefix("(let ")?.split_whitespace().next())
        .collect();
    text.lines()
        .filter(|line| {
            line.strip_prefix("(let ")
                .and_then(|rest| rest.split_whitespace().next())
                .is_none_or(|name| !names.contains(name))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Saturate the model's cone and record its size, then add the expected
/// spelling, saturate again and check each `(model, expected)` pair is one
/// class and each fact holds (`{name}` placeholders are the values' lets).
fn check(
    cx: &Graph,
    model: &[NodeIndex],
    pairs: &[(GraphTensor, GraphTensor)],
    facts: &[String],
    budget: usize,
    label: &str,
) {
    let model_text = cx.logical.render(model).unwrap();
    let (mut egraph, rows, secs) = saturate(&model_text);
    eprintln!("{label}: {rows} rows after saturation in {secs:.2} s (budget {budget})");
    let all = cx.logical.render_all().unwrap();
    egraph
        .parse_and_run_program(None, &new_lets(&all, &model_text))
        .unwrap();
    egraph.parse_and_run_program(None, SCHEDULE).unwrap();
    let mut failed = vec![];
    let equalities = pairs.iter().map(|(got, want)| {
        format!(
            "(= {} {})",
            cx.logical.let_name(got.id),
            cx.logical.let_name(want.id)
        )
    });
    for fact in equalities.chain(facts.iter().cloned()) {
        if egraph
            .parse_and_run_program(None, &format!("(check {fact})"))
            .is_err()
        {
            failed.push(fact);
        }
    }
    assert!(failed.is_empty(), "{label}: not reached: {failed:?}");
    assert!(
        rows <= budget,
        "{label}: {rows} rows exceed the budget {budget}"
    );
}

/// FLUX's single blocks as upstream vllm-omni traces them: each block
/// concatenates the two streams on entry and slices them apart on exit.
/// Saturated, each block's entry concatenation is the previous block's
/// joint output, so the chain is the blocks over the streams joined once.
#[test]
fn concatenating_blocks_chain_without_their_slices() {
    let mut cx = Graph::new();
    let txt = cx.named_tensor("txt", vec![1, A, HD], DType::F32);
    let img = cx.named_tensor("img", vec![1, B, HD], DType::F32);
    let cos = cx.named_tensor("cos", vec![S, D / 2], DType::F32);
    let sin = cx.named_tensor("sin", vec![S, D / 2], DType::F32);
    let weights: Vec<_> = (0..3)
        .map(|i| {
            let w = |cx: &mut Graph, name: &str, rows: usize, cols: usize| {
                cx.named_tensor(format!("{name}{i}"), vec![rows, cols], DType::F32)
            };
            (
                w(&mut cx, "wq", HD, HD),
                w(&mut cx, "wk", HD, HD),
                w(&mut cx, "wv", HD, HD),
                w(&mut cx, "wm", M, HD),
                w(&mut cx, "wo", HD, HD + M),
            )
        })
        .collect();
    type Weights = (
        GraphTensor,
        GraphTensor,
        GraphTensor,
        GraphTensor,
        GraphTensor,
    );
    let body = |x: GraphTensor, (wq, wk, wv, wm, wo): Weights| {
        let n = x.layer_norm(vec![2], 1e-6_f32);
        let heads = |t: GraphTensor| t.split_dims(2, D);
        let q = rope(rms(heads(linear(n, wq))), cos, sin);
        let k = rope(rms(heads(linear(n, wk))), cos, sin);
        let v = heads(linear(n, wv));
        let attn = sdpa(q, k, v);
        let mlp = linear(n, wm).gelu_fast_tanh_approximation();
        x + linear(attn.concat_along(mlp, 2), wo)
    };
    let (mut t, mut i) = (txt, img);
    let mut entries = vec![];
    for w in &weights {
        let x = t.concat_along(i, 1);
        entries.push(x);
        let y = body(x, *w);
        t = y.slice_along(..A, 1);
        i = y.slice_along(A.., 1);
    }
    // Expected: the streams joined once and the blocks applied to the join.
    let mut joint = vec![txt.concat_along(img, 1)];
    for w in &weights {
        let next = body(*joint.last().unwrap(), *w);
        joint.push(next);
    }
    let last = *joint.last().unwrap();
    check(
        &cx,
        &[t.id, i.id],
        &[
            (entries[1], joint[1]),
            (entries[2], joint[2]),
            (t, last.slice_along(..A, 1)),
            (i, last.slice_along(A.., 1)),
        ],
        &[],
        // 14115 rows without the concatenation rules, 15545 with them.
        17_000,
        "three concatenating blocks",
    );
}

/// Saturate the whole graph and run each command against it.
fn holds(cx: &Graph, commands: &[String]) {
    let (mut egraph, _, _) = saturate(&cx.logical.render_all().unwrap());
    let mut failed = vec![];
    for command in commands {
        if egraph.parse_and_run_program(None, command).is_err() {
            failed.push(command.clone());
        }
    }
    assert!(failed.is_empty(), "failed: {failed:?}");
}

fn same(cx: &Graph, a: GraphTensor, b: GraphTensor) -> String {
    format!(
        "(check (= {} {}))",
        cx.logical.let_name(a.id),
        cx.logical.let_name(b.id)
    )
}

fn apart(cx: &Graph, a: GraphTensor, b: GraphTensor) -> String {
    format!(
        "(fail (check (= {} {})))",
        cx.logical.let_name(a.id),
        cx.logical.let_name(b.id)
    )
}

/// select/concat_slices.egg: a select spelled as a concatenation is
/// certified with its parts, its two row slices (`split_with_sizes`) are
/// those parts, and a padding select is not a concatenation.
#[test]
fn a_concatenation_is_certified_and_its_row_slices_are_its_parts() {
    let mut cx = Graph::new();
    let a = cx.named_tensor("a", vec![2, A, 3], DType::F32);
    let b = cx.named_tensor("b", vec![2, B, 3], DType::F32);
    let x = a.concat_along(b, 1);
    let (low, high) = (x.slice_along(..A, 1), x.slice_along(A.., 1));
    let padded = a.pad_along(0, B, 1, 0.0);
    let name = |t: GraphTensor| cx.logical.let_name(t.id);
    holds(
        &cx,
        &[
            format!(
                "(check (concat-parts {} 1 {A} {} {}))",
                name(x),
                name(a),
                name(b)
            ),
            same(&cx, low, a),
            same(&cx, high, b),
            format!(
                "(fail (check (concat-template {} ?k ?s ?c ?t ?f ?shape)))",
                name(padded)
            ),
        ],
    );
}

/// select/concat_slices.egg: the concatenation of a tensor's two row slices
/// is the tensor; slices of two different tensors are not either one.
#[test]
fn the_concatenation_of_a_tensors_row_slices_is_the_tensor() {
    let mut cx = Graph::new();
    let x = cx.named_tensor("x", vec![2, S, 3], DType::F32);
    let y = cx.named_tensor("y", vec![2, S, 3], DType::F32);
    let joined = x.slice_along(..A, 1).concat_along(x.slice_along(A.., 1), 1);
    let mixed = x.slice_along(..A, 1).concat_along(y.slice_along(A.., 1), 1);
    holds(
        &cx,
        &[
            same(&cx, joined, x),
            apart(&cx, mixed, x),
            apart(&cx, mixed, y),
        ],
    );
}
