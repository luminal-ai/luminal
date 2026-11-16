//! Tests for the concatenation rules (concat_slices.egg, concat_rowwise.egg,
//! concat_sink.egg): one per rule on small graphs, and saturation
//! regressions on transformer-shaped graphs: a joint attention block whose
//! query, key and value streams are concatenated along the sequence before
//! a row-indexed rotary table, a per-head norm and one attention, then
//! split back into two linears; and a chain of blocks that each concatenate
//! two streams on entry and slice them apart on exit. Each regression
//! saturates the model alone and records the e-graph's size against a
//! budget, then adds the expected per-stream (or joined) spelling and checks
//! it lands in the model's classes.

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

/// FLUX's double block as upstream vllm-omni traces it: the text and image
/// query, key and value streams concatenated along the sequence (text
/// first), the rotary embedding over ONE joint table, the per-head norm,
/// one attention over all queries, the result split back into the streams
/// and each projected. Saturated, each stream's output is one attention of
/// its own queries (its rows of the table, its own norm) over the keys
/// spelled as the concatenation of the two streams' keys, which is the
/// spelling a backend's joint attention over two key sets reads.
#[test]
fn joint_attention_block_splits_into_per_stream_attention() {
    let mut cx = Graph::new();
    let stream = |cx: &mut Graph, name: &str, rows: usize| {
        cx.named_tensor(name, vec![1, rows, H, D], DType::F32)
    };
    let (qa, ka, va) = (
        stream(&mut cx, "qa", A),
        stream(&mut cx, "ka", A),
        stream(&mut cx, "va", A),
    );
    let (qb, kb, vb) = (
        stream(&mut cx, "qb", B),
        stream(&mut cx, "kb", B),
        stream(&mut cx, "vb", B),
    );
    let cos = cx.named_tensor("cos", vec![S, D / 2], DType::F32);
    let sin = cx.named_tensor("sin", vec![S, D / 2], DType::F32);
    let w_txt = cx.named_tensor("w_txt", vec![HD, HD], DType::F32);
    let w_img = cx.named_tensor("w_img", vec![HD, HD], DType::F32);

    let q = rms(rope(qa.concat_along(qb, 1), cos, sin));
    let k = rms(rope(ka.concat_along(kb, 1), cos, sin));
    let v = va.concat_along(vb, 1);
    let o = sdpa(q, k, v);
    let txt = linear(o.slice_along(..A, 1), w_txt);
    let img = linear(o.slice_along(A.., 1), w_img);

    let lo = |t: GraphTensor| t.slice_along(..A, 0);
    let hi = |t: GraphTensor| t.slice_along(A.., 0);
    let (ka_exp, kb_exp) = (
        rms(rope(ka, lo(cos), lo(sin))),
        rms(rope(kb, hi(cos), hi(sin))),
    );
    let k_exp = ka_exp.concat_along(kb_exp, 1);
    let qa_exp = rms(rope(qa, lo(cos), lo(sin)));
    let qb_exp = rms(rope(qb, hi(cos), hi(sin)));
    let txt_exp = linear(sdpa(qa_exp, k_exp, v), w_txt);
    let img_exp = linear(sdpa(qb_exp, k_exp, v), w_img);
    let name = |t: GraphTensor| cx.logical.let_name(t.id);
    // The keys hold a certified concatenation whose parts are the
    // per-stream keys: a select over clamped views of exactly those parts.
    let keys = format!(
        "(concat-parts {} 2 {A} {} {})",
        name(k),
        name(ka_exp),
        name(kb_exp)
    );
    check(
        &cx,
        &[txt.id, img.id],
        &[
            (q.slice_along(..A, 1), qa_exp),
            (q.slice_along(A.., 1), qb_exp),
            (k, k_exp),
            (txt, txt_exp),
            (img, img_exp),
        ],
        &[keys],
        // 11367 rows without the concatenation rules, 53445 with them: two
        // query streams' attentions beside the joint one.
        60_000,
        "joint attention block",
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
        // 14115 rows without the concatenation rules, 22230 with them.
        25_000,
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

/// select/concat_rowwise.egg: an elementwise op of a concatenation and a
/// row-indexed table is the concatenation of the op over the parts and the
/// table's rows, on either side, and the result is again a concatenation
/// the next op pushes through.
#[test]
fn an_elementwise_op_of_a_concatenation_is_the_concatenation_over_its_parts() {
    let mut cx = Graph::new();
    let a = cx.named_tensor("a", vec![2, A, 3], DType::F32);
    let b = cx.named_tensor("b", vec![2, B, 3], DType::F32);
    let t = cx.named_tensor("t", vec![2, S, 3], DType::F32);
    let x = a.concat_along(b, 1);
    let (t_lo, t_hi) = (t.slice_along(..A, 1), t.slice_along(A.., 1));
    let product = x * t;
    let flipped = t + x;
    let chained = (x * t).exp();
    let product_exp = (a * t_lo).concat_along(b * t_hi, 1);
    let flipped_exp = (t_lo + a).concat_along(t_hi + b, 1);
    let chained_exp = (a * t_lo).exp().concat_along((b * t_hi).exp(), 1);
    holds(
        &cx,
        &[
            same(&cx, product, product_exp),
            same(&cx, flipped, flipped_exp),
            same(&cx, chained, chained_exp),
        ],
    );
}

/// select/concat_sink.egg: a norm over the feature axis of a concatenation
/// is the concatenation of the parts' norms (the statistics' row slices
/// sink through their reduction), and a projection of a concatenation read
/// by its rows is the projection of the part; a projection of a projection
/// is left as the slice of the joint result.
#[test]
fn row_slices_sink_through_norms_and_a_projection_of_a_concatenation() {
    let mut cx = Graph::new();
    let a = cx.named_tensor("a", vec![1, A, 4], DType::F32);
    let b = cx.named_tensor("b", vec![1, B, 4], DType::F32);
    let w = cx.named_tensor("w", vec![4, 5], DType::F32);
    let w2 = cx.named_tensor("w2", vec![5, 6], DType::F32);
    let w3 = cx.named_tensor("w3", vec![5, 5], DType::F32);
    let x = a.concat_along(b, 1);
    let normed = x.std_norm(vec![2], 1e-6_f32);
    let projected = x.matmul(w);
    let twice = projected.matmul(w3);
    let _readers = (
        projected.slice_along(..A, 1).matmul(w2),
        twice.slice_along(..A, 1).matmul(w2),
    );
    let normed_exp = a
        .std_norm(vec![2], 1e-6_f32)
        .concat_along(b.std_norm(vec![2], 1e-6_f32), 1);
    holds(
        &cx,
        &[
            same(&cx, normed, normed_exp),
            same(&cx, projected.slice_along(..A, 1), a.matmul(w)),
            apart(&cx, twice.slice_along(..A, 1), a.matmul(w).matmul(w3)),
        ],
    );
}

/// select/concat_sink.egg: a row slice sinks through a reduction over an
/// earlier axis (the row axis keeps its index from the end) and through a
/// permute that moves the rows.
#[test]
fn row_slices_sink_through_reductions_over_earlier_axes_and_permutes() {
    let mut cx = Graph::new();
    let a = cx.named_tensor("a", vec![2, 3, A], DType::F32);
    let b = cx.named_tensor("b", vec![2, 3, B], DType::F32);
    let w = cx.named_tensor("w", vec![3, 4], DType::F32);
    let x = a.concat_along(b, 2);
    let summed = x.exp().sum(vec![0]).permute(vec![1, 0]);
    let _reader = summed.slice_along(..A, 0).matmul(w);
    let summed_exp = a.exp().sum(vec![0]).permute(vec![1, 0]);
    holds(&cx, &[same(&cx, summed.slice_along(..A, 0), summed_exp)]);
}

/// select/concat_views.egg: a row-indexed table read by a concatenation's
/// stream is the table of that stream's rows, through the table's own views
/// and a conversion between them (a rotary table repeated per pair, widened,
/// broadcast over the heads).
#[test]
fn a_rotary_table_read_by_a_stream_is_the_table_of_its_rows() {
    let mut cx = Graph::new();
    let a = cx.named_tensor("a", vec![1, A, H, D], DType::F32);
    let b = cx.named_tensor("b", vec![1, B, H, D], DType::F32);
    let t = cx.named_tensor("t", vec![S, D / 2], DType::Bf16);
    let table = |t: GraphTensor| {
        t.expand_dim(2, 2)
            .merge_dims(1, 2)
            .cast(DType::F32)
            .expand_dim(1, H)
            .expand_dim(0, 1)
    };
    let x = a.concat_along(b, 1);
    let product = x * table(t);
    let expected = (a * table(t.slice_along(..A, 0))).concat_along(b * table(t.slice_along(A.., 0)), 1);
    holds(&cx, &[same(&cx, product, expected)]);
}
