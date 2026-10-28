//! Integer counts, indices, offsets, and coordinates are exact integers and
//! are never carried in a float: F32 holds integers exactly only to 2^24, so
//! a longer axis saturates a count, rounds an index onto its neighbour, or is
//! refused at the int -> float cast. The cast policy makes float -> int an
//! explicit `TruncCast`, so a lowering whose graph holds no `TruncCast` has
//! no integer that passed through a float and back. Each test pins that
//! structure for one lowering and executes it on the reference runtime (the
//! planner must prove every Int op in width) against hand-computed eager
//! answers.

use luminal::prelude::petgraph::Direction;
use luminal::prelude::*;
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

use super::Translation;
use super::test_support::*;
use crate::pt2_schema::Argument;

const LONG: u32 = 5;

fn count(t: &Translation, op: impl Fn(&LogicalOp) -> bool) -> usize {
    let graph = t.graph.logical.petgraph();
    graph.node_indices().filter(|&i| op(&graph[i].op)).count()
}

fn assert_no_float_to_int(t: &Translation, what: &str) {
    assert_eq!(
        count(t, |op| matches!(op, LogicalOp::TruncCast(_))),
        0,
        "{what}: an integer passes through a float truncation"
    );
}

/// Casts that convert an integer value to a float. A Bool indicator is an
/// exact 0/1 and does not count.
fn int_to_float_casts(t: &Translation) -> usize {
    let graph = t.graph.logical.petgraph();
    graph
        .node_indices()
        .filter(|&id| {
            matches!(
                graph[id].op,
                LogicalOp::Cast(DType::F32 | DType::F16 | DType::Bf16 | DType::F64)
            ) && graph
                .neighbors_directed(id, Direction::Incoming)
                .any(|source| {
                    matches!(
                        graph[source].dtype,
                        DType::Int | DType::I64 | DType::U64 | DType::I8 | DType::U8 | DType::I16
                    )
                })
        })
        .count()
}

fn assert_integers_stay_integers(t: &Translation, what: &str) {
    assert_no_float_to_int(t, what);
    assert_eq!(
        int_to_float_casts(t),
        0,
        "{what}: an integer is converted to a float"
    );
}

fn nodes_of(t: &Translation, op: impl Fn(&LogicalOp) -> bool) -> Vec<(DType, Vec<IntExpr>)> {
    let graph = t.graph.logical.petgraph();
    graph
        .node_indices()
        .filter(|&i| op(&graph[i].op))
        .map(|i| (graph[i].dtype, graph[i].dims.clone()))
        .collect()
}

/// Search and execute with the real input data, binding every declared
/// output (an output may also feed other nodes, so it need not be a leaf).
fn run(t: &Translation, inputs: Vec<TypedBuffer>) -> ReferenceRuntime {
    let mut outputs: Vec<_> = Vec::new();
    for output in &t.outputs {
        if !outputs.contains(&output.tensor) {
            outputs.push(output.tensor);
        }
    }
    let bindings = ReferenceBindings::dense(&t.graph.logical, &outputs);
    let mut runtime = ReferenceRuntime::load_with(&t.graph, bindings).unwrap();
    let dims = t.dims.iter().map(|(&s, &hint)| (s, hint)).collect();
    let bounds = luminal::shape::SymbolBounds::exact(&dims).unwrap();
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(inputs)
        .map(|(input, values)| (input.tensor, values))
        .collect();
    runtime
        .search(&bounds, &dims, &data, &harness_search_options())
        .unwrap();
    for (input, values) in data {
        runtime.set_data(input, values);
    }
    runtime.execute().unwrap();
    runtime
}

fn string_arg(name: &str, value: &str) -> crate::pt2_schema::NodeInput {
    input(name, Argument::Other(serde_json::json!(value)))
}

fn tensors_arg(name: &str, names: &[&str]) -> crate::pt2_schema::NodeInput {
    let names: Vec<_> = names
        .iter()
        .map(|name| serde_json::json!({ "name": name }))
        .collect();
    input(
        name,
        serde_json::from_value(serde_json::json!({ "as_tensors": names })).unwrap(),
    )
}

/// `nonzero_static` and `tril_indices`/`triu_indices` share the true
/// coordinate lowering: the slot validity reads the sorted truth, and each
/// coordinate gathers an axis iota, so no count or flat div/mod is formed.
#[test]
fn true_coordinates_are_exact_integers() {
    let t = translate_one(
        node(
            "torch.ops.aten.nonzero_static.default",
            vec![
                tensor_input("input", "x"),
                scalar_int("size", 7),
                scalar_int("fill_value", -2),
            ],
            &["y"],
        ),
        &[("x", FLOAT, &[2, 3]), ("y", LONG, &[7, 2])],
        &["x"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "nonzero_static");
    let runtime = run(
        &t,
        vec![TypedBuffer::F32(vec![0.0, 1.5, 0.0, -0.0, -4.0, 2.0])],
    );
    assert_eq!(
        runtime.get_i64(t.outputs[0].tensor).unwrap(),
        &[0, 1, 1, 1, 1, 2, -2, -2, -2, -2, -2, -2, -2, -2]
    );

    for (target, rows, columns, offset, expected) in [
        (
            "torch.ops.aten.tril_indices.default",
            4,
            5,
            1,
            vec![
                vec![0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4],
            ],
        ),
        (
            "torch.ops.aten.triu_indices.default",
            5,
            3,
            -1,
            vec![
                vec![0, 0, 0, 1, 1, 1, 2, 2, 3],
                vec![0, 1, 2, 0, 1, 2, 1, 2, 2],
            ],
        ),
    ] {
        let width = expected[0].len() as i64;
        let t = translate_one(
            node(
                target,
                vec![
                    scalar_int("row", rows),
                    scalar_int("col", columns),
                    scalar_int("offset", offset),
                ],
                &["y"],
            ),
            &[("y", LONG, &[2, width])],
            &[],
            &["y"],
        );
        assert_integers_stay_integers(&t, target);
        let runtime = run(&t, Vec::new());
        assert_eq!(
            runtime.get_i64(t.outputs[0].tensor).unwrap(),
            &expected.concat(),
            "{target}"
        );
    }
}

/// An accumulating `index_put` addresses each destination with one exact
/// Int coordinate per data axis: every gather and scatter on the data reads
/// the rank-2 data with two coordinates, never one flat index.
#[test]
fn accumulated_destinations_use_per_axis_coordinates() {
    let t = translate_one(
        node(
            "torch.ops.aten.index_put.default",
            vec![
                tensor_input("self", "x"),
                tensors_arg("indices", &["r", "c"]),
                tensor_input("values", "v"),
                scalar_bool("accumulate", true),
            ],
            &["y"],
        ),
        &[
            ("x", FLOAT, &[3, 4]),
            ("r", LONG, &[4]),
            ("c", LONG, &[4]),
            ("v", FLOAT, &[4]),
            ("y", FLOAT, &[3, 4]),
        ],
        &["x", "r", "c", "v"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "index_put accumulate");
    // One read and one write of the data per update: a gather takes the
    // data and two coordinates, a scatter the data, two coordinates, and
    // the source.
    let graph = t.graph.logical.petgraph();
    let mut steps = 0;
    for id in graph.node_indices() {
        let arity = match graph[id].op {
            LogicalOp::Gather => 3,
            LogicalOp::Scatter => 4,
            _ => continue,
        };
        if graph[id].dtype != DType::F32 {
            continue;
        }
        let operands = graph.edges_directed(id, Direction::Incoming).count();
        assert_eq!(operands, arity, "{:?} addresses the data", graph[id].op);
        steps += 1;
    }
    assert_eq!(steps, 8, "four updates, each one gather and one scatter");
    // (2, 1) twice, (-1 -> 2, 3), and (0, -4 -> 0).
    let runtime = run(
        &t,
        vec![
            TypedBuffer::F32((0..12).map(|v| v as f32).collect()),
            TypedBuffer::I64(vec![2, -1, 0, 2]),
            TypedBuffer::I64(vec![1, 3, -4, 1]),
            TypedBuffer::F32(vec![0.5, -1.5, 2.0, 4.0]),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 13.5, 10.0, 9.5]
    );
}

/// A scatter along a middle axis permutes the structural axes to the front
/// of a view and merges them into one plane axis: each step addresses the
/// data with the plane coordinate and the caller index only.
#[test]
fn scatter_reduce_addresses_a_plane_view() {
    let t = translate_one(
        node(
            "torch.ops.aten.scatter_add.default",
            vec![
                tensor_input("self", "x"),
                scalar_int("dim", 1),
                tensor_input("index", "i"),
                tensor_input("src", "s"),
            ],
            &["y"],
        ),
        &[
            ("x", FLOAT, &[2, 3, 4]),
            ("i", LONG, &[2, 2, 4]),
            ("s", FLOAT, &[2, 2, 4]),
            ("y", FLOAT, &[2, 3, 4]),
        ],
        &["x", "i", "s"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "scatter_add");
    let graph = t.graph.logical.petgraph();
    let plane_dims = vec![IntExpr::from(8usize), IntExpr::from(3usize)];
    let mut steps = 0;
    for id in graph.node_indices() {
        let arity = match graph[id].op {
            LogicalOp::Gather => 3,
            LogicalOp::Scatter => 4,
            _ => continue,
        };
        let operands = graph.edges_directed(id, Direction::Incoming).count();
        assert_eq!(
            operands, arity,
            "{:?} addresses the plane view",
            graph[id].op
        );
        if graph[id].op == LogicalOp::Scatter {
            assert_eq!(
                graph[id].dims, plane_dims,
                "the scatter writes the plane view"
            );
        }
        steps += 1;
    }
    assert_eq!(
        steps, 32,
        "sixteen updates, each one gather and one scatter"
    );
    let runtime = run(
        &t,
        vec![
            TypedBuffer::F32((0..24).map(|v| v as f32).collect()),
            TypedBuffer::I64(vec![2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2]),
            TypedBuffer::F32((1..=16).map(|v| v as f32 * 0.5).collect()),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[
            0.0, 2.0, 5.5, 3.0, 4.0, 5.0, 7.5, 13.0, 11.0, 12.0, 10.0, 11.0, 18.5, 13.0, 14.0,
            21.0, 20.5, 22.0, 23.5, 19.0, 20.0, 28.0, 29.5, 31.0
        ]
    );
}

/// `put` wraps its caller-data flat indices with bounded Int remainders.
#[test]
fn put_wraps_flat_indices_in_exact_int() {
    let t = translate_one(
        node(
            "torch.ops.aten.put.default",
            vec![
                tensor_input("self", "x"),
                tensor_input("index", "i"),
                tensor_input("source", "s"),
                scalar_bool("accumulate", false),
            ],
            &["y"],
        ),
        &[
            ("x", FLOAT, &[3, 4]),
            ("i", LONG, &[4]),
            ("s", FLOAT, &[4]),
            ("y", FLOAT, &[3, 4]),
        ],
        &["x", "i", "s"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "put");
    assert!(count(&t, |op| matches!(op, LogicalOp::TruncRem)) > 0);
    let runtime = run(
        &t,
        vec![
            TypedBuffer::F32(vec![0.0; 12]),
            TypedBuffer::I64(vec![-1, 0, 6, -7]),
            TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
}

/// A true slot's source rank is the exclusive prefix count of the mask: an
/// Int scan, shifted, never an F32 cumsum.
#[test]
fn masked_scatter_rank_is_an_int_scan() {
    let t = translate_one(
        node(
            "torch.ops.aten.masked_scatter.default",
            vec![
                tensor_input("self", "x"),
                tensor_input("mask", "m"),
                tensor_input("source", "s"),
            ],
            &["y"],
        ),
        &[
            ("x", FLOAT, &[2, 3]),
            ("m", BOOL, &[2, 3]),
            ("s", FLOAT, &[4]),
            ("y", FLOAT, &[2, 3]),
        ],
        &["x", "m", "s"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "masked_scatter");
    let scans = nodes_of(&t, |op| {
        matches!(op, LogicalOp::UnspecifiedOrderScanSum { .. })
    });
    assert_eq!(scans.len(), 1);
    assert_eq!(scans[0].0, DType::Int, "the prefix count is an Int scan");
    let runtime = run(
        &t,
        vec![
            TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TypedBuffer::Bool8(vec![0, 1, 1, 0, 0, 1]),
            TypedBuffer::F32(vec![10.0, 20.0, 30.0, 40.0]),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[1.0, 10.0, 20.0, 4.0, 5.0, 30.0]
    );
}

/// `repeat_interleave.Tensor` compares Int interval ends with an Int iota:
/// no integer is ever converted to a float.
#[test]
fn repeat_interleave_intervals_are_int() {
    let t = translate_one(
        node(
            "torch.ops.aten.repeat_interleave.Tensor",
            vec![tensor_input("repeats", "r"), scalar_int("output_size", 7)],
            &["y"],
        ),
        &[("r", LONG, &[4]), ("y", LONG, &[7])],
        &["r"],
        &["y"],
    );
    assert_integers_stay_integers(&t, "repeat_interleave");
    let runtime = run(&t, vec![TypedBuffer::I64(vec![2, 0, 3, 2])]);
    assert_eq!(
        runtime.get_i64(t.outputs[0].tensor).unwrap(),
        &[0, 0, 2, 2, 2, 3, 3]
    );
}

fn segment(reduce: &str, offsets: bool) -> Translation {
    let bounds = if offsets { "offsets" } else { "lengths" };
    let extent = if offsets { 4 } else { 3 };
    translate_one(
        node(
            "torch.ops.aten.segment_reduce.default",
            vec![
                tensor_input("data", "x"),
                string_arg("reduce", reduce),
                tensor_input(bounds, "b"),
            ],
            &["y"],
        ),
        &[
            ("x", FLOAT, &[6]),
            ("b", LONG, &[extent]),
            ("y", FLOAT, &[3]),
        ],
        &["x", "b"],
        &["y"],
    )
}

/// Segment bounds are an Int scan (starts: the same scan shifted), the
/// positions an Int iota, the sizes Int indicator counts, and the product's
/// sign the Int parity of the negative-factor count.
#[test]
fn segment_reduce_bookkeeping_is_int() {
    let data = vec![-1.0, 2.0, -3.0, 4.0, 5.0, -6.0];
    // The only integer -> float conversions left are the mean's divisor and
    // the product's sign, each an exact small integer at the very end.
    for (reduce, offsets, bounds, expected, converted) in [
        ("mean", false, vec![2, 1, 3], vec![0.5, -3.0, 1.0], 1),
        ("mean", true, vec![0, 2, 3, 6], vec![0.5, -3.0, 1.0], 1),
        ("prod", false, vec![3, 0, 3], vec![6.0, 1.0, -120.0], 1),
        ("sum", false, vec![1, 4, 1], vec![-1.0, 8.0, -6.0], 0),
    ] {
        let t = segment(reduce, offsets);
        assert_no_float_to_int(&t, reduce);
        assert_eq!(
            int_to_float_casts(&t),
            converted,
            "{reduce} offsets={offsets}"
        );
        assert_eq!(
            count(&t, |op| matches!(op, LogicalOp::Floor)),
            0,
            "{reduce}: the sign parity is a float floor"
        );
        let runtime = run(
            &t,
            vec![TypedBuffer::F32(data.clone()), TypedBuffer::I64(bounds)],
        );
        assert_eq!(
            runtime.get_f32(t.outputs[0].tensor).unwrap(),
            &expected,
            "{reduce} offsets={offsets}"
        );
    }
}

/// Adaptive-pool windows are structural `IntExpr` iotas (no float bound, no
/// float ceiling, no truncation), and the window size is an Int count that
/// becomes a float only as the mean's divisor.
#[test]
fn adaptive_pool_windows_are_structural() {
    let t = translate_one(
        node(
            "torch.ops.aten._adaptive_avg_pool2d.default",
            vec![tensor_input("self", "x"), ints("output_size", &[1, 3])],
            &["y"],
        ),
        &[("x", FLOAT, &[1, 1, 1, 5]), ("y", FLOAT, &[1, 1, 1, 3])],
        &["x"],
        &["y"],
    );
    assert_no_float_to_int(&t, "adaptive_avg_pool2d");
    assert_eq!(count(&t, |op| matches!(op, LogicalOp::Ceil)), 0);
    assert_eq!(int_to_float_casts(&t), 1, "only the divisor converts");
    assert!(
        nodes_of(&t, |op| matches!(op, LogicalOp::ReduceSum { .. }))
            .iter()
            .any(|(dtype, _)| *dtype == DType::Int),
        "the window size is an Int count"
    );
    // Windows [0, 2), [1, 4), [3, 5) of 0..5.
    let runtime = run(&t, vec![TypedBuffer::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0])]);
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[0.5, 2.0, 3.5]
    );
}

/// The nanmedian slot `(count - 1) // 2` is exact Int arithmetic on an Int
/// count of non-NaN values.
#[test]
fn nanmedian_slot_is_exact_int() {
    let t = translate_one(
        node(
            "torch.ops.aten.nanmedian.dim",
            vec![
                tensor_input("self", "x"),
                scalar_int("dim", 1),
                scalar_bool("keepdim", false),
            ],
            &["values", "indices"],
        ),
        &[
            ("x", FLOAT, &[3, 4]),
            ("values", FLOAT, &[3]),
            ("indices", LONG, &[3]),
        ],
        &["x"],
        &["values", "indices"],
    );
    assert_integers_stay_integers(&t, "nanmedian");
    let nan = f32::NAN;
    let runtime = run(
        &t,
        vec![TypedBuffer::F32(vec![
            3.0, 1.0, 2.0, nan, 4.0, 1.0, 3.0, 2.0, nan, nan, nan, nan,
        ])],
    );
    let values = runtime.get_f32(t.outputs[0].tensor).unwrap();
    assert_eq!(&values[..2], &[2.0, 2.0]);
    assert!(values[2].is_nan());
    assert_eq!(runtime.get_i64(t.outputs[1].tensor).unwrap(), &[2, 3, 0]);
}

fn embedding_bag(mode: i64) -> Translation {
    translate_one(
        node(
            "torch.ops.aten._embedding_bag_forward_only.default",
            vec![
                tensor_input("weight", "w"),
                tensor_input("indices", "i"),
                tensor_input("offsets", "o"),
                scalar_bool("scale_grad_by_freq", false),
                scalar_int("mode", mode),
                scalar_int("padding_idx", 2),
            ],
            &["out", "offset2bag", "bag_size", "max_indices"],
        ),
        &[
            ("w", FLOAT, &[4, 2]),
            ("i", LONG, &[5]),
            ("o", LONG, &[3]),
            ("out", FLOAT, &[3, 2]),
            ("offset2bag", LONG, &[5]),
            ("bag_size", LONG, &[3]),
            ("max_indices", LONG, if mode == 2 { &[3, 2] } else { &[3] }),
        ],
        &["w", "i", "o"],
        &["out", "offset2bag", "bag_size", "max_indices"],
    )
}

/// Bag sizes are I64 indicator sums and offset-to-bag ids an I64 sum of
/// natively selected bag iotas.
#[test]
fn embedding_bag_counts_are_i64_sums() {
    // Bags [1, 2], [], [0, 3, 2] of rows [[0, 1], [2, 3], [4, 5], [6, 7]];
    // index 2 is padding.
    let inputs = || {
        vec![
            TypedBuffer::F32(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            TypedBuffer::I64(vec![1, 2, 0, 3, 2]),
            TypedBuffer::I64(vec![0, 2, 2]),
        ]
    };
    for (mode, out) in [
        (1, [2.0, 3.0, 0.0, 0.0, 3.0, 4.0]),
        (2, [2.0, 3.0, 0.0, 0.0, 6.0, 7.0]),
    ] {
        let t = embedding_bag(mode);
        assert_no_float_to_int(&t, "embedding_bag");
        assert!(
            nodes_of(&t, |op| matches!(op, LogicalOp::ReduceSum { .. }))
                .iter()
                .any(|(dtype, _)| *dtype == DType::I64),
            "bag sizes are an I64 sum"
        );
        let runtime = run(&t, inputs());
        assert_eq!(runtime.get_f32(t.outputs[0].tensor).unwrap(), &out);
        assert_eq!(
            runtime.get_i64(t.outputs[1].tensor).unwrap(),
            &[0, 0, 2, 2, 2]
        );
        assert_eq!(runtime.get_i64(t.outputs[2].tensor).unwrap(), &[1, 0, 2]);
        let max_indices: &[i64] = if mode == 2 {
            &[1, 1, 0, 0, 3, 3]
        } else {
            &[1, 0, 2]
        };
        assert_eq!(runtime.get_i64(t.outputs[3].tensor).unwrap(), max_indices);
    }
}
