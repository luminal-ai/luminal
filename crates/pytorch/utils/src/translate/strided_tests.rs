use super::test_support::*;
use crate::{
    pt2_schema::{DimInt, DimSize},
    translate::Translation,
};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;
fn run(t: &Translation, values: &[TypedBuffer]) -> ReferenceRuntime {
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(values)
        .map(|(i, v)| (i.tensor, v.clone()))
        .collect();
    let mut rt = ReferenceRuntime::load(&t.graph).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    for (i, v) in t.inputs.iter().zip(values) {
        rt.set_data(i.tensor, v.clone());
    }
    rt.execute().unwrap();
    rt
}
fn geometry(p: &mut crate::pt2_parser::ParsedPT2, name: &str, strides: &[i64], offset: i64) {
    let m = p
        .program
        .graph_module
        .graph
        .tensor_values
        .get_mut(name)
        .unwrap();
    m.strides = Some(sizes(strides));
    m.storage_offset = Some(DimSize::Int(DimInt { as_int: offset }));
}

#[test]
fn reshape_preserves_logical_order_with_shared_prefixes_and_singletons() {
    for (source, target) in [
        (vec![2, 3, 4], vec![2, 3, 2, 2]),
        (vec![2, 3, 2, 2], vec![2, 3, 4]),
        (vec![2, 3, 4], vec![2, 2, 6]),
        (vec![2, 3], vec![2, 1, 3]),
        (vec![2, 1, 3], vec![2, 3]),
        (vec![2, 1, 1], vec![2]),
        (vec![2], vec![2, 1, 1]),
        (vec![], vec![1, 1]),
        (vec![1, 1], vec![]),
        (vec![2, 0, 4], vec![0, 8]),
    ] {
        let t = translate_nodes(
            vec![node(
                "torch.ops.aten.reshape.default",
                vec![tensor_input("self", "a"), ints("shape", &target)],
                &["y"],
            )],
            &[("a", 5, &source), ("y", 5, &target)],
            &["a"],
            &["y"],
        );
        let values = (0..source.iter().product::<i64>())
            .map(|i| (1 << 54) + i)
            .collect::<Vec<_>>();
        let rt = run(&t, &[TypedBuffer::I64(values.clone())]);
        assert_eq!(
            rt.get_i64(t.outputs[0].tensor).unwrap(),
            &values,
            "{source:?} -> {target:?}"
        );
    }
}
/// A zero-element source reaches every target torch accepts, including a
/// target that is a strict prefix of the source and a zero extent past the
/// first axis, and the view has exactly the target shape: the operand it
/// is added to has that shape, so a mis-shaped view refuses the add.
#[test]
fn view_of_an_empty_tensor_takes_the_target_shape() {
    for (source, target) in [
        (vec![0, 3], vec![0]),
        (vec![0, 3], vec![3, 0]),
        (vec![0, 6], vec![0, 3, 2]),
        (vec![2, 0], vec![0]),
        (vec![0], vec![0, 1, 5]),
    ] {
        let t = translate_nodes(
            vec![
                node(
                    "torch.ops.aten.view.default",
                    vec![tensor_input("self", "a"), ints("size", &target)],
                    &["v"],
                ),
                node(
                    "torch.ops.aten.add.Tensor",
                    vec![tensor_input("self", "v"), tensor_input("other", "b")],
                    &["y"],
                ),
            ],
            &[
                ("a", FLOAT, &source),
                ("b", FLOAT, &target),
                ("v", FLOAT, &target),
                ("y", FLOAT, &target),
            ],
            &["a", "b"],
            &["y"],
        );
        let rt = run(&t, &[TypedBuffer::F32(vec![]), TypedBuffer::F32(vec![])]);
        assert!(
            rt.get_f32(t.outputs[0].tensor).unwrap().is_empty(),
            "{source:?} -> {target:?}"
        );
    }
}
#[test]
fn strided_copy_uses_storage_order_and_allows_repeated_reads() {
    for (shape, stride, offset, want) in [
        (vec![2, 2], vec![1, 3], 1, vec![1, 4, 2, 5]),
        (
            vec![2, 2, 2],
            vec![1, 1, 0],
            0,
            vec![0, 0, 1, 1, 1, 1, 2, 2],
        ),
        (vec![], vec![], 3, vec![3]),
        (vec![0, 2], vec![3, 1], 200, vec![]),
    ] {
        let mut p = parsed_nodes(
            vec![node(
                "torch.ops.aten.as_strided_copy.default",
                vec![
                    tensor_input("self", "a"),
                    ints("size", &shape),
                    ints("stride", &stride),
                    scalar_int("storage_offset", offset),
                ],
                &["y"],
            )],
            &[("a", 5, &[3, 2]), ("y", 5, &shape)],
            &["a"],
            &["y"],
        );
        geometry(&mut p, "a", &[1, 3], 0);
        let t = super::translate(&p).unwrap();
        let rt = run(&t, &[TypedBuffer::I64(vec![0, 3, 1, 4, 2, 5])]);
        assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &want);
    }
}
#[test]
fn strided_scatter_preserves_self_layout_and_exact_wide_values() {
    let mut p = parsed_nodes(
        vec![node(
            "torch.ops.aten.as_strided_scatter.default",
            vec![
                tensor_input("self", "a"),
                tensor_input("src", "b"),
                ints("size", &[2, 2]),
                ints("stride", &[1, 3]),
                scalar_int("storage_offset", 1),
            ],
            &["y"],
        )],
        &[("a", 5, &[3, 2]), ("b", 5, &[2, 2]), ("y", 5, &[3, 2])],
        &["a", "b"],
        &["y"],
    );
    geometry(&mut p, "a", &[1, 3], 0);
    geometry(&mut p, "y", &[1, 3], 0);
    let t = super::translate(&p).unwrap();
    let rt = run(
        &t,
        &[
            TypedBuffer::I64(vec![0, 3, 1, 4, 2, 5]),
            TypedBuffer::I64(vec![i64::MIN, i64::MAX, (1 << 54) + 1, -1]),
        ],
    );
    assert_eq!(
        rt.get_i64(t.outputs[0].tensor).unwrap(),
        &[0, 3, i64::MIN, i64::MAX, (1 << 54) + 1, -1]
    );
}
#[test]
fn strided_scatter_keeps_backing_updates_outside_visible_slice() {
    let mut p = parsed_nodes(
        vec![
            node(
                "torch.ops.aten.slice.Tensor",
                vec![
                    tensor_input("self", "a"),
                    scalar_int("dim", 0),
                    scalar_int("start", 2),
                    scalar_int("end", 4),
                    scalar_int("step", 1),
                ],
                &["v"],
            ),
            node(
                "torch.ops.aten.as_strided_scatter.default",
                vec![
                    tensor_input("self", "v"),
                    tensor_input("src", "b"),
                    ints("size", &[2]),
                    ints("stride", &[1]),
                    scalar_int("storage_offset", 0),
                ],
                &["s"],
            ),
            node(
                "torch.ops.aten.as_strided_copy.default",
                vec![
                    tensor_input("self", "s"),
                    ints("size", &[6]),
                    ints("stride", &[1]),
                    scalar_int("storage_offset", 0),
                ],
                &["y"],
            ),
        ],
        &[
            ("a", 5, &[6]),
            ("b", 5, &[2]),
            ("v", 5, &[2]),
            ("s", 5, &[2]),
            ("y", 5, &[6]),
        ],
        &["a", "b"],
        &["s", "y"],
    );
    geometry(&mut p, "v", &[1], 2);
    geometry(&mut p, "s", &[1], 0);
    let t = super::translate(&p).unwrap();
    let rt = run(
        &t,
        &[
            TypedBuffer::I64(vec![0, 1, 2, 3, 4, 5]),
            TypedBuffer::I64(vec![9, 8]),
        ],
    );
    assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &[2, 3]);
    assert_eq!(
        rt.get_i64(t.outputs[1].tensor).unwrap(),
        &[9, 8, 2, 3, 4, 5]
    );
}
#[test]
fn isfinite_classifies_encodings_including_f64_extrema() {
    let t = translate_one(
        node(
            "torch.ops.aten.isfinite.default",
            vec![tensor_input("self", "a")],
            &["y"],
        ),
        &[("a", 8, &[10]), ("y", BOOL, &[10])],
        &["a"],
        &["y"],
    );
    let rt = run(
        &t,
        &[TypedBuffer::F64(vec![
            0.,
            -0.,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfff0000000000001),
        ])],
    );
    assert_eq!(
        rt.get_bool8(t.outputs[0].tensor).unwrap(),
        &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    );
}
#[test]
fn isfinite_integer_inputs_are_true_without_lossy_conversion() {
    let t = translate_one(
        node(
            "torch.ops.aten.isfinite.default",
            vec![tensor_input("self", "a")],
            &["y"],
        ),
        &[("a", 5, &[4]), ("y", BOOL, &[4])],
        &["a"],
        &["y"],
    );
    let rt = run(&t, &[TypedBuffer::I64(vec![i64::MIN, i64::MAX, 0, -1])]);
    assert_eq!(rt.get_bool8(t.outputs[0].tensor).unwrap(), &[1, 1, 1, 1]);
}
#[test]
fn non_dense_roots_and_inaccessible_storage_are_refused() {
    for (strides, offset) in [(vec![2], 0), (vec![1], 8)] {
        let mut p = parsed_nodes(
            vec![node(
                "torch.ops.aten.as_strided_copy.default",
                vec![
                    tensor_input("self", "a"),
                    ints("size", &[2]),
                    ints("stride", &[1]),
                    scalar_int("storage_offset", 0),
                ],
                &["y"],
            )],
            &[("a", FLOAT, &[4]), ("y", FLOAT, &[2])],
            &["a"],
            &["y"],
        );
        geometry(&mut p, "a", &strides, offset);
        let err = super::translate(&p)
            .err()
            .expect("must not invent inaccessible storage values");
        assert!(format!("{err:#}").contains("backing"));
    }
}

#[test]
fn strided_copy_of_stepped_slice_reads_the_original_backing() {
    let mut p = parsed_nodes(
        vec![
            node(
                "torch.ops.aten.slice.Tensor",
                vec![
                    tensor_input("self", "a"),
                    scalar_int("dim", 0),
                    scalar_int("start", 3),
                    scalar_int("end", 8),
                    scalar_int("step", 2),
                ],
                &["v"],
            ),
            node(
                "torch.ops.aten.as_strided_copy.default",
                vec![
                    tensor_input("self", "v"),
                    ints("size", &[4]),
                    ints("stride", &[1]),
                    scalar_int("storage_offset", 1),
                ],
                &["y"],
            ),
        ],
        &[("a", 5, &[12]), ("v", 5, &[3]), ("y", 5, &[4])],
        &["a"],
        &["y"],
    );
    geometry(&mut p, "v", &[2], 3);
    let t = super::translate(&p).unwrap();
    let rt = run(&t, &[TypedBuffer::I64((0..12).collect())]);
    assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &[1, 2, 3, 4]);
}

#[test]
fn scatter_alias_defaults_use_native_offset_despite_stale_fake_metadata() {
    let mut p = parsed_nodes(
        vec![
            node(
                "torch.ops.aten.slice.Tensor",
                vec![
                    tensor_input("self", "a"),
                    scalar_int("dim", 0),
                    scalar_int("start", 2),
                    scalar_int("end", 5),
                    scalar_int("step", 1),
                ],
                &["v"],
            ),
            node(
                "torch.ops.aten.as_strided_scatter.default",
                vec![
                    tensor_input("self", "v"),
                    tensor_input("src", "b"),
                    ints("size", &[1]),
                    ints("stride", &[1]),
                    scalar_int("storage_offset", 0),
                ],
                &["s"],
            ),
            node(
                "torch.ops.aten.alias.default",
                vec![tensor_input("self", "s")],
                &["alias"],
            ),
            node(
                "torch.ops.aten.as_strided_copy.default",
                vec![
                    tensor_input("self", "alias"),
                    ints("size", &[2]),
                    ints("stride", &[1]),
                ],
                &["y"],
            ),
        ],
        &[
            ("a", 5, &[12]),
            ("b", 5, &[1]),
            ("v", 5, &[3]),
            ("s", 5, &[3]),
            ("alias", 5, &[3]),
            ("y", 5, &[2]),
        ],
        &["a", "b"],
        &["y"],
    );
    geometry(&mut p, "v", &[1], 2);
    geometry(&mut p, "s", &[1], 0);
    geometry(&mut p, "alias", &[1], 0);
    let t = super::translate(&p).unwrap();
    let rt = run(
        &t,
        &[
            TypedBuffer::I64((0..12).collect()),
            TypedBuffer::I64(vec![90]),
        ],
    );
    assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &[2, 3]);
}
