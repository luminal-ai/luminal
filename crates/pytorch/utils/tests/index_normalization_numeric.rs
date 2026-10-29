//! End-to-end negative indexing: `table[idx]` and `index_put(table, [idx], v)`
//! with indices in `[-dim, dim)`, translated, planned and executed on the
//! reference runtime WITHOUT any attested value range on the index. The
//! translator's normalization `((idx rem dim) + dim) rem dim` discharges its
//! own proof obligations (a truncated remainder is bounded by its divisor),
//! so no `bind_value_range` is needed, and the arithmetic is exact over the
//! whole int32 range rather than below 2^24.
//!
//! The fixture lives in `<workspace>/target/luminal_fixtures`; the test skips
//! when it is absent.
use luminal_pytorch_utils::{InputKind, parse_pt2, translate};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

#[test]
fn negative_indices_wrap_without_an_attested_range() {
    let path = format!(
        "{}/../../../target/luminal_fixtures/negative_indexing.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    let translation = translate(&parsed).expect("translate negative indexing");
    let user_inputs: Vec<_> = translation
        .inputs
        .iter()
        .filter(|input| matches!(input.kind, InputKind::UserInput { .. }))
        .collect();
    assert_eq!(user_inputs.len(), 3, "table, idx, values");

    let table: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let idx: Vec<i64> = vec![-1, -6, 2, -3];
    let values: Vec<f32> = vec![-1.0; 16];

    let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(user_inputs[0].tensor, TypedBuffer::F32(table.clone()));
    data.insert(user_inputs[1].tensor, TypedBuffer::I64(idx.clone()));
    data.insert(user_inputs[2].tensor, TypedBuffer::F32(values.clone()));
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .expect("negative indexing plans with no attested index range");

    runtime.set_data(user_inputs[0].tensor, TypedBuffer::F32(table.clone()));
    runtime.set_data(user_inputs[1].tensor, TypedBuffer::I64(idx.clone()));
    runtime.set_data(user_inputs[2].tensor, TypedBuffer::F32(values));
    runtime.execute().expect("negative indexing executes");

    let rows = runtime
        .get_f32(translation.outputs[0].tensor)
        .expect("rows");
    let expected_rows: Vec<f32> = idx
        .iter()
        .map(|i| i.rem_euclid(6) as usize)
        .flat_map(|row| (0..4).map(move |c| (row * 4 + c) as f32))
        .collect();
    assert_eq!(rows, &expected_rows, "table[idx] with negative idx");

    let written = runtime
        .get_f32(translation.outputs[1].tensor)
        .expect("written");
    let mut expected_written = table;
    for i in &idx {
        let row = i.rem_euclid(6) as usize;
        for c in 0..4 {
            expected_written[row * 4 + c] = -1.0;
        }
    }
    assert_eq!(written, &expected_written, "index_put at negative idx");
}
