//! An Int64 -> Int cast keeps its source bounds only when they fit in Int.
use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

/// `n / (x + offset)`, summed in Int64 and narrowed to Int.
fn divide_by_narrowed_sum(offset: i64) -> (Graph, GraphTensor, GraphTensor, GraphTensor) {
    let mut g = Graph::new();
    let x = g.tensor(vec![2], DType::Int);
    let n = g.tensor(vec![2], DType::Int);
    let wide = x.cast(DType::I64) + g.constant_i64(offset).expand_rhs(vec![2]);
    let q = n.trunc_div(wide.cast(DType::Int));
    (g, x, n, q)
}

#[test]
fn a_divisor_that_may_not_fit_after_narrowing_is_not_selected() {
    // x + 2^31 + 1 spans [1, 2^32] and 2^32 stores as 0, so trunc_div must
    // not take it as a positive divisor even though the profile values fit.
    let (g, x, n, _) = divide_by_narrowed_sum((1 << 31) + 1);
    let mut rt = ReferenceRuntime::load(&g).unwrap();
    let data = FxHashMap::from_iter([
        (x.id, TypedBuffer::I32(vec![-5, -10])),
        (n.id, TypedBuffer::I32(vec![100, 100])),
    ]);
    let err = rt
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap_err();
    assert!(err.to_string().contains("no candidate genome"), "{err}");
}

#[test]
fn a_divisor_that_fits_after_narrowing_is_still_selected() {
    // With x in [1, 100] the divisor x + 2 stays in [3, 102].
    let (g, x, n, q) = divide_by_narrowed_sum(2);
    let mut rt = ReferenceRuntime::load(&g).unwrap();
    rt.bind_value_range(x.id, 1, 100).unwrap();
    let data = FxHashMap::from_iter([
        (x.id, TypedBuffer::I32(vec![1, 7])),
        (n.id, TypedBuffer::I32(vec![100, 100])),
    ]);
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    for (id, values) in &data {
        rt.set_data(*id, values.clone());
    }
    rt.execute().unwrap();
    assert_eq!(rt.get_i32(q.id).unwrap(), &vec![33, 11]);
}
