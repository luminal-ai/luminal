//! UInt64 is an unsigned, wrapping ring at every arithmetic boundary.
use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

#[test]
fn unsigned_arithmetic_and_casts_retain_all_bits() {
    let mut g = Graph::new();
    let a = g.tensor(vec![5], DType::U64);
    let b = g.tensor(vec![5], DType::U64);
    let outputs = [a + b, a * b, a.cast(DType::I64).cast(DType::U64)];
    let aa = vec![u64::MAX, 1 << 63, (1 << 63) + 1, 0, (1 << 32) + 1];
    let bb = vec![1, 2, u64::MAX, 1, u64::MAX];
    let mut rt = ReferenceRuntime::load(&g).unwrap();
    let data = FxHashMap::from_iter([
        (a.id, TypedBuffer::U64(aa.clone())),
        (b.id, TypedBuffer::U64(bb.clone())),
    ]);
    rt.search(
        &SymbolBounds::default(),
        &FxHashMap::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    rt.set_data(a.id, TypedBuffer::U64(aa.clone()));
    rt.set_data(b.id, TypedBuffer::U64(bb.clone()));
    rt.execute().unwrap();
    assert_eq!(
        *rt.get_u64(outputs[0].id).unwrap(),
        aa.iter()
            .zip(&bb)
            .map(|(a, b)| a.wrapping_add(*b))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        *rt.get_u64(outputs[1].id).unwrap(),
        aa.iter()
            .zip(&bb)
            .map(|(a, b)| a.wrapping_mul(*b))
            .collect::<Vec<_>>()
    );
    assert_eq!(*rt.get_u64(outputs[2].id).unwrap(), aa);
}

#[test]
fn binary64_division_preserves_subnormal_and_large_quotients() {
    let mut g = Graph::new();
    let a = g.tensor(vec![3], DType::F64);
    let b = g.tensor(vec![3], DType::F64);
    let out = a / b;
    let aa = vec![f64::from_bits(1), 1e300, 1e-300];
    let bb = aa.clone();
    let mut rt = ReferenceRuntime::load(&g).unwrap();
    let data = FxHashMap::from_iter([
        (a.id, TypedBuffer::F64(aa.clone())),
        (b.id, TypedBuffer::F64(bb.clone())),
    ]);
    rt.search(
        &SymbolBounds::default(),
        &FxHashMap::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    rt.set_data(a.id, TypedBuffer::F64(aa));
    rt.set_data(b.id, TypedBuffer::F64(bb));
    rt.execute().unwrap();
    assert_eq!(*rt.get_f64(out.id).unwrap(), vec![1.; 3]);
}
