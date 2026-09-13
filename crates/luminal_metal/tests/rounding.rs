#![cfg(target_os = "macos")]
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options};

/// floor / ceil / trunc / round (half-to-even) execute on device with the
/// same semantics as the reference runtime: NaN/±inf propagate for the
/// dtype-preserving ops.
#[test]
fn rounding_ops_execute_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(9, DType::F32);
    let floor = x.floor().output();
    let ceil = x.ceil().output();
    let trunc = x.trunc().output();
    let round = x.round().output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();

    let values: Vec<f32> = vec![1.9, 1.5, 0.5, -0.5, -1.5, -1.9, 2.5, 3.5, -2.5];
    rt.set_data(x.id, values.clone());
    rt.execute().unwrap();

    assert_eq!(
        rt.get_f32(floor.id).unwrap(),
        values.iter().map(|v| v.floor()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.get_f32(ceil.id).unwrap(),
        values.iter().map(|v| v.ceil()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.get_f32(trunc.id).unwrap(),
        values.iter().map(|v| v.trunc()).collect::<Vec<_>>()
    );
    assert_eq!(
        rt.get_f32(round.id).unwrap(),
        values
            .iter()
            .map(|v| v.round_ties_even())
            .collect::<Vec<_>>()
    );
}

/// `trunc_cast` is the explicit lossy float -> int conversion: truncate
/// toward zero, in the integer target dtype.
#[test]
fn trunc_cast_executes_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(6, DType::F32);
    let out = x.trunc_cast(DType::Int).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();

    rt.set_data(x.id, vec![1.9, 1.5, 0.5, -0.5, -1.5, -1.9]);
    rt.execute().unwrap();
    assert_eq!(rt.get_i32(out.id).unwrap(), vec![1, 1, 0, 0, -1, -1]);
}
