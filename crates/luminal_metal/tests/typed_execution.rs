#![cfg(target_os = "macos")]
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::{HostBuffer, MetalRuntime, harness_search_options};

#[test]
fn bool_output_uses_byte_storage_with_odd_lengths() {
    let mut g = Graph::new();
    let x = g.tensor(5, DType::F32);
    let out = x.lt(g.constant_float(0.).expand_dim(0, 5)).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for values in [vec![-2., 0., 1., -1., 4.], vec![2., -2., -1., 0., -4.]] {
        let expected: Vec<u8> = values.iter().map(|v| u8::from(*v < 0.)).collect();
        rt.set_data(x.id, values);
        rt.execute().unwrap();
        assert_eq!(rt.get_bool8(out.id).unwrap(), expected);
    }
}
#[test]
fn f16_inputs_cast_and_compute_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(5, DType::F16);
    let out = (x * x).cast(DType::F32).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    let values: Vec<half::f16> = [-2., -0.5, 0., 1., 3.]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let bytes = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    rt.set_data(
        x.id,
        HostBuffer::new(luminal::dtype::PlanDtype::F16, bytes).unwrap(),
    );
    rt.execute().unwrap();
    assert_eq!(rt.get_f32(out.id).unwrap(), vec![4., 0.25, 0., 1., 9.]);
}
#[test]
fn integer_max_and_i64_copy_preserve_extremes() {
    let mut g = Graph::new();
    let x = g.tensor((2, 3), DType::Int);
    let y = g.tensor(3, DType::I64);
    let out = x.max(1).output();
    let wide = y.output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    rt.set_data(x.id, vec![i32::MIN, -7, -2, -99, -8, -15]);
    rt.set_data(y.id, vec![i64::MIN, 0, i64::MAX]);
    rt.execute().unwrap();
    assert_eq!(rt.get_i32(out.id).unwrap(), vec![-2, -8]);
    assert_eq!(rt.get_i64(wide.id).unwrap(), vec![i64::MIN, 0, i64::MAX]);
}
#[test]
fn empty_reduction_and_replay_initialize_recycled_storage() {
    let mut g = Graph::new();
    let x = g.tensor((3, 'n'), DType::F32);
    let out = x.sum(1).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.bind_dyn_range('n', 0, 4).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for n in [4, 0, 1, 0, 3] {
        rt.set_dim('n', n);
        rt.set_data(x.id, vec![2f32; n * 3]);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![2. * n as f32; 3]);
    }
}
#[test]
fn missing_or_mistyped_input_refuses_then_recovers() {
    let mut g = Graph::new();
    let x = g.tensor(3, DType::F32);
    let out = (x + 1.).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    assert!(
        rt.execute()
            .unwrap_err()
            .to_string()
            .contains("missing input")
    );
    rt.set_data(x.id, vec![1i32; 3]);
    assert!(
        rt.execute()
            .unwrap_err()
            .to_string()
            .contains("dtype mismatch")
    );
    rt.set_data(x.id, vec![1f32; 2]);
    assert!(rt.execute().unwrap_err().to_string().contains("bytes"));
    rt.set_data(x.id, vec![2f32; 3]);
    rt.execute().unwrap();
    assert_eq!(rt.get_f32(out.id).unwrap(), vec![3f32; 3]);
}
