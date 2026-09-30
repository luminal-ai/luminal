#![cfg(target_os = "macos")]
mod support;
use luminal::{dtype::DType, graph::Graph};
use luminal_metal::bindings::MetalBindings;
use luminal_metal::{HostBuffer, MetalRuntime, harness_search_options, metal_registry};
#[cfg(target_os = "macos")]
use support::TestTransfers;

#[test]
fn bool_output_uses_byte_storage_with_odd_lengths() {
    let mut g = Graph::new();
    let x = g.tensor(5, DType::F32);
    let out = x.lt(g.constant_f32(0.).expand_dim(0, 5));
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(x.id, vec![1f32; 5].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    for values in [vec![-2., 0., 1., -1., 4.], vec![2., -2., -1., 0., -4.]] {
        let expected: Vec<u8> = values.iter().map(|v| u8::from(*v < 0.)).collect();
        rt.upload(&mut arena_rt, x.id, values).unwrap();
        rt.execute(arena_rt.buffer()).unwrap();
        assert_eq!(rt.read_bool8(&arena_rt, out.id).unwrap(), expected);
    }
}
#[test]
fn f16_inputs_cast_and_compute_on_device() {
    let mut g = Graph::new();
    let x = g.tensor(5, DType::F16);
    let out = (x * x).cast(DType::F32);
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(
            x.id,
            HostBuffer::new(luminal::dtype::PlanDtype::F16, vec![0; 10]).unwrap(),
        )]
        .into_iter()
        .collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let values: Vec<half::f16> = [-2., -0.5, 0., 1., 3.]
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let bytes = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    rt.upload(
        &mut arena_rt,
        x.id,
        HostBuffer::new(luminal::dtype::PlanDtype::F16, bytes).unwrap(),
    )
    .unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(
        rt.read_f32(&arena_rt, out.id).unwrap(),
        vec![4., 0.25, 0., 1., 9.]
    );
}
#[test]
fn integer_max_and_i64_copy_preserve_extremes() {
    let mut g = Graph::new();
    let x = g.tensor((2, 3), DType::Int);
    let y = g.tensor(3, DType::I64);
    let out = x.max(1);
    // `wide` hands an input straight back out: not a leaf, so bind it by hand.
    let wide = y;
    let mut rt = MetalRuntime::load_with(
        &g,
        MetalBindings::dense(&g.logical, &[out.id, wide.id]),
        metal_registry(),
    )
    .unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(x.id, vec![1i32; 6].into()), (y.id, vec![1i64; 3].into())]
            .into_iter()
            .collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    rt.upload(&mut arena_rt, x.id, vec![i32::MIN, -7, -2, -99, -8, -15])
        .unwrap();
    rt.upload(&mut arena_rt, y.id, vec![i64::MIN, 0, i64::MAX])
        .unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(rt.read_i32(&arena_rt, out.id).unwrap(), vec![-2, -8]);
    assert_eq!(
        rt.read_i64(&arena_rt, wide.id).unwrap(),
        vec![i64::MIN, 0, i64::MAX]
    );
}
#[test]
fn empty_reduction_and_replay_initialize_recycled_storage() {
    let mut g = Graph::new();
    let x = g.tensor((3, 'n'), DType::F32);
    let out = x.sum(1);
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('n'.into(), (0, 4))]).unwrap(),
        &[('n'.into(), 2)].into_iter().collect(),
        &[(x.id, vec![1f32; 6].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    for n in [4, 0, 1, 0, 3] {
        rt.set_dim('n', n);
        rt.upload(&mut arena_rt, x.id, vec![2f32; n * 3]).unwrap();
        rt.execute(arena_rt.buffer()).unwrap();
        assert_eq!(
            rt.read_f32(&arena_rt, out.id).unwrap(),
            vec![2. * n as f32; 3]
        );
    }
}
#[test]
fn application_validates_payloads_and_executor_uses_device_contents() {
    let mut g = Graph::new();
    let x = g.tensor(3, DType::F32);
    let out = x + 1.;
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(x.id, vec![1f32; 3].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    // Payload validation belongs to the test's explicit transfer utility.
    assert!(
        rt.upload(&mut arena_rt, x.id, vec![1i32; 3])
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    assert!(
        rt.upload(&mut arena_rt, x.id, vec![1f32; 2])
            .unwrap_err()
            .to_string()
            .contains("byte")
    );
    rt.upload(&mut arena_rt, x.id, vec![2f32; 3]).unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![3f32; 3]);
    // Changing device contents needs no runtime data API or saved host payload.
    let input = rt.input_arena_range(x.id).unwrap();
    arena_rt
        .write(input.offset, &HostBuffer::from(vec![4f32; 3]).bytes)
        .unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![5f32; 3]);
}
