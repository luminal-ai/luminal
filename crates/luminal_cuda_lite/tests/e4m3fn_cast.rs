//! Byte-level CUDA parity for F32 narrowing to E4M3FN.
#![cfg(feature = "device")]

mod support;

use luminal::{dtype::PlanDtype, prelude::*};
use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer, harness_search_options};
use support::TestTransfers;

#[test]
fn f32_to_e4m3fn_matches_reference_at_every_midpoint() {
    // Codes 0..=126 are the nonnegative finite values in increasing order.
    let levels: Vec<_> = (0..=126)
        .map(|code| float8::F8E4M3::from_bits(code).to_f32())
        .collect();
    let mut positive = vec![
        0.0,
        f32::from_bits(1),
        449.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7fc0_1234), // quiet NaN, noncanonical payload
        f32::from_bits(0x7f80_0001), // signaling NaN
    ];
    positive.extend_from_slice(&levels);
    for pair in levels.windows(2) {
        let midpoint = (pair[0] + pair[1]) * 0.5;
        let bits = midpoint.to_bits();
        positive.extend([f32::from_bits(bits - 1), midpoint, f32::from_bits(bits + 1)]);
    }
    let values: Vec<_> = positive.iter().flat_map(|&v| [v, -v]).collect();
    assert_cast_parity(&values, false);
    assert_cast_parity(&values, true);
}

fn assert_cast_parity(values: &[f32], transpose: bool) {
    let mut graph = Graph::new();
    let x = graph.tensor([values.len() / 2, 2], DType::F32);
    let source = if transpose { x.permute(vec![1, 0]) } else { x };
    let out = source.cast(DType::F8E4M3FN);
    use luminal_reference::{ReferenceBindings, ReferenceRuntime};
    let mut reference =
        ReferenceRuntime::load_with(&graph, ReferenceBindings::dense(&graph.logical, &[out.id]))
            .unwrap();
    reference
        .compile(&Default::default(), &Default::default())
        .unwrap();
    reference.set_data(x.id, values.to_vec());
    reference.execute().unwrap();
    let expected = reference.get_f8e4m3(out.id).unwrap();
    assert_eq!(expected.len(), values.len());

    let mut runtime = CudaRuntime::load_with(
        &graph,
        CudaBindings::dense(&graph.logical, &[out.id]),
        luminal_cuda_lite::cuda_registry(),
    )
    .unwrap();
    let input = HostBuffer::from(values.to_vec());
    let samples = [(x.id, input.clone())].into_iter().collect();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &samples,
            &harness_search_options(),
        )
        .unwrap();
    let mut arena = support::allocate_arena(&runtime).unwrap();
    let mut staging = runtime.allocate_staging().unwrap();
    runtime.upload(&mut arena, x.id, input).unwrap();
    runtime.execute(arena.arena(), &mut staging).unwrap();
    let (data, _) = runtime.download(&arena, out.id).unwrap();
    assert_eq!(data.dtype, PlanDtype::F8E4M3FN);
    assert_eq!(data.bytes.len(), values.len());
    for (i, (&got, want)) in data.bytes.iter().zip(expected).enumerate() {
        let input_index = if transpose {
            (i % (values.len() / 2)) * 2 + i / (values.len() / 2)
        } else {
            i
        };
        assert_eq!(
            got,
            want.to_bits(),
            "input bits {:08x}, transpose={transpose}",
            values[input_index].to_bits()
        );
    }
}
