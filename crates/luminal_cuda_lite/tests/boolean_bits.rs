#![cfg(feature = "device")]
//! End-to-end CUDA coverage for exact encoding conversion and Boolean algebra.

mod support;

use luminal::bufferize::BufferNode;
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, HostBuffer, harness_search_options};
use rustc_hash::FxHashMap;
use support::TestTransfers;

#[test]
fn boolean_algebra_and_encoding_round_trips_preserve_bits() {
    let mut graph = Graph::new();
    let bytes = graph.tensor(vec![2, 2], DType::U8);
    let bits = bytes.bits_to_bools();
    let inverse = bits.logical_not().bools_to_bits(DType::U8);
    let zeros = bits
        .logical_and(bits.logical_not())
        .bools_to_bits(DType::U8);
    let ones = bits.logical_or(bits.logical_not()).bools_to_bits(DType::U8);
    let interleaved = bits
        .permute(vec![0, 2, 1])
        .merge_dims(1, 2)
        .bools_to_bits(DType::I16);
    let transposed_roundtrip = bytes
        .permute(vec![1, 0])
        .bits_to_bools()
        .bools_to_bits(DType::U8);

    let floats = graph.tensor(vec![6], DType::F32);
    let float_roundtrip = floats.bits_to_bools().bools_to_bits(DType::F32);
    let byte_values = vec![0x00, 0x81, 0xa5, 0xff];
    let float_codes = [0, 0x8000_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_1234, 1];
    let float_values = float_codes.map(f32::from_bits).to_vec();
    let byte_data = HostBuffer::new(PlanDtype::U8, byte_values.clone()).unwrap();
    let float_data = HostBuffer::from(float_values);
    let inputs: FxHashMap<_, _> = [
        (bytes.id, byte_data.clone()),
        (floats.id, float_data.clone()),
    ]
    .into_iter()
    .collect();

    let mut runtime = CudaRuntime::load(&graph).expect("load");
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .expect("search Boolean bit plan");

    let labels = runtime
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => Some(op.label()),
            _ => None,
        })
        .collect::<Vec<_>>();
    for expected in [
        "BitsToBoolsGeneric",
        "BoolsToBitsGeneric",
        "AndGeneric",
        "OrGeneric",
        "NotGeneric",
    ] {
        assert!(labels.contains(&expected), "missing {expected}: {labels:?}");
    }
    assert!(
        labels
            .iter()
            .all(|label| !label.contains("BitsToBoolsView") && !label.contains("BoolsToBitsView")),
        "CUDA Lite byte-per-Bool storage must not elect packed bit views: {labels:?}"
    );

    let mut arena = support::allocate_arena(&runtime).expect("allocate execution arena");
    let mut staging = runtime.allocate_staging().expect("allocate staging arena");
    runtime.upload(&mut arena, bytes.id, byte_data).unwrap();
    runtime.upload(&mut arena, floats.id, float_data).unwrap();
    runtime
        .execute(arena.arena(), &mut staging)
        .expect("execute Boolean bit plan");

    assert_eq!(
        runtime.download(&arena, inverse.id).unwrap().0.bytes,
        byte_values.iter().map(|v| !v).collect::<Vec<_>>()
    );
    assert_eq!(
        runtime.download(&arena, zeros.id).unwrap().0.bytes,
        vec![0; byte_values.len()]
    );
    assert_eq!(
        runtime.download(&arena, ones.id).unwrap().0.bytes,
        vec![0xff; byte_values.len()]
    );
    let expected_interleaved = byte_values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let mut code = 0u16;
            for bit in 0..8 {
                code |= u16::from((pair[0] >> bit) & 1) << (2 * bit);
                code |= u16::from((pair[1] >> bit) & 1) << (2 * bit + 1);
            }
            code as i16
        })
        .collect::<Vec<_>>();
    let got_interleaved = runtime.download(&arena, interleaved.id).unwrap().0;
    assert_eq!(
        got_interleaved
            .bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| i16::from_ne_bytes(*bytes))
            .collect::<Vec<_>>(),
        expected_interleaved
    );
    assert_eq!(
        runtime
            .download(&arena, transposed_roundtrip.id)
            .unwrap()
            .0
            .bytes,
        vec![
            byte_values[0],
            byte_values[2],
            byte_values[1],
            byte_values[3]
        ]
    );
    let roundtrip = runtime.download(&arena, float_roundtrip.id).unwrap().0;
    let got_codes = roundtrip
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_ne_bytes(*bytes))
        .collect::<Vec<_>>();
    assert_eq!(got_codes, float_codes);
}
