#![cfg(target_os = "macos")]

use luminal::prelude::*;
use luminal_metal::{MetalRuntime, harness_search_options};

/// The full-checkpoint rotary path exposed unbounded ring expansion when the
/// two sliced halves rejoin and their head/sequence axes are transposed back.
/// Keep its real head geometry and a nontrivial dynamic sequence interval.
#[test]
fn rotary_split_rejoin_with_dynamic_sequence_matches_scalar() {
    let mut graph = Graph::default();
    let input = graph.tensor(('s', 2048), DType::F32);
    let heads = input.split_dims(1, 64).transpose(0, 1);
    let lo = heads.slice((.., .., ..32));
    let hi = heads.slice((.., .., 32..));
    let output = (lo * 2. - hi * 3.)
        .concat_along(hi * 2. + lo * 3., 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
        .output();
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .bind_dim_buckets('s', vec![DimBucket::new(2, 64).representative(30)])
        .unwrap();
    runtime
        .search(&Default::default(), &harness_search_options())
        .unwrap();
    for sequence in [2, 30, 64] {
        let values: Vec<f32> = (0..sequence * 2048)
            .map(|i| (i % 251) as f32 / 128. - 1.)
            .collect();
        let mut expected = vec![0.; values.len()];
        for base in (0..values.len()).step_by(64) {
            for lane in 0..32 {
                let lo = values[base + lane];
                let hi = values[base + lane + 32];
                expected[base + lane] = lo * 2. - hi * 3.;
                expected[base + lane + 32] = hi * 2. + lo * 3.;
            }
        }
        runtime.set_dim('s', sequence);
        runtime.set_data(input.id, values);
        runtime.execute().unwrap();
        let (data, binding) = runtime.fetch(output.id).unwrap();
        let actual =
            luminal_metal::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap();
        assert_eq!(actual, expected, "sequence length {sequence}");
    }
}
