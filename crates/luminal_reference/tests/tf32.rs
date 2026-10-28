use luminal::prelude::*;
use luminal_reference::{
    ReferenceBindings, ReferenceRuntime, Tf32, TypedBuffer, harness_search_options,
};
use rustc_hash::FxHashMap;

// A numerical oracle independent of the codec's integer rounding algorithm.
fn rounded(value: f32) -> f32 {
    assert!(value.is_finite());
    if value == 0.0 {
        return value;
    }
    let exponent = f64::from(value).abs().log2().floor().max(-126.0) as i32;
    let step = 2.0f64.powi(exponent - 10);
    ((f64::from(value) / step).round_ties_even() * step) as f32
}

#[test]
fn explicit_casts_materialize_tf32_values_and_compiled_plans_keep_the_rounding() {
    let mut graph = Graph::new();
    let values = [
        0x3f80_1000u32,
        0x3f80_3000,
        0xbf80_1001,
        0x8000_0000,
        0x0000_1001,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
    ]
    .map(f32::from_bits)
    .to_vec();
    let expected = [
        0x3f80_0000u32,
        0x3f80_4000,
        0xbf80_2000,
        0x8000_0000,
        0x0000_2000,
        0x7f80_0000,
        0x7f80_0000,
        0x7fc0_0000,
    ];
    let input = graph.tensor([values.len()], DType::F32);
    let narrow = input.cast(DType::TF32);
    let wide = narrow.cast(DType::F32);
    let mut bindings = ReferenceBindings::new();
    bindings.input(input.id);
    let narrow_buffer = bindings.output(narrow.id);
    let wide_buffer = bindings.output(wide.id);
    let mut runtime = ReferenceRuntime::load_with(&graph, bindings).unwrap();
    let inputs = FxHashMap::from_iter([(input.id, TypedBuffer::F32(values.clone()))]);
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    runtime.set_data(input.id, values.clone());
    runtime.execute().unwrap();
    assert_eq!(
        runtime
            .get_tf32(narrow.id)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        runtime
            .get_f32(wide.id)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    assert!(
        runtime.get_f32(narrow.id).is_err(),
        "typed readback must require widening"
    );
    assert_eq!(
        runtime.get_buffer(narrow_buffer).unwrap().byte_len(),
        values.len() * 4
    );

    let bytes = runtime
        .serialize_compiled(&[input.id], &[narrow_buffer, wide_buffer])
        .unwrap();
    let mut restored = ReferenceRuntime::default();
    restored
        .deserialize_compiled(&bytes, &[input.id], &[narrow.id, wide.id], 1024 * 1024)
        .unwrap();
    restored.set_data(input.id, values);
    restored.execute().unwrap();
    assert_eq!(
        restored
            .get_f32(wide.id)
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        restored.get_tf32(narrow.id).unwrap(),
        runtime.get_tf32(narrow.id).unwrap()
    );
}

#[test]
fn views_and_selection_preserve_the_tf32_payload() {
    let mut graph = Graph::new();
    let input = graph.tensor([2, 3], DType::F32);
    let condition = graph.tensor([3, 2], DType::Bool);
    let narrow = input.cast(DType::TF32).permute(&[1, 0]);
    let other = (input * -1.0).permute(&[1, 0]).cast(DType::TF32);
    let output = condition.select(narrow, other).cast(DType::F32);
    let values: Vec<_> = (0..6).map(|i| 1.0 + (i as f32 + 1.0) / 4096.0).collect();
    let codes = vec![1, 0, 0, 1, 1, 0];
    let inputs = FxHashMap::from_iter([
        (input.id, TypedBuffer::F32(values.clone())),
        (condition.id, TypedBuffer::bool8(codes.clone()).unwrap()),
    ]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    for (id, data) in inputs {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    let expected: Vec<_> = (0..3)
        .flat_map(|row| {
            let (values, codes) = (&values, &codes);
            (0..2).map(move |col| {
                let value = rounded(values[col * 3 + row]);
                if codes[row * 2 + col] == 1 {
                    value
                } else {
                    -value
                }
            })
        })
        .collect();
    assert_eq!(runtime.get_f32(output.id).unwrap(), &expected);
}

#[test]
fn batched_products_use_rounded_inputs_f32_arithmetic_and_explicit_final_rounding() {
    let mut graph = Graph::new();
    let a = graph.tensor([2, 2, 3], DType::F32);
    let b = graph.tensor([2, 3, 2], DType::F32);
    let c = graph.tensor([2, 2, 2], DType::F32);
    let product = a
        .cast(DType::TF32)
        .cast(DType::F32)
        .matmul(b.cast(DType::TF32).cast(DType::F32));
    let output = (product + c).cast(DType::TF32).cast(DType::F32);
    let av: Vec<_> = (0..12)
        .map(|i| 1.0 + (i / 6) as f32 + (3.0 + (i % 3) as f32 / 4.0) / 8192.0)
        .collect();
    let bv: Vec<_> = (0..12)
        .map(|i| 1.0 + (i / 6) as f32 + (3.0 + (i % 4) as f32 / 8.0) / 8192.0)
        .collect();
    let cv: Vec<_> = (0..8).map(|i| (i as f32 + 1.0) / 8192.0).collect();
    let inputs = FxHashMap::from_iter([
        (a.id, TypedBuffer::F32(av.clone())),
        (b.id, TypedBuffer::F32(bv.clone())),
        (c.id, TypedBuffer::F32(cv.clone())),
    ]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    for (id, data) in inputs {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    let mut expected = Vec::new();
    let mut unrounded = Vec::new();
    for batch in 0..2 {
        for row in 0..2 {
            for col in 0..2 {
                let mut sum = 0.0f32;
                let mut raw_sum = 0.0f32;
                for inner in 0..3 {
                    let x = av[batch * 6 + row * 3 + inner];
                    let y = bv[batch * 6 + inner * 2 + col];
                    sum += rounded(x) * rounded(y);
                    raw_sum += x * y;
                }
                expected.push(rounded(sum + cv[batch * 4 + row * 2 + col]));
                unrounded.push(rounded(raw_sum + cv[batch * 4 + row * 2 + col]));
            }
        }
    }
    assert_ne!(
        expected, unrounded,
        "fixture must detect discarded input rounding"
    );
    assert_eq!(runtime.get_f32(output.id).unwrap(), &expected);
}

#[test]
fn casts_after_multiply_or_reduce_do_not_move_to_the_inputs() {
    let mut graph = Graph::new();
    let x = graph.tensor([2], DType::F32);
    let product = (x * x).cast(DType::TF32).cast(DType::F32);
    let sum = x.sum(&[0]).cast(DType::TF32).cast(DType::F32);
    let bindings = ReferenceBindings::dense(&graph.logical, &[product.id, sum.id]);
    let mut runtime = ReferenceRuntime::load_with(&graph, bindings).unwrap();
    let values = vec![1.0 + 3.0 / 8192.0, -1.0];
    let data = FxHashMap::from_iter([(x.id, TypedBuffer::F32(values.clone()))]);
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    runtime.set_data(x.id, values.clone());
    runtime.execute().unwrap();
    let expected: Vec<_> = values.iter().map(|x| rounded(x * x)).collect();
    assert_ne!(expected[0], rounded(values[0]) * rounded(values[0]));
    assert_eq!(runtime.get_f32(product.id).unwrap(), &expected);
    assert_eq!(runtime.get_f32(sum.id).unwrap(), &[3.0 / 8192.0]);
}

#[test]
fn tf32_staging_is_typed_and_memory_limits_use_32_bit_backing() {
    let mut graph = Graph::new();
    let x = graph.tensor([4], DType::F32);
    let output = x.cast(DType::TF32).cast(DType::F32);
    let values = vec![1.0 + 1.0 / 4096.0; 4];
    let data = FxHashMap::from_iter([(x.id, TypedBuffer::F32(values.clone()))]);
    let mut options = harness_search_options();
    options.max_intermediate_bytes = 15;
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    assert!(
        runtime
            .search(&Default::default(), &Default::default(), &data, &options)
            .is_err(),
        "four TF32 elements need sixteen physical bytes, not twelve"
    );
    options.max_intermediate_bytes = 16;
    runtime
        .search(&Default::default(), &Default::default(), &data, &options)
        .unwrap();
    runtime.set_data(x.id, values);
    runtime.execute().unwrap();
    assert_eq!(runtime.get_f32(output.id).unwrap(), &[1.0; 4]);
    let peak = runtime.peak_live_bytes();
    runtime.set_memory_budget_bytes(peak - 1);
    assert!(
        format!("{:#}", runtime.execute().unwrap_err()).contains("live memory budget exceeded")
    );

    let mut graph = Graph::new();
    let input = graph.tensor([2], DType::TF32);
    let out = input.cast(DType::F32);
    let payload = vec![Tf32::from_f32(1.0 + 1.0 / 4096.0), Tf32::from_f32(-1.0)];
    let data = FxHashMap::from_iter([(input.id, TypedBuffer::TF32(payload.clone()))]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    runtime.set_data(input.id, payload);
    runtime.execute().unwrap();
    assert_eq!(runtime.get_f32(out.id).unwrap(), &[1.0, -1.0]);
    runtime.set_data(input.id, vec![1.0f32, -1.0]);
    assert!(format!("{:#}", runtime.execute().unwrap_err()).contains("wrong type"));
}
