use luminal::prelude::*;
use luminal_reference::{
    ReferenceBindings, ReferenceRuntime, TypedBuffer, bf16, f16, harness_search_options,
};

fn payload(dtype: DType, values: &[f32]) -> TypedBuffer {
    match dtype {
        DType::F16 => TypedBuffer::F16(values.iter().copied().map(f16::from_f32).collect()),
        DType::Bf16 => TypedBuffer::Bf16(values.iter().copied().map(bf16::from_f32).collect()),
        _ => unreachable!(),
    }
}
fn widened(data: &TypedBuffer) -> Vec<f32> {
    match data {
        TypedBuffer::F16(v) => v.iter().map(|v| v.to_f32()).collect(),
        TypedBuffer::Bf16(v) => v.iter().map(|v| v.to_f32()).collect(),
        _ => unreachable!(),
    }
}
fn run(
    graph: &Graph,
    outputs: &[GraphTensor],
    inputs: &[(GraphTensor, TypedBuffer)],
) -> ReferenceRuntime {
    let mut runtime = ReferenceRuntime::load_with(
        graph,
        ReferenceBindings::dense(
            &graph.logical,
            &outputs.iter().map(|x| x.id).collect::<Vec<_>>(),
        ),
    )
    .unwrap();
    let data = inputs.iter().map(|(x, v)| (x.id, v.clone())).collect();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    for (x, v) in inputs {
        runtime.set_data(x.id, v.clone());
    }
    runtime.execute().unwrap();
    runtime
}

#[test]
fn half_casts_keep_16_bit_storage_and_explicit_rounding_boundaries() {
    for dtype in [DType::F16, DType::Bf16] {
        let step = if dtype == DType::F16 {
            1.0 / 1024.0
        } else {
            1.0 / 128.0
        };
        let values = vec![
            0.0,
            -0.0,
            1.0,
            1.0 + step / 2.0,
            1.0 + 1.5 * step,
            f32::INFINITY,
        ];
        let expected: Vec<u16> = if dtype == DType::F16 {
            vec![0, 0x8000, 0x3c00, 0x3c00, 0x3c02, 0x7c00]
        } else {
            vec![0, 0x8000, 0x3f80, 0x3f80, 0x3f82, 0x7f80]
        };
        let mut graph = Graph::new();
        let x = graph.tensor([values.len()], DType::F32);
        let narrow = x.cast(dtype);
        let wide = narrow.cast(DType::F32);
        let mut runtime = run(&graph, &[narrow, wide], &[(x, TypedBuffer::F32(values))]);
        let bits: Vec<_> = match dtype {
            DType::F16 => runtime
                .get_f16(narrow.id)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            _ => runtime
                .get_bf16(narrow.id)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        };
        assert_eq!(bits, expected);
        assert_eq!(
            payload(dtype, runtime.get_f32(wide.id).unwrap()).byte_len(),
            expected.len() * 2
        );
        assert_eq!(
            runtime.get_f32(wide.id).unwrap(),
            &widened(&payload(dtype, runtime.get_f32(wide.id).unwrap()))
        );
        runtime.set_data(x.id, payload(dtype, &[0.; 6]));
        assert!(format!("{:#}", runtime.execute().unwrap_err()).contains("wrong type"));
    }
}

#[test]
fn narrow_inputs_residual_bias_and_output_surround_only_f32_arithmetic() {
    for dtype in [DType::Bf16, DType::F16] {
        let mut graph = Graph::new();
        let a = graph.tensor([2, 3], dtype);
        let b = graph.tensor([3, 2], dtype);
        let c = graph.tensor([2, 2], dtype);
        let bias = graph.tensor([2], dtype);
        let out = (a.cast(DType::F32).matmul(b.cast(DType::F32))
            + c.cast(DType::F32)
            + bias.cast(DType::F32).expand_dim(0, 2))
        .cast(dtype);
        let inputs = [
            (
                a,
                payload(dtype, &[1.001, -2.003, 0.333, 0.25, 1.125, -0.125]),
            ),
            (b, payload(dtype, &[0.337, 0.125, 1.003, -0.5, 0.75, 2.007])),
            (c, payload(dtype, &[0.003, 1.007, -0.003, 2.003])),
            (bias, payload(dtype, &[0.013, -0.029])),
        ];
        let runtime = run(&graph, &[out], &inputs);
        let values: Vec<_> = inputs.iter().map(|(_, x)| widened(x)).collect();
        let mut expected = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                let product = (0..3)
                    .map(|k| f64::from(values[0][row * 3 + k]) * f64::from(values[1][k * 2 + col]))
                    .sum::<f64>() as f32;
                expected.push((product + values[2][row * 2 + col]) + values[3][col]);
            }
        }
        let actual = match dtype {
            DType::F16 => TypedBuffer::F16(runtime.get_f16(out.id).unwrap().clone()),
            _ => TypedBuffer::Bf16(runtime.get_bf16(out.id).unwrap().clone()),
        };
        assert_eq!(actual, payload(dtype, &expected));
    }
}

#[test]
fn narrow_payload_views_and_select_preserve_bits() {
    for dtype in [DType::F16, DType::Bf16] {
        let raw = [0x8000u16, 0x7fc1, 0x0001, 0x3c00, 0x7e23, 0x1234];
        let input = match dtype {
            DType::F16 => TypedBuffer::F16(raw.map(f16::from_bits).to_vec()),
            _ => TypedBuffer::Bf16(raw.map(bf16::from_bits).to_vec()),
        };
        let mut graph = Graph::new();
        let x = graph.tensor([2, 3], dtype);
        let condition = graph.tensor([3, 2], DType::Bool);
        let transposed = x.permute(&[1, 0]);
        let out = condition.select(transposed, transposed);
        let runtime = run(
            &graph,
            &[out],
            &[
                (x, input),
                (
                    condition,
                    TypedBuffer::bool8(vec![1, 0, 1, 0, 1, 0]).unwrap(),
                ),
            ],
        );
        let bits: Vec<_> = match dtype {
            DType::F16 => runtime
                .get_f16(out.id)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
            _ => runtime
                .get_bf16(out.id)
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect(),
        };
        assert_eq!(bits, vec![raw[0], raw[3], raw[1], raw[4], raw[2], raw[5]]);
    }
}
