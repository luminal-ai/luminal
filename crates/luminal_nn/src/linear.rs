use luminal::prelude::*;

/// Apply a linear projection with a canonical `(in_features, out_features)` weight.
pub fn linear(input: GraphTensor, weight: GraphTensor, bias: Option<GraphTensor>) -> GraphTensor {
    assert_eq!(weight.rank(), 2, "linear weight must be rank two");
    assert_eq!(
        input.dtype, weight.dtype,
        "linear input/weight dtype mismatch"
    );
    assert_eq!(
        input.dims().last(),
        weight.dims().first(),
        "linear input width does not match weight"
    );
    let output = input.matmul(weight);
    if let Some(bias) = bias {
        assert_eq!(bias.rank(), 1, "linear bias must be rank one");
        assert_eq!(
            bias.dtype, output.dtype,
            "linear output/bias dtype mismatch"
        );
        assert_eq!(
            bias.dims().first(),
            output.dims().last(),
            "linear bias width does not match output"
        );
        output + bias.expand_lhs(&output.dims()[..output.dims().len() - 1])
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use super::linear;
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal_reference::ReferenceRuntime;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// The M3 ladder end-to-end (load → search → execute → read): the
    /// first nn-module test on the native path. Hand-computed reference.
    #[test]
    fn linear_forward_matches_hand_reference() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 3), DType::F32);
        let weight = cx.tensor((3, 4), DType::F32);
        let out = linear(x, weight, None).output();

        let x_data = vec![1., 2., 3., 4., 5., 6.];
        let w_data: Vec<f32> = (1..=12).map(|v| v as f32 * 0.1).collect();
        let mut expected = vec![0f32; 8];
        for r in 0..2 {
            for c in 0..4 {
                expected[r * 4 + c] = (0..3).map(|k| x_data[r * 3 + k] * w_data[k * 4 + c]).sum();
            }
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone().into());
        data.insert(weight.id, w_data.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(weight.id, w_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Bias broadcasts over the batch dimension.
    #[test]
    fn linear_bias_broadcasts_over_the_batch() {
        let mut cx = Graph::new();
        let x = cx.tensor((2, 2), DType::F32);
        let weight = cx.tensor((2, 3), DType::F32);
        let bias = cx.tensor(3, DType::F32);
        let out = linear(x, weight, Some(bias)).output();

        let x_data = vec![1., 2., 3., 4.];
        let w_data = vec![1., 0., 2., 0., 1., 3.];
        let b_data = vec![0.5, -1.0, 0.25];
        let mut expected = vec![0f32; 6];
        for r in 0..2 {
            for c in 0..3 {
                expected[r * 3 + c] = (0..2)
                    .map(|k| x_data[r * 2 + k] * w_data[k * 3 + c])
                    .sum::<f32>()
                    + b_data[c];
            }
        }

        let mut data = FxHashMap::default();
        data.insert(x.id, x_data.clone().into());
        data.insert(weight.id, w_data.clone().into());
        data.insert(bias.id, b_data.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(weight.id, w_data);
        rt.set_data(bias.id, b_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}

/// Apply a per-tensor-quantized FP8 linear projection.
///
/// The weight is `(out_features, in_features)` E4M3FN and two F32 scalars
/// accompany it: a static input scale and a weight scale. The computation is:
///   q = cast_f8(x / input_scale)            (RNE, saturating ±448)
///   y = (cast_f32(q) @ cast_f32(Wᵀ)) · (input_scale · weight_scale)
/// This is numerically an FP8×FP8 GEMM with F32 accumulation.
pub fn fp8_linear(
    input: GraphTensor,
    weight: GraphTensor,
    input_scale: GraphTensor,
    weight_scale: GraphTensor,
) -> GraphTensor {
    assert_eq!(weight.rank(), 2, "FP8 linear weight must be rank two");
    assert_eq!(weight.dtype, DType::F8E4M3, "FP8 linear weight dtype");
    assert!(input_scale.dims().is_empty(), "input scale must be scalar");
    assert!(
        weight_scale.dims().is_empty(),
        "weight scale must be scalar"
    );
    assert_eq!(input_scale.dtype, DType::F32, "input scale dtype");
    assert_eq!(weight_scale.dtype, DType::F32, "weight scale dtype");
    let dims = input.dims();
    assert_eq!(
        dims.last(),
        weight.dims().get(1),
        "FP8 linear input width does not match weight"
    );
    let in_scale = input_scale.expand_lhs(&dims[..]).reciprocal();
    let quantized = (input * in_scale).cast(DType::F8E4M3);
    let wide = quantized.cast(DType::F32);
    let weight_wide = weight.cast(DType::F32).permute((1, 0));
    let raw = wide.matmul(weight_wide);
    let out_dims = raw.dims();
    let rescale = input_scale.expand_lhs(&out_dims[..]) * weight_scale.expand_lhs(&out_dims[..]);
    raw * rescale
}

#[cfg(test)]
mod fp8_tests {
    use super::fp8_linear;
    use luminal::prelude::float8::F8E4M3;
    use luminal::prelude::*;

    /// The whole fp8 story through the runtime: an E4M3FN weight STAGED
    /// as an F8 buffer, the model's explicit quantize (clamped RNE),
    /// widening dequant reads, f32 accumulation, and the two-scale
    /// rescale — against a host reference computed with the same
    /// quantization math.
    #[test]
    fn fp8_linear_matches_scalar_reference() {
        const IN: usize = 3;
        const OUT: usize = 2;
        let mut cx = Graph::new();
        let x = cx.tensor((1, IN), DType::F32);
        let weight = cx.tensor((OUT, IN), DType::F8E4M3);
        let input_scale_tensor = cx.tensor((), DType::F32);
        let weight_scale_tensor = cx.tensor((), DType::F32);
        let out = fp8_linear(x, weight, input_scale_tensor, weight_scale_tensor).output();

        let x_vals = vec![0.37f32, -1.42, 2.6];
        let input_scale = 0.5f32;
        let weight_scale = 2.0f32;
        // (out, in) row-major weight codes.
        let weight_f32 = [0.5f32, -1.5, 2.0, 3.5, -0.0625, 448.0];
        let weight_codes: Vec<F8E4M3> = weight_f32.iter().map(|w| F8E4M3::from_f32(*w)).collect();

        // Host reference with the identical quantization math.
        let quant = |v: f32| F8E4M3::from_f32((v / input_scale).clamp(-448.0, 448.0)).to_f32();
        let mut expected = vec![0.0f32; OUT];
        for (o, row) in expected.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..IN {
                acc += quant(x_vals[i]) * weight_codes[o * IN + i].to_f32();
            }
            *row = acc * (input_scale * weight_scale);
        }

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (x.id, x_vals.into()),
                (weight.id, weight_codes.into()),
                (input_scale_tensor.id, vec![input_scale].into()),
                (weight_scale_tensor.id, vec![weight_scale].into()),
            ],
        );
        let ours = rt.get_f32(out.id).expect("fp8 linear out");
        for (index, (a, b)) in ours.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }
}
