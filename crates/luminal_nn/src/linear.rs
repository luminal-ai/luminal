use luminal::prelude::*;

/// A simple linear layer
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    permute: bool,
}

impl Linear {
    pub fn new(inp: usize, out: usize, bias: bool, ns: &Ns, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (inp, out)),
            bias: if bias {
                Some(cx.named_tensor(ns.leaf("bias"), out))
            } else {
                None
            },
            permute: false,
        }
    }

    pub fn new_permuted(inp: usize, out: usize, bias: bool, ns: &Ns, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(ns.leaf("weight"), (out, inp)),
            bias: if bias {
                Some(cx.named_tensor(ns.leaf("bias"), out))
            } else {
                None
            },
            permute: true,
        }
    }
}

impl Linear {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let output = input.matmul(if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        });
        if let Some(bias) = self.bias {
            output + bias.expand_lhs(&output.dims()[..output.dims().len() - 1])
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::reference::ReferenceRuntime;
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
        let model = Linear::new(3, 4, false, &Ns::root().child("fc"), &mut cx);
        let x = cx.tensor((2, 3));
        let out = model.forward(x).output();

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
        data.insert(model.weight.id, w_data.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(model.weight.id, w_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Bias broadcasts over the batch dimension.
    #[test]
    fn linear_bias_broadcasts_over_the_batch() {
        let mut cx = Graph::new();
        let model = Linear::new(2, 3, true, &Ns::root().child("fc"), &mut cx);
        let x = cx.tensor((2, 2));
        let out = model.forward(x).output();

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
        data.insert(model.weight.id, w_data.clone().into());
        data.insert(model.bias.unwrap().id, b_data.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&data, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(x.id, x_data);
        rt.set_data(model.weight.id, w_data);
        rt.set_data(model.bias.unwrap().id, b_data);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}

/// Per-tensor-quantized fp8 linear (the nvidia/modelopt form —
/// quantization is MODEL DEFINITION, ruling 2026-08-12): the weight is
/// an E4M3FN tensor and TWO F32 scalars accompany it — a static input
/// scale (calibration) and the weight scale. The forward spells the
/// quantization explicitly in model text:
///   q = cast_f8(x / input_scale)            (RNE, saturating ±448)
///   y = (cast_f32(q) @ cast_f32(Wᵀ)) · (input_scale · weight_scale)
/// which is numerically the fp8×fp8 GEMM with f32 accumulation. The
/// weight STAYS an F8E4M3 buffer end to end; the widening casts are
/// the explicit dequant reads.
pub struct Fp8Linear {
    /// (out, in) E4M3FN weight — HF orientation, bound untransposed.
    pub weight: GraphTensor,
    /// () F32 static input scale.
    pub input_scale: GraphTensor,
    /// () F32 weight scale.
    pub weight_scale: GraphTensor,
    inp: usize,
    out: usize,
}

impl Fp8Linear {
    pub fn new(inp: usize, out: usize, ns: &Ns, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor_dtyped(ns.leaf("weight"), (out, inp), DType::F8E4M3),
            input_scale: cx.named_tensor(ns.leaf("input_scale"), ()),
            weight_scale: cx.named_tensor(ns.leaf("weight_scale"), ()),
            inp,
            out,
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let dims = input.dims();
        assert_eq!(
            dims.last().and_then(|d| d.to_usize()),
            Some(self.inp),
            "Fp8Linear input width"
        );
        let in_scale = self.input_scale.expand_lhs(&dims[..]).reciprocal();
        let quantized = (input * in_scale).cast(DType::F8E4M3);
        let wide = quantized.cast(DType::F32);
        let weight_wide = self.weight.cast(DType::F32).permute((1, 0));
        let raw = wide.matmul(weight_wide);
        let out_dims = raw.dims();
        let rescale = (self.input_scale.expand_lhs(&out_dims[..]))
            * (self.weight_scale.expand_lhs(&out_dims[..]));
        raw * rescale
    }

    pub fn out_features(&self) -> usize {
        self.out
    }
}

#[cfg(test)]
mod fp8_tests {
    use super::Fp8Linear;
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
        let layer = Fp8Linear::new(IN, OUT, &Ns::root().child("fc"), &mut cx);
        let x = cx.tensor((1, IN));
        let out = layer.forward(x).output();

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

        let rt = luminal::test_support::run_reference(
            &cx,
            &[
                (x.id, x_vals.into()),
                (layer.weight.id, weight_codes.into()),
                (layer.input_scale.id, vec![input_scale].into()),
                (layer.weight_scale.id, vec![weight_scale].into()),
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
