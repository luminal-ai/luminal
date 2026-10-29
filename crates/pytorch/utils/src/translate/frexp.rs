//! Exact radix-two normalization using comparison, selection and power-of-two
//! scaling. No logarithm approximations, bit reinterpretation, or lossy casts.
use super::Translator;
use crate::pt2_schema::Node;
use anyhow::{Result, ensure};
use luminal::prelude::*;

impl Translator<'_> {
    pub(super) fn translate_frexp(&mut self, node: &Node) -> Result<()> {
        let input = self.operand(&node.inputs[0])?;
        ensure!(
            matches!(
                input.dtype,
                DType::F16 | DType::Bf16 | DType::F32 | DType::F64
            ),
            "frexp requires floating point input"
        );
        let dtype = input.dtype;
        let value = if matches!(dtype, DType::F16 | DType::Bf16) {
            input.cast(DType::F32)
        } else {
            input
        };
        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let nan = self.is_nan(value);
        let inf = self.is_inf(value);
        let invalid = self.bool_or(nan, inf);
        let finite = self.bool_not(invalid);
        let active = self.bool_and(finite, value.ne(zero));
        let mut mantissa = active.select(value, one);
        let integer_zero = self.cx.constant_i32(0).expand_rhs(value.dims());
        let mut exponent = integer_zero;
        let steps: &[i32] = if dtype == DType::F64 {
            &[512, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        } else {
            &[64, 64, 32, 16, 8, 4, 2, 1]
        };
        for &step in steps {
            let high = self.constant_like(value, 2.0_f64.powi(step));
            let low = self.constant_like(value, 2.0_f64.powi(-step));
            let magnitude = mantissa.abs();
            let down = magnitude.ge(high);
            mantissa = down.select(mantissa * low, mantissa);
            let delta = self.cx.constant_i32(step).expand_rhs(value.dims());
            exponent = down.select(exponent + delta, exponent);
            let threshold = self.constant_like(value, 2.0_f64.powi(1 - step));
            let up = mantissa.abs().lt(threshold);
            mantissa = up.select(mantissa * high, mantissa);
            exponent = up.select(exponent - delta, exponent);
        }
        let half = self.constant_like(value, 0.5);
        let integer_one = self.cx.constant_i32(1).expand_rhs(value.dims());
        mantissa = active.select(mantissa * half, value).cast(dtype);
        exponent = active.select(exponent + integer_one, integer_zero);
        self.bind_outputs(node, vec![mantissa, exponent])
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
    use rustc_hash::FxHashMap;

    fn expected(x: f32) -> (f32, i32) {
        let bits = x.to_bits();
        let e = (bits >> 23) & 255;
        let f = bits & 0x7fffff;
        let sign = bits & 0x80000000;
        if e == 255 || (e == 0 && f == 0) {
            return (x, 0);
        }
        if e != 0 {
            return (f32::from_bits(sign | (126 << 23) | f), e as i32 - 126);
        }
        let p = 31 - f.leading_zeros();
        (
            f32::from_bits(sign | (126 << 23) | ((f << (23 - p)) & 0x7fffff)),
            p as i32 - 148,
        )
    }

    #[test]
    fn frexp_aten_and_prims_preserve_exponent_boundaries_and_special_values() {
        let mut values = vec![
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        for e in -149..128 {
            let x = 2.0_f64.powi(e) as f32;
            for bits in [x.to_bits(), x.to_bits().saturating_sub(1), x.to_bits() + 1] {
                let v = f32::from_bits(bits);
                values.push(v);
                values.push(-v);
            }
        }
        for target in [
            "torch.ops.aten.frexp.Tensor",
            "torch.ops.prims.frexp.default",
        ] {
            let shape = [values.len() as i64];
            let t = translate_one(
                node(target, vec![tensor_input("self", "x")], &["m", "e"]),
                &[
                    ("x", FLOAT, &shape),
                    ("m", FLOAT, &shape),
                    ("e", INT, &shape),
                ],
                &["x"],
                &["m", "e"],
            );
            let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
            let mut data = FxHashMap::default();
            data.insert(t.inputs[0].tensor, TypedBuffer::F32(values.clone()));
            runtime
                .search(
                    &Default::default(),
                    &Default::default(),
                    &data,
                    &harness_search_options(),
                )
                .unwrap();
            runtime.set_data(t.inputs[0].tensor, TypedBuffer::F32(values.clone()));
            runtime.execute().unwrap();
            let m = runtime.get_f32(t.outputs[0].tensor).unwrap();
            let e = runtime.get_i32(t.outputs[1].tensor).unwrap();
            for (i, &x) in values.iter().enumerate() {
                let (want_m, want_e) = expected(x);
                if want_m.is_nan() {
                    assert!(m[i].is_nan());
                } else {
                    assert_eq!(
                        m[i].to_bits(),
                        want_m.to_bits(),
                        "input {x}, mantissa {}",
                        m[i]
                    );
                }
                assert_eq!(e[i], want_e, "input {x}");
            }
        }
    }
}
