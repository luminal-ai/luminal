//! ATen shifts use the generic Boolean encoding decompositions in the frontend.
use super::{Translator, util};
use crate::pt2_schema::Node;
use anyhow::Result;
use luminal::prelude::*;

impl Translator<'_> {
    pub(super) fn translate_shift(&mut self, node: &Node, right: bool) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        anyhow::ensure!(
            matches!(
                dtype,
                DType::I8 | DType::U8 | DType::I16 | DType::Int | DType::I64 | DType::U64
            ),
            "bitwise shifts require integer operands"
        );
        let a = if let Some(a) = self.optional_tensor_operand(&node.inputs[0])? {
            a.cast(dtype)
        } else {
            self.scalar(&node.inputs[0], dtype)?
        };
        // A weak Python scalar is converted to the computation dtype by ATen.
        // Preserve that conversion before choosing the fixed index-map path.
        if let Some(count) = node.inputs[1]
            .arg
            .as_int()
            .or_else(|| node.inputs[1].arg.as_bool().map(i64::from))
        {
            let count = match dtype {
                DType::I8 => count as i8 as i64,
                DType::U8 => count as u8 as i64,
                DType::I16 => count as i16 as i64,
                DType::Int => count as i32 as i64,
                _ => count,
            };
            let count = if count < 0 {
                dtype.logical_bits()
            } else {
                (count as u64).min(dtype.logical_bits() as u64) as usize
            };
            return Ok(if right {
                a.shift_right(count)
            } else {
                a.shift_left(count)
            });
        }
        let b = self.operand(&node.inputs[1])?.cast(dtype);
        let (a, b) = util::broadcast_binary(a, b);
        Ok(if right {
            a.shift_right_by(b)
        } else {
            a.shift_left_by(b)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::translate::test_support::*;
    use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
    use rustc_hash::FxHashMap;
    fn run(t: &crate::translate::Translation, values: &[TypedBuffer]) -> ReferenceRuntime {
        let data: FxHashMap<_, _> = t
            .inputs
            .iter()
            .zip(values)
            .map(|(i, v)| (i.tensor, v.clone()))
            .collect();
        let outputs = t.outputs.iter().map(|o| o.tensor).collect::<Vec<_>>();
        let bindings = luminal_reference::ReferenceBindings::dense(&t.graph.logical, &outputs);
        let mut rt = ReferenceRuntime::load_with(&t.graph, bindings).unwrap();
        rt.search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
        for (i, v) in t.inputs.iter().zip(values) {
            rt.set_data(i.tensor, v.clone());
        }
        rt.execute().unwrap();
        rt
    }
    fn oracle(x: i64, n: i64, right: bool) -> i64 {
        if !(0..64).contains(&n) {
            if right && x < 0 { -1 } else { 0 }
        } else if right {
            x >> n
        } else {
            x.wrapping_shl(n as u32)
        }
    }
    #[test]
    fn tensor_counts_broadcast_without_truncating_wide_values_or_counts() {
        let x = [
            i64::MIN,
            i64::MAX,
            -1,
            0,
            1,
            (1 << 54) + 3,
            -7,
            7,
            -3,
            3,
            i64::MAX,
            i64::MIN,
        ];
        let counts = [0, 1, 63, 64, i64::MAX, i64::MIN];
        for right in [false, true] {
            let name = if right {
                "bitwise_right_shift.Tensor"
            } else {
                "bitwise_left_shift.Tensor"
            };
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{name}"),
                    vec![tensor_input("self", "a"), tensor_input("other", "b")],
                    &["y"],
                ),
                &[("a", 5, &[2, 6]), ("b", 5, &[6]), ("y", 5, &[2, 6])],
                &["a", "b"],
                &["y"],
            );
            let rt = run(
                &t,
                &[
                    TypedBuffer::I64(x.to_vec()),
                    TypedBuffer::I64(counts.to_vec()),
                ],
            );
            let expected = x
                .iter()
                .enumerate()
                .map(|(i, x)| oracle(*x, counts[i % 6], right))
                .collect::<Vec<_>>();
            assert_eq!(
                rt.get_i64(t.outputs[0].tensor).unwrap(),
                &expected,
                "{name}"
            );
        }
    }
    #[test]
    fn fixed_shifts_use_views_and_preserve_signed_encodings() {
        let x = [i64::MIN, i64::MAX, -1, 0, 1, (1 << 54) + 3];
        for right in [false, true] {
            for count in [-1, 0, 1, 31, 63, 64, i64::MAX] {
                let name = if right {
                    "bitwise_right_shift.Tensor_Scalar"
                } else {
                    "bitwise_left_shift.Tensor_Scalar"
                };
                let t = translate_one(
                    node(
                        &format!("torch.ops.aten.{name}"),
                        vec![tensor_input("self", "a"), scalar_int("other", count)],
                        &["y"],
                    ),
                    &[("a", 5, &[2, 3]), ("y", 5, &[2, 3])],
                    &["a"],
                    &["y"],
                );
                let rt = run(&t, &[TypedBuffer::I64(x.to_vec())]);
                assert_eq!(
                    rt.get_i64(t.outputs[0].tensor).unwrap(),
                    &x.map(|v| oracle(v, count, right)),
                    "{name} {count}"
                );
                assert!(
                    t.graph
                        .logical
                        .petgraph()
                        .node_weights()
                        .all(|node| !matches!(node.op, luminal::graph::LogicalOp::Gather)),
                    "fixed shifts must be views"
                );
            }
        }
    }
    #[test]
    fn prims_shift_aliases_use_the_same_decomposition() {
        let x = [i64::MIN, i64::MAX, (1 << 54) + 3];
        let counts = [1, 31, 63];
        for right in [false, true] {
            let name = if right {
                "shift_right_arithmetic"
            } else {
                "shift_left"
            };
            let t = translate_one(
                node(
                    &format!("torch.ops.prims.{name}.default"),
                    vec![tensor_input("self", "a"), tensor_input("other", "b")],
                    &["y"],
                ),
                &[("a", 5, &[3]), ("b", 5, &[3]), ("y", 5, &[3])],
                &["a", "b"],
                &["y"],
            );
            let rt = run(
                &t,
                &[
                    TypedBuffer::I64(x.to_vec()),
                    TypedBuffer::I64(counts.to_vec()),
                ],
            );
            assert_eq!(
                rt.get_i64(t.outputs[0].tensor).unwrap(),
                &x.iter()
                    .zip(counts)
                    .map(|(x, n)| oracle(*x, n, right))
                    .collect::<Vec<_>>()
            );
        }
    }
    #[test]
    fn narrow_scalar_counts_follow_aten_conversion() {
        let t = translate_one(
            node(
                "torch.ops.aten.bitwise_left_shift.Tensor_Scalar",
                vec![tensor_input("self", "a"), scalar_int("other", 257)],
                &["y"],
            ),
            &[("a", 2, &[4]), ("y", 2, &[4])],
            &["a"],
            &["y"],
        );
        let rt = run(&t, &[TypedBuffer::I8(vec![1, -1, 127, -128])]);
        assert_eq!(rt.get_i8(t.outputs[0].tensor).unwrap(), &[2, -2, -2, 0]);
    }
    #[test]
    fn unsigned_shift_counts_are_bounded_before_index_conversion() {
        let x = [
            u64::MAX,
            1 << 63,
            (1 << 54) + 3,
            7,
            1,
            u64::MAX,
            u64::MAX,
            0,
        ];
        let counts = [0, 1, 31, 63, 64, u64::MAX, (1 << 32) + 1, 1 << 63];
        for right in [false, true] {
            let name = if right {
                "bitwise_right_shift.Tensor"
            } else {
                "bitwise_left_shift.Tensor"
            };
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{name}"),
                    vec![tensor_input("self", "a"), tensor_input("other", "b")],
                    &["y"],
                ),
                &[("a", 35, &[8]), ("b", 35, &[8]), ("y", 35, &[8])],
                &["a", "b"],
                &["y"],
            );
            let rt = run(
                &t,
                &[
                    TypedBuffer::U64(x.to_vec()),
                    TypedBuffer::U64(counts.to_vec()),
                ],
            );
            let expected = x
                .iter()
                .zip(counts)
                .map(|(x, n)| {
                    if n >= 64 {
                        0
                    } else if right {
                        x >> n
                    } else {
                        x.wrapping_shl(n as u32)
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(rt.get_u64(t.outputs[0].tensor).unwrap(), &expected);
        }
    }
    #[test]
    fn reflected_scalar_and_promoted_tensor_counts_translate() {
        let counts = [0, 1, 7, 63, 64, -1];
        for right in [false, true] {
            let name = if right {
                "bitwise_right_shift.Scalar_Tensor"
            } else {
                "bitwise_left_shift.Scalar_Tensor"
            };
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{name}"),
                    vec![scalar_int("self", -3), tensor_input("other", "b")],
                    &["y"],
                ),
                &[("b", 5, &[6]), ("y", 5, &[6])],
                &["b"],
                &["y"],
            );
            let rt = run(&t, &[TypedBuffer::I64(counts.to_vec())]);
            assert_eq!(
                rt.get_i64(t.outputs[0].tensor).unwrap(),
                &counts.map(|n| oracle(-3, n, right))
            );
        }
        let t = translate_one(
            node(
                "torch.ops.aten.__rshift__.Tensor",
                vec![tensor_input("self", "a"), tensor_input("other", "b")],
                &["y"],
            ),
            &[("a", 2, &[6]), ("b", 5, &[6]), ("y", 5, &[6])],
            &["a", "b"],
            &["y"],
        );
        let rt = run(
            &t,
            &[
                TypedBuffer::I8(vec![-128, 127, -1, 0, 1, 3]),
                TypedBuffer::I64(vec![0, 1, 7, 31, 32, 63]),
            ],
        );
        assert_eq!(
            rt.get_i64(t.outputs[0].tensor).unwrap(),
            &[-128, 63, -1, 0, 0, 0]
        );
    }
}
