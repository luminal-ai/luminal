use anyhow::{Result, anyhow, ensure};
use luminal::prelude::*;

use super::convert;

/// ATen cat geometry, shared by ordinary tensors and each complex component.
/// Only a statically known `(0,)` is rank-neutral. Export hints must never
/// decide whether an operand contributes to a later execution.
pub(super) fn concatenate(
    values: Vec<GraphTensor>,
    raw_axis: i64,
    dtype: DType,
) -> Result<GraphTensor> {
    let first = *values.first().ok_or_else(|| anyhow!("cat: empty list"))?;
    let mut values = values
        .into_iter()
        .filter(|value| !(value.rank() == 1 && value.dims()[0].to_usize() == Some(0)));
    let Some(first_shaped) = values.next() else {
        // PyTorch's legacy all-(0,) case accepts any axis. Empty operands
        // still take part in promotion, encoded by the exported result dtype.
        return Ok(convert(first, dtype));
    };
    let rank = first_shaped.rank();
    let axis = if raw_axis < 0 {
        raw_axis + rank as i64
    } else {
        raw_axis
    };
    let axis = usize::try_from(axis)
        .ok()
        .filter(|axis| *axis < rank)
        .ok_or_else(|| anyhow!("cat: axis {raw_axis} out of range for rank {rank}"))?;
    let shape = first_shaped.dims();
    let mut acc = convert(first_shaped, dtype);
    for next in values {
        ensure!(
            next.rank() == rank,
            "cat: rank {} differs from rank {rank}",
            next.rank()
        );
        for (dim, (left, right)) in shape.iter().zip(next.dims()).enumerate() {
            if dim != axis
                && let (Some(left), Some(right)) = (left.to_usize(), right.to_usize())
            {
                ensure!(
                    left == right,
                    "cat: dimension {dim} differs ({left} vs {right})"
                );
            }
        }
        // Validate geometry before discarding same-rank empties. Symbolic
        // equality constraints remain the exported program's responsibility.
        if acc.dims()[axis].to_usize() == Some(0) {
            acc = convert(next, dtype);
        } else if next.dims()[axis].to_usize() != Some(0) {
            acc = acc.concat_along(convert(next, dtype), axis);
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use luminal::prelude::petgraph::algo::has_path_connecting;
    use luminal::prelude::*;

    use super::super::test_support::*;
    use super::super::{Translation, translate};
    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, DimExpr, DimSize, ExprValue, IntArg, RangeConstraint, TensorName, TensorsArg,
    };

    const COMPLEX_FLOAT: u32 = 10;

    fn program(tensors: &[(&str, u32, &[i64])], names: &[&str], axis: Option<i64>) -> ParsedPT2 {
        let mut args = vec![input(
            "tensors",
            Argument::Tensors(TensorsArg {
                as_tensors: names
                    .iter()
                    .map(|name| TensorName {
                        name: name.to_string(),
                    })
                    .collect(),
            }),
        )];
        if let Some(axis) = axis {
            args.push(scalar_int("dim", axis));
        }
        parsed_nodes(
            vec![node("torch.ops.aten.cat.default", args, &["y"])],
            tensors,
            names,
            &["y"],
        )
    }

    // Check the recorded value, not just the independently declared output metadata.
    fn assert_output(t: &Translation, shape: &[usize], dtype: DType) {
        assert_eq!(
            t.graph
                .logical
                .value_dims(t.outputs[0].tensor)
                .iter()
                .map(|d| d.to_usize().unwrap())
                .collect::<Vec<_>>(),
            shape
        );
        assert_eq!(t.graph.logical.petgraph()[t.outputs[0].tensor].dtype, dtype);
    }

    #[test]
    fn rank_one_empty_before_after_and_between_higher_rank_operands() {
        for dtype in [FLOAT, COMPLEX_FLOAT] {
            for axis in [2, -2] {
                for names in [
                    vec!["empty", "a", "b"],
                    vec!["a", "empty", "b"],
                    vec!["a", "b", "empty"],
                ] {
                    let parsed = program(
                        &[
                            ("empty", dtype, &[0]),
                            ("a", dtype, &[1, 2, 3, 4]),
                            ("b", dtype, &[1, 2, 2, 4]),
                            ("y", dtype, &[1, 2, 5, 4]),
                        ],
                        &names,
                        Some(axis),
                    );
                    let t = translate(&parsed).unwrap();
                    let shape = if dtype == FLOAT {
                        vec![1, 2, 5, 4]
                    } else {
                        vec![1, 2, 5, 4, 2]
                    };
                    assert_output(&t, &shape, DType::F32);
                }
            }
        }
    }

    #[test]
    fn all_rank_one_empties_preserve_torch_legacy_axis_behavior() {
        for dtype in [FLOAT, COMPLEX_FLOAT] {
            for axis in [None, Some(-20), Some(-1), Some(0), Some(20)] {
                let t = translate(&program(
                    &[("a", dtype, &[0]), ("b", dtype, &[0]), ("y", dtype, &[0])],
                    &["a", "b"],
                    axis,
                ))
                .unwrap();
                assert_output(&t, if dtype == FLOAT { &[0] } else { &[0, 2] }, DType::F32);
            }
        }
    }

    #[test]
    fn empty_operands_still_participate_in_dtype_promotion() {
        for shape in [&[0][..], &[2, 0][..]] {
            for names in [["empty", "a"], ["a", "empty"]] {
                let t = translate(&program(
                    &[
                        ("empty", FLOAT, shape),
                        ("a", HALF, &[2, 3]),
                        ("y", FLOAT, &[2, 3]),
                    ],
                    &names,
                    Some(-1),
                ))
                .unwrap();
                assert_output(&t, &[2, 3], DType::F32);
            }
        }
        let t = translate(&program(
            &[("a", HALF, &[0]), ("b", FLOAT, &[0]), ("y", FLOAT, &[0])],
            &["a", "b"],
            None,
        ))
        .unwrap();
        assert_output(&t, &[0], DType::F32);
    }

    #[test]
    fn mixed_nonempty_dtypes_are_promoted_before_concatenation() {
        let t = translate(&program(
            &[("a", HALF, &[2]), ("b", FLOAT, &[3]), ("y", FLOAT, &[5])],
            &["a", "b"],
            None,
        ))
        .unwrap();
        assert_output(&t, &[5], DType::F32);
    }

    #[test]
    fn mixed_real_and_complex_empties_preserve_component_promotion() {
        for (empty_dtype, value_dtype) in [(FLOAT, 9), (COMPLEX_FLOAT, FLOAT)] {
            for names in [["empty", "a"], ["a", "empty"]] {
                let t = translate(&program(
                    &[
                        ("empty", empty_dtype, &[0]),
                        ("a", value_dtype, &[2, 3]),
                        ("y", COMPLEX_FLOAT, &[2, 3]),
                    ],
                    &names,
                    Some(-1),
                ))
                .unwrap();
                assert_output(&t, &[2, 3, 2], DType::F32);
            }
        }
    }

    #[test]
    fn higher_rank_empties_keep_their_geometry() {
        for dtype in [FLOAT, COMPLEX_FLOAT] {
            // Empty on a non-concatenated axis: the concat axis must still grow.
            let t = translate(&program(
                &[
                    ("a", dtype, &[0, 2]),
                    ("b", dtype, &[0, 3]),
                    ("y", dtype, &[0, 5]),
                ],
                &["a", "b"],
                Some(1),
            ))
            .unwrap();
            assert_output(
                &t,
                if dtype == FLOAT { &[0, 5] } else { &[0, 5, 2] },
                DType::F32,
            );
            // Empty on the concat axis, including a rank-one sentinel.
            let t = translate(&program(
                &[
                    ("e", dtype, &[0]),
                    ("a", dtype, &[2, 0]),
                    ("b", dtype, &[2, 0]),
                    ("y", dtype, &[2, 0]),
                ],
                &["e", "a", "b"],
                Some(-1),
            ))
            .unwrap();
            assert_output(
                &t,
                if dtype == FLOAT { &[2, 0] } else { &[2, 0, 2] },
                DType::F32,
            );
        }
    }

    fn assert_error(parsed: ParsedPT2, expected: &str) {
        let error = match translate(&parsed) {
            Ok(_) => panic!("invalid cat unexpectedly translated"),
            Err(error) => format!("{error:#}"),
        };
        assert!(error.contains(expected), "{error}");
    }

    #[test]
    fn invalid_ranks_axes_and_nonconcat_extents_return_errors() {
        for dtype in [FLOAT, COMPLEX_FLOAT] {
            assert_error(program(&[("y", dtype, &[0])], &[], None), "cat");
            assert_error(
                program(&[("a", dtype, &[]), ("y", dtype, &[])], &["a"], None),
                "cat",
            );
            for axis in [-3, 2] {
                assert_error(
                    program(
                        &[
                            ("e", dtype, &[0]),
                            ("a", dtype, &[2, 3]),
                            ("y", dtype, &[2, 3]),
                        ],
                        &["e", "a"],
                        Some(axis),
                    ),
                    "axis",
                );
            }
            assert_error(
                program(
                    &[
                        ("a", dtype, &[1]),
                        ("b", dtype, &[2, 3]),
                        ("y", dtype, &[2, 4]),
                    ],
                    &["a", "b"],
                    Some(0),
                ),
                "rank",
            );
            // Cannot drop a higher-rank empty before checking its other extents.
            assert_error(
                program(
                    &[
                        ("a", dtype, &[3, 0]),
                        ("b", dtype, &[2, 4]),
                        ("y", dtype, &[2, 4]),
                    ],
                    &["a", "b"],
                    Some(1),
                ),
                "dimension",
            );
            assert_error(
                program(
                    &[
                        ("a", dtype, &[0, 3]),
                        ("b", dtype, &[2, 3]),
                        ("y", dtype, &[2, 6]),
                    ],
                    &["a", "b"],
                    Some(1),
                ),
                "dimension",
            );
        }
    }

    fn symbolic(expr: &str, hint: i64) -> DimSize {
        DimSize::Expr(DimExpr {
            as_expr: ExprValue {
                expr_str: expr.to_string(),
                hint: Some(Box::new(Argument::Int(IntArg { as_int: hint }))),
            },
        })
    }

    #[test]
    fn zero_hint_never_eliminates_symbolic_operands() {
        for dtype in [FLOAT, COMPLEX_FLOAT] {
            for rank in [1, 2] {
                for names in [["a", "b"], ["b", "a"]] {
                    let mut p = program(
                        &[("a", dtype, &[1]), ("b", dtype, &[2]), ("y", dtype, &[3])],
                        &names,
                        Some(-1),
                    );
                    let graph = &mut p.program.graph_module.graph;
                    graph.tensor_values.get_mut("a").unwrap().sizes =
                        vec![symbolic("Symbol('s0', integer=True, nonnegative=True)", 0)];
                    graph.tensor_values.get_mut("y").unwrap().sizes = vec![symbolic(
                        "Add(Symbol('s0', integer=True, nonnegative=True), Integer(2))",
                        2,
                    )];
                    if rank == 2 {
                        for meta in graph.tensor_values.values_mut() {
                            meta.sizes.insert(0, sizes(&[3]).remove(0));
                        }
                    }
                    p.program.range_constraints.insert(
                        "s0".to_string(),
                        RangeConstraint {
                            min_val: Some(0),
                            max_val: Some(5),
                        },
                    );
                    let t = translate(&p).unwrap();
                    let s = t.symbols["s0"];
                    assert_eq!(t.dims[&s], 0);
                    let out = t.outputs[0].tensor;
                    let a = t
                        .inputs
                        .iter()
                        .find(|i| i.graph_name == "a")
                        .unwrap()
                        .tensor;
                    assert!(
                        has_path_connecting(t.graph.logical.petgraph(), a, out, None),
                        "symbolic input must still contribute data"
                    );
                    let extent = t.graph.logical.value_dims(out)[rank - 1];
                    assert!(extent.to_usize().is_none());
                    for n in [0, 1, 5] {
                        let dims = [(s, n)].into_iter().collect();
                        assert_eq!(extent.exec(&dims), Some(n + 2));
                    }
                }
            }
        }
    }
}
