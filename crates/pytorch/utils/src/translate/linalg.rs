//! Real-valued triangular substitution and Cholesky helper decompositions.
//! The matrix order is specialized; batch and RHS dimensions remain symbolic.
use super::{Translator, util};
use crate::pt2_schema::Node;
use anyhow::{Context, Result, ensure};
use luminal::prelude::*;

fn transpose_matrix(value: GraphTensor) -> GraphTensor {
    value.transpose(value.rank() - 2, value.rank() - 1)
}

impl Translator<'_> {
    pub(super) fn translate_lu_unpack(&mut self, node: &Node) -> Result<()> {
        let lu = self.raw_operand(&node.inputs[0])?;
        let pivots = self.raw_operand(&node.inputs[1])?;
        let rank = lu.rank();
        ensure!(rank >= 2, "LU unpack requires a matrix");
        let dims = lu.dims();
        let m = dims[rank - 2]
            .to_usize()
            .context("LU unpack requires static row count")?;
        let n = dims[rank - 1]
            .to_usize()
            .context("LU unpack requires static column count")?;
        let k = m.min(n);
        let mut vector = dims[..rank - 2].to_vec();
        vector.push(m.into());
        let mut perm = self.cx.iota(vector.clone(), |c| c[rank - 2]);
        let row = self.cx.iota(vector.clone(), |c| c[rank - 2]);
        let one_based = self
            .cx
            .iota(vector.clone(), |c| c[rank - 2] + 1)
            .cast(pivots.dtype);
        let zero = self.cx.constant_i32(0).expand_rhs(vector.clone());
        for i in 0..k {
            let pivot = pivots
                .slice_along(i..i + 1, rank - 2)
                .expand(vector.clone());
            let pivot_row = one_based.eq(pivot);
            // Select by one-based coordinate equality instead of doing
            // unchecked arithmetic on caller-provided pivot values.
            let at_pivot = self
                .select(pivot_row, perm, zero)
                .sum(vec![rank - 2])
                .expand_dim(rank - 2, m);
            let at_i = perm.slice_along(i..i + 1, rank - 2).expand(vector.clone());
            let index = self.cx.constant_i32(i).expand_rhs(vector.clone());
            let swapped = self.select(pivot_row, at_i, perm);
            perm = self.select(row.eq(index), at_pivot, swapped);
        }
        let mut pdims = dims[..rank - 2].to_vec();
        pdims.extend([IntExpr::from(m), IntExpr::from(m)]);
        let rows = self.cx.iota(pdims.clone(), |c| c[rank - 2]);
        let p = rows
            .eq(perm.unsqueeze(rank - 2).expand(pdims))
            .cast(lu.dtype);
        let l = lu.slice_along(0..k, rank - 1);
        let rows = self.cx.iota(l.dims(), |c| c[rank - 2]);
        let cols = self.cx.iota(l.dims(), |c| c[rank - 1]);
        let zero = self.constant_like(l, 0.0);
        let one = self.constant_like(l, 1.0);
        let lower = self.select(rows.gt(cols), l, zero);
        let lower = self.select(rows.eq(cols), one, lower);
        let u = lu.slice_along(0..k, rank - 2);
        let rows = self.cx.iota(u.dims(), |c| c[rank - 2]);
        let cols = self.cx.iota(u.dims(), |c| c[rank - 1]);
        let zero = self.constant_like(u, 0.0);
        let upper = self.select(rows.le(cols), u, zero);
        let empty = self
            .typed_scalar(0.0, lu.dtype)
            .expand_rhs(vec![IntExpr::from(0)]);
        let data = self.named_bool_arg(node, "unpack_data").unwrap_or(true);
        let permutation = self.named_bool_arg(node, "unpack_pivots").unwrap_or(true);
        self.bind_outputs(
            node,
            vec![
                if permutation { p } else { empty },
                if data { lower } else { empty },
                if data { upper } else { empty },
            ],
        )
    }
    pub(super) fn translate_legacy_triangular_solve(&mut self, node: &Node) -> Result<()> {
        let b = self.operand(&node.inputs[0])?;
        let a = self.operand(&node.inputs[1])?;
        let upper = self.named_bool_arg(node, "upper").unwrap_or(true);
        let transpose = self.named_bool_arg(node, "transpose").unwrap_or(false);
        let unit = self.named_bool_arg(node, "unitriangular").unwrap_or(false);
        let solution = self.triangular_solve(
            if transpose { transpose_matrix(a) } else { a },
            b,
            upper != transpose,
            true,
            unit,
        )?;
        self.bind_outputs(node, vec![solution, a])
    }
    fn triangular_solve(
        &mut self,
        mut a: GraphTensor,
        mut b: GraphTensor,
        mut upper: bool,
        left: bool,
        unit: bool,
    ) -> Result<GraphTensor> {
        ensure!(
            a.rank() >= 2 && b.rank() >= 2,
            "triangular solve requires matrix operands"
        );
        ensure!(
            a.dtype == b.dtype,
            "triangular solve operand dtypes must agree"
        );
        ensure!(
            matches!(a.dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64),
            "triangular solve requires floating point operands"
        );
        if !left {
            a = transpose_matrix(a);
            b = transpose_matrix(b);
            upper = !upper;
        }
        let n = a.dims()[a.rank() - 1]
            .to_usize()
            .context("triangular solve requires a static matrix order")?;
        ensure!(
            a.dims()[a.rank() - 2].to_usize() == Some(n),
            "triangular solve matrix must be square"
        );
        ensure!(
            b.dims()[b.rank() - 2].to_usize() == Some(n),
            "triangular solve RHS order differs"
        );
        let rank = a.rank().max(b.rank());
        while a.rank() < rank {
            a = a.unsqueeze(0);
        }
        while b.rank() < rank {
            b = b.unsqueeze(0);
        }
        let mut ad = a.dims();
        let mut bd = b.dims();
        for axis in 0..rank - 2 {
            let dim = if ad[axis].to_usize() == Some(1) {
                bd[axis]
            } else if bd[axis].to_usize() == Some(1) || util::same_dim(ad[axis], bd[axis]) {
                ad[axis]
            } else {
                anyhow::bail!("triangular solve batch dimensions do not broadcast")
            };
            ad[axis] = dim;
            bd[axis] = dim;
        }
        a = a.expand(ad);
        b = b.expand(bd);
        let dtype = b.dtype;
        if matches!(dtype, DType::F16 | DType::Bf16) {
            a = a.cast(DType::F32);
            b = b.cast(DType::F32);
        }
        if n == 0 {
            return Ok(if left {
                b.cast(dtype)
            } else {
                transpose_matrix(b).cast(dtype)
            });
        }
        let mut rows: Vec<Option<GraphTensor>> = vec![None; n];
        for step in 0..n {
            let i = if upper { n - 1 - step } else { step };
            let coefficients = a.slice_along(i..i + 1, rank - 2).squeeze(rank - 2);
            let mut residual = b.slice_along(i..i + 1, rank - 2).squeeze(rank - 2);
            for (j, row) in rows.iter().enumerate() {
                if let Some(row) = row {
                    let coefficient = coefficients
                        .slice_along(j..j + 1, rank - 2)
                        .squeeze(rank - 2)
                        .expand_dim(rank - 2, residual.dims()[rank - 2]);
                    residual -= coefficient * *row;
                }
            }
            if !unit {
                let diagonal = coefficients
                    .slice_along(i..i + 1, rank - 2)
                    .squeeze(rank - 2)
                    .expand_dim(rank - 2, residual.dims()[rank - 2]);
                residual /= diagonal;
            }
            rows[i] = Some(residual);
        }
        let rows: Vec<_> = rows.into_iter().map(Option::unwrap).collect();
        let result = self.cx.stack(&rows, rank - 2).cast(dtype);
        Ok(if left {
            result
        } else {
            transpose_matrix(result)
        })
    }

    pub(super) fn translate_triangular_solve(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = self.operand(&node.inputs[1])?;
        let upper = self
            .named_bool_arg(node, "upper")
            .context("triangular solve requires upper")?;
        self.triangular_solve(
            a,
            b,
            upper,
            self.named_bool_arg(node, "left").unwrap_or(true),
            self.named_bool_arg(node, "unitriangular").unwrap_or(false),
        )
    }

    fn cholesky_solve(
        &mut self,
        b: GraphTensor,
        factor: GraphTensor,
        upper: bool,
    ) -> Result<GraphTensor> {
        let (first, second) = if upper {
            (transpose_matrix(factor), factor)
        } else {
            (factor, transpose_matrix(factor))
        };
        let y = self.triangular_solve(first, b, false, true, false)?;
        self.triangular_solve(second, y, true, true, false)
    }

    pub(super) fn translate_cholesky_solve(&mut self, node: &Node) -> Result<GraphTensor> {
        let b = self.operand(&node.inputs[0])?;
        let factor = self.operand(&node.inputs[1])?;
        self.cholesky_solve(
            b,
            factor,
            self.named_bool_arg(node, "upper").unwrap_or(false),
        )
    }

    pub(super) fn translate_cholesky_inverse(&mut self, node: &Node) -> Result<GraphTensor> {
        let factor = self.operand(&node.inputs[0])?;
        ensure!(factor.rank() >= 2, "cholesky inverse requires a matrix");
        let n = factor.dims()[factor.rank() - 1]
            .to_usize()
            .context("cholesky inverse requires a static matrix order")?;
        let coordinate = self.cx.arange(n);
        let identity = coordinate
            .unsqueeze(1)
            .expand(vec![n, n])
            .eq(coordinate.unsqueeze(0).expand(vec![n, n]))
            .cast(factor.dtype);
        self.cholesky_solve(
            identity,
            factor,
            self.named_bool_arg(node, "upper").unwrap_or(false),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
    use rustc_hash::FxHashMap;

    fn run(t: &super::super::Translation, values: &[Vec<f32>]) -> Vec<f32> {
        let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
        let mut data = FxHashMap::default();
        for (input, values) in t.inputs.iter().zip(values) {
            data.insert(input.tensor, TypedBuffer::F32(values.clone()));
        }
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &harness_search_options(),
            )
            .unwrap();
        for (input, values) in t.inputs.iter().zip(values) {
            runtime.set_data(input.tensor, TypedBuffer::F32(values.clone()));
        }
        runtime.execute().unwrap();
        runtime.get_f32(t.outputs[0].tensor).unwrap().clone()
    }

    #[test]
    fn triangular_solve_handles_sides_triangles_unit_diagonals_and_batch_broadcasting() {
        for upper in [false, true] {
            for left in [false, true] {
                for unit in [false, true] {
                    let mut a = vec![f32::NAN; 9];
                    for i in 0..3 {
                        for j in 0..3 {
                            if (upper && j >= i) || (!upper && j <= i) {
                                a[i * 3 + j] = if i == j {
                                    if unit { f32::NAN } else { 2. + i as f32 }
                                } else {
                                    0.25
                                };
                            }
                        }
                    }
                    let x = vec![1., 2., 3., 4., 5., 6., -1., 0., 2., 3., -4., 2.];
                    let mut b = vec![0.; 12];
                    for batch in 0..2 {
                        for row in 0..if left { 3 } else { 2 } {
                            for col in 0..if left { 2 } else { 3 } {
                                let mut sum = 0.;
                                for k in 0..3 {
                                    let (i, j) = if left { (row, k) } else { (k, col) };
                                    if (upper && j >= i) || (!upper && j <= i) {
                                        let coefficient =
                                            if unit && i == j { 1. } else { a[i * 3 + j] };
                                        sum += coefficient
                                            * x[batch * 6
                                                + if left { k * 2 + col } else { row * 3 + k }];
                                    }
                                }
                                b[batch * 6 + if left { row * 2 + col } else { row * 3 + col }] =
                                    sum;
                            }
                        }
                    }
                    let shape = if left { vec![2, 3, 2] } else { vec![2, 2, 3] };
                    let t = translate_one(
                        node(
                            "torch.ops.aten.linalg_solve_triangular.default",
                            vec![
                                tensor_input("self", "a"),
                                tensor_input("B", "b"),
                                scalar_bool("upper", upper),
                                scalar_bool("left", left),
                                scalar_bool("unitriangular", unit),
                            ],
                            &["y"],
                        ),
                        &[
                            ("a", FLOAT, &[1, 3, 3]),
                            ("b", FLOAT, &shape),
                            ("y", FLOAT, &shape),
                        ],
                        &["a", "b"],
                        &["y"],
                    );
                    let actual = run(&t, &[a, b]);
                    for (got, want) in actual.iter().zip(&x) {
                        assert!((got - want).abs() < 1e-5, "{got} != {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn cholesky_helpers_solve_and_invert_both_factor_orientations() {
        for upper in [false, true] {
            let factor = if upper {
                vec![2., 1., f32::NAN, 2.]
            } else {
                vec![2., f32::NAN, 1., 2.]
            };
            let solve = translate_one(
                node(
                    "torch.ops.aten.cholesky_solve.default",
                    vec![
                        tensor_input("self", "b"),
                        tensor_input("input2", "l"),
                        scalar_bool("upper", upper),
                    ],
                    &["y"],
                ),
                &[
                    ("b", FLOAT, &[2, 1]),
                    ("l", FLOAT, &[2, 2]),
                    ("y", FLOAT, &[2, 1]),
                ],
                &["b", "l"],
                &["y"],
            );
            let got = run(&solve, &[vec![8., 12.], factor.clone()]);
            assert!(
                (got[0] - 1.).abs() < 1e-6 && (got[1] - 2.).abs() < 1e-6,
                "{got:?}"
            );
            let inverse = translate_one(
                node(
                    "torch.ops.aten.cholesky_inverse.default",
                    vec![tensor_input("self", "l"), scalar_bool("upper", upper)],
                    &["y"],
                ),
                &[("l", FLOAT, &[2, 2]), ("y", FLOAT, &[2, 2])],
                &["l"],
                &["y"],
            );
            let got = run(&inverse, &[factor]);
            for (got, want) in got.iter().zip([0.3125, -0.125, -0.125, 0.25]) {
                assert!((got - want).abs() < 1e-6, "{got} != {want}");
            }
        }
    }
}
