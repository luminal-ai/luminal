use super::{CpuKernelOp, CpuMulInfo, CpuSumReduceInfo};
use luminal::{dtype::DType, egglog_utils::{
    SerializedEGraph, api::{Rule, SortDef, eq, rule, sort, union, v}, base::{EXPRESSION, IR, dtype, iter, new_op_call, op_term}
}, op::*, prelude::*};
use matrixmultiply::sgemm;


// ---------------------------------------------------------------------------------------------------
// MatMulDescriptor
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct CpuMatmulDescriptor {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lda: Expression,
    pub ldb: Expression,
    pub ldd: Expression,
}

impl CpuMatmulDescriptor {
    pub fn from_mul_and_sum(
        mul_info: &CpuMulInfo,
        sum_info: &CpuSumReduceInfo,
    ) -> Option<Self> {
        let zero = Expression::from(0);
        let z = Expression::from('z');

        let ok = mul_info.shape.len() == 3
            && sum_info.shape.len() == 2
            && mul_info.a_strides.len() == 3
            && mul_info.b_strides.len() == 3
            && sum_info.strides.len() == 2
            && mul_info.shape[0] == sum_info.shape[0]
            && mul_info.shape[1] == sum_info.shape[1]
            && mul_info.shape[2] == sum_info.iters
            && mul_info.a_strides[1] == zero
            && mul_info.a_strides[2] == z
            && mul_info.b_strides[0] == zero
            && mul_info.b_strides[1] == z
            && sum_info.strides[1] == z
            && sum_info.iter_stride == z;

        if !ok { return None; }

        Some(Self { m: sum_info.shape[0], n: sum_info.shape[1], k: sum_info.iters, lda: mul_info.a_strides[0].clone(), ldb: mul_info.b_strides[2].clone(), ldd: sum_info.strides[0].clone() })
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuMatmul op
// 
// The rewrite fires when egglog sees CPUSumReduce.
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Clone, Default)]
pub struct CpuMatmul {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lda: Expression,
    pub ldb: Expression,
    pub ldd: Expression,
}

impl EgglogOp for CpuMatmul {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        sort(IR, "CpuMatmul", &[
            ("m", EXPRESSION),
            ("n", EXPRESSION),
            ("k", EXPRESSION),
            ("lhs", IR),
            ("lda", EXPRESSION),
            ("rhs", IR),
            ("ldb", EXPRESSION),
            ("ldd", EXPRESSION),
        ])
    }
    fn rewrites(&self) -> Vec<luminal::egglog_utils::api::Rule> {
        let sum_sort = super::ops::CpuSumReduce::default().sort();
        let mul_sort = super::ops::CpuMul::default().sort();
        let matmul_sort = self.sort();

        let (sum_args, sum_match) = new_op_call(&sum_sort, &["inp"]);
        let (mul_args, mul_match) = new_op_call(&mul_sort, &["lhs", "rhs"]);

        let matmul_args = [
            ("m".to_string(),   sum_args["out_shape"].clone()), 
            ("n".to_string(),   sum_args["out_shape"].clone()),
            ("k".to_string(),   sum_args["iters"].clone()),
            ("lhs".to_string(), mul_args["lhs"].clone()),
            ("lda".to_string(), mul_args["a_strides"].clone()), 
            ("rhs".to_string(), mul_args["rhs"].clone()),
            ("ldb".to_string(), mul_args["b_strides"].clone()), 
            ("ldd".to_string(), sum_args["out_stride"].clone()),
        ];
        let matmul_op = op_term(matmul_sort.call(matmul_args), mul_args["__inputs"].clone());

        let dt = v("?__dt");
        vec![
            rule(union(sum_match, matmul_op.clone())).fact(eq(mul_match, sum_args["inp"].clone())).set(dtype(matmul_op), dt.clone()).fact(eq(dt, dtype(mul_args["lhs"].clone())))
        ]
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn extract<'a>(
            &'a self,
            egraph: &'a SerializedEGraph,
            kind_children: &[&'a ENodeId],
            input_enodes: Vec<&'a ENodeId>,
            list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
            expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
        ) -> (luminal::op::LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};

        let out_shape = extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let iters = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let a_strides = extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
        let b_strides = extract_expr_list(egraph, kind_children[6], list_cache, expr_cache).unwrap();
        let out_strides = extract_expr_list(egraph, kind_children[7], list_cache, expr_cache).unwrap();

        let zero = Expression::from(0);
        let z = Expression::from('z');

        let valid = out_shape.len() == 2
            && a_strides.len() == 3
            && b_strides.len() >= 3
            && out_strides.len() >=2
            && a_strides[1] == zero
            && a_strides[2] == z
            && b_strides[0] == zero
            && b_strides[1] == z
            && out_strides[1] == z;

        if !valid {
            panic!("CpuMatmul::extract: stride pattern is not a plain 2D matmul");
        }

        let matmul = CpuMatmul {
            m: out_shape[0].clone(),
            n: out_shape[1].clone(),
            k: iters,
            lda: a_strides[0].clone(),
            ldb: b_strides[2].clone(),
            ldd: out_strides[0].clone(),
        };

        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(matmul)),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuMatmul {
    fn output_size(&self) -> Expression {
        self.m.clone() * self.n.clone()
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        assert_eq!(inputs.len(), 2, "CpuMatmul expected exactly 2 inputs");

        let mut map = dyn_map.clone();
        map.entry('z').or_insert(1);

        let m = self.m.exec(&map).expect("CpuMatmul: m unresolved") as usize;
        let n = self.n.exec(&map).expect("CpuMatmul: n unresolved") as usize;
        let k = self.k.exec(&map).expect("CpuMatmul: k unresolved") as usize;
        let lda = self.lda.exec(&map).expect("CpuMatmul: lda unresolved") as usize;
        let ldb = self.ldb.exec(&map).expect("CpuMatmul: ldb unresolved") as usize;
        let ldd = self.ldd.exec(&map).expect("CpuMatmul: ldd unresolved") as usize;

        let a = inputs[0].0;
        let b = inputs[1].0;

        let mut out = vec![0.0f32; m * n];

        unsafe {
            sgemm(m, k, n, 1.0, a.as_ptr(), lda as isize, 1, b.as_ptr(), ldb as isize, 1, 0.0, out.as_mut_ptr(), ldd as isize, 1);
        }

        out
    }

    fn is_matmul(&self) -> bool {
        true
    }
}


// ---------------------------------------------------------------------------------------------------
// Unit tests for the descriptor
// ---------------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_recognises_simple_2d_matmul() {
        let mul = CpuMulInfo {
            shape: vec![Expression::from(4), Expression::from(8), Expression::from(16)],
            a_strides: vec![
                Expression::from('z') * 16,
                Expression::from(0),
                Expression::from('z'),
            ],
            b_strides: vec![
                Expression::from(0),
                Expression::from('z'),
                Expression::from('z') * 8,
            ],
            output_strides: vec![
                Expression::from('z') * 16,
                Expression::from('z') * 8,
                Expression::from('z'),
            ],
        };
        let sum = CpuSumReduceInfo {
            shape:       vec![Expression::from(4), Expression::from(8)],
            strides:     vec![Expression::from('z') * 8, Expression::from('z')],
            iters:       Expression::from(16),
            iter_stride: Expression::from('z'),
        };
 
        let desc = CpuMatmulDescriptor::from_mul_and_sum(&mul, &sum).unwrap();
        assert_eq!(desc.m, Expression::from(4));
        assert_eq!(desc.n, Expression::from(8));
        assert_eq!(desc.k, Expression::from(16));
    }

    #[test]
    fn descriptor_rejects_batched_matmul() {
        // batch dim would make a_strides[1] non-zero → should return None
        let mul = CpuMulInfo {
            shape: vec![Expression::from(2), Expression::from(4), Expression::from(8)],
            a_strides: vec![
                Expression::from('z') * 8,
                Expression::from('z') * 4,  // non-zero → batched
                Expression::from('z'),
            ],
            b_strides: vec![
                Expression::from(0),
                Expression::from('z'),
                Expression::from('z') * 4,
            ],
            output_strides: vec![
                Expression::from('z') * 8,
                Expression::from('z') * 4,
                Expression::from('z'),
            ],
        };
        let sum = CpuSumReduceInfo {
            shape:       vec![Expression::from(2), Expression::from(4)],
            strides:     vec![Expression::from('z') * 4, Expression::from('z')],
            iters:       Expression::from(8),
            iter_stride: Expression::from('z'),
        };
        assert!(CpuMatmulDescriptor::from_mul_and_sum(&mul, &sum).is_none());
    }
}

