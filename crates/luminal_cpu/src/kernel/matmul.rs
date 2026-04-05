use super::{CpuKernelOp, CpuMulInfo, CpuSumReduceInfo};
use luminal::{dtype::DType, op::EgglogOp, prelude::*};
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
            && mul_info.b_strides[1] == zero
            && mul_info.b_strides[2] == z
            && sum_info.strides[1] == z
            && sum_info.iter_strides == z;

        if !ok { return None; }

        Some(Self { m: sum_info.shape[0], n: sum_info.shape[1], k: sum_info.iters, lda: mul_info.a_strides[0].clone(), ldb: mul_info.b_strides[2].clone(), ldd: sum_info.strides[0].clone() })
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuMatmul op
// 
// This is the fused mode that replaces Mul+SumReduce in the LLIR graph
// It uses the 'matrixmultiply' crate for a well-optimized BLAS-like sgemm
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
        // CpuMatmul is injected by fuse_matmuls, not via egglog rewrites,
        // so we can return placeholder sort.
        luminal::hlir::reduce_sort("CpuMatmul")
    }
    fn rewrites(&self) -> Vec<luminal::egglog_utils::api::Rule> {
        vec![]
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn extract<'a>(
            &'a self,
            _egraph: &'a SerializedEGraph,
            _kind_children: &[&'a ENodeId],
            _input_enodes: Vec<&'a ENodeId>,
            _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
            _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
        ) -> (luminal::op::LLIROp, Vec<&'a ENodeId>) {
        panic!("CpuMatmul::extract should never be called")
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

        let m = self.m.exec(dyn_map).expect("CpuMatmul: m unresolved") as usize;
        let n = self.n.exec(dyn_map).expect("CpuMatmul: n unresolved") as usize;
        let k = self.k.exec(dyn_map).expect("CpuMatmul: k unresolved") as usize;
        let lda = self.lda.exec(dyn_map).expect("CpuMatmul: lda unresolved") as usize;
        let ldb = self.ldb.exec(dyn_map).expect("CpuMatmul: ldb unresolved") as usize;
        let ldd = self.ldd.exec(dyn_map).expect("CpuMatmul: ldd unresolved") as usize;

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
            iter_strides: Expression::from('z'),
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
            iter_strides: Expression::from('z'),
        };
        assert!(CpuMatmulDescriptor::from_mul_and_sum(&mul, &sum).is_none());
    }
}

