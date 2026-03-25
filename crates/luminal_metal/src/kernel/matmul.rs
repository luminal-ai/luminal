use super::{MetalMulInfo, MetalSumReduceInfo};
use luminal::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalMatmulFamily {
    #[default]
    Naive,
    RegularTiled,
}

#[derive(Debug, Clone)]
pub struct MatmulDescriptor {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub batch_shape: Vec<Expression>,
    pub lhs_strides: Vec<Expression>,
    pub rhs_strides: Vec<Expression>,
    pub out_strides: Vec<Expression>,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

impl MatmulDescriptor {
    pub fn from_mul_and_sum(
        mul_info: &MetalMulInfo,
        sum_info: &MetalSumReduceInfo,
    ) -> Option<Self> {
        let zero = Expression::from(0);
        let z = Expression::from('z');

        let is_simple_2d_matmul = mul_info.shape.len() == 3
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

        if !is_simple_2d_matmul {
            return None;
        }

        Some(Self {
            m: sum_info.shape[0],
            n: sum_info.shape[1],
            k: sum_info.iters,
            batch_shape: Vec::new(),
            lhs_strides: mul_info.a_strides.clone(),
            rhs_strides: mul_info.b_strides.clone(),
            out_strides: sum_info.strides.clone(),
            transpose_lhs: false,
            transpose_rhs: false,
        })
    }
}

#[derive(Debug, Clone)]
pub struct MatmulPlan {
    pub family: MetalMatmulFamily,
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lda: Expression,
    pub ldb: Expression,
    pub ldd: Expression,
    pub batch_size: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
    pub bm: u16,
    pub bn: u16,
    pub bk: u16,
    pub wm: u16,
    pub wn: u16,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MetalMatmulPlanner;

impl MetalMatmulPlanner {
    pub fn plan(&self, desc: &MatmulDescriptor) -> MatmulPlan {
        let family = if desc.batch_shape.is_empty()
            && desc.m.as_num().is_some_and(|m| m >= 32)
            && desc.n.as_num().is_some_and(|n| n >= 32)
            && desc.k.as_num().is_some_and(|k| k >= 32)
        {
            MetalMatmulFamily::RegularTiled
        } else {
            MetalMatmulFamily::Naive
        };
        MatmulPlan {
            family,
            m: desc.m,
            n: desc.n,
            k: desc.k,
            lda: desc.lhs_strides[0],
            ldb: desc.rhs_strides[2],
            ldd: desc.out_strides[0],
            batch_size: 1,
            batch_stride_a: 0,
            batch_stride_b: 0,
            batch_stride_d: 0,
            bm: 16,
            bn: 16,
            bk: 8,
            wm: 2,
            wn: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_recovers_simple_2d_matmul() {
        let mul = MetalMulInfo {
            shape: vec![
                Expression::from(4),
                Expression::from(8),
                Expression::from(16),
            ],
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
        let sum = MetalSumReduceInfo {
            shape: vec![Expression::from(4), Expression::from(8)],
            strides: vec![Expression::from('z') * 8, Expression::from('z')],
            iters: Expression::from(16),
            iter_stride: Expression::from('z'),
        };

        let desc = MatmulDescriptor::from_mul_and_sum(&mul, &sum).unwrap();
        assert_eq!(desc.m, Expression::from(4));
        assert_eq!(desc.n, Expression::from(8));
        assert_eq!(desc.k, Expression::from(16));
    }

    #[test]
    fn planner_keeps_small_problems_on_naive_path() {
        let desc = MatmulDescriptor {
            m: Expression::from(4),
            n: Expression::from(8),
            k: Expression::from(16),
            batch_shape: Vec::new(),
            lhs_strides: vec![
                Expression::from('z') * 16,
                Expression::from(0),
                Expression::from('z'),
            ],
            rhs_strides: vec![
                Expression::from(0),
                Expression::from('z'),
                Expression::from('z') * 8,
            ],
            out_strides: vec![Expression::from('z') * 8, Expression::from('z')],
            transpose_lhs: false,
            transpose_rhs: false,
        };

        let planner = MetalMatmulPlanner;
        let plan = planner.plan(&desc);
        assert_eq!(plan.family, MetalMatmulFamily::Naive);
        assert_eq!(plan.bm, 16);
        assert_eq!(plan.bn, 16);
        assert_eq!(plan.bk, 8);
        assert_eq!(plan.wm, 2);
        assert_eq!(plan.wn, 2);
        assert_eq!(plan.lda, Expression::from('z') * 16);
        assert_eq!(plan.ldb, Expression::from('z') * 8);
        assert_eq!(plan.ldd, Expression::from('z') * 8);
    }

    #[test]
    fn planner_promotes_large_problems_to_regular_tiled() {
        let desc = MatmulDescriptor {
            m: Expression::from(64),
            n: Expression::from(64),
            k: Expression::from(64),
            batch_shape: Vec::new(),
            lhs_strides: vec![
                Expression::from('z') * 64,
                Expression::from(0),
                Expression::from('z'),
            ],
            rhs_strides: vec![
                Expression::from(0),
                Expression::from('z'),
                Expression::from('z') * 64,
            ],
            out_strides: vec![Expression::from('z') * 64, Expression::from('z')],
            transpose_lhs: false,
            transpose_rhs: false,
        };

        let planner = MetalMatmulPlanner;
        let plan = planner.plan(&desc);
        assert_eq!(plan.family, MetalMatmulFamily::RegularTiled);
        assert_eq!(plan.bm, 16);
        assert_eq!(plan.bn, 16);
        assert_eq!(plan.bk, 8);
        assert_eq!(plan.wm, 2);
        assert_eq!(plan.wn, 2);
    }
}
