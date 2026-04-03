mod matmul;
mod ops;

pub use matmul::*;
pub use ops::*;

use luminal::dtype::DType;
use luminal::op::EgglogOp;
use luminal::prelude::*;


// ---------------------------------------------------------------------------------------------------
// CpuKernelOp trait
// 
// Every CPU op must implement this.
// Adding a new op in future:
//  1. Add the struct + EgglogOP + CpuKernelOp impls in ops.rs
//  2. Add the type to the CpuOps tuple below
// ---------------------------------------------------------------------------------------------------
pub trait CpuKernelOp: EgglogOp {
    /// How many f32 elements does the output contain?
    /// Uses the same symbolic Expression system as the rest code base so that
    /// dynamic dimensions are resolved at runtime
    fn output_size(&self) -> Expression;

    /// dtype for output. Defaults to the dtype of the first input - correct
    /// for every op implementation here. Override for Cast-style ops.
    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    /// Run the op. 'inputs' is a slice of (data, dtype) pairs, one per incoming
    /// edge. Returns the output as a plain Vec<f32>.
    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32>;

    // --- hooks used by fuse_matmul()
    fn mul_info(&self) -> Option<CpuMulInfo> { None }
    fn sum_reduce_info(&self) -> Option<CpuSumReduceInfo> { None }
    fn is_matmul(&self) -> bool { false }
}

#[derive(Debug, Clone)]
pub struct CpuMulInfo {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub output_strides: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct CpuSumReduceInfo {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_strides: Expression,
}

// ---------------------------------------------------------------------------------------------------
// CpuUps - the set of ops that this backend understands
// ---------------------------------------------------------------------------------------------------
pub type CpuOps = (
    CpuAdd,
    CpuMul,
    CpuMod,
    CpuLessThan,
    CpuExp2,
    CpuLog2,
    CpuSin,
    CpuSqrt,
    CpuRecip,
    // CpuSumReduce,
    // CpuMaxReduce,
    // CpuConstant,
    // CpuIota,
    // CpuGather,
    // CpuMatmul,
);

// Glue macro to Luminal's op
luminal::impl_into_ops!(CpuKernelOp);
