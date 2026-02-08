#![allow(unused)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::*;

pub mod ops;
pub use ops::Ops;

pub trait KernelOp: luminal::op::EgglogOp {
    #[allow(clippy::type_complexity)]
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    );

    /// Returns the output buffer size in elements.
    fn output_size(&self) -> Expression;

    /// Returns the number of bytes this kernel will load from global memory.
    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    /// Returns the number of bytes this kernel will store to global memory.
    fn bytes_stored(&self) -> Expression {
        0.into()
    }

    /// Returns the number of floating point operations this kernel performs.
    fn flops(&self) -> Expression {
        0.into()
    }

    /// Returns the name of this kernel for profiling display.
    fn kernel_name(&self) -> &'static str {
        "Unknown"
    }
}

luminal::impl_into_ops!(KernelOp);
