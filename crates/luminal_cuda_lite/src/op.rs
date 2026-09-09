//! Borrow CUDA execution traits from operations carried by the shared IR.
//!
//! Each executable op supplies its own interface adapter through
//! `BufferTensorIrOp::runtime_interface`. These adapters only recover a trait
//! reference; code generation and execution belong to the traits themselves.
//! There is no global operation table, label matching, allocation, or op clone.

use luminal::buffer_tensor_ir::BufferTensorIrOp;

use crate::{host::HostOp, kernels::KernelOp};

pub enum CudaOpInterface {
    Kernel(fn(&dyn BufferTensorIrOp) -> &dyn KernelOp),
    Host(fn(&dyn BufferTensorIrOp) -> &dyn HostOp),
}

impl CudaOpInterface {
    /// Return this adapter from a kernel op's `runtime_interface` method.
    pub fn kernel<T: KernelOp + 'static>() -> &'static Self {
        &Self::Kernel(|op| {
            op.as_any()
                .downcast_ref::<T>()
                .expect("kernel interface must describe its concrete op type")
        })
    }

    /// Return this adapter from a host op's `runtime_interface` method.
    pub fn host<T: HostOp + 'static>() -> &'static Self {
        &Self::Host(|op| {
            op.as_any()
                .downcast_ref::<T>()
                .expect("host interface must describe its concrete op type")
        })
    }
}

pub fn as_kernel_op(op: &dyn BufferTensorIrOp) -> Option<&dyn KernelOp> {
    match op.runtime_interface()?.downcast_ref::<CudaOpInterface>()? {
        CudaOpInterface::Kernel(borrow) => Some(borrow(op)),
        CudaOpInterface::Host(_) => None,
    }
}

pub fn as_host_op(op: &dyn BufferTensorIrOp) -> Option<&dyn HostOp> {
    match op.runtime_interface()?.downcast_ref::<CudaOpInterface>()? {
        CudaOpInterface::Host(borrow) => Some(borrow(op)),
        CudaOpInterface::Kernel(_) => None,
    }
}
