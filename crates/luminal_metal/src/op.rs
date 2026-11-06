use crate::{host::HostOp, kernels::KernelOp};
use luminal::buffer_tensor_ir::BufferTensorIrOp;
/// Borrow execution through the concrete operation, without label dispatch.
pub enum MetalOpInterface {
    Kernel(fn(&dyn BufferTensorIrOp) -> &dyn KernelOp),
    Host(fn(&dyn BufferTensorIrOp) -> &dyn HostOp),
}
impl MetalOpInterface {
    pub fn kernel<T: KernelOp + 'static>() -> &'static Self {
        &Self::Kernel(|op| {
            op.as_any()
                .downcast_ref::<T>()
                .expect("kernel interface type mismatch")
        })
    }
    pub fn host<T: HostOp + 'static>() -> &'static Self {
        &Self::Host(|op| {
            op.as_any()
                .downcast_ref::<T>()
                .expect("host interface type mismatch")
        })
    }
}
pub fn as_kernel_op(op: &dyn BufferTensorIrOp) -> Option<&dyn KernelOp> {
    match op.runtime_interface()?.downcast_ref::<MetalOpInterface>()? {
        MetalOpInterface::Kernel(borrow) => Some(borrow(op)),
        MetalOpInterface::Host(_) => None,
    }
}
pub fn as_host_op(op: &dyn BufferTensorIrOp) -> Option<&dyn HostOp> {
    match op.runtime_interface()?.downcast_ref::<MetalOpInterface>()? {
        MetalOpInterface::Host(borrow) => Some(borrow(op)),
        MetalOpInterface::Kernel(_) => None,
    }
}
