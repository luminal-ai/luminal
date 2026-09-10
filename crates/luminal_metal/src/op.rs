use crate::kernels::KernelOp;
use luminal::buffer_tensor_ir::BufferTensorIrOp;
/// Borrow execution through the concrete operation, without label dispatch.
pub struct MetalOpInterface(fn(&dyn BufferTensorIrOp) -> &dyn KernelOp);
impl MetalOpInterface {
    pub fn kernel<T: KernelOp + 'static>() -> &'static Self {
        &Self(|op| {
            op.as_any()
                .downcast_ref::<T>()
                .expect("kernel interface type mismatch")
        })
    }
}
pub fn as_kernel_op(op: &dyn BufferTensorIrOp) -> Option<&dyn KernelOp> {
    let interface = op.runtime_interface()?.downcast_ref::<MetalOpInterface>()?;
    Some((interface.0)(op))
}
