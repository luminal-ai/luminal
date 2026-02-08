use std::{fmt::Debug, sync::Arc};

use crate::cudarc::driver::{CudaSlice, CudaStream};
use luminal::{op::EgglogOp, prelude::*};
mod cublas;

pub type Ops = (cublas::CuBlasSgemmV2,);

pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// With the convention, and this is a bad way to do it, that the first is the output buffer.
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        node_inputs: &[&CudaSlice<u8>],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    fn output_size(&self) -> Expression;
}
