//! Borrow CUDA execution traits from operations carried by the shared IR.
//!
//! Each executable op supplies its own interface adapter through
//! `BufferTensorIrOp::runtime_interface`. These adapters only recover a trait
//! reference; code generation and execution belong to the traits themselves.
//! There is no global operation table, label matching, allocation, or op clone.

use luminal::buffer_tensor_ir::BufferTensorIrOp;

use crate::{host::HostOp, kernels::KernelOp};

pub(crate) type DecodeWireOp = fn(&[u8]) -> anyhow::Result<Box<dyn BufferTensorIrOp>>;
pub(crate) type EncodeWireOp = fn(&dyn BufferTensorIrOp) -> anyhow::Result<Vec<u8>>;
pub(crate) type WireParts = (&'static str, EncodeWireOp, DecodeWireOp);

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct WireInterface {
    name: fn() -> &'static str,
    encode: EncodeWireOp,
    decode: DecodeWireOp,
}

pub enum CudaOpInterface {
    Kernel {
        borrow: fn(&dyn BufferTensorIrOp) -> &dyn KernelOp,
        wire: Option<WireInterface>,
    },
    Host {
        borrow: fn(&dyn BufferTensorIrOp) -> &dyn HostOp,
        wire: Option<WireInterface>,
    },
}

impl CudaOpInterface {
    fn encode<T>(op: &dyn BufferTensorIrOp) -> anyhow::Result<Vec<u8>>
    where
        T: BufferTensorIrOp + serde::Serialize + 'static,
    {
        Ok(serde_json::to_vec(op.as_any().downcast_ref::<T>().expect(
            "CUDA wire interface must describe its concrete op type",
        ))?)
    }

    fn decode<T>(bytes: &[u8]) -> anyhow::Result<Box<dyn BufferTensorIrOp>>
    where
        T: BufferTensorIrOp + serde::de::DeserializeOwned + 'static,
    {
        Ok(Box::new(serde_json::from_slice::<T>(bytes)?))
    }

    fn name<T>() -> &'static str {
        std::any::type_name::<T>()
    }

    /// Return this adapter from a kernel op's `runtime_interface` method.
    pub fn kernel<T: KernelOp + 'static>() -> &'static Self {
        &Self::Kernel {
            borrow: |op| {
                op.as_any()
                    .downcast_ref::<T>()
                    .expect("kernel interface must describe its concrete op type")
            },
            wire: None,
        }
    }

    /// A kernel op that can also cross the rank-zero physical-plan wire.
    pub fn kernel_wire<T>() -> &'static Self
    where
        T: KernelOp + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        &Self::Kernel {
            borrow: |op| {
                op.as_any()
                    .downcast_ref::<T>()
                    .expect("kernel interface must describe its concrete op type")
            },
            wire: Some(WireInterface {
                name: Self::name::<T>,
                encode: Self::encode::<T>,
                decode: Self::decode::<T>,
            }),
        }
    }

    /// Return this adapter from a host op's `runtime_interface` method.
    pub fn host<T: HostOp + 'static>() -> &'static Self {
        &Self::Host {
            borrow: |op| {
                op.as_any()
                    .downcast_ref::<T>()
                    .expect("host interface must describe its concrete op type")
            },
            wire: None,
        }
    }

    /// A host op that can also cross the rank-zero physical-plan wire.
    pub fn host_wire<T>() -> &'static Self
    where
        T: HostOp + serde::Serialize + serde::de::DeserializeOwned + 'static,
    {
        &Self::Host {
            borrow: |op| {
                op.as_any()
                    .downcast_ref::<T>()
                    .expect("host interface must describe its concrete op type")
            },
            wire: Some(WireInterface {
                name: Self::name::<T>,
                encode: Self::encode::<T>,
                decode: Self::decode::<T>,
            }),
        }
    }

    pub(crate) fn wire_parts(&self) -> Option<WireParts> {
        let wire = match self {
            Self::Kernel { wire, .. } | Self::Host { wire, .. } => wire.as_ref()?,
        };
        Some(((wire.name)(), wire.encode, wire.decode))
    }
}

pub fn as_kernel_op(op: &dyn BufferTensorIrOp) -> Option<&dyn KernelOp> {
    match op.runtime_interface()?.downcast_ref::<CudaOpInterface>()? {
        CudaOpInterface::Kernel { borrow, .. } => Some(borrow(op)),
        CudaOpInterface::Host { .. } => None,
    }
}

pub fn as_host_op(op: &dyn BufferTensorIrOp) -> Option<&dyn HostOp> {
    match op.runtime_interface()?.downcast_ref::<CudaOpInterface>()? {
        CudaOpInterface::Host { borrow, .. } => Some(borrow(op)),
        CudaOpInterface::Kernel { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_executable_op_round_trips_over_the_plan_wire() {
        for entry in crate::ops::cuda_registry() {
            let dps = entry.prototype.to_dps();
            let executable = dps.as_deref().unwrap_or(entry.prototype.as_ref());
            let Some(interface) = executable
                .runtime_interface()
                .and_then(|interface| interface.downcast_ref::<CudaOpInterface>())
            else {
                continue;
            };
            let Some((name, encode, decode)) = interface.wire_parts() else {
                continue;
            };
            let payload = encode(executable).unwrap_or_else(|error| {
                panic!("encode {} ({name}): {error:#}", executable.label())
            });
            let decoded = decode(&payload).unwrap_or_else(|error| {
                panic!("decode {} ({name}): {error:#}", executable.label())
            });
            assert_eq!(decoded.label(), executable.label(), "{name}");
        }
    }
}
