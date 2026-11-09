//! Native NCCL communicators. Applications bootstrap the unique ID and bind the
//! same named group on every rank before executing a compiled collective graph.
#[cfg(feature = "device")]
mod native {
    use anyhow::{Result, anyhow, ensure};
    use cudarc::driver::DevicePtr;
    use cudarc::nccl::{result, sys};
    use std::{cell::RefCell, collections::HashMap, rc::Rc};

    pub struct NcclCommunicator {
        pub(crate) handle: sys::ncclComm_t,
        pub(crate) width: usize,
        rank: usize,
        ordinal: usize,
        _context: std::sync::Arc<cudarc::driver::CudaContext>,
        owns: bool,
    }
    impl Drop for NcclCommunicator {
        fn drop(&mut self) {
            if self.owns {
                unsafe {
                    let _ = result::comm_abort(self.handle);
                }
            }
        }
    }
    impl NcclCommunicator {
        /// Generate on one rank and distribute these bytes to all participants.
        pub fn unique_id() -> Result<[u8; 128]> {
            Ok(result::get_uniqueid()
                .map_err(|e| anyhow!("NCCL unique id: {e:?}"))?
                .internal
                .map(|byte| byte.to_ne_bytes()[0]))
        }
        /// Collective initialization: all ranks must enter concurrently. Rank
        /// order determines the leading output axis. No tensor data uses TCP.
        pub fn init(
            device: &crate::device::CudaDevice,
            id: [u8; 128],
            rank: usize,
            width: usize,
        ) -> Result<Rc<Self>> {
            ensure!(width > 0 && rank < width, "invalid NCCL rank/width");
            device.stream().context().bind_to_thread()?;
            let mut handle = std::ptr::null_mut();
            unsafe {
                result::comm_init_rank(
                    &mut handle,
                    i32::try_from(width)?,
                    sys::ncclUniqueId {
                        internal: id.map(|v| v as std::ffi::c_char),
                    },
                    i32::try_from(rank)?,
                )
                .map_err(|e| anyhow!("NCCL communicator initialization: {e:?}"))?;
            }
            Ok(Rc::new(Self {
                handle,
                width,
                rank,
                ordinal: device.stream().context().ordinal(),
                _context: device.stream().context().clone(),
                owns: true,
            }))
        }
        /// Borrow an existing communicator. Its owner must keep it alive until
        /// this handle and all compiled graphs retaining it have been dropped.
        /// # Safety
        /// `handle` must be a live NCCL communicator on `device`, with `width`
        /// members, and must not be used concurrently by unrelated callers.
        pub unsafe fn from_raw(
            device: &crate::device::CudaDevice,
            handle: sys::ncclComm_t,
            width: usize,
        ) -> Result<Rc<Self>> {
            ensure!(!handle.is_null() && width > 0, "invalid NCCL communicator");
            let rank = usize::try_from(
                unsafe { result::comm_user_rank(handle) }
                    .map_err(|e| anyhow!("NCCL communicator rank: {e:?}"))?,
            )?;
            ensure!(rank < width, "NCCL communicator rank exceeds group width");
            Ok(Rc::new(Self {
                handle,
                width,
                rank,
                ordinal: device.stream().context().ordinal(),
                _context: device.stream().context().clone(),
                owns: false,
            }))
        }

        /// This communicator's rank. Rank zero coordinates implementation
        /// search; all other ranks are profiling workers.
        pub fn rank(&self) -> usize {
            self.rank
        }

        pub(crate) fn broadcast_bytes(
            &self,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            root_payload: Option<&[u8]>,
        ) -> Result<Vec<u8>> {
            ensure!(
                stream.context().ordinal() == self.ordinal,
                "collective search communicator belongs to another CUDA device"
            );
            ensure!(
                (self.rank == 0) == root_payload.is_some(),
                "only collective search rank 0 supplies broadcast bytes"
            );
            let mut length = if let Some(payload) = root_payload {
                vec![u64::try_from(payload.len())?]
            } else {
                vec![0]
            };
            self.broadcast_slice(stream, &mut length)?;
            let length = usize::try_from(length[0])?;
            ensure!(
                length <= 1 << 34,
                "collective search control payload is unreasonably large"
            );
            let mut payload = root_payload.map_or_else(|| vec![0; length], ToOwned::to_owned);
            ensure!(
                payload.len() == length,
                "collective search payload length changed"
            );
            self.broadcast_slice(stream, &mut payload)?;
            Ok(payload)
        }

        pub(crate) fn all_true(
            &self,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            local: bool,
        ) -> Result<bool> {
            ensure!(
                stream.context().ordinal() == self.ordinal,
                "collective search communicator belongs to another CUDA device"
            );
            let input = stream.clone_htod(&[u32::from(local)])?;
            let output = stream.alloc_zeros::<u32>(1)?;
            unsafe {
                result::all_reduce(
                    input.device_ptr(stream).0 as _,
                    output.device_ptr(stream).0 as _,
                    1,
                    sys::ncclDataType_t::ncclUint32,
                    sys::ncclRedOp_t::ncclSum,
                    self.handle,
                    stream.cu_stream() as _,
                )
                .map_err(|e| anyhow!("NCCL collective-search readiness: {e:?}"))?;
            }
            stream.synchronize()?;
            Ok(stream.clone_dtoh(&output)? == [u32::try_from(self.width)?])
        }

        pub(crate) fn max_u64(
            &self,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            local: u64,
        ) -> Result<u64> {
            ensure!(
                stream.context().ordinal() == self.ordinal,
                "collective search communicator belongs to another CUDA device"
            );
            let input = stream.clone_htod(&[local])?;
            let output = stream.alloc_zeros::<u64>(1)?;
            unsafe {
                result::all_reduce(
                    input.device_ptr(stream).0 as _,
                    output.device_ptr(stream).0 as _,
                    1,
                    sys::ncclDataType_t::ncclUint64,
                    sys::ncclRedOp_t::ncclMax,
                    self.handle,
                    stream.cu_stream() as _,
                )
                .map_err(|e| anyhow!("NCCL collective-search maximum: {e:?}"))?;
            }
            stream.synchronize()?;
            Ok(stream.clone_dtoh(&output)?[0])
        }

        fn broadcast_slice<T>(
            &self,
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            values: &mut [T],
        ) -> Result<()>
        where
            T: cudarc::nccl::safe::NcclType + cudarc::driver::DeviceRepr + Clone + Default,
        {
            let device = stream.clone_htod(values)?;
            unsafe {
                result::broadcast(
                    device.device_ptr(stream).0 as _,
                    device.device_ptr(stream).0 as _,
                    values.len(),
                    T::as_nccl_type(),
                    0,
                    self.handle,
                    stream.cu_stream() as _,
                )
                .map_err(|e| anyhow!("NCCL collective-search broadcast: {e:?}"))?;
            }
            stream.synchronize()?;
            stream.memcpy_dtoh(&device, values)?;
            Ok(())
        }
    }
    thread_local! { static GROUPS: RefCell<HashMap<String, Rc<NcclCommunicator>>> = RefCell::new(HashMap::new()); }
    /// Bind for preparation/capture. Prepared operations retain an Rc, so graph
    /// replay remains valid after this scope ends. Rebinding requires compiling
    /// a fresh runtime; a warmed graph keeps its original communicator.
    pub fn with_group<R>(
        name: impl Into<String>,
        group: Rc<NcclCommunicator>,
        run: impl FnOnce() -> R,
    ) -> R {
        struct Restore(String, Option<Rc<NcclCommunicator>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                GROUPS.with(|groups| {
                    let mut groups = groups.borrow_mut();
                    if let Some(old) = self.1.take() {
                        groups.insert(self.0.clone(), old);
                    } else {
                        groups.remove(&self.0);
                    }
                });
            }
        }
        let name = name.into();
        let old = GROUPS.with(|groups| groups.borrow_mut().insert(name.clone(), group));
        let _restore = Restore(name, old);
        run()
    }
    pub(crate) fn bound(name: &str, width: usize, ordinal: usize) -> Result<Rc<NcclCommunicator>> {
        let group = GROUPS
            .with(|groups| groups.borrow().get(name).cloned())
            .ok_or_else(|| anyhow!("NCCL AllGather group {name:?} is not bound"))?;
        ensure!(group.width == width, "NCCL AllGather group size mismatch");
        ensure!(
            group.ordinal == ordinal,
            "NCCL communicator belongs to another CUDA device"
        );
        Ok(group)
    }
}
#[cfg(feature = "device")]
pub(crate) use native::bound;
#[cfg(feature = "device")]
pub use native::{NcclCommunicator, with_group};
