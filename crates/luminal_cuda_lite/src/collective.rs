//! Native NCCL communicators. Applications bootstrap the unique ID and bind the
//! same named group on every rank before executing a compiled collective graph.
#[cfg(feature = "device")]
mod native {
    use anyhow::{Result, anyhow, ensure};
    use cudarc::nccl::{result, sys};
    use std::{cell::RefCell, collections::HashMap, rc::Rc};

    pub struct NcclCommunicator {
        pub(crate) handle: sys::ncclComm_t,
        pub(crate) width: usize,
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
                .map(|v| v as u8))
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
            Ok(Rc::new(Self {
                handle,
                width,
                ordinal: device.stream().context().ordinal(),
                _context: device.stream().context().clone(),
                owns: false,
            }))
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
