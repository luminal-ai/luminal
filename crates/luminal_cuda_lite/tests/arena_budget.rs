#![cfg(feature = "device")]

use cudarc::driver::{CudaContext, result, sys};
use luminal_cuda_lite::device::CudaDevice;

#[test]
fn available_arena_includes_unused_cuda_pool_reservations() {
    let context = CudaContext::new(0).unwrap();
    if !context.has_async_alloc() {
        return;
    }
    let device = CudaDevice::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = unsafe { result::device::get_mem_pool(context.cu_device()).unwrap() };
    struct RestoreThreshold(sys::CUmemoryPool, u64);
    impl Drop for RestoreThreshold {
        fn drop(&mut self) {
            unsafe {
                result::mem_pool::set_attribute(
                    self.0,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    (&mut self.1 as *mut u64).cast(),
                )
                .unwrap();
                result::mem_pool::trim_to(self.0, 0).unwrap();
            }
        }
    }
    let mut restore = RestoreThreshold(pool, 0);
    let mut retained_threshold = u64::MAX;
    unsafe {
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut restore.1 as *mut u64).cast(),
        )
        .unwrap();
        result::mem_pool::set_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&mut retained_threshold as *mut u64).cast(),
        )
        .unwrap();
    }
    let before = device.available_arena_bytes().unwrap();
    let bytes = 64 * 1024 * 1024;
    let allocation = stream.alloc_zeros::<u8>(bytes).unwrap();
    stream.synchronize().unwrap();
    let occupied = device.available_arena_bytes().unwrap();
    assert!(before >= occupied + bytes);
    drop(allocation);
    stream.synchronize().unwrap();
    let free = context.mem_get_info().unwrap().0;
    let reusable = device.available_arena_bytes().unwrap();
    assert!(
        reusable >= free + bytes,
        "the pool must retain the freed allocation"
    );
    assert!(reusable >= occupied + bytes);
    assert!(reusable.abs_diff(before) < 1024 * 1024);
}
