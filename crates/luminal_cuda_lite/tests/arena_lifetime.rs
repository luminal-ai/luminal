#![cfg(feature = "device")]

use cudarc::driver::{result, sys};
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, device::CudaDevice, harness_search_options};

#[test]
fn repeated_search_releases_live_profiling_pool_bytes() {
    let device = CudaDevice::new(0).unwrap();
    let stream = device.stream();
    let context = stream.context();
    if !context.has_async_alloc() {
        return;
    }
    // SAFETY: the device and its current pool stay live throughout the test.
    let pool = unsafe { result::device::get_mem_pool(context.cu_device()).unwrap() };
    let live_bytes = || {
        // Dropping a profiling buffer enqueues its free on this stream.
        stream.synchronize().unwrap();
        let mut used = 0u64;
        // SAFETY: USED_MEM_CURRENT writes a uint64 to this writable slot.
        unsafe {
            result::mem_pool::get_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                (&mut used as *mut u64).cast(),
            )
            .unwrap();
        }
        used
    };

    let mut graph = Graph::new();
    let input = graph.tensor([1024], DType::F32);
    let _output = input + 1.0;
    let data = [(input.id, vec![2.0f32; 1024].into())]
        .into_iter()
        .collect();
    let options = harness_search_options();
    let mut runtime = CudaRuntime::load(&graph)
        .unwrap()
        .with_device(&device)
        .unwrap();
    let baseline = live_bytes();
    for iteration in 1..=3 {
        let outcome = runtime
            .search(&Default::default(), &Default::default(), &data, &options)
            .unwrap();
        assert!(outcome.plans_profiled > 0, "search did not profile a plan");
        // Check while the runtime is alive: a selected plan must not hold
        // temporary profiling storage. Freed pool reservations are fine.
        assert_eq!(live_bytes(), baseline, "after search {iteration}");
    }
}
