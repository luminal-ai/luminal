//! CALLER-OWNED DEVICE MEMORY IS A BINDING. The runtime takes device
//! addresses by BUFFER id — the same id the bindings use to state aliasing
//! — and only for buffers declared `Placement::External`. An execution
//! that cannot address such a buffer is refused by name, before any device
//! work, on any host.

use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_cuda_lite::{CudaBindings, CudaRuntime};

/// `a + b` with `a` on caller device memory and `b` host-staged.
fn runtime() -> (CudaRuntime, i64, i64) {
    let mut cx = Graph::new();
    let a = cx.tensor(4, DType::F32);
    let b = cx.tensor(4, DType::F32);
    let sum = a + b;
    let mut bindings = CudaBindings::new();
    let external = bindings.input_external(a.id);
    let staged = bindings.input(b.id);
    bindings.output(sum.id);
    let runtime =
        CudaRuntime::load_with(&cx, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    (runtime, external, staged)
}

#[test]
fn a_device_pointer_is_refused_for_a_buffer_that_is_not_external() {
    let (mut runtime, external, staged) = runtime();
    assert_eq!(
        runtime.externals().iter().copied().collect::<Vec<_>>(),
        vec![external]
    );
    // SAFETY: refused before the address is ever read.
    let refusal = unsafe { runtime.set_device_ptr(staged, 0x1000, 16) }
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains(&format!("buffer {staged} is not bound External")),
        "{refusal}"
    );
    // SAFETY: as above — the runtime refuses to execute on this host.
    unsafe { runtime.set_device_ptr(external, 0x1000, 16) }.unwrap();
    assert!(runtime.missing_external_pointers().is_empty());
    runtime.clear_device_ptr(external);
    assert_eq!(runtime.missing_external_pointers(), vec![external]);
}

#[test]
fn execute_refuses_an_external_buffer_with_no_pointer() {
    let (mut runtime, external, _) = runtime();
    assert_eq!(runtime.missing_external_pointers(), vec![external]);
    let refusal = runtime.execute().unwrap_err().to_string();
    assert!(
        refusal.contains(&format!("External buffer {external}"))
            && refusal.contains("has no device pointer"),
        "{refusal}"
    );
}

/// A MUTATION SINK SHARES ITS TARGET'S BUFFER, and therefore its placement
/// and its single pointer: the boundary names one External buffer, not two.
#[test]
fn a_sink_and_its_target_share_one_external_buffer() {
    let mut cx = Graph::new();
    let state = cx.tensor(4, DType::F32);
    let delta = cx.tensor(4, DType::F32);
    let next = state + delta;
    let mut bindings = CudaBindings::new();
    let home = bindings.input_external(state.id);
    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
    bindings.input_external(delta.id);
    bindings.output_on(next.id, home);
    let runtime =
        CudaRuntime::load_with(&cx, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    assert_eq!(runtime.externals().len(), 2);
    assert_eq!(runtime.input_buffer(state.id).unwrap(), home);
    assert_eq!(runtime.output_buffer(next.id).unwrap(), home);
    assert_eq!(runtime.missing_external_pointers().len(), 2);
}
