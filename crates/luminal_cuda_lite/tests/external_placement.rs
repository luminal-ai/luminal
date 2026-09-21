//! CALLER-OWNED DEVICE MEMORY IS A BINDING. The runtime takes device
//! addresses by BUFFER id — the same id the bindings use to state aliasing
//! — and only for buffers declared `Placement::External`. An execution
//! that cannot address such a buffer is refused by name, before any device
//! work, on any host.

use luminal::bufferize::BufferNode;
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_cuda_lite::{CudaBindings, CudaRuntime, harness_search_options};

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
    // SAFETY: refused before the address is ever read.
    let refusal = unsafe { runtime.set_device_ptr(external, 0, 16) }
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains(&format!(
            "buffer {external} was given the null device pointer"
        )),
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

/// STAGING AN EXTERNAL BUFFER IS REFUSED BY NAME: an External buffer has
/// no upload step, so bytes handed to `set_data` would vanish.
#[test]
fn set_data_on_an_external_binding_is_refused() {
    let mut cx = Graph::new();
    let a = cx.tensor(4, DType::F32);
    let b = cx.tensor(4, DType::F32);
    let sum = a + b;
    let mut bindings = CudaBindings::new();
    let external = bindings.input_external(a.id);
    bindings.input(b.id);
    bindings.output(sum.id);
    let mut runtime =
        CudaRuntime::load_with(&cx, bindings, luminal_cuda_lite::cuda_registry()).unwrap();

    let refusal = runtime
        .set_data(a.id, vec![1f32; 4])
        .unwrap_err()
        .to_string();
    assert!(
        refusal.contains(&format!("External buffer {external}"))
            && refusal.contains("set_device_ptr"),
        "{refusal}"
    );
    runtime
        .set_data(b.id, vec![1f32; 4])
        .expect("a host-staged input takes its payload");
    assert!(
        runtime
            .set_data(sum.id, vec![1f32; 4])
            .unwrap_err()
            .to_string()
            .contains("no input binding"),
        "an unbound tensor is refused, not panicked on"
    );
}

/// THE EXTERNAL OUTPUT GUARD, ON ANY HOST: an output bound External is
/// written in place, so the plan must have elected the caller's own
/// buffer rather than a view of it. That fact — the plan buffer under
/// the output slot carries the bound BufferLit — is what the guard
/// reads, and it is a fact of the searched plan, not of the device.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn an_external_output_slot_sits_on_the_bound_buffer() {
    let mut cx = Graph::new();
    let a = cx.tensor(4, DType::F32);
    let b = cx.tensor(4, DType::F32);
    let sum = a + b;
    let mut bindings = CudaBindings::new();
    bindings.input(a.id);
    bindings.input(b.id);
    let out = bindings.output_external(sum.id);
    let mut runtime =
        CudaRuntime::load_with(&cx, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    runtime
        .search(&Default::default(), &harness_search_options())
        .expect("host search");

    let plan = runtime.plan().expect("the search installed a plan");
    let mut slots = 0;
    for node in plan.dag.node_weights() {
        let BufferNode::BufferOutput { slots: bound } = node else {
            continue;
        };
        for slot in bound {
            assert_eq!(
                plan.buffers[&slot.buffer].lit,
                Some(out),
                "the External output slot must sit on the bound buffer, not a view of it"
            );
            slots += 1;
        }
    }
    assert_eq!(slots, 1, "one bound output, one slot");
    runtime
        .check_external_outputs()
        .expect("the guard reads the same fact");
}

/// A CALLER-OWNED OUTPUT MAY BE A VIEW OF AN ESCAPE CELL — with the cuBLASLt
/// estate available, which is what makes this a real question. The decorated
/// matmul claims its result in the sibling (transpose-sandwich) frame, so the
/// recorder-frame value is a VIEW of the op's D buffer. That view is a legal
/// fulfilment: the escaping cell under it IS the caller's storage, retargeted
/// onto the bound buffer id after the search, and the runtime discloses which
/// buffer it wrote, how far it reaches, and how the value is strided in it.
#[test]
#[cfg_attr(
    not(feature = "device"),
    ignore = "candidate search requires a CUDA device"
)]
fn an_external_output_may_be_a_view_of_an_escape_cell() {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 16usize), DType::F32);
    let w = cx.tensor((16usize, 32usize), DType::F32);
    let out = x.matmul(w);

    let mut bindings = CudaBindings::new();
    bindings.input(x.id);
    bindings.input(w.id);
    let bound = bindings.output_external(out.id);

    let mut runtime = CudaRuntime::load_with(
        &cx,
        bindings,
        luminal_cuda_lite::cuda_registry_filtered(|row| row.label() != "ReduceSumGeneric"),
    )
    .expect("a matmul with a caller-owned output loads");
    runtime
        .search(&FxHashMap::default(), &harness_search_options())
        .expect("a plan that fulfils the caller-owned output exists");

    runtime
        .check_external_outputs()
        .expect("the guard reads the retargeted plan");
    assert_eq!(
        runtime.output_backing_buffer(out.id).unwrap(),
        bound,
        "the output's backing storage is the caller's buffer"
    );
    assert_eq!(
        runtime.output_elected_strides(out.id).unwrap(),
        vec![32, 1],
        "the elected layout is the row-major (4, 32) result"
    );
    assert!(
        runtime.output_span_bytes(out.id).unwrap() >= 4 * 32 * 4,
        "the backing buffer holds at least the whole result"
    );
    assert!(
        runtime
            .plan()
            .expect("the search installed a plan")
            .dag
            .node_weights()
            .any(|node| matches!(node, BufferNode::Compute { op, .. } if op.label() == "CublasLt")),
        "the cuBLASLt estate is what makes this output a view"
    );
}
