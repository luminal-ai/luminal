//! Probe: what does the greedy-seeded search elect for a serving-shaped
//! linear layer? Prints the installed plan's op histogram for a few
//! shapes; asserts that the library route is elected for every one of
//! them (a decomposed [s, in, out] product is what a serving plan cannot
//! afford).
#![cfg(feature = "device")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn histogram(plan: &luminal_cuda_lite::CudaPlan) -> std::collections::BTreeMap<String, usize> {
    let mut ops = std::collections::BTreeMap::new();
    let bytes = |slot: &luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>| {
        let numel = slot.layout.literal_span_elements().unwrap_or(0);
        let width = slot
            .layout
            .dtype
            .and_then(|d| luminal_cuda_lite::host_buffer::dtype_bytes(d).ok())
            .unwrap_or(0);
        format!(
            "{}x{}B {:?}",
            numel,
            width,
            slot.layout.present()
        )
    };
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = node
        {
            let label = op.label().to_string();
            if label != "BufferAlloc" && label != "BufferFree" {
                let reads: Vec<String> = operand_info.iter().map(bytes).collect();
                let writes: Vec<String> = result_info.iter().map(bytes).collect();
                eprintln!("    {label}: reads {reads:?} writes {writes:?}");
            }
            *ops.entry(label).or_default() += 1;
        }
    }
    ops
}

fn probe(s: usize, inp: usize, out: usize, bias: bool) -> std::collections::BTreeMap<String, usize> {
    let mut cx = Graph::new();
    let x = cx.tensor((s, inp), DType::F32);
    let w = cx.tensor((inp, out), DType::F32);
    let b = bias.then(|| cx.tensor(out, DType::F32));
    let y = luminal_nn::linear(x, w, b).output();
    let mut data: FxHashMap<NodeIndex, HostBuffer> = FxHashMap::default();
    data.insert(x.id, HostBuffer::from(vec![0.5f32; s * inp]));
    data.insert(w.id, HostBuffer::from(vec![0.25f32; inp * out]));
    if let Some(b) = b {
        data.insert(b.id, HostBuffer::from(vec![0.1f32; out]));
    }
    let mut rt = CudaRuntime::load(&cx).expect("load");
    let options = luminal_cuda_lite::CompileOptions {
        profile_on_device: true,
        generations: 1,
        generation_size: 1,
        ..luminal_cuda_lite::harness_search_options()
    };
    rt.search(&data, &options)
        .unwrap_or_else(|e| panic!("search: {e:#}"));
    let ops = histogram(rt.plan().expect("plan"));
    eprintln!("[{s}x{inp}] @ [{inp}x{out}] bias={bias}: {ops:?}  ({} ms)", 0);
    let _ = y;
    ops
}

#[test]
fn serving_linears_elect_the_library_route() {
    for (s, inp, out, bias) in [
        (64, 2880, 4096, true),
        (64, 2880, 512, true),
        (64, 4096, 2880, true),
        (64, 2880, 128, true),
        (64, 2880, 201088, false),
        (1, 2880, 4096, true),
    ] {
        let ops = probe(s, inp, out, bias);
        let library: usize = ops
            .iter()
            .filter(|(k, _)| k.starts_with("CublasLt"))
            .map(|(_, v)| *v)
            .sum();
        assert!(
            library >= 1 && !ops.contains_key("ReduceSumGeneric"),
            "[{s}x{inp}]@[{inp}x{out}] bias={bias} elected {ops:?}"
        );
    }
}
