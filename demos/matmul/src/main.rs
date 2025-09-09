use std::{collections::HashMap, ffi::c_void, ptr::NonNull};

#[cfg(feature = "cuda")]
use cudarc::{driver::*, nvrtc::CompileOptions};
use itertools::Itertools;
use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};
use luminal_2::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    run::{assign_buffers, compile_kernels, run_graph},
    translate::{translate_graph, InitData},
    GPUArch, GraphTerm,
};
#[cfg(feature = "metal")]
use luminal_2::{Buffer, Device};
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use rand::{rng, Rng};
use rustc_hash::FxHashMap;

#[cfg(feature = "metal")]
#[inline]
fn with_autoreleasepool<F: FnOnce()>(f: F) {
    objc2::rc::autoreleasepool(|_| f());
}

#[cfg(feature = "cuda")]
#[inline]
fn with_autoreleasepool<F: FnOnce()>(f: F) {
    f();
}

fn main() {
    with_autoreleasepool(|| {
        #[cfg(feature = "metal")]
        let arch = GPUArch::Metal(HashMap::default());
        #[cfg(feature = "cuda")]
        let arch = GPUArch::CUDA;

        #[allow(non_snake_case)]
        let (M, K, N, J) = (512, 512, 512, 512);
        let mut cx = Graph::new();
        let a = cx.named_tensor("A", (M, K));
        let b = cx.named_tensor("B", (K, N));
        let c = cx.named_tensor("C", (K, N));
        let d = cx.named_tensor("D", (N, J));
        let _out = (a.matmul(b).swish() * a.matmul(c)).matmul(d);

        let (mut new_graph, mut mapping, accs) = translate_graph(&cx);
        // Search each subgraph
        for graph_node in new_graph.node_indices().collect_vec() {
            let graph = new_graph.node_weight_mut(graph_node).unwrap();
            // luminal_2::debug::display_graph(&graph);
            let inputs = make_test_inputs(graph, &cx.dyn_map, &accs);
            let searched_graph = search(graph, 3, &inputs, arch.clone(), &cx.dyn_map).unwrap();
            // adjust meta-edges
            let old_output = graph.externals(Direction::Outgoing).next().unwrap();
            let new_output = searched_graph
                .externals(Direction::Outgoing)
                .next()
                .unwrap();
            let old_inputs = graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = graph.node_weight(n).unwrap() {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            let new_inputs = searched_graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = searched_graph.node_weight(n).unwrap() {
                        Some((label.clone(), n))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            *graph = searched_graph;
            for edge in new_graph
                .edges_directed(graph_node, Direction::Outgoing)
                .map(|e| e.id())
                .collect_vec()
            {
                let (input, _) = new_graph.edge_weight_mut(edge).unwrap();
                *input = new_output;
            }
            // Update old-to-new-mappings
            for (_, (meta, v)) in &mut mapping {
                if *meta != graph_node {
                    continue;
                }
                if *v == old_output {
                    *v = new_output;
                }
                if let Some(gmem_label) = old_inputs.get(v) {
                    if let Some(new) = new_inputs.get(gmem_label) {
                        *v = *new;
                    }
                }
            }
        }
        let (graph, meta_to_final) = stitch_meta_graph_together(new_graph);
        luminal_2::debug::display_graph(&graph);
        let mut gmem_to_node_mapping = FxHashMap::default();
        for n in graph.node_indices() {
            if let Some(GraphTerm::GMEM { label }) = graph.node_weight(n) {
                gmem_to_node_mapping.insert(label.clone(), n);
            }
        }
        let mut unified_map = FxHashMap::default();
        for (k, v) in mapping {
            if let Some(m) = meta_to_final.get(&v) {
                unified_map.insert(k, *m);
            }
        }
        let (kernels, gmem_mapping) = codegen(graph.clone(), arch, &HashMap::default()).unwrap();

        let compiled = compile_kernels(&kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);

        #[cfg(feature = "metal")]
        let device = &MTLCreateSystemDefaultDevice().unwrap();
        #[cfg(feature = "cuda")]
        let device = &CudaContext::new(0).unwrap();

        let mut inputs = FxHashMap::default();
        let mut rng = rng();
        inputs.insert(
            gmem_mapping[&unified_map[&a.id]],
            (
                copy_buffer(
                    &(0..M * K)
                        .map(|_| rng.random_range(-1e-2..1e-2))
                        .collect_vec(),
                    device,
                ),
                false,
            ),
        );
        inputs.insert(
            gmem_mapping[&unified_map[&b.id]],
            (
                copy_buffer(
                    &(0..K * N)
                        .map(|_| rng.random_range(-1e-2..1e-2))
                        .collect_vec(),
                    device,
                ),
                false,
            ),
        );
        inputs.insert(
            gmem_mapping[&unified_map[&c.id]],
            (
                copy_buffer(
                    &(0..K * N)
                        .map(|_| rng.random_range(-1e-2..1e-2))
                        .collect_vec(),
                    device,
                ),
                false,
            ),
        );
        inputs.insert(
            gmem_mapping[&unified_map[&d.id]],
            (
                copy_buffer(
                    &(0..N * J)
                        .map(|_| rng.random_range(-1e-2..1e-2))
                        .collect_vec(),
                    device,
                ),
                false,
            ),
        );
        for (label, val) in &accs {
            if let Some(node) = gmem_to_node_mapping.get(label) {
                if let Some(input_index) = gmem_mapping.get(node) {
                    match val {
                        InitData::Expr(e) => {
                            let val = e.exec(&cx.dyn_map).unwrap();
                            inputs.insert(*input_index, {
                                let v = vec![val as f32];
                                (copy_buffer(&v, device), true)
                            });
                        }
                        InitData::Data(d) => {
                            inputs.insert(*input_index, (copy_buffer(d, device), true));
                        }
                    }
                }
            }
        }

        let (outputs, _) = {
            #[cfg(feature = "metal")]
            {
                run_graph(
                    &graph,
                    &mut inputs,
                    &kernels,
                    &FxHashMap::default(),
                    &compiled,
                    &int_buffers,
                    &int_buffer_map,
                )
            }

            #[cfg(feature = "cuda")]
            {
                run_graph(
                    &mut inputs,
                    &kernels,
                    &FxHashMap::default(),
                    &compiled,
                    &int_buffers,
                    &int_buffer_map,
                )
            }
        };
        println!("{:?}", &copy_buffer_back(&outputs[0])[..10]);
    });
}

#[cfg(feature = "cuda")]
pub fn copy_buffer(v: &[f32], ctx: &std::sync::Arc<CudaContext>) -> CudaSlice<f32> {
    assert!(!v.is_empty(), "Can't copy empty slice to device");

    // Then copy host data to the allocated device memory
    let stream = ctx.default_stream();
    let mut dst = stream.alloc_zeros::<f32>(v.len()).unwrap();
    stream.memcpy_htod(v, &mut dst).unwrap();
    dst
}

/// Device -> Host (like contents() memcpy back)
#[cfg(feature = "cuda")]
pub fn copy_buffer_back(buf: &CudaSlice<f32>) -> Vec<f32> {
    buf.stream().memcpy_dtov(buf).unwrap()
}

#[cfg(feature = "metal")]
pub fn copy_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                v.len() * std::mem::size_of::<f32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}

#[cfg(feature = "metal")]
pub fn copy_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}
