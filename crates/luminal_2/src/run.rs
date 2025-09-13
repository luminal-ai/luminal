use crate::{
    codegen::{codegen, stitch_meta_graph_together},
    extract::{make_test_inputs, search},
    translate::{translate_graph, InitData, OptimalGraphNodeIndex, SubGraphNodeIndex},
    GPUArch, GraphTerm,
};
#[cfg(feature = "metal")]
use crate::{Buffer, Device, Function};
#[cfg(feature = "blade")]
use blade_graphics as gpu;
#[cfg(feature = "cuda")]
use cudarc::{driver::*, nvrtc::CompileOptions};
use itertools::Itertools;
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer as _, MTLDevice as _};

use luminal::{
    prelude::{
        petgraph::{
            algo::toposort,
            prelude::StableGraph,
            visit::{EdgeRef, IntoEdgeReferences},
            Direction,
        },
        Graph, GraphTensor, NodeIndex,
    },
    shape::Expression,
};
use rustc_hash::FxHashMap;
use std::{collections::HashMap, ffi::c_void, ptr::NonNull};
use std::{fs::File, io::Read as _, io::Write as _};

use crate::Kernel;

#[cfg(feature = "blade")]
static VAR_NAMES: &[&'static str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];
#[cfg(feature = "blade")]
struct BladeShaderData {
    buffers: Vec<gpu::Buffer>,
}
#[cfg(feature = "blade")]
impl gpu::ShaderData for BladeShaderData {
    fn layout() -> gpu::ShaderDataLayout {
        Default::default()
    }
    fn fill(&self, mut ctx: gpu::PipelineContext) {
        use gpu::ShaderBindable as _;
        for (i, buffer) in self.buffers.iter().enumerate() {
            buffer.at(0).bind_to(&mut ctx, i as u32);
        }
    }
}

#[cfg(feature = "metal")]
pub fn chunk_based_search_compiler(
    device: &Device,
    original_graph: Graph,
    original_graph_input: Vec<(GraphTensor, Vec<f32>)>,
    original_graph_output: &GraphTensor,
    inits: &[(String, InitData)],
) -> Vec<f32> {
    use objc2::rc::autoreleasepool;

    autoreleasepool(|_| {
        let (mut meta_graph, mut global_map, buffers) = translate_graph(&original_graph);
        // Search each subgraph
        for graph_node in meta_graph.node_indices().collect_vec() {
            let sub_graph = meta_graph.node_weight_mut(graph_node).unwrap();
            // luminal_2::utils::display_graph(&graph, &[]);
            let inputs = make_test_inputs(sub_graph, &original_graph.dyn_map, inits);
            let best_searched_graph = search(
                &sub_graph,
                7,
                &inputs,
                GPUArch::Metal(HashMap::default()),
                &original_graph.dyn_map,
            )
            .unwrap();

            // !! screw it just say that the new only output of this graph is the only output (this doesn't work with multiple outputs)
            // adjust meta-edges
            let old_output: SubGraphNodeIndex =
                sub_graph.externals(Direction::Outgoing).next().unwrap();
            let new_output: OptimalGraphNodeIndex = best_searched_graph
                .externals(Direction::Outgoing)
                .next()
                .unwrap();

            let old_inputs: HashMap<SubGraphNodeIndex, String> = sub_graph // we could improve this with a better global_map
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = sub_graph.node_weight(n).unwrap() {
                        Some((n, label.clone()))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            let new_inputs: HashMap<String, OptimalGraphNodeIndex> = best_searched_graph
                .node_indices()
                .filter_map(|n| {
                    if let GraphTerm::GMEM { label } = best_searched_graph.node_weight(n).unwrap() {
                        Some((label.clone(), n))
                    } else {
                        None
                    }
                })
                .collect::<HashMap<_, _>>();
            *sub_graph = best_searched_graph;
            for edge in meta_graph
                .edges_directed(graph_node, Direction::Outgoing)
                .map(|e| e.id())
                .collect_vec()
            {
                let (input, _) = meta_graph.edge_weight_mut(edge).unwrap();
                *input = new_output;
            }
            // Update old-to-new-mappings
            for (_, (meta, v)) in &mut global_map {
                if *meta != graph_node {
                    continue;
                }
                if *v == old_output {
                    *v = new_output;
                }
                if let Some(gmem_label) = old_inputs.get(v) {
                    *v = new_inputs[gmem_label];
                }
            }
        }

        let outputs = vec![global_map[&original_graph_output.id]];
        let (new_graph, meta_to_unified, outputs) = stitch_meta_graph_together(meta_graph, outputs);
        let mut new_old_to_new_mapping = FxHashMap::default();
        for (k, v) in global_map {
            new_old_to_new_mapping.insert(k, meta_to_unified[&v]);
        }
        // luminal_2::utils::display_graph(&new_graph, &[]);
        let (kernels, gmem_mapping) = codegen(
            new_graph.clone(),
            outputs,
            GPUArch::Metal(HashMap::default()),
            0,
            &original_graph.dyn_map,
        )
        .unwrap();

        let mut inputs = FxHashMap::default();

        for (input, data) in original_graph_input {
            inputs.insert(
                gmem_mapping[&new_old_to_new_mapping[&input.id]],
                copy_metal_buffer(&data, &device),
            );
        }
        let mut gmem_to_node_mapping = FxHashMap::default();
        for n in new_graph.node_indices() {
            if let Some(GraphTerm::GMEM { label }) = new_graph.node_weight(n) {
                gmem_to_node_mapping.insert(label, n);
            }
        }

        for (label, val) in &buffers {
            match val {
                InitData::Expr(e) => {
                    let val = e.exec(&original_graph.dyn_map).unwrap();
                    inputs.insert(
                        gmem_mapping[&new_old_to_new_mapping[&gmem_to_node_mapping[label]]],
                        {
                            let v = vec![val as f32];
                            copy_metal_buffer(&v, &device)
                        },
                    );
                }
                InitData::Data(d) => {
                    inputs.insert(
                        gmem_mapping[&new_old_to_new_mapping[&gmem_to_node_mapping[label]]],
                        copy_metal_buffer(d, &device),
                    );
                }
            }
        }
        let compiled_kernels = compile_kernels(device, &kernels);
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let (outputs, _) = run_graph(
            device,
            &StableGraph::default(),
            &inputs,
            &kernels,
            &original_graph.dyn_map,
            &compiled_kernels,
            &int_buffers,
            &int_buffer_map,
        );
        copy_metal_buffer_back(&outputs[0])
    })
}

pub fn assign_buffers(
    graph: &StableGraph<Kernel, (usize, usize)>,
) -> (Vec<Expression>, FxHashMap<NodeIndex, Vec<usize>>) {
    // Count consumers only for producer outputs we manage (exclude "Inputs")
    let mut use_count: FxHashMap<(NodeIndex, usize), usize> = FxHashMap::default();
    for e in graph.edge_references() {
        let src = e.source();
        if graph[src].code != "Inputs" {
            let (src_out, _) = *e.weight();
            *use_count.entry((src, src_out)).or_default() += 1;
        }
    }

    let mut master = vec![]; // capacities by global buffer index
    let mut buf_map = FxHashMap::default(); // node -> output_idx -> buffer_idx
    let mut free_by_cap = FxHashMap::<Expression, Vec<usize>>::default(); // exact-size reuse

    for node in toposort(graph, None).unwrap() {
        let k = &graph[node];
        if k.code == "Inputs" {
            continue; // user-provided; ignore
        }

        // Allocate exact-size buffers for this node's outputs
        let mut outs = vec![];
        for &cap in &k.outputs {
            let buf_idx = if let Some(idx) = free_by_cap.get_mut(&cap).map(|l| l.pop()).flatten() {
                // reuse
                idx
            } else {
                // allocate new buffer
                master.push(cap);
                master.len() - 1
            };
            outs.push(buf_idx);
        }
        buf_map.insert(node, outs);

        // Free producer buffers whose last consumer just ran (exclude "Inputs")
        for e in graph.edges_directed(node, Direction::Incoming) {
            let src = e.source();
            if graph[src].code == "Inputs" {
                continue;
            }
            let (src_out_idx, _) = *e.weight();
            if let Some(c) = use_count.get_mut(&(src, src_out_idx)) {
                *c -= 1;
                if *c == 0 {
                    let buf_idx = buf_map[&src][src_out_idx];
                    free_by_cap
                        .entry(master[buf_idx])
                        .or_default()
                        .push(buf_idx);
                }
            }
        }
    }

    (master, buf_map)
}

#[cfg(feature = "cuda")]
pub fn compile_kernels(
    kernels: &StableGraph<Kernel, (usize, usize)>,
    ctx: &cudarc::driver::CudaContext,
) -> FxHashMap<String, CudaFunction> {
    let mut compiled = FxHashMap::default();

    // Open (or create) the log file, appending logs to it
    let log_path = "kernel_log.txt";
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // overwrite on each run
        .open(log_path)
        .expect("Failed to open kernel log file");

    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            writeln!(log_file, "Compiling kernel:\n{}\n", kernel.code)
                .expect("Failed to write to kernel log file");

            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                &kernel.code,
                CompileOptions {
                    include_paths: vec!["/usr/include".into()],
                    options: vec![
                        "--gpu-architecture=compute_75".into(),
                        "--relocatable-device-code=false".into(),
                        "--std=c++14".into(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap();
            let module = ctx.load_module(ptx).unwrap();
            let k = module.load_function("kernel_name").unwrap();
            compiled.insert(kernel.code.clone(), k);
        }
    }
    compiled
}

#[cfg(feature = "metal")]
pub fn compile_kernels(
    device: &Device,
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, Function> {
    // Open (or create) the log file, appending logs to it
    let log_path = "kernel_log.txt";
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // overwrite on each run
        .open(log_path)
        .expect("Failed to open kernel log file");

    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            use objc2_foundation::{ns_string, NSString};
            use objc2_metal::MTLLibrary;

            writeln!(log_file, "Compiling kernel:\n{}\n", kernel.code)
                .expect("Failed to write to kernel log file");

            let lib = device
                .newLibraryWithSource_options_error(&NSString::from_str(&kernel.code), None)
                .unwrap();
            let f = lib.newFunctionWithName(ns_string!("kernel_name")).unwrap();
            compiled.insert(kernel.code.clone(), f);
        }
    }
    compiled
}

#[cfg(feature = "blade")]
pub fn compile_kernels(
    ctx: &gpu::Context,
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, gpu::Shader> {
    let log_path = "kernel_log.txt";
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // overwrite on each run
        .open(log_path)
        .expect("Failed to open kernel log file");

    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            writeln!(log_file, "Compiling kernel:\n{}\n", kernel.code)
                .expect("Failed to write to kernel log file");

            let shader = ctx.create_shader(gpu::ShaderDesc {
                source: &kernel.code,
            });
            compiled.insert(kernel.code.clone(), shader);
        }
    }
    compiled
}

#[cfg(feature = "cuda")]
pub fn run_graph(
    ctx: &cudarc::driver::CudaContext,
    inputs: &FxHashMap<usize, CudaSlice<f32>>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, CudaFunction>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<CudaSlice<f32>>, u128) {
    let stream = ctx.default_stream();
    let start = std::time::Instant::now();

    // Allocate intermediate buffers
    let mut buffers = intermediate_buffers
        .iter()
        .map(|e| unsafe { stream.alloc(e.exec(dyn_vars).unwrap()).unwrap() })
        .collect_vec();
    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();
    for node in toposort(kernels, None).unwrap() {
        let kernel = &kernels[node];
        if kernel.code == "Inputs" {
            // Inputs should already be in the buffer map
        } else if kernel.code == "Outputs" {
            // Run
            stream.synchronize().unwrap(); // There shouldn't be any other syncs from dispatch till here
            let outputs = kernels
                .edges_directed(node, Direction::Incoming)
                .map(|e| {
                    (
                        e.weight().1,
                        intermediate_buffer_map[&e.source()][e.weight().0],
                    )
                })
                .sorted_by_key(|(_, b)| *b)
                .rev()
                .map(|(a, b)| (a, buffers.remove(b)))
                .sorted_by_key(|(a, _)| *a)
                .map(|(_, a)| a)
                .collect_vec();
            return (outputs, start.elapsed().as_micros());
        } else if kernel.code.starts_with("Diff") {
            // Load file and diff numbers
            let diff_name = kernel.code.replace("Diff", "");
            let (input, input_index) = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .next()
                .unwrap();
            let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
            let data: Vec<f32> = stream.memcpy_dtov(buffer).unwrap();
            let mut file = File::open(format!("{diff_name}.bin")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

            let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
            let floats: Vec<f32> = unsafe {
                let ptr = file_buffer.as_ptr() as *const f32;
                Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
            };
            let mut matched = true;
            println!("Diff {} | {}", data.len(), floats.len());
            for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                if (i - j).abs() > 1e-5 {
                    matched = false;
                    println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                    break;
                }
            }
            std::mem::forget(file_buffer);
            if matched {
                println!("DIFF {diff_name} MATCHED");
            }
            let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
            stream.memcpy_htod(&data, dest_buffer).unwrap();
        } else {
            let mut builder = stream.launch_builder(&compiled_kernels[&kernel.code]);
            println!("Code to run: {}", kernel.code);

            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                if input == input_node {
                    builder.arg(&inputs[&input_index]);
                } else {
                    builder.arg(&buffers[intermediate_buffer_map[&input][input_index]]);
                }
            }
            // set output
            let mut output_views = (0..kernel.outputs.len())
                .map(|o| buffers[intermediate_buffer_map[&node][o]].as_view_mut())
                .collect_vec();
            for o in &mut output_views {
                builder.arg(o);
            }
            // set dynamic dimensions
            for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                builder.arg(v);
            }

            // Set dispatch
            let grid = (
                kernel.grid.0.exec(dyn_vars).unwrap() as u32,
                kernel.grid.1.exec(dyn_vars).unwrap() as u32,
                kernel.grid.2.exec(dyn_vars).unwrap() as u32,
            );
            let tb = (
                kernel.threadblock.0.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.1.exec(dyn_vars).unwrap() as u32,
                kernel.threadblock.2.exec(dyn_vars).unwrap() as u32,
            );
            assert!(
                tb.0 * tb.1 * tb.2 <= 1024,
                "threadblock is too big: {tb:?} > 1024"
            );
            assert!(grid.1 <= 65535, "grid.y > 65535");
            assert!(grid.2 <= 65535, "grid.z > 65535");
            assert!(grid.0 <= 2147483647, "grid.x > 2147483647");
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: tb,
                    shared_mem_bytes: kernel.smem.exec(dyn_vars).unwrap() as u32,
                })
            }
            .unwrap();
        }
    }
    panic!("No output kernel detected in graph!");
}

#[cfg(feature = "metal")]
pub fn run_graph(
    device: &Device,
    graph: &StableGraph<GraphTerm, ()>,
    inputs: &FxHashMap<usize, Buffer>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, Function>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Buffer>, u128) {
    objc2::rc::autoreleasepool(|_| {
        use objc2_metal::MTLCommandQueue;

        let queue = device.newCommandQueue().expect("No command queue");
        let command_buffer = queue.commandBuffer().unwrap();
        let start = std::time::Instant::now();

        // Allocate intermediate buffers
        let mut buffers = intermediate_buffers
            .iter()
            .map(|e| {
                use objc2_metal::MTLResourceOptions;

                device
                    .newBufferWithLength_options(
                        e.exec(dyn_vars).unwrap() * size_of::<f32>(),
                        MTLResourceOptions::StorageModeShared,
                    )
                    .unwrap()
            })
            .collect_vec();
        let input_node = kernels
            .node_indices()
            .find(|n| kernels[*n].code == "Inputs")
            .unwrap();
        for node in toposort(kernels, None).unwrap() {
            let kernel = &kernels[node];
            // println!("Our wonderful kernel: {:?}", kernel);
            if kernel.code == "Inputs" {
                // Inputs should already be in the buffer map
            } else if kernel.code == "Outputs" {
                // Run
                use objc2_metal::MTLCommandBuffer;
                command_buffer.commit();
                unsafe {
                    command_buffer.waitUntilCompleted();
                }
                let outputs = kernels
                    .edges_directed(node, Direction::Incoming)
                    .map(|e| {
                        (
                            e.weight().1,
                            intermediate_buffer_map[&e.source()][e.weight().0],
                        )
                    })
                    .sorted_by_key(|(_, b)| *b)
                    .rev()
                    .map(|(a, b)| (a, buffers.remove(b)))
                    .sorted_by_key(|(a, _)| *a)
                    .map(|(_, a)| a)
                    .collect_vec();
                return (outputs, start.elapsed().as_micros());
            } else if kernel.code.starts_with("Diff") {
                // Load file and diff numbers

                use objc2_metal::MTLBuffer;
                let diff_name = kernel.code.replace("Diff", "");
                let (input, input_index) = kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                    .next()
                    .unwrap();
                let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
                let mut data = vec![0_f32; buffer.length() as usize / size_of::<f32>()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents().as_ptr() as *const _,
                        &mut data,
                        data.len(),
                    );
                }
                let mut file = File::open(format!("{diff_name}.bin")).unwrap();
                let mut file_buffer = Vec::new();
                file.read_to_end(&mut file_buffer).unwrap();
                assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

                let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
                let floats: Vec<f32> = unsafe {
                    let ptr = file_buffer.as_ptr() as *const f32;
                    Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
                };
                let mut matched = true;
                println!("Diff {} | {}", data.len(), floats.len());
                for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                    if (i - j).abs() > 1e-5 {
                        matched = false;
                        println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                        break;
                    }
                }
                std::mem::forget(file_buffer);
                if matched {
                    println!("DIFF {diff_name} MATCHED");
                }
                let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        &data,
                        dest_buffer.contents().as_ptr() as *mut _,
                        data.len(),
                    );
                }
            } else {
                use objc2_metal::{
                    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
                };
                let encoder = command_buffer.computeCommandEncoder().unwrap();
                let Ok(c) = device
                    .newComputePipelineStateWithFunction_error(&compiled_kernels[&kernel.code])
                else {
                    println!("failed to compile {}", kernel.code);
                    crate::debug::display_graph(graph);
                    panic!();
                };
                encoder.setComputePipelineState(&c);

                // set inputs
                let mut buffer_count = 0;

                for (input, input_index) in kernels
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|n| n.weight().1)
                    .map(|n| (n.source(), n.weight().0))
                {
                    if input == input_node {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(&inputs[&input_index]),
                                0,
                                buffer_count,
                            );
                        }
                    } else {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(&buffers[intermediate_buffer_map[&input][input_index]]),
                                0,
                                buffer_count,
                            );
                        }
                    }
                    buffer_count += 1;
                }
                // set output
                for o in 0..kernel.outputs.len() {
                    unsafe {
                        encoder.setBuffer_offset_atIndex(
                            Some(&buffers[intermediate_buffer_map[&node][o]]),
                            0,
                            buffer_count,
                        );
                    }
                    buffer_count += 1;
                }
                // set dynamic dimensions
                for (_, v) in dyn_vars.iter().sorted_by_key(|(k, _)| **k) {
                    let val: u64 = *v as u64;
                    let buf = unsafe {
                        use std::{ffi::c_void, ptr::NonNull};

                        use objc2_metal::MTLResourceOptions;

                        device
                            .newBufferWithBytes_length_options(
                                NonNull::new(&val as *const _ as *mut c_void).unwrap(),
                                std::mem::size_of::<u64>(),
                                MTLResourceOptions::StorageModeShared,
                            )
                            .unwrap()
                    };
                    unsafe { encoder.setBuffer_offset_atIndex(Some(&buf), 0, buffer_count) };
                    buffer_count += 1;
                }

                // Set dispatch
                let grid = (
                    kernel.grid.0.exec(dyn_vars).unwrap(),
                    kernel.grid.1.exec(dyn_vars).unwrap(),
                    kernel.grid.2.exec(dyn_vars).unwrap(),
                );
                let tb = (
                    kernel.threadblock.0.exec(dyn_vars).unwrap(),
                    kernel.threadblock.1.exec(dyn_vars).unwrap(),
                    kernel.threadblock.2.exec(dyn_vars).unwrap(),
                );
                assert!(
                    tb.0 * tb.1 * tb.2 <= 1024,
                    "threadblock is too big: {tb:?} > 1024"
                );
                assert!(grid.1 <= 65535, "grid.y > 65535");
                assert!(grid.2 <= 65535, "grid.z > 65535");
                assert!(grid.0 <= 2147483647, "grid.x > 2147483647");
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: grid.0,
                        height: grid.1,
                        depth: grid.2,
                    },
                    MTLSize {
                        width: tb.0,
                        height: tb.1,
                        depth: tb.2,
                    },
                );
                encoder.endEncoding();
            }
        }
        panic!("No output kernel detected in graph!");
    })
}

#[cfg(feature = "blade")]
pub fn run_graph(
    ctx: &gpu::Context,
    inputs: &FxHashMap<usize, super::Buffer>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, gpu::Shader>,
    intermediate_buffers: &Vec<Expression>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Vec<f32>>, u128) {
    let mut command_buffer = ctx.create_command_encoder(gpu::CommandEncoderDesc {
        name: "main",
        buffer_count: 1,
    });
    command_buffer.start();

    let start = std::time::Instant::now();
    let mut pipelines = Vec::new();
    let mut extra_buffers = Vec::new();

    // Allocate intermediate buffers
    let mut buffers = intermediate_buffers
        .iter()
        .map(|e| {
            let count = e.exec(dyn_vars).unwrap();
            let size = count * size_of::<f32>();
            let raw = ctx.create_buffer(gpu::BufferDesc {
                name: "",
                size: size as u64,
                //TODO: only share the outputs
                memory: gpu::Memory::Shared,
            });
            super::Buffer { raw, size }
        })
        .collect_vec();
    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();
    for node in toposort(kernels, None).unwrap() {
        let kernel = &kernels[node];
        // println!("Our wonderful kernel: {:?}", kernel);
        if kernel.code == "Inputs" {
            // Inputs should already be in the buffer map
        } else if kernel.code == "Outputs" {
            //TODO: schedule copies from Device memory into Host
            // Run
            let sync_point = ctx.submit(&mut command_buffer);
            ctx.wait_for(&sync_point, !0);

            let outputs = kernels
                .edges_directed(node, Direction::Incoming)
                .map(|e| {
                    (
                        e.weight().1,
                        intermediate_buffer_map[&e.source()][e.weight().0],
                    )
                })
                .sorted_by_key(|(_, b)| *b)
                .rev()
                .map(|(a, b)| (a, copy_blade_buffer_back(&buffers[b])))
                .sorted_by_key(|(a, _)| *a)
                .map(|(_, a)| a)
                .collect_vec();

            // Clean up
            ctx.destroy_command_encoder(&mut command_buffer);
            for pipeline in pipelines.iter_mut() {
                ctx.destroy_compute_pipeline(pipeline);
            }
            for buffer in buffers {
                ctx.destroy_buffer(buffer.raw);
            }
            for buffer in extra_buffers {
                ctx.destroy_buffer(buffer);
            }

            return (outputs, start.elapsed().as_micros());
        } else if kernel.code.starts_with("Diff") {
            // Load file and diff numbers

            let diff_name = kernel.code.replace("Diff", "");
            let (input, input_index) = kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
                .next()
                .unwrap();
            let buffer = &buffers[intermediate_buffer_map[&input][input_index]];
            //TODO: remove this copy, we can compare directly
            let mut data = vec![0_f32; buffer.size / size_of::<f32>()];
            unsafe {
                std::ptr::copy_nonoverlapping(buffer.raw.data() as *const _, &mut data, data.len());
            }
            let mut file = File::open(format!("{diff_name}.bin")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

            let num_floats = file_buffer.len() / std::mem::size_of::<f32>();
            let floats: Vec<f32> = unsafe {
                let ptr = file_buffer.as_ptr() as *const f32;
                Vec::from_raw_parts(ptr as *mut f32, num_floats, num_floats)
            };
            let mut matched = true;
            println!("Diff {} | {}", data.len(), floats.len());
            for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                if (i - j).abs() > 1e-5 {
                    matched = false;
                    println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                    break;
                }
            }
            std::mem::forget(file_buffer);
            if matched {
                println!("DIFF {diff_name} MATCHED");
            }
            let dest_buffer = &mut buffers[intermediate_buffer_map[&node][0]];
            unsafe {
                std::ptr::copy_nonoverlapping(&data, dest_buffer.raw.data() as *mut _, data.len());
            }
        } else {
            //Note: this is baked into the shader. Do `dyn_vars` change that?
            let tb = (
                kernel.threadblock.0.exec(dyn_vars).unwrap(),
                kernel.threadblock.1.exec(dyn_vars).unwrap(),
                kernel.threadblock.2.exec(dyn_vars).unwrap(),
            );
            assert!(
                tb.0 * tb.1 * tb.2 <= 1024,
                "threadblock is too big: {tb:?} > 1024"
            );

            //HACK: relying on `node_to_var` mapping to be constructed this way in codegen
            let num_inputs = kernels.edges_directed(node, Direction::Incoming).count();
            let mut layout = gpu::ShaderDataLayout {
                bindings: (0..num_inputs + kernel.outputs.len())
                    .map(|i| (VAR_NAMES[i], gpu::ShaderBinding::Buffer))
                    .collect(),
            };
            let mut shader_data = BladeShaderData {
                buffers: Vec::new(),
            };
            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                shader_data.buffers.push(if input == input_node {
                    inputs[&input_index].raw
                } else {
                    buffers[intermediate_buffer_map[&input][input_index]].raw
                });
            }
            // set output
            for output_index in 0..kernel.outputs.len() {
                shader_data
                    .buffers
                    .push(buffers[intermediate_buffer_map[&node][output_index]].raw);
            }
            // set dynamic dimensions
            if !dyn_vars.is_empty() {
                let temp_buffer = ctx.create_buffer(gpu::BufferDesc {
                    name: "dyn_vars",
                    size: dyn_vars.len() as u64 * 4,
                    memory: gpu::Memory::Shared,
                });
                for (i, (_k, &v)) in dyn_vars.iter().enumerate() {
                    unsafe {
                        *(temp_buffer.data() as *mut u32).add(i) = v as u32;
                    }
                }
                layout
                    .bindings
                    .push(("dyn_vars", gpu::ShaderBinding::Buffer));
                shader_data.buffers.push(temp_buffer);
                extra_buffers.push(temp_buffer);
            }

            let pipeline = ctx.create_compute_pipeline(gpu::ComputePipelineDesc {
                name: "",
                data_layouts: &[&layout],
                compute: compiled_kernels[&kernel.code].at("main"),
            });

            //TODO: share the pass between independent kernels of the same rank
            let mut pass = command_buffer.compute("");
            {
                let mut encoder = pass.with(&pipeline);
                // Bind all used resources
                encoder.bind(0, &shader_data);

                // Set dispatch
                let grid = [
                    kernel.grid.0.exec(dyn_vars).unwrap() as u32,
                    kernel.grid.1.exec(dyn_vars).unwrap() as u32,
                    kernel.grid.2.exec(dyn_vars).unwrap() as u32,
                ];
                assert!(grid[0] <= 2147483647, "grid.x > 2147483647");
                assert!(grid[1] <= 65535, "grid.y > 65535");
                assert!(grid[2] <= 65535, "grid.z > 65535");
                encoder.dispatch(grid);
            }
            pipelines.push(pipeline);
        }
    }
    panic!("No output kernel detected in graph!");
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    let buf = unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(v.as_ptr() as *mut c_void).unwrap(),
                v.len() * std::mem::size_of::<f32>(),
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    };
    buf
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer_back(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

#[cfg(feature = "blade")]
pub fn copy_blade_buffer_back(v: &super::Buffer) -> Vec<f32> {
    assert!(!v.raw.data().is_null(), "Buffer is not mappable");
    unsafe { std::slice::from_raw_parts(v.raw.data() as *const f32, v.size / size_of::<f32>()) }
        .to_vec()
}
