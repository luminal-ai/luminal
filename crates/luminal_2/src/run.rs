use itertools::Itertools;

#[cfg(feature = "cuda")]
use cudarc::{driver::*, nvrtc::CompileOptions};
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Direction,
            algo::toposort,
            prelude::StableGraph,
            visit::{EdgeRef, IntoEdgeReferences},
        },
    },
    shape::Expression,
};
use rustc_hash::FxHashMap;
use std::{fs::File, io::Read};
#[cfg(feature = "metal")]
use {
    crate::{Device, Function},
    objc2_metal::{MTLBuffer, MTLDevice},
    std::{ffi::c_void, ptr::NonNull},
};

use crate::Buffer;
#[cfg(feature = "cuda")]
use crate::GraphTerm;
use crate::Kernel;

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
        } // Can replace with look ups in the use_count -> if there, then not an Input

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
            }  // Can replace with look ups in the use_count -> if there, then not an Input
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
) -> FxHashMap<String, CudaFunction> {
    let ctx = shared_cuda_context();
    let mut compiled = FxHashMap::default();

    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
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
    kernels: &StableGraph<Kernel, (usize, usize)>,
) -> FxHashMap<String, Function> {
    use objc2_metal::MTLCreateSystemDefaultDevice;

    let device = MTLCreateSystemDefaultDevice().unwrap();
    let mut compiled = FxHashMap::default();
    for kernel in kernels.node_weights() {
        if !compiled.contains_key(&kernel.code)
            && kernel.code != "Inputs"
            && kernel.code != "Outputs"
        {
            use objc2_foundation::{NSString, ns_string};
            use objc2_metal::{MTLDevice, MTLLibrary};
            let lib = device
                .newLibraryWithSource_options_error(&NSString::from_str(&kernel.code), None)
                .unwrap();
            let f = lib.newFunctionWithName(ns_string!("kernel_name")).unwrap();
            compiled.insert(kernel.code.clone(), f);
        }
    }
    compiled
}

#[cfg(feature = "cuda")]
pub fn run_graph(
    inputs: &mut FxHashMap<usize, (&CudaSlice<f32>, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, CudaFunction>, // wonder how this impacts performance inserting a string of kernel code vs inserting an object (e.g. struct with code as string)
    intermediate_buffers: &mut Vec<Buffer>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Vec<f32>>, u128) {
    let ctx = shared_cuda_context();
    let stream = ctx.default_stream();
    let start = std::time::Instant::now();

    let input_node = kernels
        .node_indices()
        .find(|n| kernels[*n].code == "Inputs")
        .unwrap();
    for (i, node) in toposort(kernels, None).unwrap().into_iter().enumerate() {
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
                .map(|(a, b)| (a, dtoh(&intermediate_buffers[b])))
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
            let buffer = &intermediate_buffers[intermediate_buffer_map[&input][input_index]];
            let data: Vec<f32> = stream.memcpy_dtov(buffer).unwrap();
            let mut file = File::open(format!("{diff_name}.bin")).unwrap();
            let mut file_buffer = Vec::new();
            file.read_to_end(&mut file_buffer).unwrap();
            assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

            let floats: Vec<f32> = file_buffer
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            let mut matched = true;
            println!("Diff {} | {}", data.len(), floats.len());
            for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                if (i - j).abs() > 1e-5 {
                    matched = false;
                    println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                    break;
                }
            }

            if matched {
                println!("DIFF {diff_name} MATCHED");
            }
            let dest_buffer = &mut intermediate_buffers[intermediate_buffer_map[&node][0]];
            stream.memcpy_htod(&data, dest_buffer).unwrap();
        } else {
            // println!("{i}: {}", kernel.code);
            let mut builder = stream.launch_builder(&compiled_kernels[&kernel.code]);
            // set inputs
            for (input, input_index) in kernels
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|n| n.weight().1)
                .map(|n| (n.source(), n.weight().0))
            {
                if input == input_node {
                    // println!("INPUT: {}", inputs[&input_index].0.len());
                    builder.arg(inputs[&input_index].0);
                } else {
                    // println!(
                    //     "INPUT: {}",
                    //     intermediate_buffers[intermediate_buffer_map[&input][input_index]].len()
                    // );
                    builder
                        .arg(&intermediate_buffers[intermediate_buffer_map[&input][input_index]]);
                }
            }
            // set output
            // println!("Grid: {:?}", kernel.grid);
            // println!("Threadblock: {:?}", kernel.threadblock);
            // println!("OUTPUTS: {:?}", kernel.outputs);
            let mut output_views = (0..kernel.outputs.len())
                .map(|o| intermediate_buffers[intermediate_buffer_map[&node][o]].as_view_mut())
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
            // if let Err(e) = stream.synchronize() {
            //     use crate::codegen::split_kernels_marked_graph;

            //     std::fs::write(
            //         "graph.txt",
            //         crate::debug::display_graph_text(&split_kernels_marked_graph(graph), &[]),
            //     )
            //     .unwrap();
            //     panic!("{:?}", e);
            // }
        }
    }
    panic!("No output kernel detected in graph!");
}

#[cfg(feature = "metal")]
pub fn run_graph(
    inputs: &mut FxHashMap<usize, (&Buffer, bool)>,
    kernels: &StableGraph<Kernel, (usize, usize)>,
    dyn_vars: &FxHashMap<char, usize>,
    compiled_kernels: &FxHashMap<String, Function>,
    intermediate_buffers: &mut Vec<Buffer>,
    intermediate_buffer_map: &FxHashMap<NodeIndex, Vec<usize>>,
) -> (Vec<Vec<f32>>, u128) {
    objc2::rc::autoreleasepool(|_| {
        use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

        let device = MTLCreateSystemDefaultDevice().unwrap();
        let queue = device.newCommandQueue().expect("No command queue");
        let command_buffer = queue.commandBuffer().unwrap();
        let start = std::time::Instant::now();

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
                    .map(|(a, b)| (a, dtoh(&intermediate_buffers[b])))
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
                let buffer = &intermediate_buffers[intermediate_buffer_map[&input][input_index]];
                let mut data = vec![0_f32; buffer.length() as usize / size_of::<f32>()];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buffer.contents().as_ptr() as *const f32,
                        data.as_mut_ptr(),
                        data.len(),
                    );
                }
                let mut file = File::open(format!("{diff_name}.bin")).unwrap();
                let mut file_buffer = Vec::new();
                file.read_to_end(&mut file_buffer).unwrap();
                assert_eq!(file_buffer.len() % std::mem::size_of::<f32>(), 0);

                let _num_floats = file_buffer.len() / std::mem::size_of::<f32>();
                let floats: Vec<f32> = file_buffer
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                let mut matched = true;
                println!("Diff {} | {}", data.len(), floats.len());
                for (ind, (i, j)) in data.iter().zip(floats).enumerate() {
                    if (i - j).abs() > 1e-5 {
                        matched = false;
                        println!("Diff {diff_name} failed: curr: {i} != file: {j}, index {ind}");
                        break;
                    }
                }

                if matched {
                    println!("DIFF {diff_name} MATCHED");
                }
                let dest_buffer = &mut intermediate_buffers[intermediate_buffer_map[&node][0]];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        dest_buffer.contents().as_ptr() as *mut f32,
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
                    panic!("failed to compile {}", kernel.code);
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
                                Some(&inputs[&input_index].0),
                                0,
                                buffer_count,
                            );
                        }
                    } else {
                        unsafe {
                            encoder.setBuffer_offset_atIndex(
                                Some(
                                    &intermediate_buffers
                                        [intermediate_buffer_map[&input][input_index]],
                                ),
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
                            Some(&intermediate_buffers[intermediate_buffer_map[&node][o]]),
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

#[cfg(feature = "cuda")]
pub fn htod(v: &[f32], ctx: &std::sync::Arc<CudaContext>) -> CudaSlice<f32> {
    assert!(!v.is_empty(), "Can't copy empty slice to device");

    // Then copy host data to the allocated device memory
    let stream = ctx.default_stream();
    let mut dst = stream.alloc_zeros::<f32>(v.len()).unwrap();
    stream.memcpy_htod(v, &mut dst).unwrap();
    dst
}

/// Device -> Host (like contents() memcpy back)
#[cfg(feature = "cuda")]
pub fn dtoh(buf: &CudaSlice<f32>) -> Vec<f32> {
    buf.stream().memcpy_dtov(buf).unwrap()
}

#[cfg(feature = "metal")]
pub fn htod(v: &Vec<f32>, device: &Device) -> Buffer {
    assert!(v.len() > 0);
    unsafe {
        let ptr = NonNull::new(v.as_ptr() as *mut c_void).unwrap();
        device
            .newBufferWithBytes_length_options(
                ptr,
                (v.len() * 4) as _,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
}
#[cfg(feature = "metal")]
pub fn dtoh(v: &Buffer) -> Vec<f32> {
    let mut data = vec![0f32; v.length() as usize / size_of::<f32>()];
    let ptr = v.contents().as_ptr() as *mut f32;
    for (i, d) in data.iter_mut().enumerate() {
        *d = unsafe { *ptr.add(i) };
    }
    data
}

#[cfg(feature = "metal")]
pub fn new_buffer(size: usize) -> Buffer {
    objc2_metal::MTLCreateSystemDefaultDevice()
        .unwrap()
        .newBufferWithLength_options(size, objc2_metal::MTLResourceOptions::StorageModeShared)
        .unwrap()
}

#[cfg(feature = "cuda")]
pub fn new_buffer(size: usize) -> Buffer {
    let ctx = shared_cuda_context();
    let stream = ctx.default_stream();
    stream.alloc_zeros::<f32>(size / size_of::<f32>()).unwrap()
}

#[cfg(feature = "cuda")]
pub fn shared_cuda_context() -> std::sync::Arc<CudaContext> {
    static CTX: OnceLock<std::sync::Arc<CudaContext>> = OnceLock::new();
    CTX.get_or_init(|| CudaContext::new(0).unwrap()).clone()
}
