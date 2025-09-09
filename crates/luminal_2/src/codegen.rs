use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Directed, Direction,
            algo::toposort,
            prelude::StableGraph,
            unionfind::UnionFind,
            visit::{EdgeRef, NodeIndexable},
        },
    },
    shape::{Expression, Term},
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    iter::repeat,
};

use crate::{
    GMEMBuffer, GPUArch, GraphTerm, Kernel,
    debug::{display_graph, display_graph2},
    translate::{MetaGraph, SubGraph},
    utils::validate_graph,
};

pub const GRID_DIMS: usize = 2;
pub const THREADBLOCK_DIMS: usize = 2;
pub const MAX_THREADBLOCK_SIZE: usize = 1024; // this is max on mac
pub const MAX_GRID_X: usize = 2147483647;
pub const MAX_GRID_YZ: usize = 65535;

pub fn codegen(
    mut graph: StableGraph<GraphTerm, (), Directed>,
    mut arch: GPUArch,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<(
    StableGraph<Kernel, (usize, usize), Directed>,
    HashMap<NodeIndex, usize>,
)> {
    let gmems = graph
        .node_weights()
        .filter(|w| matches!(w, GraphTerm::GMEM { .. }))
        .count();
    // Remove free-standing GMEMs
    for node in graph.node_indices().collect_vec() {
        if matches!(graph.node_weight(node).unwrap(), GraphTerm::GMEM { .. })
            && graph.neighbors_undirected(node).next().is_none()
        {
            graph.remove_node(node);
        }
    }
    let (kernels, output_kernels) = split_kernels(graph.clone());
    // Create kernel meta graph to toposort
    let mut kernel_meta_graph = StableGraph::new();
    for _ in 0..kernels.len() {
        kernel_meta_graph.add_node(Kernel::default());
    }
    let global_input = kernel_meta_graph.add_node(Kernel {
        code: "Inputs".to_string(),
        outputs: vec![Expression::from('-'); gmems],
        ..Default::default()
    });
    let global_output = kernel_meta_graph.add_node(Kernel {
        code: "Outputs".to_string(),
        ..Default::default()
    });
    let mut gmem_mapping: HashMap<NodeIndex, usize> = HashMap::new();
    let mut gmem_names = FxHashMap::default();
    for (n_kernel, (_, inputs, _, _)) in kernels.iter().enumerate() {
        for (n_input, (input_kernel, _)) in inputs.into_iter().enumerate() {
            match input_kernel {
                GMEMBuffer::PrevKernel { kernel, output } => kernel_meta_graph.add_edge(
                    NodeIndex::new(*kernel),
                    NodeIndex::new(n_kernel),
                    (*output, n_input),
                ),
                GMEMBuffer::Input { node } => {
                    let index = if let Some(index) = gmem_mapping.get(node) {
                        *index
                    } else {
                        gmem_mapping.insert(*node, gmem_mapping.len());
                        gmem_mapping.len() - 1
                    };
                    let GraphTerm::GMEM { label } = graph.node_weight(*node).unwrap() else {
                        panic!()
                    };
                    gmem_names.insert(*node, label.clone());
                    kernel_meta_graph.add_edge(
                        global_input,
                        NodeIndex::new(n_kernel),
                        (index, n_input),
                    )
                }
            };
        }
    }
    for (i, (output_kernel, output_index)) in output_kernels.clone().into_iter().enumerate() {
        kernel_meta_graph.add_edge(
            NodeIndex::new(output_kernel),
            global_output,
            (output_index, i),
        );
    }
    // display_graph2(&kernel_meta_graph, &[]);
    let Ok(t) = toposort(&kernel_meta_graph, None) else {
        // TODO: for some reason sometimes there are cycles in the kernel graph
        return None;
    };
    for node in t {
        if kernels.len() <= node.index() {
            continue; // Either input node or output node
        }
        let (kernel_graph, inputs, outputs, smem_buffers) = kernels[node.index()].clone();
        // Handle custom kernels
        if kernel_graph
            .node_weights()
            .all(|(n, _)| matches!(n, GraphTerm::GMEM { .. } | GraphTerm::Custom(_)))
        {
            let (GraphTerm::Custom(custom_kernel), _) = kernel_graph
                .node_weights()
                .find(|(n, _)| matches!(n, GraphTerm::Custom(_)))
                .unwrap()
            else {
                panic!("invalid kernel!")
            };
            // Would the input ordering be valid? lets assume it is for now
            *kernel_meta_graph.node_weight_mut(node).unwrap() = custom_kernel.clone();
            continue;
        }

        // Handle diffs
        if kernel_graph
            .node_weights()
            .all(|(n, _)| matches!(n, GraphTerm::GMEM { .. } | GraphTerm::Diff(_)))
        {
            let (GraphTerm::Diff(diff), _) = kernel_graph
                .node_weights()
                .find(|(n, _)| matches!(n, GraphTerm::Diff(_)))
                .unwrap()
            else {
                panic!("invalid kernel!")
            };
            kernel_meta_graph.node_weight_mut(node).unwrap().code = format!("Diff{diff}");
            continue;
        }

        validate_graph(&kernel_graph);
        // display_graph(&kernel_graph, &[]);
        let mut node_to_var = inputs
            .iter()
            .map(|(_, n)| *n)
            .chain(outputs.iter().map(|(_, i)| *i))
            .chain(smem_buffers.iter().map(|(_, i, _)| *i))
            .enumerate()
            .map(|(v, n)| (n, (v, true)))
            .collect::<HashMap<_, _>>();
        for (_, (n, _)) in &node_to_var {
            arch.add_metal_buffer_type(*n, "device ");
        }
        for (n, _, _) in &smem_buffers {
            arch.add_metal_buffer_type(*n, "threadgroup ");
        }
        let mut loop_levels = vec![];
        let kernel = make_kernel(
            &kernel_graph,
            kernel_graph.node_indices().collect(),
            &mut node_to_var,
            &mut (inputs.len() + outputs.len()),
            &mut loop_levels,
            &mut HashMap::new(),
            &smem_buffers
                .iter()
                .map(|(ind, node, _)| (*node, *ind))
                .collect(),
            0,
            0,
            &mut arch,
        )?;
        let grid = loop_levels
            .clone()
            .into_iter()
            .take(GRID_DIMS)
            .chain(repeat(1.into()))
            .take(3) // Hardware always expects 3 dims
            .collect_vec();
        let threadblock = loop_levels
            .into_iter()
            .skip(GRID_DIMS)
            .take(THREADBLOCK_DIMS)
            .chain(repeat(1.into()))
            .take(3) // Hardware always expects 3 dims
            .collect_vec();
        // Make sure there are no dynamic dimensions in the threadblock
        if threadblock
            .iter()
            .copied()
            .product::<Expression>()
            .to_usize()
            .is_none()
        {
            println!("dyn in tb");
            return None;
        }
        let kernel_lines = kernel.into_iter().map(|s| format!("\t{s}")).join("\n");
        let kernel = match &arch {
            GPUArch::CUDA => {
                let inputs = inputs
                    .into_iter()
                    .map(|(_, a)| a)
                    .chain(outputs.iter().map(|(_, i)| *i))
                    .map(|a| format!("float* {}", var_to_char(node_to_var[&a].0)))
                    .chain(
                        dyn_vars
                            .iter()
                            .sorted_by_key(|(k, _)| **k)
                            .map(|(c, _)| format!("const size_t const_{c}")),
                    )
                    .join(", ");
                let smem_setup = if smem_buffers.is_empty() {
                    "".to_string()
                } else {
                    format!(
                        "\textern __shared__ float sm[];\n{}",
                        smem_buffers
                            .iter()
                            .scan("".to_string(), |prev_buffers, (n, _, size)| {
                                let r =
                                    format!("\tfloat* {} = sm{prev_buffers};\n", var_to_char(*n));
                                prev_buffers.push_str(&format!(" + {size}"));
                                Some(r)
                            })
                            .join("")
                    )
                };
                format!(
                    "extern \"C\" __global__ void kernel_name({inputs}) {{
{smem_setup}{kernel_lines}
}}"
                )
            }
            GPUArch::Metal { .. } => {
                let n_inputs = inputs.len();
                let n_inputs_outputs = inputs.len() + outputs.len();
                let mut input_string = inputs
                    .into_iter()
                    .enumerate()
                    .map(|(i, (buf, a))| {
                        format!(
                            "device float* {} [[buffer({i})]],{}",
                            var_to_char(node_to_var[&a].0),
                            if let GMEMBuffer::Input { node } = buf {
                                format!(" // GMEM({})", gmem_names[&node])
                            } else {
                                "".to_string()
                            }
                        )
                    })
                    .chain(outputs.iter().enumerate().map(|(i, (_, n))| {
                        format!(
                            "device float* {} [[buffer({})]], // Output",
                            var_to_char(node_to_var[&n].0),
                            n_inputs + i
                        )
                    }))
                    .chain(dyn_vars.iter().sorted().enumerate().map(|(i, (c, _))| {
                        format!(
                            "constant uint& const_{c} [[buffer({})]]",
                            i + n_inputs_outputs
                        )
                    }))
                    .collect_vec();
                *input_string.last_mut().unwrap() = input_string.last().unwrap().replace(",", " ");
                let (smem_setup, smem_input) = if smem_buffers.is_empty() {
                    ("".to_string(), "".to_string())
                } else {
                    (
                        smem_buffers
                            .iter()
                            .scan("".to_string(), |prev_buffers, (n, _, size)| {
                                let r = format!(
                                    "\tthreadgroup float* {} = sm{prev_buffers};\n",
                                    var_to_char(*n)
                                );
                                prev_buffers.push_str(&format!(" + {size}"));
                                Some(r)
                            })
                            .join(""),
                        ", threadgroup float* sm [[threadgroup(0)]]".to_string(),
                    )
                };
                format!(
                    "#include <metal_stdlib>
using namespace metal;
kernel void kernel_name(
	uint3 blockIdx [[threadgroup_position_in_grid]],
	uint3 threadIdx [[thread_position_in_threadgroup]],
	{}{smem_input}
) {{
{smem_setup}{kernel_lines}
}}",
                    input_string.join("\n\t")
                )
            }
        };
        // Check threadblock and grid restrictions
        if (threadblock[0] * threadblock[1] * threadblock[2])
            .exec(dyn_vars)
            .unwrap()
            > MAX_THREADBLOCK_SIZE
        {
            // println!("Tb too big: {:?}", threadblock);
            // println!("{kernel}");
            return None;
        }
        if grid[0].exec(dyn_vars).unwrap() > MAX_GRID_X {
            return None;
        }
        if grid[1].exec(dyn_vars).unwrap() > MAX_GRID_YZ {
            return None;
        }
        if grid[2].exec(dyn_vars).unwrap() > MAX_GRID_YZ {
            return None;
        }
        *kernel_meta_graph.node_weight_mut(node).unwrap() = Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            smem: smem_buffers.into_iter().map(|(_, _, a)| a).sum(),
            outputs: outputs.into_iter().map(|(o, _)| o.simplify()).collect(),
        };
    }
    Some((kernel_meta_graph, gmem_mapping))
}

fn var_to_char(var: usize) -> String {
    if var > 25 {
        format!("{}{}", var_to_char(var / 26), var_to_char(var % 26))
    } else {
        ((var + 97) as u8 as char).to_string()
    }
}

fn make_kernel(
    kernel_graph: &StableGraph<(GraphTerm, usize), (), Directed>,
    include_nodes: HashSet<NodeIndex>,
    node_to_var: &mut HashMap<NodeIndex, (usize, bool)>,
    prev_max_var: &mut usize, // contains the char last used
    tracked_loop_levels: &mut Vec<Expression>,
    loop_indexes: &mut HashMap<NodeIndex, usize>,
    smem_buffers: &HashMap<NodeIndex, usize>,
    mut current_loop_level: usize,
    unpadded_loop_level: usize,
    arch: &mut GPUArch,
) -> Option<Vec<String>> {
    let mut kernel_lines = vec![];
    let spacing = (0..current_loop_level.saturating_sub(GRID_DIMS + THREADBLOCK_DIMS))
        .map(|_| "\t")
        .join("");

    // Walk through the toposorted nodes
    for node in toposort_subset(kernel_graph, &include_nodes) {
        if node_to_var.contains_key(&node) || kernel_graph[node].1 != unpadded_loop_level {
            continue;
        }
        let (term, _) = &kernel_graph[node];
        match term {
            GraphTerm::LoopIn { range, stride, .. } => {
                // maybe we tried to merge a loop with an acc
                if stride.terms.read().len() > 1 && stride.is_acc() {
                    panic!("wtf is this {stride}");
                }
                // go through graph to find inputs and outputs
                let mut loop_inputs = HashSet::new();
                loop_inputs.insert((node, stride));
                let mut loop_outputs = HashSet::new();
                let mut dfs = kernel_graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .collect_vec();
                let mut body_nodes = HashSet::new();
                body_nodes.insert(node);
                while let Some(n) = dfs.pop() {
                    if body_nodes.contains(&n) {
                        continue;
                    }
                    body_nodes.insert(n);
                    if let (GraphTerm::LoopOut { stride, .. }, level) = &kernel_graph[n] {
                        if *level == unpadded_loop_level {
                            loop_outputs.insert((n, stride));
                            continue;
                        }
                    } else if let (GraphTerm::LoopIn { stride, .. }, level) = &kernel_graph[n] {
                        if *level == unpadded_loop_level {
                            loop_inputs.insert((n, stride));
                            continue;
                        }
                    }
                    body_nodes.insert(node);
                    dfs.extend(kernel_graph.neighbors_undirected(n));
                }

                if let Some((_, _)) = loop_inputs.iter().find(|(i, _)| {
                    kernel_graph
                        .neighbors_directed(*i, Direction::Incoming)
                        .next()
                        .map(|n| !node_to_var.contains_key(&n))
                        .unwrap_or_default()
                }) {
                    // Not all inputs to this loop are ready
                    continue;
                }
                for (i, _) in loop_inputs.iter().chain(&loop_outputs) {
                    body_nodes.remove(i);
                }

                if loop_inputs.iter().any(|i| i.1.is_acc())
                    && current_loop_level < GRID_DIMS + THREADBLOCK_DIMS
                {
                    current_loop_level = GRID_DIMS + THREADBLOCK_DIMS;
                }

                let inner_spacing = if current_loop_level < GRID_DIMS + THREADBLOCK_DIMS {
                    "".to_string()
                } else {
                    format!("{spacing}\t")
                };

                // Make accumulators
                let mut accs = vec![];
                for (input, stride) in &loop_inputs {
                    if stride.is_acc() {
                        let Term::Acc(acc_symbol) = stride.terms.read()[0] else {
                            panic!("Acc is not acc: {stride}");
                        };
                        // Create a new accumulator
                        // Work out the size needed by taking the max stride of the loopins and loopouts
                        let mut size = Expression::from(1);
                        let mut loads = Expression::from(1);
                        let mut in_loops = vec![];
                        let mut curr = *input;
                        loop {
                            match kernel_graph[curr].0 {
                                GraphTerm::LoopIn { range, stride, .. } => {
                                    if !stride.is_acc() {
                                        size = size.max(stride.substitute('z', range));
                                        loads *= range;
                                        in_loops.push((range, stride));
                                    }
                                }
                                _ => break,
                            }
                            curr = kernel_graph
                                .neighbors_directed(curr, Direction::Outgoing)
                                .next()
                                .unwrap();
                        }
                        let mut indexing_expression = Expression::from(0);
                        let mut current_elem_size = Expression::from(1);
                        for (range, stride) in in_loops.into_iter().rev() {
                            indexing_expression += stride.substitute(
                                'z',
                                (Expression::from('z') / current_elem_size) % range,
                            );
                            current_elem_size *= range;
                        }
                        let Some(corresponding_output) = loop_outputs
                            .iter()
                            .find(|(_, v)| v.terms.read()[0] == Term::Acc(acc_symbol))
                        else {
                            println!("Can't find output with Acc('{acc_symbol}')");
                            display_graph(&kernel_graph);
                            panic!();
                        };
                        let corresponding_output = corresponding_output.0;
                        curr = corresponding_output;
                        let mut out_loops = vec![];
                        loop {
                            match kernel_graph[curr].0 {
                                GraphTerm::LoopOut { range, stride, .. } => {
                                    if !stride.is_acc() {
                                        size = size.max(stride.substitute('z', range));
                                        out_loops.push((range, stride));
                                    }
                                }
                                _ => break,
                            }
                            curr = kernel_graph
                                .neighbors_directed(curr, Direction::Incoming)
                                .next()
                                .unwrap();
                        }
                        // Make accumulator
                        *prev_max_var += 1;
                        arch.add_metal_buffer_type(*prev_max_var, "thread ");
                        kernel_lines.push(format!(
                            "{spacing}{}float {}[{}] = {{0.0}};",
                            arch.metal_buffer_type(*prev_max_var),
                            var_to_char(*prev_max_var),
                            size.to_usize().unwrap()
                        ));

                        // Copy from source to accumulator
                        let outer_input = kernel_graph
                            .neighbors_directed(*input, Direction::Incoming)
                            .next()
                            .unwrap();
                        // Use a single loop with correct striding from the input
                        kernel_lines.push(format!(
                            "{spacing}for (int load = 0; load < {}; ++load) {{",
                            loads.to_kernel()
                        ));
                        let indexing_expression = indexing_expression
                            .simplify()
                            .to_kernel()
                            .replace("const_z", "load");
                        kernel_lines.push(format!(
                            "{inner_spacing}{}[{indexing_expression}] = *({} + {indexing_expression});",
                            var_to_char(*prev_max_var),
                            var_to_char(node_to_var[&outer_input].0),
                        ));
                        kernel_lines.push(format!("{spacing}}}"));
                        node_to_var.insert(*input, (*prev_max_var, true));
                        node_to_var.insert(corresponding_output, (*prev_max_var, true));
                        accs.push((*input, corresponding_output, size, out_loops));
                    }
                }
                // Make thread-level buffers
                let mut created_buffers = FxHashMap::default();
                for (output, stride) in &loop_outputs {
                    let dest = kernel_graph
                        .neighbors_directed(*output, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if !stride.is_acc() && !node_to_var.contains_key(&dest) {
                        // Handle the case where the dest is not the real loop output
                        let size = stride.substitute('z', range).max(1);
                        if current_loop_level < THREADBLOCK_DIMS + GRID_DIMS {
                            println!("too low?");
                            return None;
                        }
                        // We don't have a place to save this output to. Need to allocate a register buffer
                        *prev_max_var += 1;
                        kernel_lines.push(format!(
                            "{spacing}thread float {}[{}] = {{0.0}};",
                            var_to_char(*prev_max_var),
                            size.to_usize().unwrap()
                        ));
                        node_to_var.insert(*output, (*prev_max_var, true));
                        created_buffers.insert(*output, *prev_max_var);
                        arch.add_metal_buffer_type(*prev_max_var, "thread ");
                    }
                }

                // Make new parallel or for loop
                if current_loop_level < GRID_DIMS + THREADBLOCK_DIMS {
                    // parallel
                    if current_loop_level >= tracked_loop_levels.len() {
                        tracked_loop_levels.push(*range);
                    }
                    if loop_inputs
                        .iter()
                        .chain(&loop_outputs)
                        .any(|(_, st)| **st != 0)
                    {
                        *prev_max_var += 1;
                        kernel_lines.push(format!(
                            "int loop_{} = {};",
                            var_to_char(*prev_max_var),
                            if current_loop_level >= GRID_DIMS {
                                ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
                                    [current_loop_level - GRID_DIMS]
                            } else {
                                ["blockIdx.x", "blockIdx.y", "blockIdx.z"][current_loop_level]
                            }
                        ));
                    }
                } else {
                    // for
                    *prev_max_var += 1;
                    let loop_var = var_to_char(*prev_max_var);
                    kernel_lines.push(format!("{spacing}for (int loop_{loop_var} = 0; loop_{loop_var} < {}; ++loop_{loop_var}) {{", range.to_kernel()));
                };
                let loop_var = var_to_char(*prev_max_var);
                let loop_var_int = *prev_max_var;

                // Move input pointers (allocate new variables)
                for (input, stride) in &loop_inputs {
                    let src = kernel_graph
                        .neighbors_directed(*input, Direction::Incoming)
                        .next()
                        .unwrap();
                    let (real_input, is_ptr) = node_to_var[&src].clone();
                    if !stride.is_acc() {
                        if **stride == 0
                            || (*range == 1 && stride.substitute('z', 0).simplify() == 0)
                        {
                            // Either the range is 1 or the stride is zero, so no offset needs to happen
                            node_to_var.insert(*input, (real_input, is_ptr));
                        } else {
                            *prev_max_var += 1;
                            arch.add_metal_buffer_type(
                                *prev_max_var,
                                arch.metal_buffer_type(real_input),
                            );
                            kernel_lines.push(format!(
                                "{inner_spacing}{}float* {} = {} + {};",
                                arch.metal_buffer_type(*prev_max_var),
                                var_to_char(*prev_max_var),
                                var_to_char(real_input),
                                stride
                                    .to_kernel()
                                    .replace("const_z", &format!("loop_{loop_var}"))
                            ));
                            node_to_var.insert(*input, (*prev_max_var, is_ptr));
                        }
                    }
                }
                // Move output pointers (allocate new variables)
                let mut new_output_vars = vec![];
                for (output, stride) in &loop_outputs {
                    if stride.is_acc() {
                        continue; // Accumulators are set up in input handling (above) and results are copied back to output after body (below)
                    }
                    let dest = kernel_graph
                        .neighbors_directed(*output, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if let Some((real_output, is_ptr)) = node_to_var.get(&dest).copied() {
                        if **stride == 0
                            || (*range == 1 && stride.substitute('z', 0).simplify() == 0)
                        {
                            new_output_vars.push(real_output);
                            node_to_var.insert(*output, (real_output, is_ptr));
                        } else {
                            assert!(is_ptr, "Only pointers can be offset!");
                            *prev_max_var += 1;
                            arch.add_metal_buffer_type(
                                *prev_max_var,
                                arch.metal_buffer_type(real_output),
                            );
                            kernel_lines.push(format!(
                                "{inner_spacing}{}float* {} = {} + {};",
                                arch.metal_buffer_type(*prev_max_var),
                                var_to_char(*prev_max_var),
                                var_to_char(real_output),
                                stride
                                    .to_kernel()
                                    .replace("const_z", &format!("loop_{loop_var}"))
                            ));
                            new_output_vars.push(*prev_max_var);
                            node_to_var.insert(*output, (*prev_max_var, is_ptr));
                        }
                    } else if let Some(real_output) = created_buffers.get(output) {
                        *prev_max_var += 1;
                        arch.add_metal_buffer_type(
                            *prev_max_var,
                            arch.metal_buffer_type(*real_output),
                        );
                        kernel_lines.push(format!(
                            "{inner_spacing}{}float* {} = {} + {};",
                            arch.metal_buffer_type(*prev_max_var),
                            var_to_char(*prev_max_var),
                            var_to_char(*real_output),
                            stride
                                .to_kernel()
                                .replace("const_z", &format!("loop_{loop_var}"))
                        ));
                        new_output_vars.push(*prev_max_var);
                        node_to_var.insert(*output, (*prev_max_var, true));
                    }
                }
                for (inp, _) in &loop_inputs {
                    loop_indexes.insert(*inp, loop_var_int);
                }
                let loop_body = make_kernel(
                    kernel_graph,
                    body_nodes,
                    node_to_var,
                    prev_max_var,
                    tracked_loop_levels,
                    loop_indexes,
                    smem_buffers,
                    current_loop_level + 1,
                    unpadded_loop_level + 1,
                    arch,
                )?;

                // Handle no-op copy (empty body)
                if loop_body.is_empty() && loop_inputs.len() == 1 && loop_outputs.len() == 1 {
                    let (src, src_ptr) = node_to_var[&loop_inputs.iter().next().unwrap().0];
                    let (dest, dest_ptr) = node_to_var[&loop_outputs.iter().next().unwrap().0];
                    kernel_lines.push(format!(
                        "{inner_spacing}{}{} = {}{};",
                        if dest_ptr { "*" } else { "" },
                        var_to_char(dest),
                        if src_ptr { "*" } else { "" },
                        var_to_char(src)
                    ));
                }
                kernel_lines.extend(loop_body);

                // Set outputs if nessecary
                for (output_node, _) in &loop_outputs {
                    let body_out = kernel_graph
                        .neighbors_directed(*output_node, Direction::Incoming)
                        .next()
                        .unwrap();
                    let (body_out, body_out_ptr) = node_to_var[&body_out];
                    if let Some((output, output_ptr)) = node_to_var.get(&output_node).copied() {
                        if output != body_out && !body_out_ptr {
                            kernel_lines.push(format!(
                                "{inner_spacing}{}{} = {}{};",
                                if output_ptr { "*" } else { "" },
                                var_to_char(output),
                                if body_out_ptr { "*" } else { "" },
                                var_to_char(body_out),
                            ));
                        }
                    } else {
                        node_to_var.insert(*output_node, (body_out, false));
                    }
                }

                // Undo temporary remapping for register buffers
                for (node, real_output) in created_buffers {
                    node_to_var.insert(node, (real_output, true));
                }

                if current_loop_level >= GRID_DIMS + THREADBLOCK_DIMS {
                    kernel_lines.push(format!("{spacing}}}"));
                }

                // Save accumulators
                for (output_node, _) in loop_outputs.into_iter().filter(|(_, st)| st.is_acc()) {
                    let (_, _, size, out_loops) =
                        accs.iter().find(|(_, o, _, _)| *o == output_node).unwrap();
                    let outer_out = kernel_graph
                        .neighbors_directed(output_node, Direction::Outgoing)
                        .next()
                        .unwrap();
                    let (output, output_ptr) = node_to_var[&output_node];
                    if !node_to_var.contains_key(&outer_out) {
                        continue;
                        // display_graph(&kernel_graph, &[(outer_out, "yellow".to_string())]);
                    }
                    let (outer_out, outer_out_ptr) = node_to_var[&outer_out];
                    assert!(output_ptr);
                    assert!(outer_out_ptr || size.to_usize().unwrap() == 1);
                    let mut map = FxHashMap::default();
                    map.insert('s', 1);
                    map.insert('p', 0);
                    // Copy register buffer to output
                    if size.exec(&map).unwrap() == 1 {
                        // Copy single number to output
                        kernel_lines.push(format!(
                            "{spacing}{}{} = *{};",
                            if outer_out_ptr { "*" } else { "" },
                            var_to_char(outer_out),
                            var_to_char(output)
                        ));
                    } else {
                        assert!(outer_out_ptr);
                        // Work out indexing expression to save it by
                        let mut indexing_expression = Expression::from(0);
                        let mut current_elem_size = Expression::from(1);
                        for (range, stride) in out_loops.into_iter().rev() {
                            indexing_expression += stride.substitute(
                                'z',
                                (Expression::from('z') / current_elem_size) % range,
                            );
                            current_elem_size *= range;
                        }
                        kernel_lines.push(format!(
                            "{spacing}for (int save = 0; save < {}; ++save) {{",
                            size.to_kernel()
                        ));
                        let indexing_expression = indexing_expression
                            .simplify()
                            .to_kernel()
                            .replace("const_z", "save");
                        kernel_lines.push(format!(
                            "{inner_spacing}{}[{indexing_expression}] = *({} + {indexing_expression});",
                            var_to_char(outer_out),
                            var_to_char(output),
                        ));
                        kernel_lines.push(format!("{spacing}}}"));
                    }
                }
            }
            GraphTerm::LoopOut { .. } => {
                return None; // TODO: why do we ever see this? should be able to panic here.
                // panic!("found loopout range: {range} stride: {stride}")
            }
            GraphTerm::Custom(_) | GraphTerm::Diff(_) | GraphTerm::Break => {
                unreachable!("this should be handled directly in codegen!")
            }
            GraphTerm::SMEMLoad | GraphTerm::SMEMRead => {
                // Find the gmem input and smem input
                let inputs = kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .collect_vec();
                assert_eq!(inputs.len(), 2);
                let search_for_smem = |mut node| loop {
                    let Some(prev) = kernel_graph
                        .neighbors_directed(node, Direction::Incoming)
                        .next()
                    else {
                        return false;
                    };
                    match kernel_graph[prev].0 {
                        GraphTerm::LoopIn { .. } => node = prev,
                        GraphTerm::SMEM => return true,
                        _ => return false,
                    };
                };
                let (smem, gmem) = if search_for_smem(inputs[0]) {
                    (inputs[0], inputs[1])
                } else {
                    assert!(search_for_smem(inputs[1])); // ensure the other input is an smem ptr
                    (inputs[1], inputs[0])
                };
                let (gmem, gmem_ptr) = node_to_var[&gmem];
                let (smem, smem_ptr) = node_to_var[&smem];
                assert!(smem_ptr);
                let sync_barrier = match arch {
                    GPUArch::CUDA => "__syncthreads()",
                    GPUArch::Metal(_) => "threadgroup_barrier(mem_flags::mem_threadgroup)",
                };
                match term {
                    GraphTerm::SMEMLoad => {
                        kernel_lines.push(format!("{spacing}{sync_barrier};"));
                        kernel_lines.push(format!(
                            "{spacing}*{} = {}{};",
                            var_to_char(smem),
                            if gmem_ptr { "*" } else { "" },
                            var_to_char(gmem),
                        ));
                        node_to_var.insert(node, (smem, true));
                    }
                    GraphTerm::SMEMRead => {
                        // gmem ptr isn't actually gmem, it should be pointing to the smem copy
                        kernel_lines.push(format!("{spacing}{sync_barrier};"));
                        node_to_var.insert(node, (smem, true));
                    }
                    _ => panic!(),
                }
            }
            GraphTerm::GMEM { .. } => {}
            GraphTerm::SMEM => {}
            GraphTerm::Sin
            | GraphTerm::Exp2
            | GraphTerm::Log2
            | GraphTerm::Neg
            | GraphTerm::Recip
            | GraphTerm::Sqrt => {
                *prev_max_var += 1;
                let (src, src_ptr) = node_to_var[&kernel_graph
                    .neighbors_directed(node, Direction::Incoming)
                    .next()
                    .unwrap()];
                node_to_var.insert(node, (*prev_max_var, false));
                let inp = format!("{}{}", if src_ptr { "*" } else { "" }, var_to_char(src));
                let expr = match term {
                    GraphTerm::Sin => format!("sin({inp})"),
                    GraphTerm::Exp2 => format!("exp2({inp})"),
                    GraphTerm::Log2 => format!("log2({inp})"),
                    GraphTerm::Neg => format!("-{inp}"),
                    GraphTerm::Sqrt => format!("sqrt({inp})"),
                    GraphTerm::Recip => format!("1.0 / {inp}"),
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{spacing}float {} = {expr};",
                    var_to_char(*prev_max_var),
                ));
            }
            GraphTerm::Mul
            | GraphTerm::Add
            | GraphTerm::Max
            | GraphTerm::Mod
            | GraphTerm::LessThan => {
                *prev_max_var += 1;
                let mut srcs = kernel_graph.neighbors_directed(node, Direction::Incoming);
                let (src_a, src_a_ptr) = node_to_var[&srcs.next().unwrap()];
                let (src_b, src_b_ptr) = node_to_var[&srcs.next().unwrap()];
                node_to_var.insert(node, (*prev_max_var, false));
                let inp_a = format!("{}{}", if src_a_ptr { "*" } else { "" }, var_to_char(src_a));
                let inp_b = format!("{}{}", if src_b_ptr { "*" } else { "" }, var_to_char(src_b));
                let expr = match &term {
                    GraphTerm::Add => format!("{inp_a} + {inp_b}"),
                    GraphTerm::Mul => format!("{inp_a} * {inp_b}"),
                    GraphTerm::LessThan => format!("(float)({inp_a} < {inp_b})"),
                    GraphTerm::Mod => format!(
                        "{}({inp_a}, {inp_b})",
                        if matches!(arch, GPUArch::Metal(_)) {
                            "fmod"
                        } else {
                            "__mod"
                        }
                    ),
                    GraphTerm::Max => {
                        if matches!(arch, GPUArch::Metal(_)) {
                            format!("fmax({inp_a}, {inp_b})")
                        } else {
                            format!("{inp_a} > {inp_b} ? {inp_a} : {inp_b}")
                        }
                    }
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{spacing}float {} = {expr};",
                    var_to_char(*prev_max_var)
                ));
            }
            GraphTerm::TCMatmul {
                a_k_stride,
                b_k_stride,
                a_inner_stride,
                b_inner_stride,
                c_inner_stride,
                k_outer_loops,
            } => {
                if cfg!(feature = "cuda") {
                    // CUDA build: skip / fallback
                    return None; // or generate a non-TC matmul
                }

                let mut srcs = kernel_graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| e.source());
                let (src_a, src_a_ptr) = node_to_var[&srcs.next().unwrap()];
                let (src_b, src_b_ptr) = node_to_var[&srcs.next().unwrap()];
                let dest = kernel_graph
                    .neighbors_directed(node, Direction::Outgoing)
                    .next()
                    .unwrap();
                let (dest, dest_ptr) = node_to_var[&dest];
                assert!(src_a_ptr && src_b_ptr && dest_ptr);
                kernel_lines.push(
                    format!(
                        "
// TensorCore loop
{{
	simdgroup_float8x8 acc = simdgroup_float8x8(0);
	for (uint tc_loop = 0; tc_loop < {}; tc_loop++) {{
	    threadgroup_barrier(mem_flags::mem_threadgroup); // For some reason this speeds it up

	    // Load sources into simdgroup matricies
	    simdgroup_float8x8 simdA;
	    simdgroup_load(simdA, {} + {}, {});
	    simdgroup_float8x8 simdB;
	    simdgroup_load(simdB, {} + {}, {});

	    simdgroup_multiply_accumulate(acc, simdA, simdB, acc);
	}}
	simdgroup_store(acc, {}, {});
}}",
                        k_outer_loops.to_kernel(),
                        var_to_char(src_a),
                        a_k_stride.to_kernel().replace("const_z", "tc_loop"),
                        a_inner_stride.substitute('z', 1).to_kernel(),
                        var_to_char(src_b),
                        b_k_stride.to_kernel().replace("const_z", "tc_loop"),
                        b_inner_stride.substitute('z', 1).to_kernel(),
                        var_to_char(dest),
                        c_inner_stride.substitute('z', 1).to_kernel()
                    )
                    .split("\n")
                    .map(|s| format!("{spacing}\t{s}"))
                    .join("\n"),
                );
                node_to_var.insert(node, (dest, true));
            }
        }
    }
    Some(kernel_lines)
}

/// Toposort a subset of a graph
fn toposort_subset<N, E>(
    graph: &StableGraph<N, E, Directed>,
    subset: &HashSet<NodeIndex>,
) -> Vec<NodeIndex> {
    // in-degree restricted to `subset`
    let mut indeg: HashMap<NodeIndex, usize> = subset.iter().map(|&n| (n, 0)).collect();
    for &n in subset {
        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            if subset.contains(&succ) {
                *indeg.get_mut(&succ).unwrap() += 1;
            }
        }
    }

    // ready set: smallest index first for deterministic order
    let mut ready: BTreeSet<NodeIndex> = indeg
        .iter()
        .filter_map(|(&n, &d)| (d == 0).then_some(n))
        .collect();

    let mut order = Vec::with_capacity(subset.len());
    while let Some(&n) = ready.iter().next() {
        ready.remove(&n);
        order.push(n);

        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(d) = indeg.get_mut(&succ) {
                *d -= 1;
                if *d == 0 {
                    ready.insert(succ);
                }
            }
        }
    }
    order
}

/// add kernel dimensions so that all loop-to-loop dependencies are between seperate kernels or on the threadblock / thread levels
fn split_kernels(
    graph: StableGraph<GraphTerm, (), Directed>,
) -> (
    Vec<(
        StableGraph<(GraphTerm, usize), (), Directed>,
        Vec<(GMEMBuffer, NodeIndex)>, // (src buffer, current graph node)
        Vec<(Expression, NodeIndex)>, // output node
        Vec<(usize, NodeIndex, Expression)>, // (shared memory buffer name, node, buffer size)
    )>,
    Vec<(usize, usize)>, // Output kernels
) {
    let mut marked_graph = StableGraph::new();
    // Add nodes
    let mut to_remove = vec![];
    for node in graph.node_indices().sorted() {
        let mut levels = FxHashSet::default();
        if graph
            .neighbors_directed(node, Direction::Outgoing)
            .next()
            .is_none()
        {
            levels.insert(0);
        }
        // Ensure node indexes match between original and marked graph by making dummy indexes if needed
        for _ in marked_graph.node_count()..node.index() {
            to_remove.push(marked_graph.add_node((GraphTerm::Add, vec![], HashSet::default())));
        }
        marked_graph.add_node((graph[node].clone(), vec![], levels));
    }
    for n in to_remove {
        marked_graph.remove_node(n);
    }
    // Add edges
    for edge in graph.edge_indices() {
        let (start, end) = graph.edge_endpoints(edge).unwrap();
        marked_graph.add_edge(start, end, *graph.edge_weight(edge).unwrap());
    }

    // Assign loop levels
    let mut seen = graph.externals(Direction::Outgoing).collect::<HashSet<_>>();
    let mut dfs = seen
        .iter()
        .flat_map(|n| marked_graph.neighbors_directed(*n, Direction::Incoming))
        .collect_vec();
    while let Some(n) = dfs.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.insert(n);
        let curr_term = marked_graph[n].0.clone();
        if let Some(outgoing_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Outgoing)
            .find(|n| seen.contains(n))
        {
            // Base level off outgoing neighbor
            let (neighbor_weight, mut neighbor_levels, _) = marked_graph[outgoing_neighbor].clone();
            if let GraphTerm::LoopOut { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopIn { .. }) {
                if neighbor_levels.is_empty() {
                    display_graph2(&marked_graph, &[]);
                }
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else if let Some(incoming_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Incoming)
            .find(|n| seen.contains(n))
        {
            // Base level off incoming neighbor
            let (neighbor_weight, mut neighbor_levels, _) = marked_graph[incoming_neighbor].clone();
            if let GraphTerm::LoopIn { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopOut { .. }) {
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else {
            display_graph(&marked_graph);
            panic!("No seen neighbors when building loop levels!");
        }
        dfs.extend(marked_graph.neighbors_undirected(n));
    }
    // Assign kernel numbers
    for node in toposort(&marked_graph, None).unwrap().into_iter().rev() {
        let (term, curr_level, curr_kernel) = marked_graph[node].clone();
        for source in marked_graph
            .neighbors_directed(node, Direction::Incoming)
            .collect_vec()
        {
            let (src_term, _, src_kernel) = marked_graph.node_weight_mut(source).unwrap();
            // TODO: this conditional is gross
            if (matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < GRID_DIMS + THREADBLOCK_DIMS)
                || matches!(src_term, GraphTerm::Custom(_) | GraphTerm::Diff(_))
                || matches!(term, GraphTerm::Diff(_))
                || (matches!(term, GraphTerm::Custom(_))
                    && !matches!(src_term, GraphTerm::GMEM { .. }))
            {
                let min_src = *src_kernel.iter().min().unwrap_or(&0);
                let max_curr = *curr_kernel.iter().max().unwrap_or(&0);
                if min_src <= max_curr {
                    let max_kernel = curr_kernel
                        .iter()
                        .map(|i| *i + 1)
                        .chain(src_kernel.iter().copied())
                        .max()
                        .unwrap_or(0);
                    src_kernel.clear();
                    src_kernel.insert(max_kernel + 1);
                }
            } else {
                src_kernel.extend(curr_kernel.iter().copied());
            }
        }
    }

    // Split disjoint kernels into unique kernels
    reassign_disjoint_kernels(&mut marked_graph);

    // Remove holes in the kernel index count (0, 1, 3, 4) -> (0, 1, 2, 3)
    let remap: FxHashMap<usize, usize> = marked_graph
        .node_weights()
        .flat_map(|(_, _, k)| k.iter().copied())
        .sorted_unstable()
        .dedup()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    for (_, _, ks) in marked_graph.node_weights_mut() {
        *ks = ks.drain().map(|k| remap[&k]).collect();
    }

    // Add kernel barriers
    for edge in marked_graph.edge_indices().collect_vec() {
        let (mut src, mut dest) = marked_graph.edge_endpoints(edge).unwrap();
        let (_, dest_level, dest_kernel) = marked_graph[dest].clone();
        let (_, _, src_kernel) = marked_graph[src].clone();
        if dest_level.len() > 0 && dest_kernel.iter().any(|i| !src_kernel.contains(i)) {
            // Put a barrier here
            // Get buffer size before this loop
            let mut curr = src;
            let mut total_size = Expression::from(0);
            loop {
                let GraphTerm::LoopOut { range, stride, .. } = marked_graph[curr].0 else {
                    break;
                };
                total_size = total_size.max(stride.substitute('z', range));
                curr = marked_graph
                    .neighbors_directed(curr, Direction::Incoming)
                    .next()
                    .unwrap();
            }

            marked_graph.remove_edge(edge);
            for i in (0..dest_level.len()).rev() {
                let new_src = marked_graph.add_node((
                    GraphTerm::LoopOut {
                        range: dest_level[i].clone(),
                        stride: total_size * 'z',
                        marker: "".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    src_kernel.clone(),
                ));
                marked_graph.add_edge(src, new_src, ());
                src = new_src;
                let new_dest = marked_graph.add_node((
                    GraphTerm::LoopIn {
                        range: dest_level[i].clone(),
                        stride: total_size * 'z',
                        marker: "".to_string(),
                    },
                    dest_level[..i].to_vec(),
                    dest_kernel.clone(),
                ));
                marked_graph.add_edge(new_dest, dest, ());
                dest = new_dest;
            }
            marked_graph.add_edge(src, dest, ());
        }
    }

    // Place nodes in kernel graphs
    let n_kernels = 1 + *marked_graph
        .node_weights()
        .flat_map(|(_, _, k)| k)
        .max()
        .expect("No kernels found!?");
    let mut kernel_graphs = (0..n_kernels)
        .map(|_| (StableGraph::new(), vec![], vec![], vec![]))
        .collect_vec();
    let mut node_maps = (0..n_kernels).map(|_| HashMap::new()).collect_vec();
    let mut final_outputs = HashMap::new();
    for node in marked_graph.node_indices() {
        let (term, loop_level, kernels) = &marked_graph[node];
        for kernel in kernels {
            let (kernel_graph, _, curr_outputs, _) = &mut kernel_graphs[*kernel];
            node_maps[*kernel].insert(
                node,
                kernel_graph.add_node((term.clone(), loop_level.len())),
            );
            if let Some(ind) = marked_graph
                .externals(Direction::Outgoing)
                .position(|n| n == node)
            {
                final_outputs.insert(ind, (*kernel, curr_outputs.len()));
                curr_outputs.push((Expression::default(), node_maps[*kernel][&node]));
            }
        }
    }

    // Mark cross kernel dependencies in the kernel inputs / outputs
    for edge in marked_graph.edge_indices() {
        let (start, end) = marked_graph.edge_endpoints(edge).unwrap();
        let (_, _, start_kernels) = &marked_graph[start];
        let (_, _, end_kernels) = &marked_graph[end];
        for end_kernel in end_kernels {
            if matches!(marked_graph[start].0, GraphTerm::GMEM { .. }) {
                kernel_graphs[*end_kernel].1.push((
                    GMEMBuffer::Input { node: start },
                    node_maps[*end_kernel][&end],
                ));
            } else if !start_kernels.contains(end_kernel) {
                // start kernel must be outputted
                let start_kernel = start_kernels.iter().sorted().next().unwrap();
                // Start and end are in different kernels
                let src_kernel_node = node_maps[*start_kernel][&start];
                let dest_kernel_node = node_maps[*end_kernel][&end];

                // make sure we're outputting the src node from the src graph
                let src_output_index = if let Some(p) = kernel_graphs[*start_kernel]
                    .2
                    .iter()
                    .position(|(_, s)| *s == src_kernel_node)
                {
                    p
                } else {
                    kernel_graphs[*start_kernel]
                        .2
                        .push((Expression::default(), src_kernel_node));
                    kernel_graphs[*start_kernel].2.len() - 1
                };
                // add input to dest
                kernel_graphs[*end_kernel].1.push((
                    GMEMBuffer::PrevKernel {
                        kernel: *start_kernel,
                        output: src_output_index,
                    },
                    dest_kernel_node,
                ));
            } else {
                // end kernel is same as start kernel
                kernel_graphs[*end_kernel].0.add_edge(
                    node_maps[*end_kernel][&start],
                    node_maps[*end_kernel][&end],
                    (),
                );
            }
        }
    }

    // Go through SMEM buffers
    for (kernel_graph, inputs, outputs, smem_buffers) in &mut kernel_graphs {
        for node in kernel_graph.node_indices() {
            if let (GraphTerm::SMEM, _) = kernel_graph[node] {
                // Walk forward until load to find the size
                for mut curr in kernel_graph.neighbors_directed(node, Direction::Outgoing) {
                    let mut size = Expression::from(1);
                    let mut load = false;
                    loop {
                        match kernel_graph[curr].0 {
                            GraphTerm::LoopIn { range, stride, .. } => {
                                if stride != 0 {
                                    size *= range;
                                }
                                curr = kernel_graph
                                    .neighbors_directed(curr, Direction::Outgoing)
                                    .next()
                                    .unwrap();
                            }
                            GraphTerm::SMEMLoad => {
                                load = true;
                                break;
                            }
                            _ => break,
                        }
                    }
                    if load {
                        let buf_index = inputs.len() + outputs.len() + smem_buffers.len();
                        smem_buffers.push((buf_index, node, size));
                        break;
                    }
                }
            }
        }

        // Ensure GMEM is placed on the graph
        for (_, input) in inputs {
            if !matches!(kernel_graph[*input].0, GraphTerm::GMEM { .. }) {
                let new_input = kernel_graph.add_node((
                    GraphTerm::GMEM {
                        label: "Kernel Input".to_string(),
                    },
                    0,
                ));
                kernel_graph.add_edge(new_input, *input, ());
                *input = new_input;
            }
        }
        for (size, output) in outputs {
            if !matches!(kernel_graph[*output].0, GraphTerm::GMEM { .. }) {
                let new_output = kernel_graph.add_node((
                    GraphTerm::GMEM {
                        label: "Kernel Output".to_string(),
                    },
                    0,
                ));
                kernel_graph.add_edge(*output, new_output, ());
                *output = new_output;
            }
            // Loop back through all loopouts to find the size of the output
            let mut curr = *output;
            let mut new_size = Expression::from(1);
            loop {
                let term = &kernel_graph[curr].0;
                if let GraphTerm::LoopOut { range, stride, .. } = term {
                    if !stride.is_acc() && *stride != 0 {
                        new_size = new_size.max(stride.substitute('z', *range));
                        new_size = new_size.max(stride.substitute('z', *range - 1) + 1);
                    }
                } else if !matches!(term, GraphTerm::GMEM { .. }) {
                    break;
                }
                if kernel_graph
                    .neighbors_directed(curr, Direction::Incoming)
                    .next()
                    .is_none()
                {
                    display_graph2(&kernel_graph, &[]);
                    display_graph2(&marked_graph, &[]);
                    display_graph2(&graph, &[]);
                }
                if let Some(c) = kernel_graph
                    .neighbors_directed(curr, Direction::Incoming)
                    .next()
                {
                    curr = c;
                } else {
                    break;
                }
            }
            *size = new_size.simplify();
        }
    }

    (
        kernel_graphs,
        final_outputs
            .into_iter()
            .sorted_by_key(|(k, _)| *k)
            .map(|(_, v)| v)
            .collect(),
    )
}

/// if we have multiple disjoint sets of terms, they need different kernel indexes
fn reassign_disjoint_kernels(
    g: &mut StableGraph<(GraphTerm, Vec<Expression>, FxHashSet<usize>), ()>,
) {
    // remove gmem kernel ids
    for (term, _, kernels) in g.node_weights_mut() {
        if matches!(term, GraphTerm::GMEM { .. }) {
            kernels.clear();
        }
    }
    // Build i -> nodes and find current max index
    let next_kernel_idx = *g.node_weights().flat_map(|(_, _, k)| k).max().unwrap_or(&0);
    let idx_to_nodes = g
        .node_indices()
        .flat_map(|n| g[n].2.iter().copied().map(move |i| (i, n)))
        .into_group_map();

    // For each i, find connected components in the subgraph induced by nodes containing i
    for (i, nodes) in idx_to_nodes {
        if nodes.len() <= 1 {
            continue;
        }

        // Union-Find over only edges inside the induced subgraph
        let mut uf = UnionFind::new(g.node_bound());
        for e in g.edge_indices() {
            let (u, v) = g.edge_endpoints(e).unwrap();
            if nodes.contains(&u) && nodes.contains(&v) {
                uf.union(u.index(), v.index());
            }
        }

        // Collect components
        let comps = nodes
            .iter()
            .map(|n| (uf.find(n.index()), *n))
            .into_group_map();
        if comps.len() <= 1 {
            continue;
        }

        // Keep largest; rename `i` in others
        for (n_group, comp) in comps
            .into_values()
            .sorted_by_key(|c| c.len())
            .rev()
            .skip(1)
            .enumerate()
        {
            for n in comp {
                let ws = &mut g.node_weight_mut(n).unwrap().2;
                if ws.remove(&i) {
                    ws.insert(next_kernel_idx + n_group + 1);
                }
            }
        }
    }

    // re-add kernel ids
    for i in g.node_indices().collect_vec() {
        if matches!(g[i].0, GraphTerm::GMEM { .. }) {
            g.node_weight_mut(i).unwrap().2 = g
                .neighbors_directed(i, Direction::Outgoing)
                .flat_map(|n| &g[n].2)
                .copied()
                .collect();
        }
    }
}

pub fn stitch_meta_graph_together(
    meta_graph: MetaGraph,
) -> (SubGraph, FxHashMap<(NodeIndex, NodeIndex), NodeIndex>) {
    let mut out = SubGraph::new();

    // (meta_node, inner_node) -> stitched node (no entry for break_in placeholders)
    let mut map: FxHashMap<(NodeIndex, NodeIndex), NodeIndex> = FxHashMap::default();
    // record which inner nodes are break_in placeholders, per meta node
    let mut is_placeholder: FxHashMap<(NodeIndex, NodeIndex), bool> = FxHashMap::default();
    // record outgoing targets of each placeholder: (meta, placeholder_inner) -> Vec<inner_target>
    let mut placeholder_outs: FxHashMap<(NodeIndex, NodeIndex), Vec<NodeIndex>> =
        FxHashMap::default();

    let is_break_in = |term: &GraphTerm| {
        matches!(term,
            GraphTerm::GMEM { label } if label.starts_with("break_")
        )
    };

    // 1) copy nodes (skip creating nodes for break_in placeholders)
    for m in meta_graph.node_indices() {
        let inner = &meta_graph[m];
        for n in inner.node_indices() {
            let term = &inner[n];
            let ph = is_break_in(term);
            is_placeholder.insert((m, n), ph);
            if !ph {
                map.insert((m, n), out.add_node(term.clone()));
            }
        }
    }

    // 2) copy inner edges (skip edges touching placeholders; record placeholder -> target)
    for m in meta_graph.node_indices() {
        let inner = &meta_graph[m];
        for e in inner.edge_indices() {
            let (u, v) = inner.edge_endpoints(e).unwrap();
            let up = *is_placeholder.get(&(m, u)).unwrap();
            let vp = *is_placeholder.get(&(m, v)).unwrap();

            if up {
                // defer: producer will be connected to all of v later
                placeholder_outs.entry((m, u)).or_default().push(v);
                continue;
            }
            if vp {
                // edges *into* placeholder are never needed
                continue;
            }

            out.add_edge(map[&(m, u)], map[&(m, v)], ());
        }
    }

    // 3) connect across breaks using meta-edges
    // meta edge payload: (producer_inner, consumer_placeholder_inner_or_real)
    for me in meta_graph.edge_indices() {
        let (mu, mv) = meta_graph.edge_endpoints(me).unwrap();
        let (u_inner, v_inner) = meta_graph[me];

        let v_is_ph = *is_placeholder.get(&(mv, v_inner)).unwrap_or(&false);
        if v_is_ph {
            // fan-out to every recorded consumer of the placeholder
            if let Some(targets) = placeholder_outs.get(&(mv, v_inner)) {
                let src = map[&(mu, u_inner)];
                for &t_inner in targets {
                    out.add_edge(src, map[&(mv, t_inner)], ());
                }
            }
        } else {
            out.add_edge(map[&(mu, u_inner)], map[&(mv, v_inner)], ());
        }
    }

    (out, map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GPUArch, GraphTerm, translate::SubGraph};
    use std::collections::HashMap;

    #[test]
    fn test_var_to_char() {
        assert_eq!(var_to_char(0), "a");
        assert_eq!(var_to_char(25), "z");
        assert_eq!(var_to_char(26), "ba");
        assert_eq!(var_to_char(27), "bb");
    }

    #[test]
    fn test_toposort_subset_empty() {
        let graph: StableGraph<GraphTerm, (), Directed> = StableGraph::new();
        let subset = std::collections::HashSet::new();
        let result = toposort_subset(&graph, &subset);
        assert!(result.is_empty());
    }

    #[test]
    fn test_toposort_subset_single_node() {
        let mut graph: StableGraph<GraphTerm, (), Directed> = StableGraph::new();
        let node = graph.add_node(GraphTerm::GMEM {
            label: "test".to_string(),
        });
        let mut subset = std::collections::HashSet::new();
        subset.insert(node);

        let result = toposort_subset(&graph, &subset);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], node);
    }

    #[test]
    fn test_toposort_subset_linear_chain() {
        let mut graph: StableGraph<GraphTerm, (), Directed> = StableGraph::new();
        let node1 = graph.add_node(GraphTerm::GMEM {
            label: "input".to_string(),
        });
        let node2 = graph.add_node(GraphTerm::Add);
        let node3 = graph.add_node(GraphTerm::GMEM {
            label: "output".to_string(),
        });

        graph.add_edge(node1, node2, ());
        graph.add_edge(node2, node3, ());

        let mut subset = std::collections::HashSet::new();
        subset.insert(node1);
        subset.insert(node2);
        subset.insert(node3);

        let result = toposort_subset(&graph, &subset);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], node1);
        assert_eq!(result[1], node2);
        assert_eq!(result[2], node3);
    }

    #[test]
    fn test_metal_arch_creation() {
        let arch = GPUArch::Metal(HashMap::default());
        match arch {
            GPUArch::Metal(_) => {} // Success
            _ => panic!("Should create Metal arch"),
        }
    }

    #[test]
    fn test_stitch_kernel_meta_graph_empty() {
        let kernel_meta_graph: StableGraph<SubGraph, (NodeIndex, NodeIndex), Directed> =
            StableGraph::new();
        let (stitched, map) = stitch_meta_graph_together(kernel_meta_graph);
        assert_eq!(stitched.node_count(), 0);
        assert!(map.is_empty());
    }
}
