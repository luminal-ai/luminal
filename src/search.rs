use std::{
    collections::{BTreeSet, HashMap, HashSet},
    iter::repeat,
};

use itertools::Itertools;
use petgraph::{algo::toposort, prelude::StableGraph, visit::EdgeRef, Directed, Direction};
use regex::Regex;
use rustc_hash::FxHashMap;

use crate::prelude::*;

const GRID_DIMS: usize = 2;
const THREADBLOCK_DIMS: usize = 2;
const MAX_THREADBLOCK_SIZE: usize = 1024; // this is max on mac

#[derive(Clone)]
enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { label: Option<String> },
}

#[derive(Clone, PartialEq, Eq)]
pub enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub code: String,
    // launch params
    pub grid: (Expression, Expression, Expression),
    pub threadblock: (Expression, Expression, Expression),
    pub smem: Expression, // sizes of required shared memory buffers
    pub outputs: Vec<Expression>,
}

pub fn translate_graph(
    graph: &Graph,
) -> (StableGraph<GraphTerm, (), Directed>, Vec<(String, f32)>) {
    let mut new_graph = StableGraph::new();
    let mut node_mapping = FxHashMap::default();
    let mut accumulators = vec![];
    for node in toposort(&graph.graph, None).unwrap() {
        let node_weight = graph.node_weight(node).unwrap();
        let op_name_full = format!("{node_weight:?}");
        let op = op_name_full
            .split('|')
            .next()
            .unwrap_or(&op_name_full)
            .trim();
        let mut sources = graph.get_sources(node);
        match op {
            "Sqrt" | "Exp2" | "Sin" | "Contiguous" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source, output_index, shape) = sources.pop().unwrap();
                let mut new_source = node_mapping[&(source, output_index)];
                for (i, (range, stride)) in
                    shape.dims().into_iter().zip(shape.strides()).enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source, loopin, ());
                    new_source = loopin;
                }
                let mut op = if op == "Contiguous" {
                    new_source
                } else {
                    let r = new_graph.add_node(match op {
                        "Sqrt" => GraphTerm::Sqrt,
                        "Exp2" => GraphTerm::Exp,
                        "Sin" => GraphTerm::Sin,
                        _ => unreachable!(),
                    });
                    new_graph.add_edge(new_source, r, ());
                    r
                };
                // walk through output and place loopouts
                for (i, (stride, range)) in shape
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .zip(shape.dims())
                    .enumerate()
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: (shape.dims().len() - i - 1).to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                }
                node_mapping.insert((node, 0), op);
            }
            "Add" | "Mul" | "Mod" | "LessThan" => {
                // walk through input ranges and strides, making new loopins as we go
                let (source_a, output_index_a, shape_a) = sources.pop().unwrap();
                let mut new_source_a = node_mapping[&(source_a, output_index_a)];
                for (i, (range, stride)) in shape_a
                    .dims()
                    .into_iter()
                    .zip(shape_a.strides())
                    .enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source_a, loopin, ());
                    new_source_a = loopin;
                }
                let (source_b, output_index_b, shape_b) = sources.pop().unwrap();
                let mut new_source_b = node_mapping[&(source_b, output_index_b)];
                for (i, (range, stride)) in shape_b
                    .dims()
                    .into_iter()
                    .zip(shape_b.strides())
                    .enumerate()
                {
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source_b, loopin, ());
                    new_source_b = loopin;
                }
                let mut op = new_graph.add_node(match op {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Mod" => GraphTerm::Mod,
                    "LessThan" => GraphTerm::LessThan,
                    _ => unreachable!(),
                });
                new_graph.add_edge(new_source_a, op, ());
                new_graph.add_edge(new_source_b, op, ());
                // walk through output and place loopouts
                for (i, (stride, range)) in shape_a
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .zip(shape_a.dims().into_iter().rev())
                    .enumerate()
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: (shape_a.dims().len() - i - 1).to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                }
                node_mapping.insert((node, 0), op);
            }
            s if s.starts_with("SumReduce") || s.starts_with("MaxReduce") => {
                // make accumulator
                let acc_label = format!("acc_{}", accumulators.len());
                let (start_val, term, reduce_dim) = match op {
                    s if s.starts_with("SumReduce") => {
                        (0.0, GraphTerm::Add, graph.get_op::<SumReduce>(node).0)
                    }
                    s if s.starts_with("MaxReduce") => (
                        f32::NEG_INFINITY,
                        GraphTerm::Max,
                        graph.get_op::<MaxReduce>(node).0,
                    ),
                    _ => unreachable!(),
                };
                accumulators.push((acc_label.clone(), start_val));
                let mut acc = new_graph.add_node(GraphTerm::GMEM {
                    label: Some(acc_label),
                });
                // walk through input ranges and strides, making new loopins as we go
                let (source, output_index, shape) = sources.pop().unwrap();
                let mut new_source = node_mapping[&(source, output_index)];
                let mut rm_strides = shape
                    .dims()
                    .into_iter()
                    .rev()
                    .scan(Expression::from(1), |i, s| {
                        let r = *i;
                        *i *= s;
                        Some(r)
                    })
                    .collect::<Vec<_>>();
                rm_strides.reverse();
                let mut after_acc = false;
                for (i, ((range, stride), acc_stride)) in shape
                    .dims()
                    .into_iter()
                    .zip(shape.strides())
                    .zip(rm_strides)
                    .enumerate()
                {
                    if i == reduce_dim {
                        for z in i..THREADBLOCK_DIMS + GRID_DIMS + 1 {
                            let loopin = new_graph.add_node(GraphTerm::LoopIn {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(new_source, loopin, ());
                            new_source = loopin;
                            let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(acc, new_acc, ());
                            acc = new_acc;
                        }
                    }
                    let loopin = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride: Expression::from('z') * stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(new_source, loopin, ());
                    new_source = loopin;
                    let stride = if i == reduce_dim {
                        after_acc = true;
                        Expression::from(Term::Acc('a'))
                    } else if after_acc {
                        Expression::from('z') * acc_stride
                    } else {
                        Expression::from(0)
                    };
                    let new_acc = new_graph.add_node(GraphTerm::LoopIn {
                        range,
                        stride,
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(acc, new_acc, ());
                    acc = new_acc;
                }
                // Insert op
                let mut op = new_graph.add_node(term);
                new_graph.add_edge(new_source, op, ());
                new_graph.add_edge(acc, op, ());
                // walk through output and place loopouts
                for ((stride, i), range) in shape
                    .dims()
                    .into_iter()
                    .enumerate()
                    .rev()
                    .scan(Expression::from(1), |i, (ind, s)| {
                        if ind == reduce_dim {
                            Some((Expression::from(Term::Acc('a')), ind))
                        } else {
                            let r = *i;
                            *i *= s;
                            Some((r, ind))
                        }
                    })
                    .zip(shape.dims().into_iter().rev())
                {
                    let loopout = new_graph.add_node(GraphTerm::LoopOut {
                        range,
                        stride: if i == reduce_dim {
                            stride
                        } else {
                            Expression::from('z') * stride
                        },
                        marker: i.to_string(),
                    });
                    new_graph.add_edge(op, loopout, ());
                    op = loopout;
                    if i == reduce_dim {
                        for z in (i..THREADBLOCK_DIMS + GRID_DIMS + 1).rev() {
                            let loopout = new_graph.add_node(GraphTerm::LoopOut {
                                range: 1.into(),
                                stride: 0.into(),
                                marker: format!("pad{z}"),
                            });
                            new_graph.add_edge(op, loopout, ());
                            op = loopout;
                        }
                    }
                }
                node_mapping.insert((node, 0), op);
            }
            _ => {
                // Assume a load
                node_mapping.insert(
                    (node, 0),
                    new_graph.add_node(GraphTerm::GMEM {
                        label: Some(op.to_string()),
                    }),
                );
            }
        }
    }

    (new_graph, accumulators)
}

#[derive(Clone, Debug)]
pub enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: Option<String>,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    Add,
    Mul,
    Mod,
    Max,
    Exp,
    Recip,
    Sin,
    Neg,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
    Sqrt,
    LessThan,
}

impl std::fmt::Display for GraphTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphTerm::GMEM { label } => {
                if let Some(label) = label {
                    write!(f, "GMEM({label})")
                } else {
                    write!(f, "GMEM")
                }
            }
            GraphTerm::LoopIn { range, stride, .. } => {
                write!(f, "LoopIn(range: {range:?}, stride: {stride:?})")
            }
            GraphTerm::LoopOut { range, stride, .. } => {
                write!(f, "LoopOut(range: {range:?}, stride: {stride:?})")
            }
            GraphTerm::Add => write!(f, "Add"),
            GraphTerm::Mul => write!(f, "Mul"),
            GraphTerm::Mod => write!(f, "Mod"),
            GraphTerm::Max => write!(f, "Max"),
            GraphTerm::Exp => write!(f, "Exp"),
            GraphTerm::Recip => write!(f, "Recip"),
            GraphTerm::Sin => write!(f, "Sin"),
            GraphTerm::Neg => write!(f, "Neg"),
            GraphTerm::SMEM => write!(f, "SMEM"),
            GraphTerm::SMEMLoad => write!(f, "SMEMLoad"),
            GraphTerm::SMEMRead => write!(f, "SMEMRead"),

            GraphTerm::Sqrt => write!(f, "Sqrt"),
            GraphTerm::LessThan => write!(f, "LessThan"),
        }
    }
}

pub fn codegen(
    graph: StableGraph<GraphTerm, (), Directed>,
    root: NodeIndex,
    mut arch: GPUArch,
    n_graph: usize,
) -> Option<StableGraph<Kernel, (u8, u8), Directed>> {
    let (kernels, root_kernel) = split_kernels(graph, root, n_graph);
    // Create kernel meta graph to toposort
    let mut meta_graph = StableGraph::new();
    for _ in 0..kernels.len() {
        meta_graph.add_node(Kernel::default());
    }
    let global_input = meta_graph.add_node(Kernel {
        code: "Inputs".to_string(),
        ..Default::default()
    });
    let global_output = meta_graph.add_node(Kernel {
        code: "Outputs".to_string(),
        ..Default::default()
    });
    let mut gmem_mapping = HashMap::new();
    for (n_kernel, (_, inputs, _, _)) in kernels.iter().enumerate() {
        for (n_input, (input_kernel, _)) in inputs.into_iter().enumerate() {
            match input_kernel {
                GMEMBuffer::PrevKernel { kernel, output } => meta_graph.add_edge(
                    NodeIndex::new(*kernel),
                    NodeIndex::new(n_kernel),
                    (*output as u8, n_input as u8),
                ),
                GMEMBuffer::Input { label, .. } => {
                    let index = if let Some(index) = gmem_mapping.get(label.as_ref().unwrap()) {
                        *index
                    } else {
                        gmem_mapping.insert(label.clone().unwrap(), gmem_mapping.len());
                        gmem_mapping.len() - 1
                    };
                    meta_graph.add_edge(
                        global_input,
                        NodeIndex::new(n_kernel),
                        (index as u8, n_input as u8),
                    )
                }
            };
        }
    }
    meta_graph.node_weight_mut(global_input).unwrap().code =
        format!("Inputs{}", serde_json::to_string(&gmem_mapping).unwrap());
    meta_graph.add_edge(NodeIndex::new(root_kernel), global_output, (0, 0));
    for node in toposort(&meta_graph, None).unwrap() {
        if kernels.len() <= node.index() {
            continue; // Either input node or output node
        }
        let (kernel_graph, inputs, outputs, smem_buffers) = kernels[node.index()].clone();
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
        let kernel_lines = kernel.into_iter().map(|s| format!("\t{s}")).join("\n");
        // if node.index() == 0 {
        //     display_graph(&kernel_graph, &[]);
        // }
        let kernel = match &arch {
            GPUArch::CUDA => {
                let inputs = inputs
                    .into_iter()
                    .map(|(_, a)| a)
                    .chain(outputs.iter().map(|(_, i)| *i))
                    .map(|a| format!("float* {}", var_to_char(node_to_var[&a].0)))
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
                    "extern \"C\" __global__ void kernel{}({inputs}) {{
{smem_setup}{kernel_lines}
}}",
                    node.index()
                )
            }
            GPUArch::Metal { .. } => {
                let mut input_comment = inputs
                    .iter()
                    .filter_map(|(buf, ind)| {
                        if let GMEMBuffer::Input {
                            label: Some(label), ..
                        } = buf
                        {
                            Some(format!(
                                "\t// {label} = {}",
                                var_to_char(node_to_var[ind].0)
                            ))
                        } else {
                            None
                        }
                    })
                    .join("\n");
                if !input_comment.is_empty() {
                    input_comment = format!("\t// Inputs\n{input_comment}\n");
                }
                let input_string = inputs
                    .into_iter()
                    .map(|(_, a)| a)
                    .chain(outputs.iter().map(|(_, i)| *i))
                    .enumerate()
                    .map(|(i, a)| {
                        format!(
                            "device float* {} [[buffer({i})]]",
                            var_to_char(node_to_var[&a].0)
                        )
                    })
                    .join(",\n\t");
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
kernel void kernel{}(
	uint3 blockIdx [[threadgroup_position_in_grid]],
	uint3 threadIdx [[thread_position_in_threadgroup]],
	{input_string}{smem_input}
) {{
{input_comment}{smem_setup}{kernel_lines}
}}",
                    node.index(),
                )
            }
        };
        if (threadblock[0] * threadblock[1] * threadblock[2])
            .to_usize()
            .unwrap()
            > MAX_THREADBLOCK_SIZE
        {
            // Threadblock size is too large for device
            return None;
        }
        // println!("{kernel}");
        // println!("{:?}", outputs);
        *meta_graph.node_weight_mut(node).unwrap() = Kernel {
            code: kernel,
            grid: (grid[0], grid[1], grid[2]),
            threadblock: (threadblock[0], threadblock[1], threadblock[2]),
            smem: smem_buffers.into_iter().map(|(_, _, a)| a).sum(),
            outputs: outputs.into_iter().map(|(o, _)| o.simplify()).collect(),
        };
    }
    Some(meta_graph)
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
    loop_levels: &mut Vec<Expression>,
    loop_indexes: &mut HashMap<NodeIndex, usize>,
    smem_buffers: &HashMap<NodeIndex, usize>,
    current_loop_level: usize,
    arch: &mut GPUArch,
) -> Option<Vec<String>> {
    let mut kernel_lines = vec![];
    let spacing = (0..current_loop_level.saturating_sub(GRID_DIMS + THREADBLOCK_DIMS))
        .map(|_| "\t")
        .join("");

    // Walk through the toposorted nodes
    for node in toposort_subset(kernel_graph, &include_nodes) {
        if node_to_var.contains_key(&node)
            || kernel_graph.node_weight(node).unwrap().1 != current_loop_level
        {
            continue;
        }
        let (term, loop_level) = kernel_graph.node_weight(node).unwrap();
        match term {
            GraphTerm::LoopIn { range, stride, .. } => {
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
                    if let (GraphTerm::LoopOut { stride, .. }, level) =
                        &kernel_graph.node_weight(n).unwrap()
                    {
                        if *level == *loop_level {
                            loop_outputs.insert((n, stride));
                            continue;
                        }
                    } else if let (GraphTerm::LoopIn { stride, .. }, level) =
                        &kernel_graph.node_weight(n).unwrap()
                    {
                        if *level == *loop_level {
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
                // display_graph(
                //     &kernel_graph,
                //     &body_nodes
                //         .iter()
                //         .copied()
                //         .map(|i| (i, "yellow".to_string()))
                //         .chain(loop_inputs.iter().map(|(i, _)| (*i, "green".to_string())))
                //         .chain(loop_outputs.iter().map(|(i, _)| (*i, "red".to_string())))
                //         .collect_vec(),
                // );

                let inner_spacing = if current_loop_level < GRID_DIMS + THREADBLOCK_DIMS {
                    "".to_string()
                } else {
                    format!("{spacing}\t")
                };

                // Make accumulators
                let mut accs = vec![];
                for (input, stride) in &loop_inputs {
                    if stride.is_acc() {
                        if *loop_level < GRID_DIMS + THREADBLOCK_DIMS {
                            return None;
                        }
                        let Term::Acc(acc_symbol) = stride.terms.read()[0] else {
                            panic!("Acc is not acc");
                        };
                        // Create a new accumulator
                        // Work out the size needed by taking the max stride of the loopins and loopouts
                        let mut size = Expression::from(1);
                        let mut loads = Expression::from(1);
                        let mut in_loops = vec![];
                        let mut curr = *input;
                        loop {
                            match kernel_graph.node_weight(curr).unwrap().0 {
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
                        let corresponding_output = loop_outputs
                            .iter()
                            .find(|(_, v)| v.terms.read()[0] == Term::Acc(acc_symbol))
                            .unwrap()
                            .0;
                        curr = corresponding_output;
                        let mut out_loops = vec![];
                        loop {
                            match kernel_graph.node_weight(curr).unwrap().0 {
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
                            "{spacing}for (int load = 0; load < {loads}; ++load) {{"
                        ));
                        let indexing_expression = indexing_expression
                            .simplify()
                            .to_string()
                            .replace("z", "load");
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
                for (output, stride) in &loop_outputs {
                    let dest = kernel_graph
                        .neighbors_directed(*output, Direction::Outgoing)
                        .next()
                        .unwrap();
                    if !stride.is_acc() && !node_to_var.contains_key(&dest) {
                        // Handle the case where the dest is not the real loop output
                        let size = stride.substitute('z', range);
                        if current_loop_level < THREADBLOCK_DIMS + GRID_DIMS {
                            return None;
                        }
                        // assert!(
                        //     current_loop_level >= THREADBLOCK_DIMS + GRID_DIMS,
                        //     "Grid / Threadblock intermediate buffers not supported yet!"
                        // );
                        // We don't have a place to save this accumulator to. Need to allocate a register buffer
                        *prev_max_var += 1;
                        kernel_lines.push(format!(
                            "{spacing}thread float {}[{size}] = {{0.0}};",
                            var_to_char(*prev_max_var)
                        ));
                        node_to_var.insert(*output, (*prev_max_var, true));
                        arch.add_metal_buffer_type(*prev_max_var, "thread ");
                    }
                }

                // Make new loop
                if *loop_level < GRID_DIMS + THREADBLOCK_DIMS {
                    if current_loop_level >= loop_levels.len() {
                        loop_levels.push(*range);
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
                            if *loop_level >= GRID_DIMS {
                                ["threadIdx.x", "threadIdx.y", "threadIdx.z"]
                                    [*loop_level - GRID_DIMS]
                            } else {
                                ["blockIdx.x", "blockIdx.y", "blockIdx.z"][*loop_level]
                            }
                        ));
                    }
                } else {
                    *prev_max_var += 1;
                    let loop_var = var_to_char(*prev_max_var);
                    kernel_lines.push(format!("{spacing}for (int loop_{loop_var} = 0; loop_{loop_var} < {range}; ++loop_{loop_var}) {{"));
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
                                stride.to_string().replace('z', &format!("loop_{loop_var}"))
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
                                stride.to_string().replace('z', &format!("loop_{loop_var}"))
                            ));
                            new_output_vars.push(*prev_max_var);
                            node_to_var.insert(*output, (*prev_max_var, is_ptr));
                        }
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
                    loop_levels,
                    loop_indexes,
                    smem_buffers,
                    current_loop_level + 1,
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

                if *loop_level >= GRID_DIMS + THREADBLOCK_DIMS {
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

                    // Copy register buffer to output
                    if size.to_usize().unwrap() == 1 {
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
                            "{spacing}for (int save = 0; save < {size}; ++save) {{"
                        ));
                        let indexing_expression = indexing_expression
                            .simplify()
                            .to_string()
                            .replace("z", "save");
                        kernel_lines.push(format!(
                            "{inner_spacing}{}[{indexing_expression}] = *({} + {indexing_expression});",
                            var_to_char(outer_out),
                            var_to_char(output),
                        ));
                        kernel_lines.push(format!("{spacing}}}"));
                    }
                }
            }
            GraphTerm::LoopOut { range, stride, .. } => {
                panic!("found loopout range: {range} stride: {stride}")
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
                    match kernel_graph.node_weight(prev).unwrap().0 {
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
            | GraphTerm::Exp
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
                    GraphTerm::Exp => format!("exp({inp})"),
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
                    GraphTerm::LessThan => format!("{inp_a} < {inp_b} ? 1.0 : 0.0"),
                    GraphTerm::Mod => format!(
                        "{}({inp_a}, {inp_b})",
                        if matches!(arch, GPUArch::Metal(_)) {
                            "fmod"
                        } else {
                            "__mod"
                        }
                    ),
                    GraphTerm::Max => format!(
                        "{}({inp_a}, {inp_b})",
                        if matches!(arch, GPUArch::Metal(_)) {
                            "fmax"
                        } else {
                            "__max"
                        }
                    ),
                    _ => panic!(),
                };
                kernel_lines.push(format!(
                    "{spacing}float {} = {expr};",
                    var_to_char(*prev_max_var)
                ));
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
    mut root: NodeIndex,
    _n_graph: usize,
) -> (
    Vec<(
        StableGraph<(GraphTerm, usize), (), Directed>,
        Vec<(GMEMBuffer, NodeIndex)>, // (src buffer, current graph node)
        Vec<(Expression, NodeIndex)>, // output node
        Vec<(usize, NodeIndex, Expression)>, // (shared memory buffer name, node, buffer size)
    )>,
    usize, // Root kernel
) {
    // Mark level of ops in graph
    let mut marked_graph = StableGraph::new();
    let mut map = HashMap::<NodeIndex, NodeIndex>::default();
    // Add nodes
    for node in graph.node_indices() {
        let new_node = marked_graph.add_node((
            graph.node_weight(node).unwrap().clone(),
            vec![],
            if node == root { vec![0] } else { vec![] },
        ));
        map.insert(node, new_node);
    }
    // Add edges
    for edge in graph.edge_indices() {
        let (start, end) = graph.edge_endpoints(edge).unwrap();
        marked_graph.add_edge(map[&start], map[&end], *graph.edge_weight(edge).unwrap());
    }
    root = map[&root];

    let mut dfs = marked_graph.neighbors_undirected(root).collect_vec();
    let mut seen = HashSet::new();
    seen.insert(root);
    while let Some(n) = dfs.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.insert(n);
        let curr_term = marked_graph.node_weight(n).unwrap().0.clone();
        if let Some(outgoing_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Outgoing)
            .find(|n| seen.contains(n))
        {
            // Base level off outgoing neighbor
            let (neighbor_weight, mut neighbor_levels, _) =
                marked_graph.node_weight(outgoing_neighbor).unwrap().clone();
            if let GraphTerm::LoopOut { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopIn { .. }) {
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else if let Some(incoming_neighbor) = marked_graph
            .neighbors_directed(n, Direction::Incoming)
            .find(|n| seen.contains(n))
        {
            // Base level off incoming neighbor
            let (neighbor_weight, mut neighbor_levels, _) =
                marked_graph.node_weight(incoming_neighbor).unwrap().clone();
            if let GraphTerm::LoopIn { range, .. } = neighbor_weight {
                neighbor_levels.push(range);
            }
            if matches!(curr_term, GraphTerm::LoopOut { .. }) {
                neighbor_levels.pop().unwrap();
            }
            marked_graph.node_weight_mut(n).unwrap().1 = neighbor_levels;
        } else {
            // display_graph(&marked_graph, &[(n, "yellow".to_string())]);
            panic!("No seen neighbors when building loop levels!");
        }
        dfs.extend(marked_graph.neighbors_undirected(n));
    }

    // Assign kernel numbers
    let mut dfs_stack = vec![root];
    let mut n_kernels = 1;
    while let Some(node) = dfs_stack.pop() {
        let (term, curr_level, curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        for source in marked_graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| e.source())
            .collect_vec()
        {
            let (src_term, src_level, src_kernel) = marked_graph.node_weight_mut(source).unwrap();
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < GRID_DIMS + THREADBLOCK_DIMS
            {
                let max_kernel = *curr_kernel.iter().max().unwrap();
                n_kernels = n_kernels.max(max_kernel + 2);
                *src_kernel = vec![max_kernel + 1];
            } else {
                for k in &curr_kernel {
                    if !src_kernel.contains(k) {
                        src_kernel.push(*k);
                    }
                }
            }
            if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(curr_level, *src_level); // Edges need to go between the same levels if neither op is a LoopIn or LoopOut
            }
            dfs_stack.push(source);
        }
    }
    // Run forward from all inputs to catch any nodes we didn't hit already
    dfs_stack = marked_graph
        .node_indices()
        .filter(|n| {
            marked_graph
                .neighbors_directed(*n, Direction::Incoming)
                .next()
                .is_none()
        })
        .collect_vec();
    while let Some(node) = dfs_stack.pop() {
        dfs_stack.extend(marked_graph.neighbors_directed(node, Direction::Outgoing));
        let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        if !curr_kernel.is_empty() {
            continue;
        }
        for source in marked_graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| e.source())
            .collect_vec()
        {
            let (src_term, src_level, src_kernel) = marked_graph.node_weight(source).unwrap();
            if matches!(term, GraphTerm::LoopIn { .. })
                && matches!(src_term, GraphTerm::LoopOut { .. })
                && curr_level.len() < GRID_DIMS + THREADBLOCK_DIMS
            {
                let max_kernel = *src_kernel.iter().max().unwrap();
                n_kernels = n_kernels.max(max_kernel + 2);
                curr_kernel = vec![max_kernel + 1];
            } else {
                for k in src_kernel {
                    if !curr_kernel.contains(k) {
                        curr_kernel.push(*k);
                    }
                }
            }
            if !matches!(term, GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. })
                && !matches!(
                    src_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                )
            {
                assert_eq!(curr_level, *src_level); // Edges need to go between the same levels if neither op is a LoopIn or LoopOut
            }
        }
        marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
    }
    // Prune out unnessecary kernel members
    let mut dfs_stack = marked_graph
        .neighbors_directed(root, Direction::Incoming)
        .collect_vec();
    while let Some(node) = dfs_stack.pop() {
        dfs_stack.extend(marked_graph.neighbors_directed(node, Direction::Incoming));
        let (term, curr_level, mut curr_kernel) = marked_graph.node_weight(node).unwrap().clone();
        let split_cond = curr_level.len() < GRID_DIMS + THREADBLOCK_DIMS
            && matches!(term, GraphTerm::LoopOut { .. });
        curr_kernel.retain(|k| {
            marked_graph
                .neighbors_directed(node, Direction::Outgoing)
                .any(|n| {
                    (split_cond
                        && matches!(
                            marked_graph.node_weight(n).unwrap().0,
                            GraphTerm::LoopIn { .. }
                        ))
                        || marked_graph.node_weight(n).unwrap().2.contains(k)
                })
        });
        marked_graph.node_weight_mut(node).unwrap().2 = curr_kernel;
    }

    // Disallow disjoint nodes to be in the same kernel
    let mut by_kernel: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
    for n in marked_graph.node_indices() {
        for &k in &marked_graph[n].2 {
            // field 2 == Vec<usize> of kernel IDs
            by_kernel.entry(k).or_default().push(n);
        }
    }
    use std::collections::{HashSet, VecDeque};
    let mut next_kernel = n_kernels; // keep issuing fresh IDs

    for (k, nodes) in by_kernel {
        let mut seen: HashSet<NodeIndex> = HashSet::new();

        for &seed in &nodes {
            if seen.contains(&seed) {
                continue;
            }

            // BFS restricted to nodes that still carry -k-
            let mut q = VecDeque::from([seed]);
            let mut comp: Vec<NodeIndex> = Vec::new();
            while let Some(v) = q.pop_front() {
                if !seen.insert(v) {
                    continue;
                }
                comp.push(v);
                for nb in marked_graph.neighbors_undirected(v) {
                    if marked_graph[nb].2.contains(&k) {
                        q.push_back(nb);
                    }
                }
            }

            // first component keeps k; extra components get fresh IDs
            if !comp.is_empty() && seed != nodes[0] {
                let new_k = {
                    next_kernel += 1;
                    next_kernel - 1
                };
                for v in comp {
                    let kernels = &mut marked_graph.node_weight_mut(v).unwrap().2;
                    kernels.retain(|id| *id != k);
                    kernels.push(new_k);
                }
            }
        }
    }
    n_kernels = next_kernel;
    // display_graph(&marked_graph, &[]);

    // Add kernel barriers
    for edge in marked_graph.edge_indices().collect_vec() {
        let (mut src, mut dest) = marked_graph.edge_endpoints(edge).unwrap();
        let (_, dest_level, dest_kernel) = marked_graph.node_weight(dest).unwrap().clone();
        let (_, _, src_kernel) = marked_graph.node_weight(src).unwrap().clone();
        if dest_level.len() > 0 && dest_kernel.iter().any(|i| !src_kernel.contains(i)) {
            // Put a barrier here
            // Get buffer size before this loop
            let mut curr = src;
            let mut total_size = Expression::from(0);
            loop {
                match marked_graph.node_weight(curr).unwrap().0 {
                    GraphTerm::LoopOut { range, stride, .. } => {
                        total_size = total_size.max(stride.substitute('z', range));
                    }
                    _ => break,
                }
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
    let mut kernel_graphs = (0..n_kernels)
        .map(|_| (StableGraph::new(), vec![], vec![], vec![]))
        .collect_vec();
    let mut node_maps = (0..n_kernels).map(|_| HashMap::new()).collect_vec();
    for node in marked_graph.node_indices() {
        let (term, loop_level, kernels) = marked_graph.node_weight(node).unwrap();
        for kernel in kernels {
            let (kernel_graph, _, outputs, _) = &mut kernel_graphs[*kernel];
            node_maps[*kernel].insert(
                node,
                kernel_graph.add_node((term.clone(), loop_level.len())),
            );
            if node == root {
                outputs.push((Expression::default(), node_maps[*kernel][&node]));
            }
        }
    }
    // Go through graph inputs and add them to the kernel hashmap as inputs
    for input in marked_graph
        .node_indices()
        // Must not have any input nodes
        .filter(|n| {
            marked_graph
                .neighbors_directed(*n, Direction::Incoming)
                .next()
                .is_none()
        })
        // Must not be an SMEM
        .filter(|n| !matches!(marked_graph.node_weight(*n).unwrap().0, GraphTerm::SMEM))
        .sorted()
    {
        let (term, _, kernels) = marked_graph.node_weight(input).unwrap();
        let label = if let GraphTerm::GMEM { label } = term {
            label.clone()
        } else {
            None
        };
        for kernel in kernels {
            kernel_graphs[*kernel].1.push((
                GMEMBuffer::Input {
                    label: label.clone(),
                },
                node_maps[*kernel][&input],
            ));
        }
    }
    // Mark cross kernel dependencies in the kernel inputs / outputs
    for edge in marked_graph.edge_indices() {
        let (start, end) = marked_graph.edge_endpoints(edge).unwrap();
        let weight = marked_graph.edge_weight(edge).unwrap();
        let (_, _, start_kernels) = marked_graph.node_weight(start).unwrap();
        let (_, _, end_kernels) = marked_graph.node_weight(end).unwrap();
        for end_kernel in end_kernels {
            if !start_kernels.contains(end_kernel) {
                // start kernel must be outputted
                let start_kernel = start_kernels.first().unwrap();
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
                    *weight,
                );
            }
        }
    }

    // Go through SMEM buffers
    for (kernel_graph, inputs, outputs, smem_buffers) in &mut kernel_graphs {
        // Get all ZeroStrideLoad ones at thread level 0..3
        for node in kernel_graph.node_indices() {
            if let (GraphTerm::SMEM, _) = kernel_graph.node_weight(node).unwrap() {
                // Walk forward until load to find the size
                for mut curr in kernel_graph.neighbors_directed(node, Direction::Outgoing) {
                    let mut size = Expression::from(1);
                    let mut load = false;
                    loop {
                        match kernel_graph.node_weight(curr).unwrap().0 {
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
        for (buf, input) in inputs {
            if !matches!(
                kernel_graph.node_weight(*input).unwrap().0,
                GraphTerm::GMEM { .. }
            ) {
                let label = match buf {
                    GMEMBuffer::Input { label, .. } => label.clone(),
                    _ => None,
                };
                let new_input = kernel_graph.add_node((GraphTerm::GMEM { label }, 0));
                kernel_graph.add_edge(new_input, *input, ());
                *input = new_input;
            }
        }
        for (size, output) in outputs {
            if !matches!(
                kernel_graph.node_weight(*output).unwrap().0,
                GraphTerm::GMEM { .. }
            ) {
                let new_output = kernel_graph.add_node((GraphTerm::GMEM { label: None }, 0));
                kernel_graph.add_edge(*output, new_output, ());
                *output = new_output;
            }
            // Loop back through all loopouts to find the size of the output
            let mut curr = *output;
            let mut new_size = Expression::from(1);
            loop {
                let term = &kernel_graph.node_weight(curr).unwrap().0;
                if let GraphTerm::LoopOut { range, stride, .. } = term {
                    if !stride.is_acc() && *stride != 0 {
                        new_size = new_size.max(stride.substitute('z', *range));
                    }
                } else if !matches!(term, GraphTerm::GMEM { .. }) {
                    break;
                }
                curr = kernel_graph
                    .neighbors_directed(curr, Direction::Incoming)
                    .next()
                    .unwrap();
            }
            *size = new_size;
        }
    }
    let root_kernel = marked_graph.node_weight(root).unwrap().2[0];
    (kernel_graphs, root_kernel)
}

pub trait TermToString {
    fn term_to_string(&self) -> String;
}

pub trait EdgeToString {
    fn edge_to_string(&self) -> String;
}

impl EdgeToString for u8 {
    fn edge_to_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeToString for () {
    fn edge_to_string(&self) -> String {
        "".to_string()
    }
}

impl EdgeToString for (u8, u8) {
    fn edge_to_string(&self) -> String {
        format!("{}, {}", self.0, self.1)
    }
}

// impl TermToString for Term {
//     fn term_to_string(&self) -> String {
//         match self {
//             Term::App(a, _) => a.to_string(),
//             Term::Lit(l) => l.to_string(),
//             Term::Var(v) => v.to_string(),
//         }
//     }
// }

impl TermToString for usize {
    fn term_to_string(&self) -> String {
        self.to_string()
    }
}

impl TermToString for String {
    fn term_to_string(&self) -> String {
        self.clone()
    }
}

// impl TermToString for (Term, usize) {
//     fn term_to_string(&self) -> String {
//         let s = match &self.0 {
//             Term::App(a, _) => a.to_string(),
//             Term::Lit(l) => l.to_string(),
//             Term::Var(v) => v.to_string(),
//         };
//         format!("{s}[{}]", self.1)
//     }
// }

impl TermToString for Kernel {
    fn term_to_string(&self) -> String {
        if self.code == "Inputs" || self.code == "Outputs" {
            return self.code.clone();
        } else {
            format!(
                "Kernel ({}, {}, {}), ({}, {}, {}) -> {:?}",
                self.grid.0,
                self.grid.1,
                self.grid.2,
                self.threadblock.0,
                self.threadblock.1,
                self.threadblock.2,
                self.outputs
            )
        }
    }
}

impl TermToString for GraphTerm {
    fn term_to_string(&self) -> String {
        match self {
            GraphTerm::Add => "Add".to_string(),
            GraphTerm::Mul => "Mul".to_string(),
            GraphTerm::Max => "Max".to_string(),
            GraphTerm::Exp => "Exp".to_string(),
            GraphTerm::Sin => "Sin".to_string(),
            GraphTerm::Recip => "Recip".to_string(),
            GraphTerm::Neg => "Neg".to_string(),
            GraphTerm::Sqrt => "Sqrt".to_string(),
            GraphTerm::Mod => "Mod".to_string(),
            GraphTerm::LessThan => "LessThan".to_string(),
            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => format!("LoopIn ({range}; {stride}; -{marker}-)"),
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => format!("LoopOut ({range}; {stride}; -{marker}-)"),
            GraphTerm::GMEM { label } => {
                if let Some(label) = label {
                    format!("GMEM ({label})")
                } else {
                    "GMEM".to_string()
                }
            }
            GraphTerm::SMEM => "SMEM".to_string(),
            GraphTerm::SMEMLoad => "SMEMLoad".to_string(),
            GraphTerm::SMEMRead => "SMEMRead".to_string(),
        }
    }
}

impl TermToString for (GraphTerm, usize) {
    fn term_to_string(&self) -> String {
        format!("{} [{}]", self.0.term_to_string(), self.1)
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!("{} {:?} {{{:?}}}", self.0.term_to_string(), self.1, self.2)
    }
}

impl TermToString for (GraphTerm, Vec<String>, Vec<usize>) {
    fn term_to_string(&self) -> String {
        format!(
            "{} [{}] {{{:?}}}",
            self.0.term_to_string(),
            self.1.len(),
            self.2
        )
    }
}

/// View a debug graph in the browser
pub fn display_graph_2<G: TermToString, E: EdgeToString>(
    graph: &petgraph::stable_graph::StableGraph<G, E, petgraph::Directed, u32>,
    mark_nodes: &[(NodeIndex, String)],
) {
    let mut new_graph = StableGraph::new();
    let mut map = HashMap::new();
    for node in graph.node_indices() {
        map.insert(
            node,
            new_graph.add_node(graph.node_weight(node).unwrap().term_to_string()),
        );
    }
    for edge in graph.edge_indices() {
        let weight = graph.edge_weight(edge).unwrap();
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        new_graph.add_edge(map[&src], map[&dest], weight.edge_to_string());
    }
    let mut graph_string =
        petgraph::dot::Dot::with_config(&new_graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    graph_string = re.replace_all(&graph_string, "").to_string();
    for (n, color) in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"{color}\" label =",
                n.index()
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {:?}", e);
    }
}
