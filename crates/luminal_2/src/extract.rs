use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::usize;

use crate::Kernel;
use crate::debug::display_graph;
use crate::run::{assign_buffers, compile_kernels, run_graph};
use crate::translate::InitData;
use crate::utils::{build_search_space, generate_proof, print_kernels};
#[cfg(feature = "metal")]
use crate::{Buffer, Device};
use crate::{GPUArch, GraphTerm};
#[cfg(feature = "cuda")]
use anyhow::Result;
use colored::Colorize;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, DriverError};
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use luminal::prelude::NodeIndex;
use luminal::prelude::petgraph::prelude::StableGraph;
use luminal::prelude::petgraph::{Directed, Direction};
use luminal::shape::{Expression, Term};
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use rand::{Rng, rng};
use rustc_hash::{FxHashMap, FxHashSet};
#[cfg(feature = "cuda")]
use std::sync::Arc;

const WARMUP_TRIALS: usize = 0;
const TRIALS: usize = 1;
const MAX_SEARCHED_GRAPHS: usize = 100_000;
const MAX_CYCLES: usize = 1;
const INVALID_IR: &[&str] = &[
    "SwapLoops",
    "TileLoop",
    "UnpadLoop",
    "MReplace",
    "MergeLoops",
    "TiledMatmulInputA",
    "TiledMatmulInputB",
    "TiledMatmulAcc",
    "loop_level",
];

#[cfg(feature = "metal")]
#[inline]
fn with_autoreleasepool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    objc2::rc::autoreleasepool(|_| f())
}

#[cfg(feature = "cuda")]
#[inline]
fn with_autoreleasepool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

type Cost = u128; // Execution time in microseconds

fn is_expression_enode(enode_label: &str) -> bool {
    matches!(
        enode_label,
        "MNum"
            | "MVar"
            | "MAdd"
            | "MSub"
            | "MMul"
            | "MDiv"
            | "MMod"
            | "MMin"
            | "MMax"
            | "MAnd"
            | "MOr"
            | "MGte"
            | "MLt"
            | "MFloorTo"
            | "MReplace"
            | "MAccum"
    ) || enode_label.starts_with("MNum:")
        || enode_label.starts_with("MVar:")
}

fn shortest_from_enode<'a>(
    egraph: &'a EGraph,
    enode: &'a NodeId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    if let Some(cached) = cache.get(enode) {
        return cached.clone();
    }
    if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
        cache.insert(enode, None);
        return None;
    }
    if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
        cache.insert(enode, None);
        return None;
    }

    *seen.entry(enode).or_insert(0) += 1;

    let out = if egraph.nodes[enode].children.is_empty() {
        // Leaf → path is just this enode
        Some(vec![enode])
    } else {
        // For each child class, take its shortest; if any child has no path → this enode invalid
        let mut acc: Vec<&'a NodeId> = vec![enode];
        let mut ok = true;

        for child in &egraph.nodes[enode].children {
            let child_class = egraph.nid_to_cid(child);
            if let Some(child_path) = extract_shortest(egraph, child_class, seen, junk, cache) {
                acc.extend(child_path);
            } else {
                ok = false;
                break;
            }
        }

        if ok { Some(acc) } else { None }
    };

    *seen.get_mut(&enode).unwrap() -= 1;

    if out.is_none() {
        junk.insert(enode);
    }
    cache.insert(enode, out.clone());
    out
}

pub fn extract_shortest<'a>(
    egraph: &'a EGraph,
    class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    // Try all enodes in the class and keep the shortest
    let mut best: Option<Vec<&'a NodeId>> = None;
    for enode in &egraph.classes()[class].nodes {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) || junk.contains(enode) {
            junk.insert(enode);
            continue;
        }
        if seen.get(&enode).copied().unwrap_or(0) >= MAX_CYCLES {
            continue;
        }

        if let Some(path) = shortest_from_enode(egraph, enode, seen, junk, cache) {
            if best.as_ref().map_or(true, |b| path.len() < b.len()) {
                best = Some(path);
            }
        } else {
            junk.insert(enode);
        }
    }
    best
}

fn extract_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, Vec<Vec<&'a NodeId>>>,
    waiting: usize,
) -> Vec<Vec<&'a NodeId>> {
    let mut trajectories = vec![];
    'enode_loop: for enode in egraph.classes()[current_class].nodes.iter().rev() {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) {
            junk_cache.insert(enode);
            continue;
        } else if junk_cache.contains(&enode)
            || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
        {
            continue;
        }
        let mut enode_trajectories = vec![];
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            let child_first_enode = child;
            // let child = egraph.nid_to_cid(child);
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                let child_trajectories = if is_expression_enode(&egraph.nodes[child_first_enode].op)
                {
                    extract_shortest(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        &mut FxHashMap::default(),
                    )
                    .map(|i| vec![i])
                    .unwrap_or_default()
                } else if egraph.nodes[child_first_enode].op == "Loop" {
                    // Pull just the range out for the loop
                    extract_shortest(
                        egraph,
                        egraph.nid_to_cid(&egraph.nodes[child_first_enode].children[1]),
                        seen,
                        junk_cache,
                        &mut FxHashMap::default(),
                    )
                    .map(|i| vec![i])
                    .unwrap_or_default()
                } else {
                    extract_trajectories(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        trajectory_cache,
                        (waiting * enode_trajectories.len().max(1)) + trajectories.len(),
                    )
                };
                if child_trajectories.is_empty() {
                    // bad enode
                    junk_cache.insert(enode);
                    *seen.get_mut(&enode).unwrap() -= 1;
                    continue 'enode_loop;
                }
                trajectory_cache.insert(child, child_trajectories.clone());
            }

            if enode_trajectories.is_empty() {
                // First child
                for mut child_trajectory in trajectory_cache[child].clone() {
                    child_trajectory.insert(0, enode);
                    enode_trajectories.push(child_trajectory);
                }
            } else if !trajectory_cache[child].is_empty() {
                // Cartisian product the current trajectories with the new trajectories
                let n_enode_traj = enode_trajectories.len();
                enode_trajectories = enode_trajectories
                    .into_iter()
                    .cartesian_product(
                        trajectory_cache[child]
                            .iter()
                            .take(MAX_SEARCHED_GRAPHS / n_enode_traj),
                    )
                    .map(|(p, n)| [p, n.clone()].concat())
                    .collect();
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;

        if egraph.nodes[enode].children.is_empty() {
            // Leaf node → single-element trajectory
            trajectories.push(vec![enode]);
        } else {
            // Add combined trajectories
            trajectories.extend(enode_trajectories);
        }
        if trajectories.len() * waiting > MAX_SEARCHED_GRAPHS {
            break; // Only pick the first valid (non cycling) enode for expressions
        }
    }
    trajectories
}

pub fn search(
    graph: &StableGraph<GraphTerm, ()>,
    steps: usize,
    inputs: &[(String, InitData)],
    arch: GPUArch,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<StableGraph<GraphTerm, ()>> {
    let og = graph.clone();
    let egraph = build_search_space(graph, steps);
    let trajectories = extract_trajectories(
        &egraph,
        &egraph.root_eclasses[0],
        &mut FxHashMap::default(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
        1,
    );
    // build loop level -> enode mapping
    let mut loop_level_values = FxHashMap::default();
    for (id, _) in &egraph.class_data {
        if egraph.classes()[id]
            .nodes
            .iter()
            .any(|n| egraph.nodes[n].op == "loop_level")
        {
            loop_level_values.insert(
                id,
                egraph.classes()[id]
                    .nodes
                    .iter()
                    .find_map(|n| egraph.nodes[n].op.parse::<i32>().ok())
                    .unwrap(),
            );
        }
    }
    let mut loop_level_map = FxHashMap::default();
    for (id, node) in &egraph.nodes {
        if node.op == "loop_level" {
            for child in &node.children {
                loop_level_map.insert(child, loop_level_values[egraph.nid_to_cid(id)]);
            }
        }
    }

    // Now we have DFS trajectories
    let mut ref_outputs: Vec<Vec<f32>> = vec![];
    let mut best_time = u128::MAX;
    let mut fastest = "".to_string();
    let mut best_graph = None;
    let mut valid_graphs = 0;
    let total_trajectories = trajectories.len().min(MAX_SEARCHED_GRAPHS);
    let mut prev_graphs = vec![];
    let mut prev_traj = vec![];
    let mut og_kernels = "".to_string();
    let mut ui_functions = None;
    if option_env!("DEBUG").is_none() {
        ui_functions = Some(crate::utils::search_ui());
    };
    let mut seen = FxHashSet::default();
    let mut possibles = 0;
    'trajectory_loop: for (n, trajectory) in trajectories
        .into_iter()
        .take(MAX_SEARCHED_GRAPHS)
        .enumerate()
    {
        // Build termdag
        let mut graph = extraction_to_graph(&egraph, &trajectory, &loop_level_map);
        // crate::debug::display_graph2(&graph, &[]);
        prev_graphs.push(graph.clone());
        prev_traj.push(trajectory.clone());

        // Dedup GMEMs (don't think we need this?)
        let mut canon: FxHashMap<String, NodeIndex> = FxHashMap::default();

        for n in graph.node_indices().collect::<Vec<_>>() {
            if let GraphTerm::GMEM { label } = &graph[n] {
                match canon.entry(label.clone()) {
                    Entry::Vacant(e) => {
                        e.insert(n);
                    }
                    Entry::Occupied(e) => {
                        let c = *e.get();
                        for src in graph
                            .neighbors_directed(n, Direction::Incoming)
                            .collect::<Vec<_>>()
                        {
                            graph.update_edge(src, c, ());
                        }
                        for dst in graph
                            .neighbors_directed(n, Direction::Outgoing)
                            .collect::<Vec<_>>()
                        {
                            graph.update_edge(c, dst, ());
                        }
                        graph.remove_node(n);
                    }
                }
            }
        }

        // Build input mapping
        let node_index_to_init_data: Vec<(NodeIndex, InitData)> = inputs
            .iter()
            .filter_map(|(label, data)| canon.get(label).map(|&n| (n, data.clone())))
            .collect();

        let Some((kernels, gmem_mapping)) =
            crate::codegen::codegen(graph.clone(), arch.clone(), dyn_vars)
        else {
            continue;
        };
        possibles += 1;
        // let inputs = inputs.into_iter().filter_map(|(l, d)| graph.node_indices().find(|n| matches!(graph.node_weight(*n).unwrap(), GraphTerm::GMEM { label } if label == l)).map(|i| (i, d.clone()))).collect_vec();
        match &arch {
            GPUArch::CUDA => {
                let k = print_kernels(&kernels);
                if seen.contains(&k) {
                    continue;
                } else {
                    seen.insert(k);
                }
                if let Some((us, outs)) = cost(
                    &graph,
                    &kernels,
                    &node_index_to_init_data,
                    &gmem_mapping,
                    dyn_vars,
                ) {
                    valid_graphs += 1;
                    if let Some((progress, logs, title, _)) = &ui_functions {
                        progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                        logs(print_kernels(&kernels));
                        title(format!("Graph {valid_graphs} {us}µs"));
                    } else if option_env!("DEBUG").is_some() {
                        println!("{}", print_kernels(&kernels));
                        println!("Graph {valid_graphs} {us}µs");
                        if ref_outputs.is_empty() {
                            ref_outputs = outs;
                        } else {
                            for (a, b) in ref_outputs.iter().zip(&outs) {
                                for (x, y) in a.iter().zip(b) {
                                    if (x - y).abs() >= 0.1 {
                                        if option_env!("DEBUG").is_some() {
                                            // display_graph(&graph, &[]);
                                            println!(
                                                "REF: {:?}",
                                                &ref_outputs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            println!(
                                                "New: {:?}",
                                                &outs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            crate::debug::display_graph(&og);
                                            crate::debug::display_graph(&graph);
                                            generate_proof(&og, &graph);
                                            println!("{}", og_kernels);
                                            println!("{}", print_kernels(&kernels));
                                            panic!(
                                                "{} {x} != {y} {}",
                                                "Output Mismatch".bold().on_bright_red(),
                                                (x - y).abs()
                                            );
                                        }
                                        continue 'trajectory_loop;
                                    }
                                }
                            }
                            println!("{}", "Outputs Validated".bold().on_bright_green());
                        }
                    }
                    let kernel_string = print_kernels(&kernels);
                    if og_kernels.is_empty() {
                        og_kernels = kernel_string.clone();
                    }
                    if us < best_time {
                        best_time = us;
                        best_graph = Some(graph);
                        fastest = kernel_string;
                    }
                }
            }
            GPUArch::Metal(_) => {
                let k = print_kernels(&kernels);
                if seen.contains(&k) {
                    continue;
                } else {
                    seen.insert(k);
                }
                if let Some((us, outs)) = cost(
                    &graph,
                    &kernels,
                    &node_index_to_init_data,
                    &gmem_mapping,
                    dyn_vars,
                ) {
                    valid_graphs += 1;
                    if let Some((progress, logs, title, _)) = &ui_functions {
                        progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                        logs(print_kernels(&kernels));
                        title(format!("Graph {valid_graphs} {us}µs"));
                    } else if option_env!("DEBUG").is_some() {
                        println!("{}", print_kernels(&kernels));
                        println!("Graph {valid_graphs} {us}µs");
                        if ref_outputs.is_empty() {
                            ref_outputs = outs;
                            println!("{}", "Initial".bold().on_bright_green());
                        } else {
                            for (a, b) in ref_outputs.iter().zip(&outs) {
                                for (x, y) in a.iter().zip(b) {
                                    if (x - y).abs() >= 1e-1 {
                                        if option_env!("DEBUG").is_some() {
                                            // display_graph(&graph, &[]);
                                            println!(
                                                "REF: {:?}",
                                                &ref_outputs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            println!(
                                                "New: {:?}",
                                                &outs
                                                    .iter()
                                                    .map(|v| &v[..v.len().min(20)])
                                                    .collect_vec()
                                            );
                                            // crate::utils::generate_proof(&og, &graph);
                                            println!("{}", og_kernels);
                                            println!("{}", print_kernels(&kernels));
                                            crate::debug::display_multiple_graphs(&[&og, &graph]);
                                            panic!(
                                                "{} {x} != {y}",
                                                "Output Mismatch".bold().on_bright_red()
                                            );
                                        }
                                        continue 'trajectory_loop;
                                    }
                                }
                            }
                            println!("{}", "Outputs Validated".bold().on_bright_green());
                        }
                    }
                    let kernel_string = print_kernels(&kernels);
                    if og_kernels.is_empty() {
                        og_kernels = kernel_string.clone();
                    }
                    if us < best_time {
                        best_time = us;
                        best_graph = Some(graph);
                        fastest = kernel_string;
                    }
                }
            }
        }
    }
    if let Some((_, _, _, e)) = &ui_functions {
        e();
    }
    println!("FASTEST ({}ms): {fastest}", best_time / 1000);
    println!("Valids: {:?} / {:?}", possibles, total_trajectories);
    best_graph
}

pub fn extraction_to_graph(
    egraph: &EGraph,
    trajectory: &[&NodeId],
    loop_level_map: &FxHashMap<&NodeId, i32>,
) -> StableGraph<GraphTerm, (), Directed> {
    let mut g: StableGraph<GraphTerm, (), Directed> = StableGraph::new();

    #[derive(Debug, Clone)]
    enum Ret {
        Expr(NodeIndex),
        Math(Expression),
        Op(GraphTerm),
    }

    fn recurse<'b, 'a: 'b>(
        egraph: &EGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
        g: &mut StableGraph<GraphTerm, (), Directed>,
        loop_level_map: &FxHashMap<&NodeId, i32>,
        prev_placed: &'b mut FxHashMap<&'a NodeId, NodeIndex>,
        no_place: bool,
    ) -> Ret {
        let node_choice = trajectory[*current];
        let enode = &egraph.nodes[node_choice];
        match enode.op.as_str() {
            "GMEM" => {
                *current += 1;
                if no_place {
                    Ret::Expr(NodeIndex::default())
                } else {
                    Ret::Expr(*prev_placed.entry(node_choice).or_insert_with(|| {
                        g.add_node(GraphTerm::GMEM {
                            label: egraph.nodes[&enode.children[0]]
                                .op
                                .replace("Boxed(\"", "")
                                .replace("\")", ""),
                        })
                    }))
                }
            }
            "SMEM" => todo!(),

            // LoopIn  = (LoopIn <expr> <Math> <Math>)
            "LoopIn" | "LoopOut" => {
                *current += 1;
                let already_placed = prev_placed.contains_key(node_choice);
                let Ret::Expr(child_one) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(range) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!();
                };
                if no_place {
                    Ret::Expr(NodeIndex::default())
                } else if let Some(n) = prev_placed.get(node_choice) {
                    Ret::Expr(*n)
                } else {
                    let r = g.add_node(match enode.op.as_str() {
                        "LoopIn" => GraphTerm::LoopIn {
                            range,
                            stride,
                            marker: loop_level_map
                                .get(node_choice)
                                .map(|i| i.to_string())
                                .unwrap_or_default(),
                        },
                        "LoopOut" => GraphTerm::LoopOut {
                            range,
                            stride,
                            marker: loop_level_map
                                .get(node_choice)
                                .map(|i| i.to_string())
                                .unwrap_or_default(),
                        },
                        _ => panic!(),
                    });
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    Ret::Expr(r)
                }
            }

            "TCMatmul" => {
                *current += 1;
                let already_placed = prev_placed.contains_key(node_choice);
                let Ret::Expr(src_a) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Expr(src_b) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(a_k_stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(b_k_stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(a_inner_stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(b_inner_stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(c_inner_stride) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(k_outer_loops) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                if no_place {
                    Ret::Expr(NodeIndex::default())
                } else if let Some(n) = prev_placed.get(node_choice) {
                    Ret::Expr(*n)
                } else {
                    let r = g.add_node(GraphTerm::TCMatmul {
                        a_k_stride,
                        b_k_stride,
                        a_inner_stride,
                        b_inner_stride,
                        c_inner_stride,
                        k_outer_loops,
                    });
                    prev_placed.insert(node_choice, r);
                    g.add_edge(src_a, r, ());
                    g.add_edge(src_b, r, ());
                    Ret::Expr(r)
                }
            }
            "Binary" => {
                *current += 1;
                let already_placed = prev_placed.contains_key(node_choice);
                let Ret::Op(op) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                let Ret::Expr(child_one) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Expr(child_two) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                if no_place {
                    Ret::Expr(NodeIndex::default())
                } else if let Some(n) = prev_placed.get(node_choice) {
                    Ret::Expr(*n)
                } else {
                    let r = g.add_node(op);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    g.add_edge(child_two, r, ());
                    Ret::Expr(r)
                }
            }
            "Add" | "Mul" | "Max" => {
                *current += 1;
                Ret::Op(match enode.op.as_str() {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    _ => panic!(),
                })
            }
            "Unary" => {
                *current += 1;
                let already_placed = prev_placed.contains_key(node_choice);
                let Ret::Op(op) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                let Ret::Expr(child_one) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already_placed || no_place,
                ) else {
                    panic!()
                };
                if no_place {
                    Ret::Expr(NodeIndex::default())
                } else if let Some(n) = prev_placed.get(node_choice) {
                    Ret::Expr(*n)
                } else {
                    let r = g.add_node(op);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    Ret::Expr(r)
                }
            }
            "Exp2" | "Sin" | "Recip" | "Neg" | "Sqrt" => {
                *current += 1;
                Ret::Op(match enode.op.as_str() {
                    "Exp2" => GraphTerm::Exp2,
                    "Log2" => GraphTerm::Log2,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    "Sqrt" => GraphTerm::Sqrt,
                    _ => panic!(),
                })
            }
            "Fused" => {
                *current += 1;
                recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                )
            }
            // ----------- literals & vars -----------
            op if op.starts_with("MNum:") => {
                let num: i64 = op["MNum:".len()..].parse().expect("invalid MNum literal");
                Ret::Math(Expression::from(num as usize))
            }
            op if op.starts_with("MVar:") => {
                let name = op["MVar:".len()..].to_owned();
                Ret::Math(Expression::from(name.chars().next().unwrap()))
            }
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                Ret::Math(Expression::from(name.chars().next().unwrap()))
            }

            // ----------- unary ops -----------
            "MNeg" | "MRecip" => {
                *current += 1;
                let Ret::Math(c0) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                ) else {
                    panic!()
                };
                Ret::Math(match enode.op.as_str() {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                })
            }

            // ----------- binary ops -----------
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                *current += 1;
                let Ret::Math(lhs) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                ) else {
                    panic!()
                };
                *current += 1;
                let Ret::Math(rhs) = recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                ) else {
                    panic!()
                };
                Ret::Math(match enode.op.as_str() {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MFloorTo" => lhs / rhs * rhs, // NOT CORRECT, NEED FLOORTO IN EXPRESSIONS
                    _ => unreachable!(),
                })
            }
            "MAccum" => {
                *current += 1;
                Ret::Math(Expression::from(Term::Acc('a')))
            }
            "MNum" | "MVar" => {
                *current += 1;
                recurse(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                )
            }
            _ => {
                if let Ok(n) = enode.op.parse::<usize>() {
                    Ret::Math(Expression::from(n))
                } else {
                    panic!("unsupported op '{}'", enode.op)
                }
            }
        }
    }

    recurse(
        egraph,
        trajectory,
        &mut 0,
        &mut g,
        loop_level_map,
        &mut FxHashMap::default(),
        false,
    );
    for n in g.node_indices() {
        if g.neighbors_undirected(n).next().is_none() {
            display_graph(&g);
        }
    }
    g
}

fn cost<'a>(
    graph: &StableGraph<GraphTerm, ()>,
    kernels: &StableGraph<Kernel, (usize, usize), Directed>,
    inputs: &[(NodeIndex, InitData)],
    gmem_mapping: &HashMap<NodeIndex, usize>,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<(Cost, Vec<Vec<f32>>)> {
    with_autoreleasepool(|| {
        // Get buffer info
        let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
        let compiled_kernels = compile_kernels(&kernels);
        #[cfg(feature = "metal")]
        let device = MTLCreateSystemDefaultDevice().unwrap();
        #[cfg(feature = "cuda")]
        let ctx = CudaContext::new(0).unwrap(); // will need to expand beyond single host
        // Copy input buffers over
        let mut inputs = inputs
            .into_iter()
            .filter(|(n, _)| gmem_mapping.contains_key(n))
            .map(|(n, b)| {
                (
                    gmem_mapping[n],
                    (
                        #[cfg(feature = "metal")]
                        copy_metal_buffer(&b.clone().to_vec(dyn_vars), &device),
                        #[cfg(feature = "cuda")]
                        copy_cuda_buffer(&b.clone().to_vec(dyn_vars), ctx.clone()),
                        false,
                    ),
                )
            })
            .collect::<FxHashMap<_, _>>();
        // Warm up resources (buffer allocation, kernel compiler, etc.)
        for _ in 0..WARMUP_TRIALS {
            #[cfg(feature = "metal")]
            run_graph(
                &graph,
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
            #[cfg(feature = "cuda")]
            run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &int_buffers,
                &int_buffer_map,
            );
        }
        // Test runtime
        let mut micros = vec![];
        let mut outputs = vec![];

        for _ in 0..TRIALS {
            let (o, m_val) = {
                #[cfg(feature = "metal")]
                {
                    run_graph(
                        &graph,
                        &mut inputs,
                        &kernels,
                        dyn_vars,
                        &compiled_kernels,
                        &int_buffers,
                        &int_buffer_map,
                    )
                }

                #[cfg(feature = "cuda")]
                {
                    run_graph(
                        &mut inputs,
                        &kernels,
                        dyn_vars,
                        &compiled_kernels,
                        &int_buffers,
                        &int_buffer_map,
                    )
                }
            };
            outputs = o;
            micros.push(m_val);
        }
        Some((
            micros.into_iter().sum::<u128>() / TRIALS as u128,
            #[cfg(feature = "metal")]
            outputs.iter().map(copy_metal_buffer_back).collect_vec(),
            #[cfg(feature = "cuda")]
            outputs.iter().map(copy_cuda_buffer_back).collect_vec(),
        ))
    })
}

#[cfg(feature = "cuda")]
pub fn copy_cuda_buffer(v: &[f32], ctx: Arc<CudaContext>) -> CudaSlice<f32> {
    assert!(!v.is_empty(), "Can't copy empty slice to device");

    // Then copy host data to the allocated device memory
    let stream = ctx.default_stream();
    let mut dst = stream.alloc_zeros::<f32>(v.len()).unwrap();
    stream.memcpy_htod(v, &mut dst).unwrap();
    dst
}

/// Device -> Host (like contents() memcpy back)
#[cfg(feature = "cuda")]
pub fn copy_cuda_buffer_back(buf: &CudaSlice<f32>) -> Vec<f32> {
    buf.stream().memcpy_dtov(buf).unwrap()
}

#[cfg(feature = "metal")]
pub fn copy_metal_buffer(v: &Vec<f32>, device: &Device) -> Buffer {
    assert!(v.len() > 0);
    unsafe {
        let ptr = NonNull::new(v.as_ptr() as *mut c_void).unwrap();
        device
            .newBufferWithBytes_length_options(
                ptr,
                (v.len() * 4) as _,
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap()
    }
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

pub fn make_test_inputs(
    graph: &StableGraph<GraphTerm, ()>,
    dyn_map: &FxHashMap<char, usize>,
    inits: &[(String, InitData)],
) -> Vec<(String, InitData)> {
    // Go through each GMEM and work out the size
    let mut inputs = vec![];
    let mut rng = rng();
    for node in graph.externals(Direction::Incoming) {
        if let GraphTerm::GMEM { label } = graph.node_weight(node).unwrap() {
            if let Some(init) = inits.iter().find(|(n, _)| n == label) {
                inputs.push(init.clone());
                continue;
            }
            // Walk down the loopins to find the max size
            let mut size = Expression::from(1);
            let mut curr = graph
                .neighbors_directed(node, Direction::Outgoing)
                .next()
                .unwrap();
            loop {
                if let GraphTerm::LoopIn { range, stride, .. } = graph.node_weight(curr).unwrap() {
                    size = size.max(stride.substitute('z', *range));
                    // size = size.max(stride.substitute('z', *range - 1) + 1); // why were we doing this?
                    curr = graph
                        .neighbors_directed(curr, Direction::Outgoing)
                        .next()
                        .unwrap();
                } else {
                    break;
                }
            }
            inputs.push((
                label.clone(),
                InitData::Data(
                    (0..size.exec(&dyn_map).unwrap())
                        .map(|_| rng.random_range(-1e-3..1e-3))
                        .collect(),
                ),
            ));
        }
    }
    inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        translate::{MetaGraph, SubGraph, translate_graph},
        utils::build_search_space,
    };

    fn build_minimal_add_graph() -> (luminal::graph::Graph, MetaGraph, SubGraph) {
        use luminal::graph::Graph;

        let mut cx = Graph::new();
        let a = cx.tensor(3).set([1., 2., 3.]);
        let b = cx.tensor(3).set([4., 5., 6.]);
        let c = (a + b).sqrt();
        let d = c * a;
        let _e = d.sum(0).retrieve();

        let (meta_graph, _global_map, _inits) = translate_graph(&cx);
        let meta_node = meta_graph
            .node_indices()
            .next()
            .expect("MetaGraph unexpectedly empty");
        let sub = meta_graph
            .node_weight(meta_node)
            .expect("Missing subgraph at meta node")
            .clone();

        (cx, meta_graph, sub)
    }

    fn build_nonempty_egraph() -> EGraph {
        // Keep `cx` and `meta_graph` alive while we build the egraph
        let (_cx, meta_graph, sub) = build_minimal_add_graph();
        let e = build_search_space(&sub, /*iters=*/ 2);
        // `_cx` and `meta_graph` can drop now; `e` no longer needs them
        drop(meta_graph);
        e
    }

    #[test]
    fn test_egraph_is_nonempty_and_has_root() {
        let egraph = build_nonempty_egraph();
        assert!(!egraph.classes().is_empty(), "EGraph should have classes");
        assert!(
            !egraph.root_eclasses.is_empty(),
            "EGraph should have a root"
        );
    }

    #[test]
    fn test_extract_trajectories_invalid_ir_filtering() {
        let egraph = build_nonempty_egraph();

        if egraph.classes().is_empty() || egraph.root_eclasses.is_empty() {
            return;
        }

        let root_class = &egraph.root_eclasses[0];
        let mut seen = FxHashMap::default();
        let mut junk_cache = FxHashSet::default();
        let mut trajectory_cache = FxHashMap::default();

        let trajectories = extract_trajectories(
            &egraph,
            root_class,
            &mut seen,
            &mut junk_cache,
            &mut trajectory_cache,
            1,
        );

        // Check that trajectories don't contain INVALID_IR operations
        for trajectory in trajectories {
            for &node in &trajectory {
                let op_name = &egraph.nodes[node].op;
                assert!(
                    !INVALID_IR.contains(&op_name.as_str()),
                    "Trajectory contains invalid IR operation: {}",
                    op_name
                );
            }
        }
    }

    #[test]
    fn test_is_expression_enode() {
        // Test that expression enodes are correctly identified
        assert!(is_expression_enode("MNum"));
        assert!(is_expression_enode("MVar"));
        assert!(is_expression_enode("MAdd"));
        assert!(is_expression_enode("MNum:42"));
        assert!(is_expression_enode("MVar:x"));

        // Test that non-expression enodes are not identified
        assert!(!is_expression_enode("GMEM"));
        assert!(!is_expression_enode("LoopIn"));
        assert!(!is_expression_enode("Add"));
        assert!(!is_expression_enode("Invalid"));
    }
}
