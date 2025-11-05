use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::usize;

use crate::codegen::codegen;
use crate::debug::display_graph;
use crate::run::{assign_buffers, compile_kernels, htod, new_buffer, run_graph};
use crate::translate::InitData;
use crate::utils::{build_search_space, print_kernels};
use crate::{Buffer, Kernel};
use crate::{GPUArch, GraphTerm};
use colored::Colorize;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use egraph_serialize::{ClassId, EGraph, NodeId};
use itertools::Itertools;
use luminal::prelude::petgraph::prelude::StableGraph;
use luminal::prelude::petgraph::{Directed, Direction};
use luminal::prelude::NodeIndex;
use luminal::shape::{Expression, Term};
#[cfg(feature = "metal")]
use objc2_metal::MTLCreateSystemDefaultDevice;
use rand::{rng, Rng};
use rustc_hash::{FxHashMap, FxHashSet};

const WARMUP_TRIALS: usize = 0;
const TRIALS: usize = 1;
const MAX_SEARCHED_GRAPHS: usize = 10_000;
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
    "vec-of",
    "set-of",
];

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

pub fn extract_shortest<'a>(
    egraph: &'a EGraph,
    class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk: &mut FxHashSet<&'a NodeId>,
    cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
) -> Option<Vec<&'a NodeId>> {
    egraph.classes()[class]
        .nodes
        .iter()
        .filter_map(|en| {
            if INVALID_IR.contains(&egraph.nodes[en].op.as_str()) || junk.contains(en) {
                junk.insert(en);
                return None;
            }
            if *seen.get(en).unwrap_or(&0) >= MAX_CYCLES {
                return None;
            }
            if let Some(c) = cache.get(en) {
                return c.clone();
            }
            *seen.entry(en).or_insert(0) += 1;
            let out = if egraph.nodes[en].children.is_empty() {
                Some(vec![en])
            } else {
                egraph.nodes[en]
                    .children
                    .iter()
                    .try_fold(vec![en], |mut acc, ch| {
                        extract_shortest(egraph, &egraph.nid_to_cid(ch), seen, junk, cache).map(
                            |p| {
                                acc.extend(p);
                                acc
                            },
                        )
                    })
            };
            *seen.get_mut(en).unwrap() -= 1;
            if out.is_none() {
                junk.insert(en);
            }
            cache.insert(en, out.clone());
            out
        })
        .min_by_key(|p| p.len())
}

fn extract_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, Vec<Vec<&'a NodeId>>>,
    trajectory_counts: &FxHashMap<&'a NodeId, usize>,
    mut budget: usize,
) -> Vec<Vec<&'a NodeId>> {
    let mut trajectories = vec![];
    let enodes = egraph.classes()[current_class]
        .nodes
        .iter()
        .filter(|e| !INVALID_IR.contains(&egraph.nodes[*e].op.as_str()))
        .filter(|e| !junk_cache.contains(*e))
        .filter(|e| seen.get(*e).copied().unwrap_or_default() < MAX_CYCLES)
        .collect_vec();
    let enode_budgets = sample_enodes(
        &enodes
            .iter()
            .map(|e| {
                // if egraph.nodes[*e].op == "Fused" {
                //     trajectory_counts[e] * 2
                // } else {
                trajectory_counts[e]
                // }
            })
            .collect_vec(),
        budget,
    );
    'enode_loop: for (enode, enode_budget) in enodes.into_iter().zip(enode_budgets).collect_vec() {
        if enode_budget == 0 {
            continue;
        }
        let mut enode_trajectories = vec![];
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                let child_trajectories = if is_expression_enode(&egraph.nodes[child].op) {
                    extract_shortest(
                        egraph,
                        egraph.nid_to_cid(child),
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
                        trajectory_counts,
                        enode_budget,
                    )
                };
                if child_trajectories.is_empty() {
                    // bad enode
                    junk_cache.insert(enode);
                    *seen.get_mut(&enode).unwrap() -= 1;
                    continue 'enode_loop;
                }
                trajectory_cache.insert(child, child_trajectories);
            }

            if enode_trajectories.is_empty() {
                // First child
                for mut child_trajectory in trajectory_cache[child].clone() {
                    child_trajectory.insert(0, enode);
                    enode_trajectories.push(child_trajectory);
                }
            } else if !trajectory_cache[child].is_empty() {
                // Cartisian product the current trajectories with the new trajectories
                enode_trajectories = enode_trajectories
                    .into_iter()
                    .cartesian_product(&trajectory_cache[child])
                    .take(enode_budget)
                    .map(|(p, n)| [p, n.clone()].concat())
                    .collect();
            }
            if egraph.nodes[enode].op == "Fused" {
                break;
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;

        if egraph.nodes[enode].children.is_empty() {
            // Leaf node → single-element trajectory
            trajectories.push(vec![enode]);
            budget -= 1;
        } else {
            // Add combined trajectories
            budget -= enode_trajectories.len();
            trajectories.extend(enode_trajectories);
        }
        if budget == 0 {
            break; // Only pick the first enode for expressions
        }
    }
    trajectories
}

/// Given total trajectories each enode can yield, and a budget, return how many trajectories should be yielded from each enode
fn sample_enodes(enode_traj_counts: &[usize], traj_budget: usize) -> Vec<usize> {
    let total: f64 = enode_traj_counts.iter().map(|&c| c as f64).sum();
    if total == 0.0 {
        return vec![0; enode_traj_counts.len()];
    }

    // Expected allocation proportional to counts
    let mut alloc: Vec<usize> = enode_traj_counts
        .iter()
        .map(|&c| ((traj_budget as f64) * (c as f64) / total).floor() as usize)
        .collect();

    // Distribute remainder by largest fractional parts
    let remainder = traj_budget.saturating_sub(alloc.iter().sum());
    let mut frac_parts: Vec<(usize, f64)> = enode_traj_counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let expected = (traj_budget as f64) * (c as f64) / total;
            (i, expected - alloc[i] as f64)
        })
        .collect();

    frac_parts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for &(i, _) in frac_parts.iter().take(remainder) {
        alloc[i] += 1;
    }

    alloc
}

fn count_trajectories<'a>(
    egraph: &'a EGraph,
    current_class: &'a ClassId,
    seen: &mut FxHashMap<&'a NodeId, usize>,
    junk_cache: &mut FxHashSet<&'a NodeId>,
    trajectory_cache: &mut FxHashMap<&'a NodeId, usize>,
) -> usize {
    let mut trajectories = 0;
    'enode_loop: for enode in egraph.classes()[current_class].nodes.iter().rev() {
        if INVALID_IR.contains(&egraph.nodes[enode].op.as_str()) {
            junk_cache.insert(enode);
            continue;
        } else if junk_cache.contains(&enode)
            || seen.get(&enode).copied().unwrap_or_default() >= MAX_CYCLES
        {
            continue;
        }
        let mut enode_trajectories = 1;
        *seen.entry(enode).or_insert(0) += 1;
        for child in &egraph.nodes[enode].children {
            // Ask what's the child's trajectories
            if !trajectory_cache.contains_key(child) {
                if is_expression_enode(&egraph.nodes[child].op) {
                    trajectory_cache.insert(child, 1);
                } else {
                    let child_trajectories = count_trajectories(
                        egraph,
                        egraph.nid_to_cid(child),
                        seen,
                        junk_cache,
                        trajectory_cache,
                    );
                    if child_trajectories == 0 {
                        // bad enode
                        junk_cache.insert(enode);
                        *seen.get_mut(&enode).unwrap() -= 1;
                        continue 'enode_loop;
                    }
                    trajectory_cache.insert(child, child_trajectories);
                }
            }

            // Cartisian product the current trajectories with the new trajectories
            enode_trajectories *= trajectory_cache[child].max(1);
            if egraph.nodes[enode].op == "Fused" {
                break;
            }
        }
        *seen.get_mut(&enode).unwrap() -= 1;
        trajectory_cache.insert(enode, enode_trajectories);
        trajectories += enode_trajectories;
    }
    trajectories
}

pub fn human_readable(n: usize) -> String {
    const THOUSAND: f64 = 1_000.0;
    const MILLION: f64 = 1_000_000.0;
    const BILLION: f64 = 1_000_000_000.0;
    const TRILLION: f64 = 1_000_000_000_000.0;

    let n_f = n as f64;

    if n_f >= TRILLION {
        format!("{:.1} trillion", n_f / TRILLION)
    } else if n_f >= BILLION {
        format!("{:.1} billion", n_f / BILLION)
    } else if n_f >= MILLION {
        format!("{:.1} million", n_f / MILLION)
    } else if n_f >= THOUSAND {
        format!("{:.1} thousand", n_f / THOUSAND)
    } else {
        n.to_string()
    }
}

pub fn search(
    graph: &StableGraph<GraphTerm, ()>,
    steps: usize,
    inputs: &[(String, InitData)],
    arch: GPUArch,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<StableGraph<GraphTerm, ()>> {
    // Get reference outputs
    let mut ref_graph = graph.clone();
    let mut canon: FxHashMap<String, NodeIndex> = FxHashMap::default();
    for n in ref_graph.node_indices().collect_vec() {
        if let GraphTerm::GMEM { label } = &ref_graph[n] {
            match canon.entry(label.clone()) {
                Entry::Vacant(e) => {
                    e.insert(n);
                }
                Entry::Occupied(e) => {
                    let c = *e.get();
                    for src in ref_graph
                        .neighbors_directed(n, Direction::Incoming)
                        .collect_vec()
                    {
                        ref_graph.update_edge(src, c, ());
                    }
                    for dst in ref_graph
                        .neighbors_directed(n, Direction::Outgoing)
                        .collect_vec()
                    {
                        ref_graph.update_edge(c, dst, ());
                    }
                    ref_graph.remove_node(n);
                }
            }
        }
    }
    let (ref_kernels, ref_gmem_map) = codegen(ref_graph.clone(), arch.clone(), dyn_vars).unwrap();
    #[cfg(feature = "metal")]
    let device = MTLCreateSystemDefaultDevice().unwrap();
    #[cfg(feature = "cuda")]
    let device = crate::run::shared_cuda_context();
    let inputs = inputs
        .into_iter()
        .map(|(n, init)| {
            (
                n.clone(),
                match init {
                    InitData::Data(d) => htod(d, &device),
                    InitData::Expr(e) => htod(&vec![e.exec(dyn_vars).unwrap() as f32], &device),
                },
            )
        })
        .collect_vec();
    let (_, ref_outputs) = cost(
        &ref_kernels,
        &map_inputs_into_kernel_graph(&inputs, &ref_graph),
        &ref_gmem_map,
        dyn_vars,
    )
    .unwrap();

    // Build search space and get trajectories
    let egraph = build_search_space(graph, steps);
    let mut trajectory_counts = FxHashMap::default();
    let total_trajectories = count_trajectories(
        &egraph,
        &egraph.root_eclasses[0],
        &mut FxHashMap::default(),
        &mut FxHashSet::default(),
        &mut trajectory_counts,
    );
    if option_env!("PRINT_EGGLOG").is_some() {
        println!(
            "Total Trajectories in E-Graph: {}",
            human_readable(total_trajectories).bold()
        );
    }
    // display_egraph(&egraph);
    let mut trajectories = extract_trajectories(
        &egraph,
        &egraph.root_eclasses[0],
        &mut FxHashMap::default(),
        &mut FxHashSet::default(),
        &mut FxHashMap::default(),
        &trajectory_counts,
        MAX_SEARCHED_GRAPHS,
    );
    trajectories.sort_by_key(|t| t.len());
    // trajectories.shuffle(&mut rng());
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
                for node in &egraph.classes()[egraph.nid_to_cid(child)].nodes {
                    loop_level_map.insert(node, loop_level_values[egraph.nid_to_cid(id)]);
                }
            }
        }
    }

    // if std::env::var("DEBUG").is_ok() {
    //     // make sure all IR nodes loop levels
    //     for (id, node) in &egraph.nodes {
    //         if node.eclass.to_string().starts_with("IR-") {
    //             assert!(
    //                 loop_level_map.contains_key(id),
    //                 "Loop level not found for {}",
    //                 node.op
    //             );
    //         }
    //     }
    // }

    // Run through each trajectory, codegen and test graph
    let mut best_time = u128::MAX;
    let mut fastest = "".to_string();
    let mut best_graph = None;
    let mut valid_graphs = 0;
    let total_trajectories = trajectories.len();
    let mut ui_functions = None;
    if option_env!("DEBUG").is_none() {
        ui_functions = Some(crate::utils::search_ui());
    };
    let mut seen = FxHashSet::default();
    let mut valids = 0;
    'trajectory_loop: for (n, trajectory) in trajectories.into_iter().enumerate() {
        let graph = extraction_to_graph(&egraph, &trajectory, &loop_level_map);
        let Some((kernels, gmem_mapping)) =
            crate::codegen::codegen(graph.clone(), arch.clone(), dyn_vars)
        else {
            continue;
        };
        valids += 1;
        let k = print_kernels(&kernels);
        if seen.contains(&k) {
            continue;
        } else {
            seen.insert(k.clone());
        }
        if let Some((us, outs)) = cost(
            &kernels,
            &map_inputs_into_kernel_graph(&inputs, &graph),
            &gmem_mapping,
            dyn_vars,
        ) {
            valid_graphs += 1;
            if let Some((progress, logs, title, _)) = &ui_functions {
                progress(((n as f32 / total_trajectories as f32) * 100.0) as u16);
                logs(k);
                title(format!(
                    "Graph {valid_graphs} Best {best_time}µs Current {us}µs"
                ));
            } else if option_env!("DEBUG").is_some() {
                println!("{k}");
                println!("Graph {valid_graphs} Best {best_time}µs Current {us}µs");
                for (a, b) in ref_outputs.iter().zip(&outs) {
                    for (x, y) in a.iter().zip(b) {
                        if (x - y).abs() >= 1e-4 {
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
                                    &outs.iter().map(|v| &v[..v.len().min(20)]).collect_vec()
                                );
                                println!("{}", print_kernels(&ref_kernels));
                                println!("{}", print_kernels(&kernels));
                                crate::debug::display_multiple_graphs(&[&ref_graph, &graph]);
                                panic!("{} {x} != {y}", "Output Mismatch".bold().on_bright_red());
                            }
                            continue 'trajectory_loop;
                        }
                    }
                }
                println!("{}", "Outputs Validated".bold().on_bright_green());
            }
            let kernel_string = print_kernels(&kernels);
            // let us = kernels.node_count() as u128;
            if us < best_time {
                best_time = us;
                best_graph = Some(graph);
                fastest = kernel_string;
            }
        }
    }
    if let Some((_, _, _, exit)) = &ui_functions {
        exit();
    }
    println!("FASTEST ({}ms): {fastest}", best_time / 1000);
    println!("Valids: {valids} / {total_trajectories}");
    best_graph
}

pub fn extraction_to_graph(
    egraph: &EGraph,
    trajectory: &[&NodeId],
    loop_level_map: &FxHashMap<&NodeId, i32>,
) -> StableGraph<GraphTerm, (), Directed> {
    let mut g = StableGraph::new();

    fn build_expression<'a>(
        egraph: &EGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
    ) -> Expression {
        let nid = trajectory[*current];
        let enode = &egraph.nodes[nid];
        let op = enode.op.as_str();

        match op {
            // unary math
            "MNeg" | "MRecip" => {
                *current += 1;
                let c0 = build_expression(egraph, trajectory, current);
                match op {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }

            // binary math
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" => {
                *current += 1;
                let lhs = build_expression(egraph, trajectory, current);
                *current += 1;
                let rhs = build_expression(egraph, trajectory, current);
                match op {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MGte" => lhs.gte(rhs),
                    "MLt" => lhs.lt(rhs),
                    "MFloorTo" => lhs / rhs * rhs, // TODO: real floorto in Expression
                    _ => unreachable!(),
                }
            }

            // wrappers around a literal/var child
            "MNum" | "MVar" => {
                *current += 1;
                build_expression(egraph, trajectory, current)
            }

            // “accumulator” token
            "MAccum" => {
                *current += 1;
                Expression::from(Term::Acc('a'))
            }
            op if op.starts_with("\"") => {
                let name = op.replace("\"", "");
                Expression::from(name.chars().next().unwrap())
            }
            op => op
                .parse::<usize>()
                .map(|i| i.into())
                .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
        }
    }

    // --- IR builder: places nodes in `g` and returns NodeIndex ---
    fn build_ir<'a, 'b>(
        egraph: &EGraph,
        trajectory: &[&'a NodeId],
        current: &mut usize,
        g: &mut StableGraph<GraphTerm, (), Directed>,
        loop_level_map: &FxHashMap<&'a NodeId, i32>,
        prev_placed: &'b mut FxHashMap<&'a NodeId, NodeIndex>,
        no_place: bool,
    ) -> NodeIndex {
        let node_choice = trajectory[*current];
        let enode = &egraph.nodes[node_choice];
        let op = enode.op.as_str();

        match op {
            // Leaf-ish memory node
            "GMEM" => {
                *current += 1;
                if no_place {
                    NodeIndex::default()
                } else {
                    *prev_placed.entry(node_choice).or_insert_with(|| {
                        let label = egraph.nodes[&enode.children[0]].op.replace("\"", "");
                        g.add_node(GraphTerm::GMEM { label })
                    })
                }
            }

            // LoopIn/LoopOut = (Loop* <expr> <Math> <Math>)
            "LoopIn" | "LoopOut" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                // child expr
                let child_one = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                // range, stride (expressions)
                *current += 1;
                let range = build_expression(egraph, trajectory, current);
                *current += 1;
                let stride = build_expression(egraph, trajectory, current);

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let term = match op {
                        "LoopIn" => GraphTerm::LoopIn { range, stride },
                        "LoopOut" => GraphTerm::LoopOut { range, stride },
                        _ => unreachable!(),
                    };
                    let r = g.add_node(term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    r
                }
            }

            // TCMatmul = (TCMatmul <A> <B> <a_k> <b_k> <a_in> <b_in> <c_in> <k_outer>)
            "TCMatmul" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);

                let src_a = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );
                *current += 1;
                let src_b = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                *current += 1;
                let a_k_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let b_k_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let a_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let b_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let c_inner_stride = build_expression(egraph, trajectory, current);
                *current += 1;
                let k_outer_loops = build_expression(egraph, trajectory, current);

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
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
                    r
                }
            }

            // Binary = (Binary <Op> <Expr> <Expr>)
            "Binary" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                let op_term = match egraph.nodes[trajectory[*current]].op.as_str() {
                    "Add" => GraphTerm::Add,
                    "Mul" => GraphTerm::Mul,
                    "Max" => GraphTerm::Max,
                    other => panic!("unknown binary IR operator: {other}"),
                };
                *current += 1;

                let child_one = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );
                *current += 1;
                let child_two = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let r = g.add_node(op_term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child_one, r, ());
                    g.add_edge(child_two, r, ());
                    r
                }
            }

            // Unary = (Unary <Op> <Expr>)
            "Unary" => {
                *current += 1;
                let already = prev_placed.contains_key(node_choice);
                let op_term = match egraph.nodes[trajectory[*current]].op.as_str() {
                    "Exp2" => GraphTerm::Exp2,
                    "Log2" => GraphTerm::Log2,
                    "Sin" => GraphTerm::Sin,
                    "Recip" => GraphTerm::Recip,
                    "Neg" => GraphTerm::Neg,
                    "Sqrt" => GraphTerm::Sqrt,
                    other => panic!("unknown unary IR operator: {other}"),
                };
                *current += 1;

                let child = build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    already || no_place,
                );

                if no_place {
                    NodeIndex::default()
                } else if let Some(&n) = prev_placed.get(node_choice) {
                    n
                } else {
                    let r = g.add_node(op_term);
                    prev_placed.insert(node_choice, r);
                    g.add_edge(child, r, ());
                    r
                }
            }
            "Fused" => {
                *current += 1;
                build_ir(
                    egraph,
                    trajectory,
                    current,
                    g,
                    loop_level_map,
                    prev_placed,
                    no_place,
                )
            }
            op => panic!("unsupported IR op: {op}"),
        }
    }

    build_ir(
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
            panic!("free-standing node found in graph");
        }
    }
    g
}

fn map_inputs_into_kernel_graph<'a>(
    inputs: &'a [(String, Buffer)],
    graph: &StableGraph<GraphTerm, ()>,
) -> Vec<(NodeIndex, &'a Buffer)> {
    inputs
    	.iter()
     	.filter_map(|(l, d)|
    		graph.node_indices().find(|n| matches!(graph.node_weight(*n).unwrap(), GraphTerm::GMEM { label } if label == l)).map(|i| (i, d))
      	)
      	.collect()
}

fn cost<'a>(
    kernels: &StableGraph<Kernel, (usize, usize), Directed>,
    inputs: &[(NodeIndex, &Buffer)],
    gmem_mapping: &HashMap<NodeIndex, usize>,
    dyn_vars: &FxHashMap<char, usize>,
) -> Option<(Cost, Vec<Vec<f32>>)> {
    // Get buffer info
    let (int_buffers, int_buffer_map) = assign_buffers(&kernels);
    for i in &int_buffers {
        if i.to_usize().unwrap_or_default() * size_of::<f32>() > 17_179_869_184 {
            // Can't allocate a buffer larger than 16GB
            return None;
        }
    }
    let compiled_kernels = compile_kernels(&kernels);
    // Set up input buffers
    let mut inputs = inputs
        .into_iter()
        .filter(|(n, _)| gmem_mapping.contains_key(n))
        .map(|(n, b)| (gmem_mapping[n], (*b, false)))
        .collect::<FxHashMap<_, _>>();
    // Allocate intermediate buffers
    let mut buffers = int_buffers
        .iter()
        .map(|e| new_buffer(e.exec(dyn_vars).unwrap() * size_of::<f32>()))
        .collect_vec();

    // Warm up resources (buffer allocation, kernel compiler, etc.)
    for _ in 0..WARMUP_TRIALS {
        run_graph(
            &mut inputs,
            &kernels,
            dyn_vars,
            &compiled_kernels,
            &mut buffers,
            &int_buffer_map,
        );
    }
    // Test runtime
    let mut micros = vec![];
    let mut outputs = vec![];

    for _ in 0..TRIALS {
        let (o, m_val) = {
            crate::run::run_graph(
                &mut inputs,
                &kernels,
                dyn_vars,
                &compiled_kernels,
                &mut buffers,
                &int_buffer_map,
            )
        };
        outputs = o;
        micros.push(m_val);
    }
    Some((micros.into_iter().sum::<u128>() / TRIALS as u128, outputs))
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
