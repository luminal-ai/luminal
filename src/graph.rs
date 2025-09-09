use crate::prelude::*;
use std::{
    io::Write,
    ops::{Deref, DerefMut},
    time::Duration,
};

use super::compiler_utils::{ToIds, ToIdsMut};
use colored::Colorize;
use itertools::Itertools;
use petgraph::{stable_graph::StableGraph, visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

pub type StorageGraph = StableGraph<Box<dyn Operator>, Dependency>;

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
/// All data is stored inside this object as well.
#[derive(Debug, Default)]
pub struct Graph {
    /// The store of tensors in the graph. Indexed by node index and output index.
    pub tensors: FxHashMap<(NodeIndex, u8), Tensor>,
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: StorageGraph,
    /// Tensors marked in this set will not get deleted when the graph is ran
    pub no_delete: FxHashSet<NodeIndex>,
    /// Tensors marked in this set need to be retrieved later (mostly for optimizers to insert copy back calls, the graph itself doesn't treat these differently)
    pub to_retrieve: FxHashMap<NodeIndex, (u8, ShapeTracker)>,
    /// A cached list of nodes to run, source nodes, and view nodes to delete after execution.
    #[allow(clippy::type_complexity)]
    pub(crate) linearized_graph: Option<Vec<(NodeIndex, Vec<(NodeIndex, u8, ShapeTracker)>)>>,
    /// Cached consumers (for execution only)
    consumers_map: Option<FxHashMap<(NodeIndex, u8), usize>>,
}

/// A dependency between two nodes
#[derive(Debug, Clone, Copy)]
#[allow(clippy::large_enum_variant)]
pub enum Dependency {
    /// A data dependency (transferring a tensor from one node to the next)
    Data {
        input_order: u8,
        output_order: u8,
        shape: ShapeTracker,
    },
    /// Explicit dependency for ordering. No tensors are transferred through this dependency
    Schedule,
}

impl Dependency {
    /// Try to extract dependency data
    pub fn as_data(self) -> Option<(u8, u8, ShapeTracker)> {
        if let Self::Data {
            input_order,
            output_order,
            shape,
        } = self
        {
            Some((input_order, output_order, shape))
        } else {
            None
        }
    }

    /// Is this a schedule dependency?
    pub fn is_schedule(&self) -> bool {
        matches!(self, Self::Schedule)
    }
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Try to remove the tensor data from the graph
    pub fn get_tensor(&mut self, id: NodeIndex, ind: u8) -> Option<Tensor> {
        self.tensors.remove(&(id, ind))
    }

    /// Try to get the tensor data in the graph
    pub fn get_tensor_ref(&self, id: NodeIndex, ind: u8) -> Option<&Tensor> {
        self.tensors.get(&(id, ind))
    }

    /// Delete the tensor data from the graph
    pub fn drop_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.tensors.remove(&(id, 0));
        }
    }

    /// Mark tensors to be kept
    pub fn keep_tensors<T: ToIds>(&mut self, tensors: T) {
        for id in tensors.to_ids() {
            self.no_delete.insert(id);
        }
    }

    /// Set a tensor's data
    pub fn set_tensor(&mut self, id: NodeIndex, ind: u8, tensor: Tensor) {
        self.tensors.insert((id, ind), tensor);
    }

    /// Set a dynamic dimension
    pub fn set_dyn_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("Tensor", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: &str, shape: impl ToShape) -> GraphTensor {
        let name = name.to_string();
        GraphTensor {
            id: self.graph.add_node(Box::new(Function(
                format!("{name} Load"),
                Box::new(move |_| panic!("You must set a value for this tensor! ({name})")),
            ))),
            graph_ref: self,
            shape: ShapeTracker::new(shape),
        }
    }

    /// Compile the graph using the given compiler
    pub fn compile<T: ToIdsMut, C: Compiler>(&mut self, compiler: C, remap: T) -> C::Output {
        let output = compiler.compile(self, remap);
        self.toposort();
        self.reset();
        output
    }

    /// Refresh the internally sorted graph
    pub(crate) fn toposort(&mut self) {
        self.linearized_graph = Some(
            petgraph::algo::toposort(&self.graph, None)
                .unwrap()
                .into_iter()
                .map(|node| (node, self.get_sources(node)))
                .collect(),
        );

        // Refresh the internal remaining consumers map
        self.consumers_map = Some(
            self.graph
                .node_indices()
                .flat_map(|i| {
                    self.graph
                        .edges_directed(i, Direction::Outgoing)
                        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                        .group_by(|(_, (_, i, _))| *i)
                        .into_iter()
                        .map(|(ind, g)| ((i, ind), g.count()))
                        .collect::<Vec<_>>()
                })
                .collect(),
        );
    }

    /// Swap the tensors with these ids
    pub fn swap_tensors(&mut self, a: GraphTensor, b: GraphTensor) {
        // Swap tensors
        for i in 0.. {
            let a_t = self.tensors.remove(&(a.id, i));
            let b_t = self.tensors.remove(&(b.id, i));
            if a_t.is_none() && b_t.is_none() {
                break;
            }
            if let Some(a_t) = a_t {
                self.tensors.insert((b.id, i), a_t);
            }
            if let Some(b_t) = b_t {
                self.tensors.insert((a.id, i), b_t);
            }
        }
    }

    /// Clear any remaining tensors that may be around from old executions
    pub fn reset(&mut self) {
        self.tensors.retain(|(n, _), _| self.no_delete.contains(n));
    }

    /// Execute the graph.
    pub fn execute(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut dim_stack = Vec::new();

        for (node, src_ids) in self.linearized_graph.as_ref().unwrap() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // Execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }

            // Bookkeep remaining consumers
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }
        self.reset();
    }

    /// Execute the graph without deleting intermediate tensors
    pub fn execute_no_delete(&mut self) {
        // Track the number of views pointing to each tensor so we know when to clear;
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let mut srcs = src_ids
                .iter()
                .map(|(id, ind, st)| {
                    (
                        InputTensor::Borrowed(self.tensors.get(&(*id, *ind)).unwrap()),
                        *st,
                    )
                })
                .collect_vec();

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready, execute
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
        }
    }

    /// Execute the graph with debug prints
    pub fn execute_debug(&mut self) {
        fn format_duration(duration: &Duration) -> String {
            if duration.as_secs() > 0 {
                format!("{:.2}s", duration.as_secs_f32())
            } else if duration.as_millis() > 0 {
                format!("{}ms", duration.as_millis())
            } else {
                format!("{}µs", duration.as_micros())
            }
        }
        // Track the number of views pointing to each tensor so we know when to clear
        if self.linearized_graph.is_none() {
            self.toposort();
        }
        let mut dim_stack = Vec::new();
        let mut consumers = self.consumers_map.as_ref().unwrap().clone();
        let mut op_times = FxHashMap::default();
        let width = term_size::dimensions().unwrap().0;

        println!(
            "{:->2$} Executing {:->2$}",
            "",
            "",
            (width.saturating_sub(" Executing ".len())) / 2
        );
        let start = std::time::Instant::now();
        for (node, src_ids) in self.linearized_graph.as_ref().unwrap().iter() {
            if self.tensors.contains_key(&(*node, 0)) {
                continue;
            }
            let op_name = format!("{:?} | {}", self.node_weight(*node).unwrap(), node.index());
            print!("{}", op_name.bold().bright_green());

            let mut srcs =
                get_source_tensors(&self.no_delete, &mut self.tensors, src_ids, &consumers);

            // Substitute in the dyn dims
            for (_, st) in srcs.iter_mut() {
                st.resolve_global_dyn_dims_stack(&self.dyn_map, &mut dim_stack);
            }

            // All sources are ready
            let mut shapes_string = srcs
                .iter()
                .map(|(_, s)| {
                    format!(
                        "{:?}",
                        s.dims()
                            .into_iter()
                            .map(|i| i.to_usize().unwrap())
                            .collect::<Vec<_>>()
                    )
                })
                .join(", ");
            if !shapes_string.is_empty() {
                shapes_string = format!(" ({shapes_string})");
            }
            print!("{shapes_string}");
            std::io::stdout().flush().unwrap();
            // Execute
            let now = std::time::Instant::now();
            let tensors = self.graph.node_weight_mut(*node).unwrap().process(srcs);
            let elapsed = now.elapsed();
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width
                    .saturating_sub(op_name.len())
                    .saturating_sub(shapes_string.len()),
            );
            for (i, tensor) in tensors.into_iter().enumerate() {
                self.tensors.insert((*node, i as u8), tensor);
            }
            let timed_op_name = format!("{:?}", self.node_weight(*node).unwrap());
            if let Some(t) = op_times.get_mut(&timed_op_name) {
                *t += elapsed;
            } else {
                op_times.insert(timed_op_name, elapsed);
            }

            // Check if we can delete the source tensors now
            for (id, ind, _) in src_ids {
                *consumers.get_mut(&(*id, *ind)).unwrap() -= 1;
            }
        }

        // Print out total times
        println!();
        println!(
            "{:->2$} Total Times {:->2$}",
            "",
            "",
            (width.saturating_sub(" Total Times ".len())) / 2
        );
        for (name, elapsed) in op_times.into_iter().sorted_by(|(_, a), (_, b)| b.cmp(a)) {
            print!("{}", name.bold().bright_green());
            println!(
                "{:.>1$}",
                format_duration(&elapsed).bold(),
                width.saturating_sub(name.len()),
            );
        }
        println!("Total: {}", format_duration(&start.elapsed()).bold());
        self.reset();
    }
}

impl Deref for Graph {
    type Target = StorageGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

/// Get source tensor array for a node
fn get_source_tensors<'a>(
    no_delete: &'a FxHashSet<NodeIndex>,
    tensors: *mut FxHashMap<(NodeIndex, u8), Tensor>,
    src_ids: &'a [(NodeIndex, u8, ShapeTracker)],
    consumers: &'a FxHashMap<(NodeIndex, u8), usize>,
) -> Vec<(InputTensor<'a>, ShapeTracker)> {
    let mut srcs = vec![];
    for (id, ind, sh) in src_ids {
        let id = &(*id, *ind);
        if consumers[id] == 1 && !no_delete.contains(&id.0) {
            srcs.push((
                InputTensor::Owned(unsafe { tensors.as_mut().unwrap() }.remove(id).unwrap()),
                *sh,
            ));
        } else {
            srcs.push((
                InputTensor::Borrowed(unsafe { tensors.as_ref().unwrap() }.get(id).unwrap()),
                *sh,
            ));
        }
    }
    srcs
}
