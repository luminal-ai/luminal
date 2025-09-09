// Lots of utilities used by compilers

use std::{any::TypeId, borrow::Borrow, collections::HashSet, fmt::Debug, sync::Arc};

use colored::Colorize;
use itertools::Itertools;
use petgraph::{
    algo::toposort,
    stable_graph::{EdgeIndex, EdgeReference, StableGraph},
    visit::EdgeRef,
    Direction,
};
use regex::Regex;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::prelude::*;

pub trait ToIdsMut {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex>;
}

pub trait ToIds {
    fn to_ids(&self) -> Vec<NodeIndex>;
}

pub trait ToId {
    fn to_id(&self) -> NodeIndex;
}

impl ToId for GraphTensor {
    fn to_id(&self) -> NodeIndex {
        self.id
    }
}

impl ToId for NodeIndex {
    fn to_id(&self) -> NodeIndex {
        *self
    }
}

impl ToIdsMut for GraphTensor {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.id]
    }
}
impl ToIds for GraphTensor {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![self.id]
    }
}
impl<T: ToIdsMut> ToIdsMut for Vec<T> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl<T: ToIds> ToIds for Vec<T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}
impl<T: ToIdsMut> ToIdsMut for &mut [T] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().flat_map(|i| i.to_ids_mut()).collect()
    }
}
impl ToIdsMut for &mut Vec<NodeIndex> {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl ToIdsMut for &mut [NodeIndex] {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        self.iter_mut().collect()
    }
}
impl<T: ToIds> ToIds for &mut [T] {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.iter().flat_map(|i| i.to_ids()).collect()
    }
}

impl<T: ToIdsMut> ToIdsMut for &mut T {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        (*self).to_ids_mut()
    }
}
impl<T: ToIds> ToIds for &T {
    fn to_ids(&self) -> Vec<NodeIndex> {
        <T as ToIds>::to_ids(*self)
    }
}
impl ToIds for NodeIndex {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![*self]
    }
}
impl ToIdsMut for &mut NodeIndex {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![self]
    }
}
impl ToIdsMut for () {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![]
    }
}
impl ToIds for () {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![]
    }
}

impl<T: ToIds> ToIds for FxHashMap<String, T> {
    fn to_ids(&self) -> Vec<NodeIndex> {
        self.values().flat_map(|i| i.to_ids()).collect()
    }
}

impl ToIds for (NodeIndex, ShapeTracker) {
    fn to_ids(&self) -> Vec<NodeIndex> {
        vec![self.0]
    }
}

impl ToIdsMut for (NodeIndex, ShapeTracker) {
    fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
        vec![&mut self.0]
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            ToIdsMut, )+
        > ToIdsMut for ($($name,)+) {
            fn to_ids_mut(&mut self) -> Vec<&mut NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids_mut());)+
                v
            }
        }
        impl<
        $($name:
            ToIds, )+
        > ToIds for ($($name,)+) {
            fn to_ids(&self) -> Vec<NodeIndex> {
                let mut v = vec![];
                $(v.append(&mut self.$idx.to_ids());)+
                v
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);

pub trait Compiler {
    type Output;
    /// Run a compilation pass
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, ids: T) -> Self::Output;
}

impl Compiler for () {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, _: &mut Graph, _: T) {}
}

/// Wrap this around a compiler to rerun the compiler until it doesn't change the graph anymore
#[derive(Debug)]
pub struct Looped<C: Compiler + Debug>(C);

impl<C: Compiler + Debug> Compiler for Looped<C> {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) {
        let mut linearized = None;
        loop {
            self.0.compile(graph, &mut remap);
            graph.toposort();
            if linearized == graph.linearized_graph {
                break;
            }
            linearized.clone_from(&graph.linearized_graph);
        }
    }
}

impl<C: Default + Compiler + Debug> Default for Looped<C> {
    fn default() -> Self {
        Self(C::default())
    }
}

/// Wrap this around a compiler to measure the time it takes to compile
#[derive(Debug)]
pub struct Timed<C: Compiler + Debug>(pub C);

impl<C: Compiler + Debug> Compiler for Timed<C> {
    type Output = ();
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, remap: T) {
        let compiler_name = format!("{:?}", self.0).bold();
        println!("Starting {compiler_name}");
        let start = std::time::Instant::now();
        self.0.compile(graph, remap);
        let finished_millis = start.elapsed().as_millis();
        let minutes = finished_millis / 60_000;
        let seconds = (finished_millis % 60_000) / 1000;
        let millis = finished_millis % 1000;
        println!(
            "Finished {compiler_name} in {}",
            if minutes > 0 {
                format!("{minutes}m {seconds}s {millis}ms")
            } else if seconds > 0 {
                format!("{seconds}s {millis}ms")
            } else {
                format!("{millis}ms")
            }
            .bold()
        );
    }
}

impl<C: Default + Compiler + Debug> Default for Timed<C> {
    fn default() -> Self {
        Self(C::default())
    }
}

macro_rules! tuple_impls {
    ([$($name:ident),+] , [$($idx:tt),+]) => {
        impl<
        $($name:
            Compiler, )+
        > Compiler for ($($name,)+) {
            type Output = ( $($name::Output, )+ );
            fn compile<T: ToIdsMut>(&self, graph: &mut Graph, mut remap: T) -> Self::Output {
                ( $(self.$idx.compile(graph, &mut remap), )+ )
            }
        }
    };
}

tuple_impls!([M1], [0]);
tuple_impls!([M1, M2], [0, 1]);
tuple_impls!([M1, M2, M3], [0, 1, 2]);
tuple_impls!([M1, M2, M3, M4], [0, 1, 2, 3]);
tuple_impls!([M1, M2, M3, M4, M5], [0, 1, 2, 3, 4]);
tuple_impls!([M1, M2, M3, M4, M5, M6], [0, 1, 2, 3, 4, 5]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7], [0, 1, 2, 3, 4, 5, 6]);
tuple_impls!([M1, M2, M3, M4, M5, M6, M7, M8], [0, 1, 2, 3, 4, 5, 6, 7]);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9],
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
);
tuple_impls!(
    [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15, M16, M17, M18, M19, M20],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
);

// Helpers

impl Graph {
    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// use luminal::prelude::*;
    /// let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx
    ///     .add_op(luminal::op::Mul)
    ///     .input(a.id, 0, a.shape)
    ///     .finish();
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph());
    /// ```
    pub fn add_op<O: Operator + 'static>(&mut self, op: O) -> NewOp<'_> {
        self.linearized_graph = None;
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Add op on the graph, and get back a NewOp. Just like add_op, except a boxed op is expected.
    pub fn add_boxed_op(&mut self, op: Box<dyn Operator + 'static>) -> NewOp<'_> {
        self.linearized_graph = None;
        NewOp {
            new_op_id: self.graph.add_node(op),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Create a schedule dependency between a and b
    pub fn add_schedule_dependency(&mut self, a: NodeIndex, b: NodeIndex) {
        self.graph.add_edge(a, b, Dependency::Schedule);
    }

    /// Run the custom function on a node and get an output
    pub fn node_custom<O: 'static, I: 'static>(
        &mut self,
        node: NodeIndex,
        key: &str,
        input: I,
    ) -> Option<O> {
        let node_weight = self.graph.node_weight_mut(node)?;

        node_weight
            .custom(key, Box::new(input))
            .and_then(|o| o.downcast::<O>().ok().map(|o| *o))
    }

    /// Convert to debug-viewable graph
    fn debug_graph(
        &self,
        show_shapes: bool,
    ) -> (
        StableGraph<String, u8>,
        Vec<EdgeIndex>,
        FxHashMap<NodeIndex, NodeIndex>,
    ) {
        let mut new_graph = StableGraph::default();
        let mut id_map = FxHashMap::default();
        for (id, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            id_map.insert(id, new_graph.add_node(format!("{node:?} | {}", id.index())));
        }

        let mut schedule_edges = vec![];
        for node in self.graph.node_indices() {
            for edge in self
                .graph
                .edges_directed(node, Direction::Outgoing)
                .sorted_by_key(|e| {
                    if let Some(d) = e.weight().as_data() {
                        d.0
                    } else {
                        0
                    }
                })
            {
                let new_edge = new_graph.add_edge(
                    id_map[&edge.source()],
                    id_map[&edge.target()],
                    if let Some(d) = edge.weight().as_data() {
                        d.0
                    } else {
                        0
                    },
                );
                if edge.weight().is_schedule() {
                    schedule_edges.push(new_edge);
                }
                if show_shapes
                    && new_graph.contains_node(id_map[&edge.target()])
                    && edge
                        .weight()
                        .as_data()
                        .map(|d| !d.2.is_empty())
                        .unwrap_or_default()
                {
                    new_graph
                        .node_weight_mut(id_map[&edge.target()])
                        .unwrap()
                        .push_str(&format!(
                            " | {:?}",
                            edge.weight().as_data().unwrap().2.dims()
                        ));
                }
            }
        }

        (new_graph, schedule_edges, id_map)
    }

    pub fn check_node_type<T: Operator + 'static>(&self, node: NodeIndex) -> bool {
        self.node_weight(node)
            .expect("Node not found in graph!")
            .as_any()
            .is::<T>()
    }

    pub fn display(&self) {
        let (g, e, _) = self.debug_graph(false);
        display_graph(&g, &e, &[]);
    }

    pub fn display_shapes(&self) {
        let (g, e, _) = self.debug_graph(true);
        display_graph(&g, &e, &[]);
    }

    pub fn display_set<T: ToIds>(&self, set: T) {
        let (g, e, id_map) = self.debug_graph(false);
        display_graph(
            &g,
            &e,
            &set.to_ids().iter().map(|i| id_map[i]).collect::<Vec<_>>(),
        );
    }

    /// Remove node if it only has n dests
    pub fn safe_remove_node(&mut self, node: NodeIndex, dests: usize) {
        if self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .filter(|e| !e.weight().is_schedule())
            .count()
            <= dests
        {
            self.graph.remove_node(node);
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, u8, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, (_, c, b))| (a, c, b))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn Operator>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
            .sorted_by_key(|(_, (i, _, _))| *i)
            .map(|(a, _)| (a, self.graph.node_weight(a).unwrap()))
            .collect()
    }

    pub fn try_get_op<T: Operator + 'static>(&self, node: NodeIndex) -> Option<&T> {
        self.node_weight(node).unwrap().as_any().downcast_ref::<T>()
    }
    pub fn get_op<T: Operator + 'static>(&self, node: NodeIndex) -> &T {
        self.try_get_op(node).unwrap()
    }
    pub fn try_get_op_mut<T: Operator + 'static>(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.node_weight_mut(node)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<T>()
    }
    pub fn get_op_mut<T: Operator + 'static>(&mut self, node: NodeIndex) -> &mut T {
        self.try_get_op_mut(node).unwrap()
    }
}

/// View a debug graph in the browser
pub fn display_graph(
    graph: &petgraph::stable_graph::StableGraph<String, u8, petgraph::Directed, u32>,
    schedule_edges: &[EdgeIndex],
    mark_nodes: &[NodeIndex],
) {
    let mut graph_string =
        petgraph::dot::Dot::with_config(&graph, &[petgraph::dot::Config::EdgeIndexLabel])
            .to_string();
    let re = Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    for e in schedule_edges {
        graph_string =
            graph_string.replace(&format!("label = \"{}\"", e.index()), "color=\"green\"");
    }
    graph_string = re.replace_all(&graph_string, "").to_string();
    for n in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"yellow\" label =",
                n.index()
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {e:?}");
    }
}

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl NewOp<'_> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(mut self, id: NodeIndex, from_output: u8, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(
            id,
            self.new_op_id,
            Dependency::Data {
                input_order: self.num_srcs,
                output_order: from_output,
                shape,
            },
        );
        self.num_srcs += 1;
        self
    }
}

/// Transfer all external references from one node to another (this may happen because one node is about to be removed / merged into another)
pub fn remap<T: ToIdsMut>(from: NodeIndex, to: NodeIndex, mut ids: T, graph: &mut Graph) {
    for id in ids.to_ids_mut() {
        if *id == from {
            *id = to;
        }
    }
    // Transfer no_delete
    if graph.no_delete.remove(&from) {
        graph.no_delete.insert(to);
    }
    // Transfer to_retrieve
    if let Some(w) = graph.to_retrieve.remove(&from) {
        graph.to_retrieve.insert(to, w);
    }
}

pub fn move_outgoing_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over outgoing edges from node to other_node
    for (weight, target) in graph
        .edges_directed(from, petgraph::Direction::Outgoing)
        .map(|e| (e.weight().clone(), e.target()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(to, target, weight);
    }
}

pub fn move_incoming_edge<N, E: Clone>(
    from: NodeIndex,
    to: NodeIndex,
    graph: &mut StableGraph<N, E>,
) {
    // Carry over incoming edges from node to other_node
    for (weight, source) in graph
        .edges_directed(from, petgraph::Direction::Incoming)
        .map(|e| (e.weight().clone(), e.source()))
        .collect::<Vec<_>>()
    {
        graph.add_edge(source, to, weight);
    }
}

pub struct GraphSearch {
    current: FxHashMap<Uuid, NodeIndex>,
    selector: StableGraph<(Uuid, SelectOp), Option<u8>>,
    graph: *mut Graph,
    to_return: Vec<FxHashMap<NodeIndex, NodeIndex>>,
    returned_anchors: HashSet<NodeIndex>,
    anchor: NodeIndex,
    pub matched: bool,
}

impl GraphSearch {
    pub fn next_match(&mut self) -> bool {
        // Look through graph for pattern from selector
        let graph = unsafe { self.graph.as_mut().unwrap() };

        if self.to_return.is_empty() {
            // Replenish to_return
            let (_, select_op) = self.selector.node_weight(self.anchor).unwrap();
            for node in graph.graph.node_indices().collect::<Vec<_>>() {
                if !self.returned_anchors.contains(&node)
                    && test_node(select_op, &mut graph.graph, node)
                {
                    // Backtrack to check if this is a match
                    if let Some(mapping) =
                        backtrack_match(self.anchor, &self.selector, node, &mut graph.graph)
                    {
                        self.to_return.push(mapping);
                    }
                }
            }
        }
        if let Some(mapping) = self.to_return.pop() {
            self.returned_anchors.insert(mapping[&self.anchor]);
            self.current = mapping
                .into_iter()
                .map(|(k, v)| (self.selector.node_weight(k).unwrap().0, v))
                .collect();
            self.matched = true;
            return true;
        }
        self.matched = false;
        false
    }

    pub fn clear_cached_results(&mut self) {
        self.to_return.clear();
    }
    pub fn reset(&mut self) {
        self.clear_cached_results();
        self.returned_anchors.clear();
    }
    pub fn check_no_delete(&self, exclude: &[Uuid]) -> bool {
        let graph = unsafe { self.graph.as_ref().unwrap() };
        self.current
            .iter()
            .filter(|(k, _)| !exclude.contains(k))
            .any(|(_, v)| graph.no_delete.contains(v))
    }
    pub fn get<T: Borrow<SelectGraph>>(&self, node: T) -> NodeIndex {
        *self.current.get(&node.borrow().id).unwrap()
    }
    pub fn try_delete(&self) {
        let graph = unsafe { self.graph.as_mut().unwrap() };
        for node in toposort(&self.selector, None).unwrap().into_iter().rev() {
            let id = self.selector.node_weight(node).unwrap().0;
            let node = self.current[&id];
            graph.safe_remove_node(node, 0);
        }
    }
}

fn backtrack_match(
    pattern_root: NodeIndex,
    pattern_graph: &StableGraph<(Uuid, SelectOp), Option<u8>>,
    main_root: NodeIndex,
    main_graph: &mut StorageGraph,
) -> Option<FxHashMap<NodeIndex, NodeIndex>> {
    fn get_parents<N, E>(
        graph: &petgraph::stable_graph::StableGraph<N, E>,
        node_index: NodeIndex,
        edge_filter: fn(&EdgeReference<'_, E>) -> bool,
    ) -> Vec<NodeIndex> {
        graph
            .edges_directed(node_index, Direction::Incoming)
            .filter(edge_filter)
            .map(|e| e.source())
            .collect()
    }

    if !test_node(
        &pattern_graph.node_weight(pattern_root).unwrap().1,
        main_graph,
        main_root,
    ) {
        return None;
    }

    let mut mapping = FxHashMap::default();
    mapping.insert(pattern_root, main_root);
    let main_parents = get_parents(main_graph, main_root, |e| !e.weight().is_schedule());
    'pattern_loop: for pattern_parent in get_parents(pattern_graph, pattern_root, |_| true) {
        for parent in main_parents.iter() {
            if mapping.values().any(|&v| v == *parent) {
                // This main node was used already, skip it
                continue;
            }
            if let Some(new_mapping) =
                backtrack_match(pattern_parent, pattern_graph, *parent, main_graph)
            {
                mapping.extend(new_mapping.into_iter());
                continue 'pattern_loop;
            }
        }
        return None;
    }
    Some(mapping)
}

fn test_node(
    SelectOp {
        type_id,
        check,
        shape,
        fake,
    }: &SelectOp,
    graph: &mut StorageGraph,
    graph_node: NodeIndex,
) -> bool {
    let input_shapes = graph
        .edges_directed(graph_node, petgraph::Direction::Incoming)
        .filter_map(|e| e.weight().as_data())
        .sorted_by_key(|e| e.0)
        .map(|e| e.2)
        .collect::<Vec<_>>();
    let current_weight = graph.node_weight_mut(graph_node).unwrap();
    // Test type
    if let Some(ty) = type_id {
        if current_weight.as_any().type_id() != *ty {
            return false;
        }
    }

    // Test shape
    if let Some(shape) = shape {
        let mut shape_map = FxHashMap::default();
        if shape.len() != input_shapes.len() {
            return false;
        }
        for (a_sh, b_sh) in shape.iter().zip(input_shapes.iter()) {
            if a_sh.len() != b_sh.dims.len() {
                return false;
            }
            for (a, b) in a_sh.iter().zip(b_sh.dims().into_iter()) {
                match a.to_usize() {
                    Some(n) => {
                        if b.to_usize().map(|i| i != n).unwrap_or(true) {
                            return false;
                        }
                    }
                    None => {
                        let c = a
                            .to_symbols()
                            .pop()
                            .expect("Selector dimension must be either a symbol or number");
                        if let Some(expected) = shape_map.get(&c) {
                            if b != *expected {
                                return false;
                            }
                        } else {
                            shape_map.insert(c, b);
                        }
                    }
                }
            }
        }
    }
    // Test fakes
    if let Some(fakes) = fake {
        for (a_sh, b_sh) in fakes.iter().zip(input_shapes.iter()) {
            for (a, b) in a_sh.iter().zip(b_sh.indexes.iter().map(|i| b_sh.fake[*i])) {
                if let Some(a) = a {
                    if *a != b {
                        return false;
                    }
                }
            }
        }
    }

    // Run check
    if let Some(check) = check {
        if !check(current_weight.as_mut(), &input_shapes) {
            return false;
        }
    }
    true
}

#[derive(Default, Clone)]
pub struct SelectOp {
    /// Type constraint
    pub type_id: Option<TypeId>,
    /// Check constraint
    #[allow(clippy::type_complexity)]
    pub check: Option<Arc<Box<dyn Fn(&mut dyn Operator, &[ShapeTracker]) -> bool>>>,
    /// Shape constraint
    pub shape: Option<Vec<Vec<Expression>>>,
    /// Fake constraint
    pub fake: Option<Vec<Vec<Option<bool>>>>,
}

#[macro_export]
macro_rules! select_ty {
    ($t: ty) => {
        SelectOp::new().ty::<$t>()
    };
}

impl SelectOp {
    pub fn new() -> Self {
        Self::default()
    }
}

pub fn check_no_delete(graph: &Graph, nodes: &[NodeIndex]) -> bool {
    nodes.iter().any(|n| graph.no_delete.contains(n))
}

#[derive(Clone)]
pub struct SelectGraph {
    /// The full selection graph
    graph: StableGraph<(Uuid, SelectOp), Option<u8>>,
    /// The current id for the current node in the graph
    pub id: Uuid,
    /// The index of the current node in the graph
    node: NodeIndex,
}

impl SelectGraph {
    fn new(restrictions: SelectOp) -> Self {
        let mut graph = StableGraph::new();
        let id = Uuid::new_v4();
        Self {
            node: graph.add_node((id, restrictions)),
            graph,
            id,
        }
    }

    /// Constrain the op to a type
    pub fn ty<O: Operator + 'static>(&mut self) {
        self.graph.node_weight_mut(self.node).unwrap().1.type_id = Some(TypeId::of::<O>());
    }
    /// Constrain the op to a type
    pub fn type_id(&mut self, type_id: TypeId) {
        self.graph.node_weight_mut(self.node).unwrap().1.type_id = Some(type_id);
    }
    /// Constrain the op to a checking function
    pub fn check<F: Fn(&mut dyn Operator, &[ShapeTracker]) -> bool + 'static>(&mut self, check: F) {
        self.graph.node_weight_mut(self.node).unwrap().1.check = Some(Arc::new(Box::new(check)));
    }
    /// Constrain the op to input shapes
    pub fn shapes<E: Into<Expression>, V: Into<Vec<E>>, S: Into<Vec<V>>>(&mut self, shapes: S) {
        self.graph.node_weight_mut(self.node).unwrap().1.shape = Some(
            shapes
                .into()
                .into_iter()
                .map(|i| i.into().into_iter().map(|i| i.into()).collect())
                .collect(),
        );
    }
    /// Constrain the op to input shape fakes
    pub fn fakes<V: Into<Vec<Option<bool>>>, S: Into<Vec<V>>>(&mut self, fakes: S) {
        self.graph.node_weight_mut(self.node).unwrap().1.fake =
            Some(fakes.into().into_iter().map(|i| i.into()).collect());
    }

    pub fn connect(mut self, b: Self) -> Self {
        // Add b graph to a graph
        let mut node_map = FxHashMap::default();
        let mut a_nodes = FxHashMap::default();
        for node in self.graph.node_indices() {
            let id = self.graph.node_weight(node).unwrap().0;
            a_nodes.insert(id, node);
        }
        for node in b.graph.node_indices() {
            let id = b.graph.node_weight(node).unwrap().0;
            let new_node = a_nodes.get(&id).copied().unwrap_or_else(|| {
                self.graph
                    .add_node(b.graph.node_weight(node).unwrap().clone())
            });
            node_map.insert(node, new_node);
        }
        for edge in b.graph.edge_indices() {
            let (src, trg) = b.graph.edge_endpoints(edge).unwrap();
            self.graph.add_edge(
                node_map[&src],
                node_map[&trg],
                *b.graph.edge_weight(edge).unwrap(),
            );
        }
        self.graph.add_edge(self.node, node_map[&b.node], None);
        self.node = node_map[&b.node];
        self.id = b.id;
        self
    }

    pub fn search(self, graph: &mut Graph) -> GraphSearch {
        let anchor = *toposort(&self.graph, None).unwrap().last().unwrap();
        GraphSearch {
            current: FxHashMap::default(),
            to_return: vec![],
            selector: self.graph,
            graph,
            returned_anchors: HashSet::new(),
            anchor,
            matched: false,
        }
    }
}

pub fn op<T: Operator + 'static>() -> SelectGraph {
    let mut s = SelectOp::new();
    s.type_id = Some(TypeId::of::<T>());
    SelectGraph::new(s)
}

pub fn node() -> SelectGraph {
    SelectGraph::new(SelectOp::new())
}

pub fn unary<T: Operator + 'static>(node: SelectGraph) -> SelectGraph {
    node.connect(op::<T>())
}

pub fn binary<T: Operator + 'static>(a: SelectGraph, b: SelectGraph) -> SelectGraph {
    b.connect(a.connect(op::<T>()))
}

/// Whether or not to do debug prints (env var DEBUG=1)
pub fn debug() -> bool {
    std::env::var("DEBUG")
        .unwrap_or_default()
        .parse::<i32>()
        .map(|i| i == 1)
        .unwrap_or_default()
}
