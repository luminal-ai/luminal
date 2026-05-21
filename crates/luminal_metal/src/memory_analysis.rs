#![allow(dead_code)]

use std::sync::Arc;

use itertools::Itertools;
use luminal::{
    dtype::DType,
    egglog_utils::{
        ClassId, EGraphChoiceSet, LateEgglogPass, NodeId, SerializedEGraph, api::SortDef,
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::EgglogOp,
    prelude::*,
};

const MEMORY_ANALYSIS_RULESET: &str = "metal_memory_analysis";
const MAX_EXACT_MEMORY_STATES_PER_CLASS: usize = 64;

const DTYPE_BITS: &[(&str, usize)] = &[
    ("F32", 32),
    ("F16", 16),
    ("Bf16", 16),
    ("Int", 32),
    ("Bool", 8),
    ("I4", 4),
    ("TF32", 19),
];

pub(crate) fn metal_memory_analysis_pass(
    ops: &[Arc<Box<dyn EgglogOp>>],
    max_memory_bytes: Option<usize>,
    _dyn_map: &FxHashMap<char, usize>,
) -> LateEgglogPass {
    let mut sorts = ops
        .iter()
        .map(|op| op.sort())
        .filter(|sort| sort.class == "OpKind")
        .collect_vec();
    sorts.sort_by(|a, b| a.name.cmp(&b.name));

    let mut program = String::from(
        r#"
(ruleset metal_memory_analysis)
(relation metal_output_bytes (OpKind Expression))
(relation metal_local_memory (IR Expression))

(rule ((= ?node (Input ?id ?label ?dtype)))
      ((metal_local_memory ?node (MNum 0)))
      :ruleset metal_memory_analysis
      :name "metal-memory-input")

(rule ((= ?node (Output ?inp ?id)))
      ((metal_local_memory ?node (MNum 0)))
      :ruleset metal_memory_analysis
      :name "metal-memory-output")

(rule ((= ?node (OutputJoin ?a ?b)))
      ((metal_local_memory ?node (MNum 0)))
      :ruleset metal_memory_analysis
      :name "metal-memory-output-join")

(rule ((= ?node (Op ?kind ?inputs))
       (metal_output_bytes ?kind ?bytes))
      ((metal_local_memory ?node ?bytes))
      :ruleset metal_memory_analysis
      :name "metal-memory-op-local")
"#,
    );

    for sort in &sorts {
        for rule in output_bytes_rules(sort) {
            program.push('\n');
            program.push_str(&rule);
        }
    }

    let pass = LateEgglogPass::new(
        program,
        format!("(run-schedule (saturate {MEMORY_ANALYSIS_RULESET}))"),
    );

    let _ = max_memory_bytes;
    pass
}

pub(crate) fn estimate_graph_memory_bytes<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
    dyn_map: &FxHashMap<char, usize>,
) -> Option<usize> {
    ChoiceMemoryEstimator::new(egraph, choices, dyn_map)
        .root_state()
        .map(|state| state.peak)
}

/// A memory state for one possible way to produce an IR enode.
///
/// `live` is the intermediate memory that must remain live after the enode is
/// produced. `peak` is the maximum intermediate memory needed while producing
/// it. Frontiers keep only Pareto-minimal states under the configured limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct MemoryState {
    live: usize,
    peak: usize,
}

impl MemoryState {
    fn zero() -> Self {
        Self { live: 0, peak: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ListMemoryState {
    lives: Vec<usize>,
    peak: usize,
}

impl ListMemoryState {
    fn empty() -> Self {
        Self {
            lives: Vec::new(),
            peak: 0,
        }
    }

    fn live(&self) -> usize {
        self.lives
            .iter()
            .fold(0usize, |acc, live| acc.saturating_add(*live))
    }
}

fn list_state_dominates(a: &ListMemoryState, b: &ListMemoryState) -> bool {
    a.peak <= b.peak
        && a.lives.len() == b.lives.len()
        && a.lives
            .iter()
            .zip(&b.lives)
            .all(|(a_live, b_live)| a_live <= b_live)
}

fn pareto_memory_states(states: Vec<(MemoryState, ClassId)>) -> Vec<(MemoryState, ClassId)> {
    let mut frontier: Vec<(MemoryState, ClassId)> = Vec::new();
    for (state, class) in states {
        if frontier
            .iter()
            .any(|(existing, _)| existing.live <= state.live && existing.peak <= state.peak)
        {
            continue;
        }
        frontier
            .retain(|(existing, _)| !(state.live <= existing.live && state.peak <= existing.peak));
        frontier.push((state, class));
    }
    frontier
}

fn bound_memory_states(states: Vec<(MemoryState, ClassId)>) -> Vec<(MemoryState, ClassId)> {
    if states.len() <= MAX_EXACT_MEMORY_STATES_PER_CLASS {
        return states;
    }
    let mut states = pareto_memory_states(states);
    states.sort_by_key(|(state, _)| (state.peak, state.live));
    states.truncate(MAX_EXACT_MEMORY_STATES_PER_CLASS);
    states
}

fn pareto_list_states(states: Vec<(ListMemoryState, ClassId)>) -> Vec<(ListMemoryState, ClassId)> {
    let mut frontier: Vec<(ListMemoryState, ClassId)> = Vec::new();
    for (state, class) in states {
        if frontier
            .iter()
            .any(|(existing, _)| list_state_dominates(existing, &state))
        {
            continue;
        }
        frontier.retain(|(existing, _)| !list_state_dominates(&state, existing));
        frontier.push((state, class));
    }
    frontier
}

fn bound_list_states(states: Vec<(ListMemoryState, ClassId)>) -> Vec<(ListMemoryState, ClassId)> {
    if states.len() <= MAX_EXACT_MEMORY_STATES_PER_CLASS {
        return states;
    }
    let mut states = pareto_list_states(states);
    states.sort_by_key(|(state, _)| (state.peak, state.live(), state.lives.clone()));
    states.truncate(MAX_EXACT_MEMORY_STATES_PER_CLASS);
    states
}

#[derive(Debug, Clone, Copy)]
struct KindMemory {
    output_bytes: usize,
    alias_input: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MemorySplitStats {
    pub original_enodes: usize,
    pub split_enodes: usize,
    pub original_eclasses: usize,
    pub split_eclasses: usize,
}

/// Rewrite the serialized e-graph into a product e-graph keyed by memory state.
///
/// Each remaining IR/IList enode points only to child state classes whose
/// concrete combination fits `limit`, so ordinary extraction over the rewritten
/// e-graph cannot construct an over-limit graph for the supplied dyn map.
pub(crate) fn split_egraph_by_memory_limit(
    egraph: &mut SerializedEGraph,
    limit: usize,
    dyn_map: &FxHashMap<char, usize>,
) -> MemorySplitStats {
    let original_enodes = egraph.enodes.len();
    let original_eclasses = egraph.eclasses.len();
    let splitter = StateSplitter::new(egraph, limit, dyn_map);
    let mut split = splitter.split();

    compact_egraph_after_prune(&mut split);
    validate_unique_loop_markers(&split);
    let stats = MemorySplitStats {
        original_enodes,
        split_enodes: split.enodes.len(),
        original_eclasses,
        split_eclasses: split.eclasses.len(),
    };
    *egraph = split;
    stats
}

struct StateSplitter<'a> {
    original: &'a SerializedEGraph,
    split: SerializedEGraph,
    limit: usize,
    dyn_map: &'a FxHashMap<char, usize>,
    sort_by_name: FxHashMap<String, SortDef>,
    ir_memo: FxHashMap<ClassId, Vec<(MemoryState, ClassId)>>,
    list_memo: FxHashMap<ClassId, Vec<(ListMemoryState, ClassId)>>,
    ir_states_by_owner: FxHashMap<ClassId, Vec<(MemoryState, ClassId)>>,
    list_states_by_owner: FxHashMap<ClassId, Vec<(ListMemoryState, ClassId)>>,
    visiting_ir: FxHashSet<ClassId>,
    visiting_list: FxHashSet<ClassId>,
    ir_state_classes: FxHashMap<(ClassId, MemoryState), ClassId>,
    list_state_classes: FxHashMap<(ClassId, ListMemoryState), ClassId>,
    kind_singletons: FxHashMap<NodeId, ClassId>,
    node_memo: FxHashMap<(ClassId, String, Vec<ClassId>, bool), NodeId>,
    next_class: usize,
    next_node: usize,
}

impl<'a> StateSplitter<'a> {
    fn new(
        original: &'a SerializedEGraph,
        limit: usize,
        dyn_map: &'a FxHashMap<char, usize>,
    ) -> Self {
        let mut split = SerializedEGraph {
            enodes: FxHashMap::default(),
            eclasses: FxHashMap::default(),
            node_to_class: FxHashMap::default(),
            roots: Vec::new(),
        };

        for (class, (sort, nodes)) in &original.eclasses {
            if sort == "IR" || sort == "IList" {
                continue;
            }
            let kept_nodes = nodes
                .iter()
                .filter(|node| original.enodes.contains_key(*node))
                .cloned()
                .collect_vec();
            if kept_nodes.is_empty() {
                continue;
            }
            split
                .eclasses
                .insert(class.clone(), (sort.clone(), kept_nodes.clone()));
            for node in kept_nodes {
                if let Some(enode) = original.enodes.get(&node) {
                    split.enodes.insert(node.clone(), enode.clone());
                    split.node_to_class.insert(node, class.clone());
                }
            }
        }

        Self {
            original,
            split,
            limit,
            dyn_map,
            sort_by_name: metal_sort_map(),
            ir_memo: FxHashMap::default(),
            list_memo: FxHashMap::default(),
            ir_states_by_owner: FxHashMap::default(),
            list_states_by_owner: FxHashMap::default(),
            visiting_ir: FxHashSet::default(),
            visiting_list: FxHashSet::default(),
            ir_state_classes: FxHashMap::default(),
            list_state_classes: FxHashMap::default(),
            kind_singletons: FxHashMap::default(),
            node_memo: FxHashMap::default(),
            next_class: 0,
            next_node: 0,
        }
    }

    fn split(mut self) -> SerializedEGraph {
        let Some(root) = self.original.roots.first().cloned() else {
            return self.split;
        };
        let root_states = self.split_ir_class(&root);
        if root_states.is_empty() {
            return self.split;
        }

        let root_class = self.fresh_class_id("mem_root");
        self.split
            .eclasses
            .insert(root_class.clone(), ("IR".to_string(), Vec::new()));

        for (_, state_class) in root_states {
            let root_nodes = self
                .split
                .eclasses
                .get(&state_class)
                .map(|(_, nodes)| nodes.clone())
                .unwrap_or_default();
            for node in root_nodes {
                let Some((label, children)) = self.split.enodes.get(&node).cloned() else {
                    continue;
                };
                self.add_node(
                    root_class.clone(),
                    label,
                    children,
                    node.as_ref().starts_with("synth_"),
                );
            }
        }

        self.split.roots = vec![root_class];
        self.split
    }

    fn split_ir_class(&mut self, class: &ClassId) -> Vec<(MemoryState, ClassId)> {
        if let Some(states) = self.ir_memo.get(class) {
            return states.clone();
        }
        if !self.visiting_ir.insert(class.clone()) {
            return Vec::new();
        }

        let nodes = match self.original.eclasses.get(class) {
            Some((sort, nodes)) if sort == "IR" => nodes.clone(),
            _ => {
                self.visiting_ir.remove(class);
                return Vec::new();
            }
        };

        for node in nodes {
            if self.original.enodes.contains_key(&node) {
                self.split_ir_node(class, &node);
            }
        }

        self.visiting_ir.remove(class);
        let mut states = bound_memory_states(
            self.ir_states_by_owner
                .get(class)
                .cloned()
                .unwrap_or_default(),
        );
        states.sort_by_key(|(state, _)| *state);
        self.ir_memo.insert(class.clone(), states.clone());
        states
    }

    fn split_ir_node(&mut self, owner_class: &ClassId, node: &NodeId) {
        let Some((label, children)) = self.original.enodes.get(node).cloned() else {
            return;
        };

        match label.as_str() {
            "Input" => {
                self.add_ir_state_node(owner_class, MemoryState::zero(), label, children, node);
            }
            "Output" => {
                let Some(input_class) = children.first() else {
                    return;
                };
                for (state, state_class) in self.split_ir_class(input_class) {
                    let mut split_children = children.clone();
                    split_children[0] = state_class;
                    self.add_ir_state_node(owner_class, state, label.clone(), split_children, node);
                }
            }
            "OutputJoin" => {
                let Some(a_class) = children.first() else {
                    return;
                };
                let Some(b_class) = children.get(1) else {
                    return;
                };
                for (a_state, a_split_class) in self.split_ir_class(a_class) {
                    for (b_state, b_split_class) in self.split_ir_class(b_class) {
                        let state = best_join_state(a_state, b_state);
                        if state.peak > self.limit {
                            continue;
                        }
                        let mut split_children = children.clone();
                        split_children[0] = a_split_class.clone();
                        split_children[1] = b_split_class;
                        self.add_ir_state_node(
                            owner_class,
                            state,
                            label.clone(),
                            split_children,
                            node,
                        );
                    }
                }
            }
            "Op" => self.split_op_node(owner_class, node, label, children),
            label if direct_loop_marker(label) => {
                self.split_direct_loop_marker_node(owner_class, node, label.to_string(), children)
            }
            _ => {
                let Some((idx, child_class)) =
                    first_child_with_sort_index(self.original, &children, "IR")
                else {
                    return;
                };
                for (state, state_class) in self.split_ir_class(&child_class) {
                    let mut split_children = children.clone();
                    split_children[idx] = state_class;
                    self.add_ir_state_node(owner_class, state, label.clone(), split_children, node);
                }
            }
        }
    }

    fn split_op_node(
        &mut self,
        owner_class: &ClassId,
        source_node: &NodeId,
        label: String,
        children: Vec<ClassId>,
    ) {
        let Some(kind_class) = children.first() else {
            return;
        };
        let Some(inputs_class) = children.get(1) else {
            return;
        };
        let Some((sort, kind_nodes)) = self.original.eclasses.get(kind_class) else {
            return;
        };
        if sort != "OpKind" {
            return;
        }

        let input_states = self.split_list_class(inputs_class);
        for kind_node in kind_nodes {
            let Some((kind_label, _)) = self.original.enodes.get(kind_node) else {
                continue;
            };
            let Some(kind) =
                kind_memory_for_node(self.original, &self.sort_by_name, kind_node, self.dyn_map)
            else {
                continue;
            };
            if kind.output_bytes > self.limit {
                continue;
            }
            let kind_split_class = self.kind_singleton_class(kind_node);
            if loop_op_kind(kind_label) {
                // Loop OpKinds are structural markers. Keep the marker singleton and
                // pick one feasible state for the data flowing through it.
                let Some((state, input_split_class)) = input_states
                    .iter()
                    .filter_map(|(input_state, input_split_class)| {
                        let state = op_memory_state(kind, input_state)?;
                        (state.peak <= self.limit).then(|| (state, input_split_class.clone()))
                    })
                    .min_by_key(|(state, _)| (state.peak, state.live))
                else {
                    continue;
                };

                let mut split_children = children.clone();
                split_children[0] = kind_split_class;
                split_children[1] = input_split_class;
                self.add_ir_state_node(
                    owner_class,
                    state,
                    label.clone(),
                    split_children,
                    source_node,
                );
                continue;
            }

            for (input_state, input_split_class) in &input_states {
                let Some(state) = op_memory_state(kind, input_state) else {
                    continue;
                };
                if state.peak > self.limit {
                    continue;
                }
                let mut split_children = children.clone();
                split_children[0] = kind_split_class.clone();
                split_children[1] = input_split_class.clone();
                self.add_ir_state_node(
                    owner_class,
                    state,
                    label.clone(),
                    split_children,
                    source_node,
                );
            }
        }
    }

    fn split_direct_loop_marker_node(
        &mut self,
        owner_class: &ClassId,
        source_node: &NodeId,
        label: String,
        children: Vec<ClassId>,
    ) {
        let Some((idx, child_class)) = first_child_with_sort_index(self.original, &children, "IR")
        else {
            return;
        };
        // LoopStart/LoopEnd identity is part of the loop scaffold, so state
        // splitting must not clone the marker across child-state variants.
        let Some((state, state_class)) = self
            .split_ir_class(&child_class)
            .into_iter()
            .filter(|(state, _)| state.peak <= self.limit)
            .min_by_key(|(state, _)| (state.peak, state.live))
        else {
            return;
        };

        let mut split_children = children;
        split_children[idx] = state_class;
        self.add_ir_state_node(owner_class, state, label, split_children, source_node);
    }

    fn split_list_class(&mut self, class: &ClassId) -> Vec<(ListMemoryState, ClassId)> {
        if let Some(states) = self.list_memo.get(class) {
            return states.clone();
        }
        if !self.visiting_list.insert(class.clone()) {
            return Vec::new();
        }

        let nodes = match self.original.eclasses.get(class) {
            Some((sort, nodes)) if sort == "IList" => nodes.clone(),
            _ => {
                self.visiting_list.remove(class);
                return Vec::new();
            }
        };

        for node in nodes {
            if self.original.enodes.contains_key(&node) {
                self.split_list_node(class, &node);
            }
        }

        self.visiting_list.remove(class);
        let mut states = bound_list_states(
            self.list_states_by_owner
                .get(class)
                .cloned()
                .unwrap_or_default(),
        );
        states.sort_by_key(|(state, _)| state.clone());
        self.list_memo.insert(class.clone(), states.clone());
        states
    }

    fn split_list_node(&mut self, owner_class: &ClassId, node: &NodeId) {
        let Some((label, children)) = self.original.enodes.get(node).cloned() else {
            return;
        };

        match label.as_str() {
            "INil" => {
                self.add_list_state_node(
                    owner_class,
                    ListMemoryState::empty(),
                    label,
                    children,
                    node,
                );
            }
            "ICons" => {
                let Some(head_class) = children.first() else {
                    return;
                };
                let Some(tail_class) = children.get(1) else {
                    return;
                };
                for (head_state, head_split_class) in self.split_ir_class(head_class) {
                    for (tail_state, tail_split_class) in self.split_list_class(tail_class) {
                        let state = best_cons_state(head_state, tail_state);
                        if state.peak > self.limit {
                            continue;
                        }
                        let mut split_children = children.clone();
                        split_children[0] = head_split_class.clone();
                        split_children[1] = tail_split_class;
                        self.add_list_state_node(
                            owner_class,
                            state,
                            label.clone(),
                            split_children,
                            node,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn add_ir_state_node(
        &mut self,
        owner_class: &ClassId,
        state: MemoryState,
        label: String,
        children: Vec<ClassId>,
        source_node: &NodeId,
    ) {
        let state_class = self.ir_state_class(owner_class, state);
        self.add_node(
            state_class,
            label,
            children,
            source_node.as_ref().starts_with("synth_"),
        );
    }

    fn add_list_state_node(
        &mut self,
        owner_class: &ClassId,
        state: ListMemoryState,
        label: String,
        children: Vec<ClassId>,
        source_node: &NodeId,
    ) {
        let state_class = self.list_state_class(owner_class, state);
        self.add_node(
            state_class,
            label,
            children,
            source_node.as_ref().starts_with("synth_"),
        );
    }

    fn ir_state_class(&mut self, owner_class: &ClassId, state: MemoryState) -> ClassId {
        let key = (owner_class.clone(), state);
        if let Some(class) = self.ir_state_classes.get(&key) {
            return class.clone();
        }
        let class = self.fresh_class_id("mem_ir");
        self.split
            .eclasses
            .insert(class.clone(), ("IR".to_string(), Vec::new()));
        self.ir_state_classes.insert(key, class.clone());
        self.ir_states_by_owner
            .entry(owner_class.clone())
            .or_default()
            .push((state, class.clone()));
        class
    }

    fn list_state_class(&mut self, owner_class: &ClassId, state: ListMemoryState) -> ClassId {
        let key = (owner_class.clone(), state.clone());
        if let Some(class) = self.list_state_classes.get(&key) {
            return class.clone();
        }
        let class = self.fresh_class_id("mem_ilist");
        self.split
            .eclasses
            .insert(class.clone(), ("IList".to_string(), Vec::new()));
        self.list_state_classes.insert(key, class.clone());
        self.list_states_by_owner
            .entry(owner_class.clone())
            .or_default()
            .push((state, class.clone()));
        class
    }

    fn kind_singleton_class(&mut self, kind_node: &NodeId) -> ClassId {
        if let Some(class) = self.kind_singletons.get(kind_node) {
            return class.clone();
        }
        let Some((label, children)) = self.original.enodes.get(kind_node).cloned() else {
            return self.fresh_class_id("missing_kind");
        };
        let class = self.fresh_class_id("mem_kind");
        let node = self.fresh_node_id("mem_kind_node");
        self.split
            .eclasses
            .insert(class.clone(), ("OpKind".to_string(), vec![node.clone()]));
        self.split.enodes.insert(node.clone(), (label, children));
        self.split.node_to_class.insert(node, class.clone());
        self.kind_singletons
            .insert(kind_node.clone(), class.clone());
        class
    }

    fn add_node(
        &mut self,
        class: ClassId,
        label: String,
        children: Vec<ClassId>,
        prefer_synth: bool,
    ) -> NodeId {
        let key = (class.clone(), label.clone(), children.clone(), prefer_synth);
        if let Some(node) = self.node_memo.get(&key) {
            return node.clone();
        }
        let node = self.fresh_node_id(if prefer_synth {
            "synth_mem_node"
        } else {
            "mem_node"
        });
        self.split.enodes.insert(node.clone(), (label, children));
        self.split.node_to_class.insert(node.clone(), class.clone());
        if let Some((_, nodes)) = self.split.eclasses.get_mut(&class) {
            nodes.push(node.clone());
        }
        self.node_memo.insert(key, node.clone());
        node
    }

    fn fresh_class_id(&mut self, prefix: &str) -> ClassId {
        let id = ClassId::from(format!("{prefix}_{}", self.next_class));
        self.next_class += 1;
        id
    }

    fn fresh_node_id(&mut self, prefix: &str) -> NodeId {
        let id = NodeId::from(format!("{prefix}_{}", self.next_node));
        self.next_node += 1;
        id
    }
}

struct ChoiceMemoryEstimator<'a> {
    egraph: &'a SerializedEGraph,
    choices: &'a EGraphChoiceSet<'a>,
    dyn_map: &'a FxHashMap<char, usize>,
    sort_by_name: FxHashMap<String, SortDef>,
    ir_cache: FxHashMap<ClassId, MemoryState>,
    list_cache: FxHashMap<ClassId, ListMemoryState>,
    visiting_ir: FxHashSet<ClassId>,
    visiting_list: FxHashSet<ClassId>,
}

impl<'a> ChoiceMemoryEstimator<'a> {
    fn new(
        egraph: &'a SerializedEGraph,
        choices: &'a EGraphChoiceSet<'a>,
        dyn_map: &'a FxHashMap<char, usize>,
    ) -> Self {
        Self {
            egraph,
            choices,
            dyn_map,
            sort_by_name: metal_sort_map(),
            ir_cache: FxHashMap::default(),
            list_cache: FxHashMap::default(),
            visiting_ir: FxHashSet::default(),
            visiting_list: FxHashSet::default(),
        }
    }

    fn root_state(&mut self) -> Option<MemoryState> {
        let root = self.egraph.roots.first()?;
        self.ir_class_state(root)
    }

    fn ir_class_state(&mut self, class: &ClassId) -> Option<MemoryState> {
        if let Some(state) = self.ir_cache.get(class) {
            return Some(*state);
        }
        if !self.visiting_ir.insert(class.clone()) {
            return Some(MemoryState::zero());
        }
        let Some(&node) = self.choices.get(class) else {
            self.visiting_ir.remove(class);
            return None;
        };
        let mut state = self.ir_node_state(node);
        if state.is_none()
            && let Some((_, nodes)) = self.egraph.eclasses.get(class)
        {
            state = nodes
                .iter()
                .filter(|candidate| **candidate != *node)
                .filter_map(|candidate| self.ir_node_state(candidate))
                .min_by_key(|state| (state.peak, state.live));
        }
        self.visiting_ir.remove(class);
        if let Some(state) = state {
            self.ir_cache.insert(class.clone(), state);
        }
        state
    }

    fn ir_node_state(&mut self, node: &NodeId) -> Option<MemoryState> {
        let (label, children) = self.egraph.enodes.get(node)?.clone();
        match label.as_str() {
            "Input" => Some(MemoryState::zero()),
            "Output" => self.ir_class_state(children.first()?),
            "OutputJoin" => {
                let a = self.ir_class_state(children.first()?)?;
                let b = self.ir_class_state(children.get(1)?)?;
                Some(best_join_state(a, b))
            }
            "Op" => self.op_state(&children),
            _ => {
                let child = first_child_with_sort(self.egraph, &children, "IR")?;
                self.ir_class_state(&child)
            }
        }
    }

    fn op_state(&mut self, children: &[ClassId]) -> Option<MemoryState> {
        let kind_class = children.first()?;
        let inputs_class = children.get(1)?;
        let kind_node = choose_kind_node(self.egraph, kind_class)?;
        let kind = kind_memory_for_node(self.egraph, &self.sort_by_name, kind_node, self.dyn_map)?;
        let Some(inputs) = self.list_class_state(inputs_class) else {
            return Some(MemoryState {
                live: kind.output_bytes,
                peak: kind.output_bytes,
            });
        };
        op_memory_state(kind, &inputs)
    }

    fn list_class_state(&mut self, class: &ClassId) -> Option<ListMemoryState> {
        if let Some(state) = self.list_cache.get(class) {
            return Some(state.clone());
        }
        if !self.visiting_list.insert(class.clone()) {
            return Some(ListMemoryState::empty());
        }
        let Some(&node) = self.choices.get(class) else {
            self.visiting_list.remove(class);
            return None;
        };
        let mut state = self.list_node_state(node);
        if state.is_none()
            && let Some((_, nodes)) = self.egraph.eclasses.get(class)
        {
            state = nodes
                .iter()
                .filter(|candidate| **candidate != *node)
                .filter_map(|candidate| self.list_node_state(candidate))
                .min_by_key(|state| (state.peak, state.live()));
        }
        self.visiting_list.remove(class);
        if let Some(state) = &state {
            self.list_cache.insert(class.clone(), state.clone());
        }
        state
    }

    fn list_node_state(&mut self, node: &NodeId) -> Option<ListMemoryState> {
        let (label, children) = self.egraph.enodes.get(node)?.clone();
        match label.as_str() {
            "INil" => Some(ListMemoryState::empty()),
            "ICons" => {
                let head = self.ir_class_state(children.first()?)?;
                let tail = self.list_class_state(children.get(1)?)?;
                Some(best_cons_state(head, tail))
            }
            _ => None,
        }
    }
}

fn best_join_state(a: MemoryState, b: MemoryState) -> MemoryState {
    let live = a.live.saturating_add(b.live);
    let a_first_peak = a.peak.max(a.live.saturating_add(b.peak));
    let b_first_peak = b.peak.max(b.live.saturating_add(a.peak));
    MemoryState {
        live,
        peak: a_first_peak.min(b_first_peak),
    }
}

fn best_cons_state(head: MemoryState, tail: ListMemoryState) -> ListMemoryState {
    let mut lives = Vec::with_capacity(tail.lives.len() + 1);
    lives.push(head.live);
    lives.extend_from_slice(&tail.lives);
    let head_first_peak = head.peak.max(head.live.saturating_add(tail.peak));
    let tail_first_peak = tail.peak.max(tail.live().saturating_add(head.peak));
    ListMemoryState {
        lives,
        peak: head_first_peak.min(tail_first_peak),
    }
}

fn op_memory_state(kind: KindMemory, inputs: &ListMemoryState) -> Option<MemoryState> {
    if let Some(alias_input) = kind.alias_input {
        return inputs.lives.get(alias_input).map(|live| MemoryState {
            live: *live,
            peak: inputs.peak,
        });
    }

    Some(MemoryState {
        live: kind.output_bytes,
        peak: inputs
            .peak
            .max(inputs.live().saturating_add(kind.output_bytes)),
    })
}

fn kind_memory_for_node(
    egraph: &SerializedEGraph,
    sort_by_name: &FxHashMap<String, SortDef>,
    node: &NodeId,
    dyn_map: &FxHashMap<char, usize>,
) -> Option<KindMemory> {
    let (kind_label, kind_child_classes) = egraph.enodes.get(node)?;
    if zero_local_op_kind(kind_label) {
        return Some(KindMemory {
            output_bytes: 0,
            alias_input: None,
        });
    }
    let sort = sort_by_name.get(kind_label.as_str())?;
    let kind_children = kind_child_classes
        .iter()
        .map(|class| first_data_node(egraph, class))
        .collect::<Option<Vec<_>>>()?;
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();
    let bytes_expr = local_output_bytes(
        egraph,
        sort,
        &kind_children,
        &mut list_cache,
        &mut expr_cache,
    )?;
    Some(KindMemory {
        output_bytes: eval_bytes(bytes_expr, dyn_map)?,
        alias_input: output_alias_input_for_kind(kind_label),
    })
}

fn first_data_node<'a>(egraph: &'a SerializedEGraph, class: &'a ClassId) -> Option<&'a NodeId> {
    let (_, nodes) = egraph.eclasses.get(class)?;
    nodes.iter().find(|node| egraph.enodes.contains_key(*node))
}

fn eval_bytes(expr: Expression, dyn_map: &FxHashMap<char, usize>) -> Option<usize> {
    let mut dyn_map = dyn_map.clone();
    dyn_map.entry('z').or_insert(1);
    expr.simplify().exec(&dyn_map)
}

fn output_alias_input_for_kind(kind: &str) -> Option<usize> {
    match kind {
        "MetalScatterNoCopy" => Some(0),
        _ => None,
    }
}

fn first_child_with_sort(
    egraph: &SerializedEGraph,
    children: &[ClassId],
    sort: &str,
) -> Option<ClassId> {
    children
        .iter()
        .find(|class| egraph.eclasses.get(*class).is_some_and(|(s, _)| s == sort))
        .cloned()
}

fn first_child_with_sort_index(
    egraph: &SerializedEGraph,
    children: &[ClassId],
    sort: &str,
) -> Option<(usize, ClassId)> {
    children
        .iter()
        .enumerate()
        .find(|(_, class)| egraph.eclasses.get(*class).is_some_and(|(s, _)| s == sort))
        .map(|(idx, class)| (idx, class.clone()))
}

fn choose_kind_node<'a>(egraph: &'a SerializedEGraph, kind_class: &ClassId) -> Option<&'a NodeId> {
    let kind_enodes = &egraph.eclasses.get(kind_class)?.1;
    let extractor_length = |eclass_id: &ClassId| -> Option<usize> {
        let mut len = 0usize;
        let mut cur_eclass = eclass_id.clone();
        let mut visited = FxHashSet::default();
        loop {
            if !visited.insert(cur_eclass.clone()) {
                return None;
            }
            let (label, enodes) = egraph.eclasses.get(&cur_eclass)?;
            if !label.contains("List") {
                return Some(len);
            }
            let head_enode = enodes.first()?;
            let head_label = &egraph.enodes.get(head_enode)?.0;
            if head_label == "ENil" || head_label == "INil" {
                return Some(len);
            }
            if head_label != "ECons" && head_label != "ICons" {
                return Some(len);
            }
            len += 1;
            let children = &egraph.enodes.get(head_enode)?.1;
            if children.len() < 2 {
                return Some(len);
            }
            cur_eclass = children[1].clone();
        }
    };
    let elist_lens_for = |node: &NodeId| -> Vec<usize> {
        egraph
            .enodes
            .get(node)
            .map(|(_, children)| {
                children
                    .iter()
                    .filter_map(|class| {
                        let label = &egraph.eclasses.get(class)?.0;
                        label.contains("List").then(|| extractor_length(class))?
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let is_consistent = |node: &&NodeId| -> bool {
        let lens = elist_lens_for(node);
        lens.is_empty() || lens.iter().all(|len| *len == lens[0])
    };
    let is_kernel = |node: &&NodeId| -> bool {
        let label = &egraph.enodes[*node].0;
        label.starts_with("Metal")
            || label.starts_with("MPS")
            || label == "GenericMatmul"
            || label == "FusionStart"
            || label == "FusionEnd"
    };

    kind_enodes
        .iter()
        .find(|node| egraph.enodes.contains_key(*node) && is_kernel(node) && is_consistent(node))
        .or_else(|| {
            kind_enodes
                .iter()
                .find(|node| egraph.enodes.contains_key(*node) && is_consistent(node))
        })
        .or_else(|| {
            kind_enodes
                .iter()
                .find(|node| egraph.enodes.contains_key(*node) && is_kernel(node))
        })
        .or_else(|| {
            kind_enodes
                .iter()
                .find(|node| egraph.enodes.contains_key(*node))
        })
}

fn compact_egraph_after_prune(egraph: &mut SerializedEGraph) {
    loop {
        let mut to_remove = Vec::new();
        for (id, (_, children)) in &egraph.enodes {
            if children.iter().any(|class| {
                egraph.eclasses.get(class).is_none_or(|(_, nodes)| {
                    !nodes.iter().any(|node| egraph.enodes.contains_key(node))
                })
            }) {
                to_remove.push(id.clone());
            }
        }
        if to_remove.is_empty() {
            break;
        }
        for node in to_remove {
            egraph.enodes.remove(&node);
        }
    }

    for (_, nodes) in egraph.eclasses.values_mut() {
        nodes.retain(|node| egraph.enodes.contains_key(node));
    }
    egraph.eclasses.retain(|_, (_, nodes)| !nodes.is_empty());
    egraph
        .node_to_class
        .retain(|node, _| egraph.enodes.contains_key(node));

    let mut reachable_classes = FxHashSet::default();
    let mut reachable_nodes = FxHashSet::default();
    let mut stack = egraph.roots.clone();
    while let Some(class) = stack.pop() {
        if !reachable_classes.insert(class.clone()) {
            continue;
        }
        let Some((_, nodes)) = egraph.eclasses.get(&class) else {
            continue;
        };
        for node in nodes {
            if !egraph.enodes.contains_key(node) || !reachable_nodes.insert(node.clone()) {
                continue;
            }
            stack.extend(egraph.enodes[node].1.iter().cloned());
        }
    }

    egraph
        .eclasses
        .retain(|class, _| reachable_classes.contains(class));
    egraph
        .enodes
        .retain(|node, _| reachable_nodes.contains(node));
    egraph
        .node_to_class
        .retain(|node, class| reachable_nodes.contains(node) && reachable_classes.contains(class));
    for (_, nodes) in egraph.eclasses.values_mut() {
        nodes.retain(|node| egraph.enodes.contains_key(node));
    }
    egraph
        .roots
        .retain(|class| egraph.eclasses.contains_key(class));
}

fn zero_local_op_kind(kind: &str) -> bool {
    loop_op_kind(kind)
}

fn loop_op_kind(kind: &str) -> bool {
    matches!(
        kind,
        "LoopInput" | "LoopInputStatic" | "LoopOutput" | "LoopOutputSelect"
    )
}

fn direct_loop_marker(kind: &str) -> bool {
    matches!(kind, "LoopStart" | "LoopEnd")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LoopMarkerKey {
    label: String,
    fields: Vec<String>,
}

fn validate_unique_loop_markers(egraph: &SerializedEGraph) {
    let mut seen = FxHashMap::default();
    for node in egraph.enodes.keys() {
        for key in loop_marker_keys_for_node(egraph, node) {
            if let Some(previous) = seen.insert(key.clone(), node.clone()) {
                panic!(
                    "Metal memory splitter duplicated loop marker {key:?}: {previous:?} and {node:?}"
                );
            }
        }
    }
}

fn loop_marker_keys_for_node(egraph: &SerializedEGraph, node: &NodeId) -> Vec<LoopMarkerKey> {
    let Some((label, children)) = egraph.enodes.get(node) else {
        return Vec::new();
    };
    if direct_loop_marker(label) {
        return vec![LoopMarkerKey {
            label: label.clone(),
            fields: field_signature(egraph, children.iter().skip(1)),
        }];
    }
    if label != "Op" {
        return Vec::new();
    }
    let Some(kind_class) = children.first() else {
        return Vec::new();
    };
    let Some((sort, kind_nodes)) = egraph.eclasses.get(kind_class) else {
        return Vec::new();
    };
    if sort != "OpKind" {
        return Vec::new();
    }

    kind_nodes
        .iter()
        .filter_map(|kind_node| {
            let (kind_label, kind_children) = egraph.enodes.get(kind_node)?;
            loop_op_kind(kind_label).then(|| LoopMarkerKey {
                label: kind_label.clone(),
                fields: field_signature(egraph, kind_children.iter()),
            })
        })
        .collect()
}

fn field_signature<'a>(
    egraph: &SerializedEGraph,
    fields: impl Iterator<Item = &'a ClassId>,
) -> Vec<String> {
    fields
        .map(|class| {
            let node_label = egraph
                .eclasses
                .get(class)
                .and_then(|(_, nodes)| {
                    nodes
                        .iter()
                        .find_map(|node| egraph.enodes.get(node).map(|(label, _)| label.clone()))
                })
                .unwrap_or_else(|| "<missing>".to_string());
            format!("{}:{node_label}", class.as_ref())
        })
        .collect()
}

fn metal_sort_map() -> FxHashMap<String, SortDef> {
    <crate::kernel::MetalOps as luminal::op::IntoEgglogOp>::into_vec()
        .into_iter()
        .map(|op| {
            let sort = op.sort();
            (sort.name.clone(), sort)
        })
        .collect()
}

fn local_output_bytes<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    match sort.name.as_str() {
        name if zero_local_op_kind(name) => Some(0.into()),
        "FusionStart" => Some(0.into()),
        "MetalConstant" => Some(4.into()),
        "MetalIota" => Some(expr_field(egraph, sort, kind_children, "range", expr_cache)? * 4),
        "MetalLessThan" => Some(n_elements_field(
            egraph,
            sort,
            kind_children,
            "shape",
            list_cache,
            expr_cache,
        )?),
        "MPSMatmul" => {
            let m = expr_field(egraph, sort, kind_children, "m", expr_cache)?;
            let n = expr_field(egraph, sort, kind_children, "n", expr_cache)?;
            Some(m * n * 4)
        }
        "MPSBatchedMatmul" => {
            let batch = expr_field(egraph, sort, kind_children, "batch", expr_cache)?;
            let m = expr_field(egraph, sort, kind_children, "m", expr_cache)?;
            let n = expr_field(egraph, sort, kind_children, "n", expr_cache)?;
            Some(batch * m * n * 4)
        }
        "MetalCast" => {
            let size = expr_field(egraph, sort, kind_children, "size", expr_cache)?;
            let dtype = dtype_field(egraph, sort, kind_children, "dtype")?;
            Some(bytes_for_elements(size, dtype))
        }
        _ => {
            let shape_field = ["shape", "out_shape", "dest_shape"]
                .into_iter()
                .find(|name| field_index(sort, name).is_some())?;
            let elems = n_elements_field(
                egraph,
                sort,
                kind_children,
                shape_field,
                list_cache,
                expr_cache,
            )?;
            Some(elems * 4)
        }
    }
}

fn bytes_for_elements(elements: Expression, dtype: DType) -> Expression {
    (elements * dtype.bits()).ceil_div(8)
}

fn field_index(sort: &SortDef, name: &str) -> Option<usize> {
    sort.fields.iter().position(|field| field.name == name)
}

fn expr_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    let node = kind_children.get(field_index(sort, field)?)?;
    extract_expr(egraph, node, expr_cache)
}

fn dtype_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
) -> Option<DType> {
    let node = kind_children.get(field_index(sort, field)?)?;
    Some(extract_dtype(egraph, node))
}

fn n_elements_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
    list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    let node = kind_children.get(field_index(sort, field)?)?;
    Some(
        extract_expr_list(egraph, node, list_cache, expr_cache)?
            .into_iter()
            .product::<Expression>()
            .max(1),
    )
}

fn output_bytes_rules(sort: &SortDef) -> Vec<String> {
    match sort.name.as_str() {
        name if zero_local_op_kind(name) => vec![output_bytes_rule(sort, "(MNum 0)", "zero")],
        "FusionStart" => vec![output_bytes_rule(sort, "(MNum 0)", "zero")],
        "MetalConstant" => vec![output_bytes_rule(sort, "(MNum 4)", "f32-scalar")],
        "MetalIota" => vec![output_bytes_rule(
            sort,
            "(MMul ?range (MNum 4))",
            "int-range",
        )],
        "MetalLessThan" => vec![output_bytes_rule_with_facts(
            sort,
            "?__metal_elems",
            "bool-shape",
            None,
            &["(= ?__metal_elems (n_elements ?shape))"],
        )],
        "MPSMatmul" => vec![output_bytes_rule(
            sort,
            "(MMul (MMul ?m ?n) (MNum 4))",
            "f32-matmul",
        )],
        "MPSBatchedMatmul" => vec![output_bytes_rule(
            sort,
            "(MMul (MMul (MMul ?batch ?m) ?n) (MNum 4))",
            "f32-batched-matmul",
        )],
        "MetalCast" => dtype_output_bytes_rules(sort, "size", "dtype"),
        _ => {
            let Some(shape_field) = ["shape", "out_shape", "dest_shape"]
                .into_iter()
                .find(|name| field_index(sort, name).is_some())
            else {
                return vec![];
            };
            output_bytes_rule_for_shape(sort, shape_field)
        }
    }
}

fn output_bytes_rule_for_shape(sort: &SortDef, shape_field: &str) -> Vec<String> {
    let elem_fact = format!("(= ?__metal_elems (n_elements ?{shape_field}))");
    vec![output_bytes_rule_with_facts(
        sort,
        "(MMul ?__metal_elems (MNum 4))",
        "f32-shape",
        None,
        &[elem_fact.as_str()],
    )]
}

fn dtype_output_bytes_rules(sort: &SortDef, elem_field: &str, dtype_field: &str) -> Vec<String> {
    dtype_output_bytes_rules_for_expr(sort, &format!("?{elem_field}"), dtype_field)
}

fn dtype_output_bytes_rules_for_expr(
    sort: &SortDef,
    elements_expr: &str,
    dtype_field: &str,
) -> Vec<String> {
    DTYPE_BITS
        .iter()
        .map(|(dtype, bits)| {
            output_bytes_rule_with_field_value(
                sort,
                &ceil_bytes_expr(elements_expr, *bits),
                &format!("{dtype}-bytes"),
                dtype_field,
                &format!("({dtype})"),
            )
        })
        .collect()
}

fn ceil_bytes_expr(elements_expr: &str, bits: usize) -> String {
    format!("(MCeilDiv (MMul {elements_expr} (MNum {bits})) (MNum 8))")
}

fn output_bytes_rule(sort: &SortDef, bytes_expr: &str, suffix: &str) -> String {
    output_bytes_rule_with_facts(sort, bytes_expr, suffix, None, &[])
}

fn output_bytes_rule_with_field_value(
    sort: &SortDef,
    bytes_expr: &str,
    suffix: &str,
    field: &str,
    value: &str,
) -> String {
    output_bytes_rule_with_facts(sort, bytes_expr, suffix, Some((field, value)), &[])
}

fn output_bytes_rule_with_facts(
    sort: &SortDef,
    bytes_expr: &str,
    suffix: &str,
    override_field: Option<(&str, &str)>,
    extra_facts: &[&str],
) -> String {
    let args = sort
        .fields
        .iter()
        .map(|field| {
            if override_field
                .as_ref()
                .is_some_and(|(name, _)| *name == field.name)
            {
                override_field.unwrap().1.to_string()
            } else {
                format!("?{}", field.name)
            }
        })
        .join(" ");
    let pattern = if args.is_empty() {
        format!("({})", sort.name)
    } else {
        format!("({} {args})", sort.name)
    };
    let facts = std::iter::once(format!("(= ?kind {pattern})"))
        .chain(extra_facts.iter().map(|fact| (*fact).to_string()))
        .join("\n     ");
    format!(
        "(rule
    ({facts})
    ((metal_output_bytes ?kind {bytes_expr}))
    :ruleset {MEMORY_ANALYSIS_RULESET}
    :name \"metal-memory-{}-{suffix}\"
)",
        sort.name
    )
}
