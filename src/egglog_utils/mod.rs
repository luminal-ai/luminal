use itertools::Itertools;
use std::{str, sync::Arc};

pub const BASE: &str = include_str!("base.egg");
pub const BASE_CLEANUP: &str = include_str!("base_cleanup.egg");
pub const RUN_SCHEDULE: &str = include_str!("run_schedule.egg");

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    let ops_str = ops
        .iter()
        .map(|o| {
            let (name, body) = o.term();
            format!(
                "({name} {})",
                body.into_iter().map(|j| format!("{j:?}")).join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "
    (datatype*
        (IR
            (OutputJoin IR IR)
            {ops_str}
        )
        (IList
            (ICons IR IList)
            (INil)
            (IConcat IList IList)
        )
    )
    (function dtype (IR) DType :merge new)
    "
    )
}

fn ilist_concat_rules() -> &'static str {
    r#"
    (rewrite (IConcat (INil) ?b) ?b :ruleset expr)
    (rewrite (IConcat (ICons ?h ?t) ?b) (ICons ?h (IConcat ?t ?b)) :ruleset expr)
    "#
}

fn op_cleanups_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    format!(
        "
    {}
    ",
        ops.iter()
            .filter(|op| op.cleanup())
            .map(|o| {
                let (name, body) = o.term();
                let body_terms = (0..body.len()).map(|i| (b'a' + i as u8) as char).join(" ");
                format!(
                    "(rule
                ((= ?m ({name} {body_terms})))
                ((delete ({name} {body_terms})))
                :ruleset cleanup
            )"
                )
            })
            .join("\n")
    )
}

pub fn early_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> String {
    [
        BASE.to_string(),
        op_defs_string(ops),
        ilist_concat_rules().to_string(),
        ops.iter().flat_map(|o| o.early_rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        format!(
            "(run-schedule
                (saturate expr)
                (run)
                (saturate base_cleanup)
            )
            (extract {root})"
        ),
    ]
    .join("\n")
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    [
        BASE.to_string(),
        op_defs_string(ops),
        ilist_concat_rules().to_string(),
        ops.iter().flat_map(|o| o.rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        RUN_SCHEDULE.to_string(),
    ]
    .join("\n")
}

use crate::{op::EgglogOp, prelude::FxHashMap};
use egglog::{ArcSort, EGraph, Value};
use egraph_serialize::{ClassId, NodeId};

#[derive(Debug)]
///  This is snapshot of an EGraph with Rust native hash maps and sets for enabling more native traversal / algorithm writing.
///  The name comes from the serialize egraph crates, which returns a ETermDAG, which caused issues, so this is a homebrew semi-static egraph
pub struct SerializedEGraph {
    pub enodes: FxHashMap<NodeId, (String, Vec<ClassId>)>,
    pub eclasses: FxHashMap<ClassId, (String, Vec<NodeId>)>,
    pub node_to_class: FxHashMap<NodeId, ClassId>,
    pub roots: Vec<ClassId>,
}

impl SerializedEGraph {
    /// This is an opinionated function which does more than strictly take the state of the egglog object.
    /// It also filters out "[...]" nodes and then changes the structure from the e-termDAG that egraph-serialize
    /// produces to a strict egraph, where the children of e-classes are e-nodes.
    pub fn new(egraph: &EGraph, root_eclasses: Vec<(ArcSort, Value)>) -> Self {
        let s = egraph.serialize(egglog::SerializeConfig {
            root_eclasses,
            max_functions: None,
            include_temporary_functions: false,
            max_calls_per_function: None,
        });
        // Convert to SerializedEGraph
        let mut classes = FxHashMap::default();
        for (node_id, node) in &s.egraph.nodes {
            classes
                .entry(node.eclass.clone())
                .or_insert(vec![])
                .push(node_id.clone())
        }
        let mut s_egraph = SerializedEGraph {
            roots: s.egraph.root_eclasses,
            node_to_class: s
                .egraph
                .nodes
                .iter()
                .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
                .collect(),
            enodes: s
                .egraph
                .nodes
                .iter()
                .map(|(n, enode)| {
                    (
                        n.clone(),
                        (
                            enode.op.clone(),
                            enode
                                .children
                                .iter()
                                .map(|n| s.egraph.nodes[n].eclass.clone())
                                .collect(),
                        ),
                    )
                })
                .collect(),
            eclasses: s
                .egraph
                .class_data
                .iter()
                .map(|(c, eclass)| (c.clone(), (eclass.typ.clone().unwrap(), classes[c].clone())))
                .collect(),
        };
        // Strip out all [...] enodes
        s_egraph.enodes.retain(|_, (label, _)| label != "[...]");
        loop {
            let mut to_remove = vec![];
            for (id, (_, children)) in &s_egraph.enodes {
                if children.iter().any(|c| {
                    !s_egraph.eclasses[c]
                        .1
                        .iter()
                        .any(|n| s_egraph.enodes.contains_key(n))
                }) {
                    to_remove.push(id.clone());
                }
            }
            for n in &to_remove {
                s_egraph.enodes.remove(n);
            }
            if to_remove.is_empty() {
                break;
            }
        }
        // Correct the eclass mapping
        for (_, enodes) in s_egraph.eclasses.values_mut() {
            enodes.retain(|n| s_egraph.enodes.contains_key(n));
        }
        s_egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
        s_egraph
            .node_to_class
            .retain(|n, _| s_egraph.enodes.contains_key(n));
        s_egraph
    }
}
