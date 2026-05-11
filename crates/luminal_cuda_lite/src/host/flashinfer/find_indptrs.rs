//! Walk the e-graph from the mask node to find qo_indptr and kv_indptr Input nodes.
//!
//! The mask is produced by `compute_attn_mask(q_pos, qo_indptr, kv_indptr)` using
//! primitive HLIR ops. This module validates the mask's structure and extracts the
//! indptr Input node IDs so FlashInfer can use them directly.

use luminal::egglog_utils::{ClassId, NodeId, SerializedEGraph};
use luminal::prelude::FxHashSet;

/// Result of walking the mask computation chain.
#[derive(Debug)]
pub struct IndptrNodes<'a> {
    pub qo_indptr: &'a NodeId,
    pub kv_indptr: &'a NodeId,
}

/// Find the qo_indptr and kv_indptr Input nodes by walking backwards from the mask.
///
/// Validates the mask structure: `allowed * 1e10 + (-1e10)`. Then does a BFS from
/// the `allowed` subtree to find all reachable Input nodes with names containing
/// "qo_indptr" and "kv_indptr".
///
/// Panics with a diagnostic message if the structure doesn't match or the
/// indptr inputs can't be found.
pub fn find_indptr_inputs<'a>(
    egraph: &'a SerializedEGraph,
    mask_node: &'a NodeId,
) -> IndptrNodes<'a> {
    // Step 1: Validate mask = Add(scaled_allowed, neg_constant)
    let (mask_label, mask_children) = &egraph.enodes[mask_node];
    assert!(
        mask_label == "Op",
        "find_indptr_inputs: mask node is not an Op (label={mask_label})"
    );
    let mask_kind = resolve_first_node(egraph, &mask_children[0]);
    let mask_kind_label = &egraph.enodes[mask_kind].0;
    assert!(
        mask_kind_label.contains("Add"),
        "find_indptr_inputs: mask is not an Add (kind={mask_kind_label})"
    );

    let mask_inputs = walk_ilist_simple(egraph, &mask_children[1]);
    assert_eq!(
        mask_inputs.len(),
        2,
        "find_indptr_inputs: mask Add should have 2 inputs, got {}",
        mask_inputs.len()
    );

    // Step 2: One of the inputs should be Mul(allowed, Constant(1e10))
    let (scaled_allowed, allowed_node) = find_1e10_mul(egraph, &mask_inputs);

    // Step 3: BFS from `allowed` to find all reachable Input nodes
    let reachable_inputs = find_reachable_inputs(egraph, allowed_node);

    // Step 4: Match by name
    let mut qo_indptr: Option<&NodeId> = None;
    let mut kv_indptr: Option<&NodeId> = None;

    for (node_id, name) in &reachable_inputs {
        if name.contains("qo_indptr") {
            qo_indptr = Some(node_id);
        } else if name.contains("kv_indptr") {
            kv_indptr = Some(node_id);
        }
    }

    let qo = qo_indptr.unwrap_or_else(|| {
        let found_names: Vec<&str> = reachable_inputs.iter().map(|(_, n)| n.as_str()).collect();
        panic!(
            "find_indptr_inputs: could not find 'qo_indptr' Input reachable from mask.\n\
             Found inputs: {:?}\n\
             Mask node: {:?}\n\
             Scaled allowed node: {:?}",
            found_names, mask_node, scaled_allowed
        );
    });

    let kv = kv_indptr.unwrap_or_else(|| {
        let found_names: Vec<&str> = reachable_inputs.iter().map(|(_, n)| n.as_str()).collect();
        panic!(
            "find_indptr_inputs: could not find 'kv_indptr' Input reachable from mask.\n\
             Found inputs: {:?}\n\
             Mask node: {:?}\n\
             Scaled allowed node: {:?}",
            found_names, mask_node, scaled_allowed
        );
    });

    IndptrNodes {
        qo_indptr: qo,
        kv_indptr: kv,
    }
}

fn find_1e10_mul<'a>(
    egraph: &'a SerializedEGraph,
    mask_add_inputs: &[&'a NodeId],
) -> (&'a NodeId, &'a NodeId) {
    for &input_node in mask_add_inputs {
        let (label, children) = &egraph.enodes[input_node];
        if label != "Op" {
            continue;
        }
        let kind = resolve_first_node(egraph, &children[0]);
        if !egraph.enodes[kind].0.contains("Mul") {
            continue;
        }
        let mul_inputs = walk_ilist_simple(egraph, &children[1]);
        if mul_inputs.len() != 2 {
            continue;
        }
        for (i, &inp) in mul_inputs.iter().enumerate() {
            if is_constant(egraph, inp, 1e10) {
                let other = mul_inputs[1 - i];
                return (input_node, other);
            }
        }
    }
    let mut debug_info = String::new();
    for (i, &input_node) in mask_add_inputs.iter().enumerate() {
        let (label, children) = &egraph.enodes[input_node];
        debug_info.push_str(&format!("\n  input[{i}]: label={label}"));
        if label == "Op" && !children.is_empty() {
            let kind = resolve_first_node(egraph, &children[0]);
            let kind_label = &egraph.enodes[kind].0;
            debug_info.push_str(&format!(" kind={kind_label}"));
            for (j, kc) in egraph.enodes[kind].1.iter().enumerate() {
                let kc_node = resolve_first_node(egraph, kc);
                debug_info.push_str(&format!(" child[{j}]={}", egraph.enodes[kc_node].0));
            }
            if kind_label.contains("Mul") && children.len() >= 2 {
                let mul_inputs = walk_ilist_simple(egraph, &children[1]);
                for (j, &mi) in mul_inputs.iter().enumerate() {
                    let (ml, mc) = &egraph.enodes[mi];
                    debug_info.push_str(&format!("\n    mul_input[{j}]: label={ml}"));
                    if ml == "Op" && !mc.is_empty() {
                        let mk = resolve_first_node(egraph, &mc[0]);
                        debug_info.push_str(&format!(" kind={}", egraph.enodes[mk].0));
                        for (k, mkc) in egraph.enodes[mk].1.iter().enumerate() {
                            let mkc_node = resolve_first_node(egraph, mkc);
                            debug_info.push_str(&format!(" ch[{k}]={}", egraph.enodes[mkc_node].0));
                        }
                    }
                }
            }
        }
    }
    panic!(
        "find_indptr_inputs: could not find Mul(allowed, Constant(1e10)) in mask Add inputs.{debug_info}"
    );
}

fn is_constant(egraph: &SerializedEGraph, node: &NodeId, expected: f32) -> bool {
    let (label, children) = &egraph.enodes[node];
    if label != "Op" {
        return false;
    }
    let kind = resolve_first_node(egraph, &children[0]);
    let kind_label = &egraph.enodes[kind].0;
    if !kind_label.contains("Constant") {
        return false;
    }
    let val_children = &egraph.enodes[kind].1;
    if val_children.is_empty() {
        return false;
    }
    let val_node = resolve_first_node(egraph, &val_children[0]);
    let val_str = &egraph.enodes[val_node].0;
    if let Ok(val) = val_str.parse::<f64>() {
        (val as f32 - expected).abs() < 1.0
    } else {
        false
    }
}

fn find_reachable_inputs<'a>(
    egraph: &'a SerializedEGraph,
    start: &'a NodeId,
) -> Vec<(&'a NodeId, String)> {
    let mut found = Vec::new();
    let mut visited = FxHashSet::default();
    let mut stack = vec![start];

    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }

        let (label, children) = &egraph.enodes[node];

        if label == "Input" {
            if children.len() >= 2 {
                let name_node = resolve_first_node(egraph, &children[1]);
                let name = egraph.enodes[name_node].0.trim_matches('"').to_string();
                found.push((node, name));
            }
            continue;
        }

        if label == "Op" && children.len() >= 2 {
            let ir_inputs = walk_ilist_simple(egraph, &children[1]);
            for inp in ir_inputs {
                stack.push(inp);
            }
        }
    }

    found
}

fn walk_ilist_simple<'a>(
    egraph: &'a SerializedEGraph,
    ilist_eclass: &'a ClassId,
) -> Vec<&'a NodeId> {
    let mut inputs = Vec::new();
    let mut current = resolve_first_node(egraph, ilist_eclass);

    loop {
        let (label, children) = &egraph.enodes[current];
        if label == "INil" {
            break;
        }
        if label != "ICons" {
            break;
        }
        let ir_node = resolve_first_ir_node(egraph, &children[0]);
        inputs.push(ir_node);
        current = resolve_first_node(egraph, &children[1]);
    }

    inputs
}

fn resolve_first_node<'a>(egraph: &'a SerializedEGraph, eclass: &ClassId) -> &'a NodeId {
    &egraph.eclasses[eclass].1[0]
}

fn resolve_first_ir_node<'a>(egraph: &'a SerializedEGraph, eclass: &ClassId) -> &'a NodeId {
    let nodes = &egraph.eclasses[eclass].1;
    for node in nodes {
        let label = &egraph.enodes[node].0;
        if label == "Op" || label == "Input" {
            return node;
        }
    }
    &nodes[0]
}
