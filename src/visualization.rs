//! Visualization interfaces, re-seated on the current pipeline's artifacts
//! (the HLIR-era display_graph machinery died with the old world): the
//! recorded [`LogicalGraph`](crate::graph::LogicalGraph), the extracted
//! plan ([`ExtractedGraph`](crate::layout_ir::ExtractedGraph)), the
//! bufferized plan ([`BufferIrGraph`](crate::bufferize::BufferIrGraph)),
//! and the [`SerializedEGraph`](crate::egglog_utils::SerializedEGraph)
//! snapshot.

use anyhow::Result;
use egglog::EGraph;
use rustc_hash::FxHashMap;

pub trait ToHtml {
    fn to_html(&self) -> Result<String>;
}

pub trait ToDot {
    fn to_dot(&self) -> Result<String>;
}

const EGRAPH_VIS_TEMPLATE: &str = r#"
<div id="egraph-visualizer"></div>
<link rel="stylesheet" href="https://esm.sh/egraph-visualizer/dist/style.css" />
<script type="module">
    import { mount } from "https://esm.sh/egraph-visualizer";
    const egraph = {{JSON_TEMPLATE}};
    const mounted = mount(document.getElementById("egraph-visualizer"));
    mounted.render([JSON.stringify(egraph)]);
</script>"#;

/// Pipeline-agnostic: any egglog e-graph, rendered by the egraph-visualizer
/// web component.
impl ToHtml for EGraph {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json = serde_json::to_string_pretty(
            &self.serialize(egglog::SerializeConfig::default()).egraph,
        )?;
        Ok(EGRAPH_VIS_TEMPLATE.replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

/// The homebrew semi-static e-graph snapshot: one cluster per e-class, one
/// node per e-node, edges from each e-node to the e-class of each child
/// (compound edges dock at the cluster border). Output is deterministic
/// (classes and nodes sorted by id string); root classes get the boundary
/// blue border.
impl ToDot for crate::egglog_utils::SerializedEGraph {
    fn to_dot(&self) -> Result<String> {
        let mut classes: Vec<_> = self.eclasses.iter().collect();
        classes.sort_by_key(|(class, _)| class.to_string());

        let mut out = String::from(
            "digraph SerializedEGraph {\n  compound=true;\n  node [shape=box, style=\"rounded,filled\", fillcolor=\"#ede9fe\", color=\"#7c3aed\", fontname=\"Helvetica\"];\n",
        );
        // (cluster index, anchor node index) per class — edges point at the
        // anchor and clip at the cluster border via lhead.
        let mut anchors = FxHashMap::default();
        let mut node_ids = FxHashMap::default();
        let mut next = 0usize;
        for (cluster, (class, (typ, members))) in classes.iter().enumerate() {
            let mut members: Vec<_> = members.iter().collect();
            members.sort_by_key(|node| node.to_string());
            let border = if self.roots.contains(class) {
                "#2563eb"
            } else {
                "#9ca3af"
            };
            out.push_str(&format!(
                "  subgraph cluster_{cluster} {{\n    label=\"{}\";\n    color=\"{border}\";\n",
                escape_dot_string(&format!("{typ} {class}"))
            ));
            for member in members {
                let id = next;
                next += 1;
                node_ids.insert(member.clone(), id);
                anchors.entry((*class).clone()).or_insert((cluster, id));
                let label = escape_dot_string(&self.enodes[member].0);
                out.push_str(&format!("    n{id} [label=\"{label}\"];\n"));
            }
            out.push_str("  }\n");
        }
        let mut enodes: Vec<_> = self.enodes.iter().collect();
        enodes.sort_by_key(|(node, _)| node.to_string());
        for (node, (_, children)) in enodes {
            let src = node_ids[node];
            for child in children {
                // Children outside the retained snapshot (stripped classes)
                // simply have no cluster to point at.
                if let Some((cluster, anchor)) = anchors.get(child) {
                    out.push_str(&format!(
                        "  n{src} -> n{anchor} [lhead=cluster_{cluster}];\n"
                    ));
                }
            }
        }
        out.push_str("}\n");
        Ok(out)
    }
}

/// The recorded logical model as a dataflow DAG: one node per LIVE value
/// (dead rows elided, matching `model_text`), operand edges into consumers.
/// Boundary rows (inputs, output designations) are blue, interior values
/// violet — the shared visual grammar. A poisoned graph refuses to render,
/// same contract as `model_text`.
impl ToDot for crate::graph::LogicalGraph {
    fn to_dot(&self) -> Result<String> {
        if let Some(reason) = self.poisoned() {
            anyhow::bail!("logical graph poisoned: {reason}");
        }
        let live = self.live_set();
        let mut out_keys: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (id, key) in self.viz_outputs() {
            out_keys.entry(id.index()).or_default().push(key);
        }
        let mut out = String::from(
            "digraph LogicalGraph {\n  node [fontname=\"Helvetica\"];\n  edge [fontname=\"Helvetica\"];\n",
        );
        for (id, node) in self.viz_nodes() {
            if !live.contains(&id) {
                continue;
            }
            let index = id.index();
            let input_label = match &node.op {
                crate::graph::LogicalOp::Input { label } => Some(label.as_str()),
                _ => None,
            };
            let dims_text = node
                .dims
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join("×");
            let mut lines = vec![
                match input_label {
                    Some(name) => format!("v{index} = input {name}"),
                    None => format!("v{index} = {}", node.op.constructor()),
                },
                format!("[{dims_text}] {:?}", node.dtype),
            ];
            if let Some(keys) = out_keys.get(&index) {
                for key in keys {
                    lines.push(format!("out {key}"));
                }
            }
            let boundary = input_label.is_some() || out_keys.contains_key(&index);
            let (fill, border) = if boundary {
                ("#dbeafe", "#2563eb")
            } else {
                ("#ede9fe", "#7c3aed")
            };
            let label = lines
                .iter()
                .map(|line| escape_dot_string(line))
                .collect::<Vec<_>>()
                .join("\\n");
            out.push_str(&format!(
                "  n{index} [shape=box, style=\"rounded,filled\", fillcolor=\"{fill}\", color=\"{border}\", label=\"{label}\"];\n"
            ));
            let operands = self.viz_operands(id);
            for (slot, operand) in &operands {
                // Position labels only where order matters (2+ operands).
                if operands.len() > 1 {
                    out.push_str(&format!(
                        "  n{} -> n{index} [label=\"{slot}\"];\n",
                        operand.index()
                    ));
                } else {
                    out.push_str(&format!("  n{} -> n{index};\n", operand.index()));
                }
            }
        }
        out.push_str("}\n");
        Ok(out)
    }
}

/// The extraction artifact — the most useful single type to render:
/// `extractor::extract_layout_ir_with_ops_and_matchers` returns
/// `Result<Option<ExtractedGraph>>` and `dps::dps_rewrite` both consumes
/// AND produces this same type, so one impl covers the whole
/// extract → DPS ladder (render before and after the rewrite).
impl ToDot for crate::layout_ir::ExtractedGraph {
    fn to_dot(&self) -> Result<String> {
        // The inherent renderer (layout_ir.rs) wins method resolution — no
        // recursion here.
        Ok(crate::layout_ir::ExtractedGraph::to_dot(self))
    }
}

/// The bufferized plan; the inherent renderer (bufferize.rs) carries the
/// slot-table grammar and buffer-name labeling.
impl<L: crate::bufferize::PlanLayout> ToDot for crate::bufferize::BufferIrGraph<L> {
    fn to_dot(&self) -> Result<String> {
        Ok(crate::bufferize::BufferIrGraph::to_dot(self))
    }
}

/// Write rendered html to a file.
pub fn save_html(html: &str, path: &str) -> Result<()> {
    std::fs::write(path, html)?;
    Ok(())
}

/// Open a dot source in the Luminal Visualizer (browser).
pub fn open_dot(dot: &str) {
    let url = format!("http://viz.luminal.com/?dot={}", urlencoding::encode(dot));
    let _ = open::that(&url);
}

fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn logical_graph_dot_smoke() {
        let mut cx = Graph::new();
        let a = cx.tensor(2, DType::F32);
        let b = cx.tensor(2, DType::F32);
        let _c = (a + b).output();
        let dot = cx.logical.to_dot().expect("recorded model renders");
        assert!(dot.contains("digraph"), "header missing:\n{dot}");
        assert!(dot.contains("LogicalAdd"), "op node missing:\n{dot}");
        assert!(dot.contains("->"), "operand edges missing:\n{dot}");
    }

    #[test]
    fn serialized_egraph_dot_smoke() {
        // var! is unhygienic — it needs Span/RustSpan in scope.
        #[allow(unused_imports)]
        use egglog::{ast::Span, prelude::RustSpan, var};
        let mut egraph = egglog::EGraph::default();
        let commands = egraph
            .parser
            .get_program_from_string(
                None,
                "(datatype Math (Num i64) (Add Math Math))\n(let root (Add (Num 1) (Num 2)))",
            )
            .expect("program parses");
        egraph.run_program(commands).expect("program runs");
        let (sort, value) = egraph.eval_expr(&var!("root")).expect("root resolves");
        let serialized = crate::egglog_utils::SerializedEGraph::new(&egraph, vec![(sort, value)]);
        let dot = serialized.to_dot().expect("snapshot renders");
        assert!(dot.contains("cluster_"), "eclass clusters missing:\n{dot}");
        assert!(dot.contains("Add"), "enode label missing:\n{dot}");
    }
}
