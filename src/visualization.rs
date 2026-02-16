use anyhow::Result;
use egglog::EGraph;
use std::string::String;

use crate::prelude::petgraph::dot::{Config, Dot};
use crate::prelude::*;
use petgraph::Directed;
use petgraph::prelude::StableGraph;
use std::{fmt::Debug, io::Write};

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

impl ToHtml for EGraph {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json = serde_json::to_string_pretty(
            &self.serialize(egglog::SerializeConfig::default()).egraph,
        )?;
        Ok(EGRAPH_VIS_TEMPLATE.replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

impl ToDot for EGraph {
    fn to_dot(&self) -> Result<String> {
        Ok(self
            .serialize(egglog::SerializeConfig::default())
            .egraph
            .to_dot())
    }
}

/// Implements `ToDot` for [`LLIRGraph`] and [`HLIRGraph`]
/// TODO: This simple extraction can be improved in the future with support for edge labels
impl<N, E> ToDot for petgraph::stable_graph::StableGraph<N, E>
where
    N: std::fmt::Debug,
    E: std::fmt::Debug,
{
    fn to_dot(&self) -> Result<String> {
        Ok(format!(
            "{:?}",
            Dot::with_config(self, &[Config::EdgeNoLabel])
        ))
    }
}

#[allow(unused)]
/// View a debug graph in the browser
///
/// # Panics
/// - Depending on the platform, this function may fail if the full directory path for `file_name` does not exist.
/// - Even if the file is properly created, there may be an unrecoverable error when writing all the data to it.
pub fn display_graph(
    graph: &StableGraph<impl Debug, impl Debug, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
    file_name: &str,
) {
    let mut file = std::fs::File::create(file_name).unwrap();
    file.write_all(display_graph_text(graph, mark_nodes).as_bytes())
        .unwrap();
}

fn display_graph_text(
    graph: &StableGraph<impl Debug, impl Debug, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
) -> String {
    let mut new_graph = StableGraph::new();
    let mut map = std::collections::HashMap::new();
    for node in graph.node_indices() {
        if mark_nodes.as_ref().map_or(true, |m| m.contains(&node)) {
            map.insert(
                node,
                new_graph.add_node(format!("{:?}", graph.node_weight(node).unwrap())),
            );
        }
    }
    for edge in graph.edge_indices() {
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        if let (Some(src), Some(dest)) = (map.get(&src), map.get(&dest)) {
            let label = format!("{:?}", graph.edge_weight(edge).unwrap());
            new_graph.add_edge(*src, *dest, label);
        }
    }
    let graph_string = crate::prelude::petgraph::dot::Dot::with_config(&new_graph, &[]).to_string();

    format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    )
}
