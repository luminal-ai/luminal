//! Search observability: progress bars, selected/failed LLIR dumps
//! (`LLIR_DUMP_DIR`), best-candidate logs (`LUMINAL_LOG_LLIR`), and the
//! per-candidate op histogram log (`LUMINAL_CANDIDATE_OPS`).

use std::fmt::Write as FmtWrite;
use std::io::Write;

use colored::Colorize;
use itertools::Itertools;
use petgraph::{
    Direction,
    dot::{Config, Dot},
    stable_graph::NodeIndex,
    visit::EdgeRef,
};
use rustc_hash::FxHashMap;

use super::genetic::Candidate;
use crate::graph::LLIRGraph;
use crate::shape::DynMap;

/// The search progress display: one `Search` bar plus, when bucketed, a
/// `Bucket` bar beneath it.
pub struct ProgressBars {
    bucket_progress: Option<(usize, usize)>,
    n_bar_lines: usize,
}

impl ProgressBars {
    pub fn new(bucket_progress: Option<(usize, usize)>) -> Self {
        Self {
            bucket_progress,
            n_bar_lines: 1 + usize::from(bucket_progress.is_some()),
        }
    }

    fn make_bar(searched: usize, total: usize) -> String {
        let bar_width = 24;
        let head = ((searched as f32 / total as f32) * bar_width as f32)
            .clamp(0.0, bar_width as f32)
            .floor() as usize;
        if head == 0 {
            format!("[>{}]", " ".repeat(bar_width - 1))
        } else if head >= bar_width {
            format!("[{}>]", "=".repeat(bar_width))
        } else {
            format!(
                "[{}>{}]",
                "=".repeat(head),
                " ".repeat(bar_width - head - 1)
            )
        }
    }

    /// Draw the bars at the cursor and flush.
    pub fn render(&self, n_graphs: usize, limit: usize) {
        print!(
            "\x1b[2K  {:>6}  {} {n_graphs}/{limit}",
            "Search".cyan().bold(),
            Self::make_bar(n_graphs, limit),
        );
        if let Some((bucket_idx, n_buckets)) = self.bucket_progress {
            print!(
                "\n\x1b[2K  {:>6}  {} {}/{n_buckets}",
                "Bucket".cyan().bold(),
                Self::make_bar(bucket_idx, n_buckets),
                bucket_idx,
            );
        }
        std::io::stdout().flush().unwrap();
    }

    /// Move back to the first bar line, clear it, and redraw in place.
    pub fn redraw(&self, n_graphs: usize, limit: usize) {
        for _ in 1..self.n_bar_lines {
            print!("\x1b[1A");
        }
        print!("\r\x1b[2K");
        self.render(n_graphs, limit);
    }

    /// Print a line above the bars. `replace_previous` overwrites the
    /// transient line a previous message left (a "slower" result replaces
    /// the last "slower" line in place; a "faster" result is appended).
    /// Call [`ProgressBars::render`] afterwards to redraw the bars.
    pub fn print_message(&self, msg: &str, replace_previous: bool) {
        for _ in 1..self.n_bar_lines {
            print!("\x1b[1A");
        }
        if replace_previous {
            print!("\x1b[1A");
        }
        print!("\r\x1b[2K");
        println!("{msg}");
    }

    /// Erase the bars.
    pub fn clear(&self) {
        for _ in 1..self.n_bar_lines {
            print!("\x1b[1A");
        }
        print!("\r");
        for _ in 0..self.n_bar_lines {
            println!("\x1b[2K");
        }
        for _ in 0..self.n_bar_lines {
            print!("\x1b[1A");
        }
        print!("\r");
        std::io::stdout().flush().unwrap();
    }
}

/// Dump a candidate whose evaluation panicked, under `LLIR_DUMP_DIR`.
pub fn dump_failed_candidate<M>(candidate: &Candidate<M>, dyn_map: &DynMap) {
    maybe_dump_selected_llir("failed-filter-candidate", dyn_map, &candidate.llir);
    if let Some(pre) = candidate.pre_collapse() {
        maybe_dump_selected_llir("failed-filter-precollapse", dyn_map, pre);
    }
}

pub fn maybe_dump_selected_llir(label: &str, dyn_map: &DynMap, llir: &LLIRGraph) {
    let Ok(dir) = std::env::var("LLIR_DUMP_DIR") else {
        return;
    };

    if let Err(err) = std::fs::create_dir_all(&dir) {
        eprintln!("failed to create LLIR_DUMP_DIR={dir}: {err}");
        return;
    }

    let dims = dyn_map
        .iter()
        .sorted_by_key(|(dim, _)| **dim)
        .map(|(dim, value)| format!("{dim}{value}"))
        .join("_");
    let stem = if dims.is_empty() {
        format!("selected-llir-{label}")
    } else {
        format!("selected-llir-{label}-{dims}")
    };
    let dot_path = format!("{dir}/{stem}.dot");
    let summary_path = format!("{dir}/{stem}.txt");

    let dot = format!("{:?}", Dot::with_config(llir, &[Config::EdgeNoLabel]));
    if let Err(err) = std::fs::write(&dot_path, dot) {
        eprintln!("failed to write {dot_path}: {err}");
    }

    let mut op_counts = std::collections::BTreeMap::<String, usize>::new();
    for node in llir.node_indices() {
        *op_counts.entry(format!("{}", llir[node])).or_default() += 1;
    }

    let mut summary = String::new();
    let _ = writeln!(
        summary,
        "selected LLIR {label}: {} nodes, {} edges",
        llir.node_count(),
        llir.edge_count()
    );
    let _ = writeln!(summary, "dyn_map: {dyn_map:?}");
    let _ = writeln!(summary, "\nop counts:");
    for (op, count) in op_counts {
        let _ = writeln!(summary, "  {count:5} {op}");
    }
    let _ = writeln!(summary, "\nnodes:");
    for node in llir.node_indices().sorted_by_key(|n| n.index()) {
        let inputs = llir
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|edge| edge.id())
            .map(|edge| edge.source().index().to_string())
            .join(", ");
        let _ = writeln!(
            summary,
            "  n{} <- [{}] {}",
            node.index(),
            inputs,
            llir[node]
        );
    }

    if let Err(err) = std::fs::write(&summary_path, summary) {
        eprintln!("failed to write {summary_path}: {err}");
    } else {
        println!("   LLIR dump {summary_path}");
    }
}

pub fn panic_initial_filter_limit(filter_fails: usize, last_rejection: Option<&str>) -> ! {
    if let Some(last_rejection) = last_rejection {
        panic!(
            "Failed to find a viable initial genome after {filter_fails} runtime filter failures; last rejection: {last_rejection}"
        );
    }
    panic!("Failed to find a viable initial genome after {filter_fails} runtime filter failures");
}

pub fn append_filter_display(display: String, filter_display: Option<&str>) -> String {
    if let Some(filter_display) = filter_display.filter(|s| !s.is_empty()) {
        format!("{display} | {filter_display}")
    } else {
        display
    }
}

/// Append one line per profiled candidate (op-type histogram + metric) to
/// the file named by `LUMINAL_CANDIDATE_OPS` — search-trajectory forensics
/// for "was family X ever generated, and what did it measure".
pub fn log_candidate_ops(llir: &LLIRGraph, tag: &str) {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var("LUMINAL_CANDIDATE_OPS").ok()) else {
        return;
    };
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for op in llir.node_weights() {
        let debug = format!("{op:?}");
        let name = debug
            .split(['{', '(', ' ', ')'])
            .find(|s| !s.is_empty() && *s != "LLIROp" && *s != "DialectOp")
            .unwrap_or("?")
            .to_string();
        *counts.entry(name).or_default() += 1;
    }
    let line = format!(
        "{tag} | {}\n",
        counts
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

/// When `LUMINAL_LOG_LLIR=1`, print a canonical, diffable dump of a
/// candidate LLIR each time the search finds a new fastest graph. Nodes are
/// numbered canonically (Kahn topological order with a deterministic
/// tie-break on op text and canonical input ids), so two runs of an
/// identical graph produce byte-identical output regardless of NodeIndex
/// assignment — best-so-far graphs from different runs can be compared with
/// plain `diff`.
pub fn log_best_llir(llir: &LLIRGraph, context: &str) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("LUMINAL_LOG_LLIR").is_some()) {
        return;
    }
    use petgraph::visit::EdgeRef;
    use std::collections::BTreeMap;

    let mut indegree: FxHashMap<NodeIndex, usize> = llir
        .node_indices()
        .map(|n| (n, llir.edges_directed(n, Direction::Incoming).count()))
        .collect();
    // Ready nodes keyed by (op text, canonical input ids) for deterministic
    // pops; true duplicates tie and are interchangeable.
    let mut ready: BTreeMap<(String, Vec<usize>), Vec<NodeIndex>> = BTreeMap::new();
    let mut canonical: FxHashMap<NodeIndex, usize> = FxHashMap::default();
    let inputs_of = |n: NodeIndex, canonical: &FxHashMap<NodeIndex, usize>| -> Vec<usize> {
        llir.edges_directed(n, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| canonical.get(&e.source()).copied().unwrap_or(usize::MAX))
            .collect()
    };
    for (&n, &d) in &indegree {
        if d == 0 {
            ready
                .entry((format!("{:?}", llir[n]), Vec::new()))
                .or_default()
                .push(n);
        }
    }
    let mut lines: Vec<String> = Vec::with_capacity(indegree.len());
    while let Some((key, nodes)) = ready.pop_first() {
        for n in nodes {
            let id = lines.len();
            canonical.insert(n, id);
            let inputs = key
                .1
                .iter()
                .map(|i| format!("n{i}"))
                .collect::<Vec<_>>()
                .join(",");
            lines.push(format!("n{id}: {} <- [{inputs}]", key.0));
            for succ in llir
                .neighbors_directed(n, Direction::Outgoing)
                .collect::<Vec<_>>()
            {
                let d = indegree.get_mut(&succ).unwrap();
                *d -= 1;
                if *d == 0 {
                    ready
                        .entry((format!("{:?}", llir[succ]), inputs_of(succ, &canonical)))
                        .or_default()
                        .push(succ);
                }
            }
        }
    }
    println!("LLIR_BEST {context} nodes={}", lines.len());
    for line in &lines {
        println!("{line}");
    }
    println!("LLIR_BEST_END");
}
