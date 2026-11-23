//! Watch an e-graph grow: run a schedule ONE RULESET APPLICATION AT A TIME and
//! draw the e-graph after each.
//!
//! ```sh
//! cargo run --release --example egraph_steps -- facts.egg [--out DIR] [--max-steps 60]
//!     [--sorts LogicalTensor] [--hide logical-of] [--max-nodes 400] [--abbreviate]
//!     [--single-file run.html] [--standalone]
//! ```
//!
//! `facts.egg` holds facts (`let`s) and, optionally, a `(run-schedule ...)`; core's
//! program (sorts, constructors, rules) is loaded first unless `--standalone`. The
//! file's facts are loaded, then its schedule (default: core's two-level saturation)
//! is interpreted here instead of by egglog, so every `(run <ruleset>)` is a step
//! that can be looked at:
//!
//! - `(saturate a b ...)` is a LOOP: run a, b, ... in order, again and again, until
//!   one whole pass changes nothing. Loops nest. In core's
//!   `(saturate (saturate (run main_ruleset) (run prop)) (run subst-walk))` the INNER
//!   loop repeats main_ruleset, prop until neither changes the database; the OUTER
//!   loop repeats "the inner loop to its fixpoint, then subst-walk once" until that
//!   whole pass changes nothing. Each step is labelled with the pass of every loop it
//!   is in (`L1 pass 3 > L2 pass 7 > run main_ruleset`).
//! - `(repeat n a b ...)` is the same, at most n passes.
//!
//! A schedule that never converges is cut at `--max-steps` steps.
//!
//! Output, in `--out` (default `egraph-steps`):
//! - `index.html`: step through the run (arrow keys; scroll zooms the drawing, drag
//!   moves it, double-click fits it). Per step: where in the loops
//!   it is, whether it changed the database, tuple and e-node counts, the rules that
//!   matched, the e-nodes it added, and the drawing.
//! - `step_NNN.dot` / `.svg` (with graphviz `dot` on the PATH): the e-graph of the
//!   chosen sorts (`--sorts`, default `LogicalTensor`). A box is an e-class, titled
//!   by its sort and its smallest term; each row in it an e-node; an edge goes from an e-node to
//!   the class of each operand of a drawn sort. Operands of other sorts (shapes,
//!   index maps, dtypes) are written into the e-node's label in full: an e-node reads
//!   as its egglog term, with `#k` where the arrow numbered k goes. `--abbreviate`
//!   writes those operands in a short arithmetic notation instead (`c0` for a
//!   `CoordVar`, `[10]` for a shape, `map[...]` for an `IndexMapLit`). E-nodes new in the
//!   step are green; a class two classes merged into is outlined in red. E-nodes of
//!   the ops in `--hide` (default `logical-of`, the layout ladder's back-pointer from
//!   a layout tensor to its logical tensor) are left out.
//! - `steps.tsv`: the per-step table.
//! - with `--single-file FILE.html`, also that one file: the same page with every
//!   drawing inside it, nothing else to copy (needs graphviz, for the drawings).

use luminal::prelude::egglog;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

const DEFAULT_SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

#[derive(Debug, Clone)]
enum Sexp {
    Atom(String),
    List(Vec<Sexp>),
}

/// The top-level forms of an egglog text (comments and strings respected).
fn top_level_forms(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut forms, mut depth, mut start, mut i) = (Vec::new(), 0usize, 0usize, 0usize);
    while i < bytes.len() {
        match bytes[i] {
            b';' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            b'(' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    forms.push(&text[start..=i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    forms
}

fn parse_sexp(text: &str) -> Sexp {
    fn parse(tokens: &[String], at: &mut usize) -> Sexp {
        if tokens[*at] == "(" {
            *at += 1;
            let mut items = Vec::new();
            while tokens[*at] != ")" {
                items.push(parse(tokens, at));
            }
            *at += 1;
            Sexp::List(items)
        } else {
            *at += 1;
            Sexp::Atom(tokens[*at - 1].clone())
        }
    }
    let tokens: Vec<String> = text
        .replace('(', " ( ")
        .replace(')', " ) ")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    parse(&tokens, &mut 0)
}

/// One `(run <ruleset>)`, as it came out.
struct Step {
    path: String,
    ruleset: String,
    changed: bool,
    tuples: usize,
    classes: usize,
    nodes: usize,
    rules: Vec<(usize, String)>,
    added: Vec<String>,
    merged: usize,
    drawn: bool,
    /// the step changed the drawn e-graph but no image could be made of it
    failed: bool,
    seconds: f64,
}

struct Options {
    out: std::path::PathBuf,
    max_steps: usize,
    max_nodes: usize,
    sorts: BTreeSet<String>,
    /// e-node ops left out of the drawing and the counts
    hide: BTreeSet<String>,
    /// write operands in the short arithmetic notation instead of as egglog terms
    abbreviate: bool,
    /// also write the whole run as one HTML file with the drawings inside it
    single_file: Option<std::path::PathBuf>,
}

struct Stepper {
    egraph: egglog::EGraph,
    options: Options,
    steps: Vec<Step>,
    /// the schedule's loops, numbered in the order they are first entered
    loops: HashMap<usize, usize>,
    /// e-node keys and class keys of the last drawing
    seen_nodes: BTreeSet<u64>,
    seen_classes: BTreeSet<u64>,
    stopped: bool,
    start: std::time::Instant,
}

type Serialized = luminal::prelude::legacy_egraph_serialize::EGraph;
type NodeId = luminal::prelude::legacy_egraph_serialize::NodeId;
type ClassId = luminal::prelude::legacy_egraph_serialize::ClassId;

/// The serialized e-graph with, per class, its smallest term (size, the e-node at
/// its root) and a hash of that term: a name for the class that survives a step.
struct View<'a> {
    graph: &'a Serialized,
    members: BTreeMap<ClassId, Vec<NodeId>>,
    best: HashMap<ClassId, (u64, NodeId)>,
    keys: HashMap<ClassId, u64>,
    /// terms in the short arithmetic notation, not as egglog
    abbreviate: bool,
}

impl<'a> View<'a> {
    fn new(graph: &'a Serialized, abbreviate: bool) -> Self {
        let mut members: BTreeMap<ClassId, Vec<NodeId>> = BTreeMap::new();
        for (id, node) in &graph.nodes {
            if !node.subsumed {
                members.entry(node.eclass.clone()).or_default().push(id.clone());
            }
        }
        for nodes in members.values_mut() {
            nodes.sort();
        }
        // smallest term per class, to a fixpoint (a class reached only through
        // itself keeps no term)
        let mut best: HashMap<ClassId, (u64, NodeId)> = HashMap::new();
        loop {
            let mut changed = false;
            for (class, nodes) in &members {
                for id in nodes {
                    let node = &graph.nodes[id];
                    let mut size = 1u64;
                    let mut known = true;
                    for child in &node.children {
                        match best.get(&graph.nodes[child].eclass) {
                            Some((s, _)) => size = size.saturating_add(*s),
                            None => {
                                known = false;
                                break;
                            }
                        }
                    }
                    if known && best.get(class).is_none_or(|(s, _)| size < *s) {
                        best.insert(class.clone(), (size, id.clone()));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let mut view = View { graph, members, best, keys: HashMap::new(), abbreviate };
        let classes: Vec<ClassId> = view.members.keys().cloned().collect();
        for class in classes {
            view.class_key(&class);
        }
        view
    }

    fn class_of(&self, node: &NodeId) -> &ClassId {
        &self.graph.nodes[node].eclass
    }

    fn sort_of(&self, class: &ClassId) -> &str {
        self.graph.class_data.get(class).and_then(|d| d.typ.as_deref()).unwrap_or("")
    }

    /// The hash of the class's smallest term.
    fn class_key(&mut self, class: &ClassId) -> u64 {
        if let Some(key) = self.keys.get(class) {
            return *key;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match self.best.get(class).cloned() {
            Some((_, root)) => {
                let node = &self.graph.nodes[&root];
                node.op.hash(&mut hasher);
                let children: Vec<ClassId> = node.children.iter().map(|c| self.class_of(c).clone()).collect();
                for child in children {
                    self.class_key(&child).hash(&mut hasher);
                }
            }
            None => class.to_string().hash(&mut hasher),
        }
        let key = hasher.finish();
        self.keys.insert(class.clone(), key);
        key
    }

    fn node_key(&self, id: &NodeId) -> u64 {
        let node = &self.graph.nodes[id];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        node.op.hash(&mut hasher);
        for child in &node.children {
            self.keys[self.class_of(child)].hash(&mut hasher);
        }
        hasher.finish()
    }

    /// The class's smallest term, written compactly, at most `budget` characters.
    fn term(&self, class: &ClassId, budget: usize) -> String {
        let mut out = String::new();
        if self.abbreviate {
            self.write_term(class, &mut out, budget);
        } else {
            self.write_egglog(class, &mut out, budget);
        }
        if out.chars().count() > budget {
            out = out.chars().take(budget.saturating_sub(1)).collect::<String>() + "…";
        }
        out
    }

    fn write_term(&self, class: &ClassId, out: &mut String, budget: usize) {
        if out.len() > budget {
            return;
        }
        let Some((_, root)) = self.best.get(class) else {
            let _ = write!(out, "<{class}>");
            return;
        };
        self.write_node(root, out, budget);
    }

    /// The class's smallest term as egglog text: `(Constructor operand ...)`, a
    /// literal as itself.
    fn write_egglog(&self, class: &ClassId, out: &mut String, budget: usize) {
        if out.len() > budget {
            return;
        }
        let Some((_, root)) = self.best.get(class) else {
            let _ = write!(out, "<{class}>");
            return;
        };
        let node = &self.graph.nodes[root];
        if node.children.is_empty() && root.to_string().starts_with("primitive-") {
            out.push_str(&node.op);
            return;
        }
        out.push('(');
        out.push_str(&node.op);
        for child in &node.children {
            out.push(' ');
            self.write_egglog(self.class_of(child), out, budget);
        }
        out.push(')');
    }

    /// An e-node over its operands' smallest terms. Core's integer, shape and map
    /// constructors are written as the arithmetic they mean.
    fn write_node(&self, id: &NodeId, out: &mut String, budget: usize) {
        let node = &self.graph.nodes[id];
        let child = |i: usize| self.class_of(&node.children[i]);
        let op = node.op.as_str();
        let infix = |symbol: &str, out: &mut String| {
            out.push('(');
            self.write_term(child(0), out, budget);
            out.push_str(symbol);
            self.write_term(child(1), out, budget);
            out.push(')');
        };
        match (op, node.children.len()) {
            ("IntLit", 1) => self.write_term(child(0), out, budget),
            ("IntAdd", 2) => infix(" + ", out),
            ("IntMul", 2) => infix("*", out),
            ("IntMin", 2) | ("IntMax", 2) => {
                out.push_str(if op == "IntMin" { "min(" } else { "max(" });
                self.write_term(child(0), out, budget);
                out.push_str(", ");
                self.write_term(child(1), out, budget);
                out.push(')');
            }
            ("CoordVar", 2) => {
                out.push('c');
                self.write_term(child(1), out, budget);
            }
            ("ShapeLit", 1) => self.write_term(child(0), out, budget),
            ("IndexMapLit", 2) => {
                out.push_str("map");
                self.write_term(child(0), out, budget);
            }
            (name, 0) if name.ends_with("Nil") => out.push_str("[]"),
            (name, 2) if name.ends_with("Cons") => {
                out.push('[');
                let mut at = id.clone();
                let mut first = true;
                loop {
                    let node = &self.graph.nodes[&at];
                    if node.children.len() != 2 || !node.op.ends_with("Cons") {
                        break;
                    }
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    self.write_term(self.class_of(&node.children[0]), out, budget);
                    let tail = self.class_of(&node.children[1]);
                    match self.best.get(tail) {
                        Some((_, next)) => at = next.clone(),
                        None => break,
                    }
                    if out.len() > budget {
                        break;
                    }
                }
                out.push(']');
            }
            (_, 0) => out.push_str(op.trim_matches('"')),
            _ => {
                out.push_str(op);
                out.push('(');
                for i in 0..node.children.len() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    self.write_term(child(i), out, budget);
                }
                out.push(')');
            }
        }
    }

    /// The e-node's label in a drawing: the e-node as egglog text, each operand of a
    /// drawn sort written `#k` (the arrow numbered k goes to its class), every other
    /// operand as its class's smallest term in full. With `--abbreviate`: its op and
    /// the operands that are not drawn, in the short notation.
    fn node_label(&self, id: &NodeId, drawn: &BTreeSet<String>) -> String {
        let node = &self.graph.nodes[id];
        if !self.abbreviate {
            let mut label = format!("({}", node.op);
            let mut arrow = 0;
            for child in &node.children {
                let class = self.class_of(child);
                label.push(' ');
                if drawn.contains(self.sort_of(class)) {
                    arrow += 1;
                    let _ = write!(label, "#{arrow}");
                } else {
                    self.write_egglog(class, &mut label, usize::MAX);
                }
            }
            label.push(')');
            return label;
        }
        let mut label = node.op.clone();
        let others: Vec<String> = node
            .children
            .iter()
            .map(|c| self.class_of(c))
            .filter(|c| !drawn.contains(self.sort_of(c)))
            .map(|c| self.term(c, 48))
            .collect();
        if !others.is_empty() {
            label.push(' ');
            label.push_str(&others.join(" → "));
        }
        label
    }
}

fn dot_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"").replace('<', "\\<").replace('>', "\\>").replace('{', "\\{").replace('}', "\\}").replace('|', "\\|")
}

/// A DOT label of `text` broken at spaces into left-justified lines of about `width`.
fn dot_wrapped(text: &str, width: usize) -> String {
    let mut lines = vec![String::new()];
    for word in text.split(' ') {
        let line = lines.last_mut().unwrap();
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            lines.push(format!("  {word}"));
        } else {
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
    }
    if lines.len() == 1 {
        return dot_escape(&lines[0]);
    }
    lines.iter().map(|line| format!("{}\\l", dot_escape(line))).collect()
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

impl Stepper {
    /// Interpret a schedule; returns whether it changed the database.
    fn run(&mut self, schedule: &Sexp, path: &str) -> bool {
        if self.stopped {
            return false;
        }
        let Sexp::List(items) = schedule else {
            let Sexp::Atom(name) = schedule else { unreachable!() };
            return self.step(name, path);
        };
        let head = match items.first() {
            Some(Sexp::Atom(head)) => head.as_str(),
            _ => "seq",
        };
        match head {
            "run" => {
                let Some(Sexp::Atom(name)) = items.get(1) else { panic!("(run <ruleset>) expected") };
                self.step(name, path)
            }
            "run-schedule" | "seq" => {
                let mut changed = false;
                for item in &items[usize::from(matches!(items.first(), Some(Sexp::Atom(_))))..] {
                    changed |= self.run(item, path);
                }
                changed
            }
            "saturate" | "repeat" => {
                let (limit, body) = if head == "repeat" {
                    let Some(Sexp::Atom(n)) = items.get(1) else { panic!("(repeat <n> ...) expected") };
                    (n.parse::<usize>().expect("repeat count"), &items[2..])
                } else {
                    (usize::MAX, &items[1..])
                };
                // a loop keeps its name every time the schedule comes back to it
                let next = self.loops.len() + 1;
                let name = format!("L{}", self.loops.entry(schedule as *const Sexp as usize).or_insert(next));
                let mut any = false;
                let mut pass = 0;
                while pass < limit && !self.stopped {
                    pass += 1;
                    let here = format!("{path}{name} pass {pass} › ");
                    let mut changed = false;
                    for item in body {
                        changed |= self.run(item, &here);
                    }
                    any |= changed;
                    if !changed {
                        break;
                    }
                }
                any
            }
            other => panic!("schedule form `{other}` is not interpreted (saturate, repeat, seq, run)"),
        }
    }

    fn step(&mut self, ruleset: &str, path: &str) -> bool {
        if self.steps.len() >= self.options.max_steps {
            self.stopped = true;
            return false;
        }
        let before = self.egraph.get_overall_run_report().iterations.len();
        self.egraph
            .parse_and_run_program(None, &format!("(run-schedule (run {ruleset}))"))
            .unwrap_or_else(|error| panic!("run {ruleset}: {error}"));
        let report = self.egraph.get_overall_run_report();
        let new = &report.iterations[before..];
        let changed = new.iter().any(|it| it.rule_set_report.changed);
        let mut rules: Vec<(usize, String)> = new
            .iter()
            .flat_map(|it| it.rule_set_report.rule_reports.iter())
            .map(|(rule, reports)| (reports.iter().map(|r| r.num_matches).sum::<usize>(), rule.split_whitespace().collect::<Vec<_>>().join(" ")))
            .filter(|(matches, _)| *matches > 0)
            .collect();
        rules.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let index = self.steps.len() + 1;
        let (classes, nodes, added, merged, drawn) = self.draw(index, changed);
        self.steps.push(Step {
            path: format!("{path}run {ruleset}"),
            ruleset: ruleset.to_string(),
            changed,
            tuples: self.egraph.num_tuples(),
            classes,
            nodes,
            rules,
            added,
            merged,
            drawn,
            failed: (changed || index == 1) && !drawn,
            seconds: self.start.elapsed().as_secs_f64(),
        });
        let step = self.steps.last().unwrap();
        eprintln!(
            "step {index:>3} {:<60} {} {:>7} tuples, {:>4} e-nodes in {:>4} classes drawn, +{} e-nodes, {} rules matched",
            step.path,
            if changed { "changed  " } else { "no change" },
            step.tuples,
            nodes,
            classes,
            step.added.len(),
            step.rules.len()
        );
        changed
    }

    /// Write `step_<index>.dot` (and `.svg`); returns (classes, e-nodes) drawn, the
    /// labels of the e-nodes new since the last step, how many drawn classes are
    /// merges of earlier ones, and whether an image was rendered.
    fn draw(&mut self, index: usize, changed: bool) -> (usize, usize, Vec<String>, usize, bool) {
        let serialized = self.egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let mut view = View::new(&serialized, self.options.abbreviate);
        let hide = &self.options.hide;
        for nodes in view.members.values_mut() {
            nodes.retain(|id| !hide.contains(&serialized.nodes[id].op));
        }
        let sorts = &self.options.sorts;
        let drawn_classes: Vec<&ClassId> = view.members.keys().filter(|c| sorts.contains(view.sort_of(c))).collect();
        let node_count: usize = drawn_classes.iter().map(|c| view.members[*c].len()).sum();
        let first = index == 1 && self.seen_nodes.is_empty();
        let mut added = Vec::new();
        let mut node_keys = BTreeSet::new();
        let mut class_keys = BTreeSet::new();
        let mut merged = 0;
        let mut dot = String::from("digraph egraph {\n  compound=true; rankdir=BT; fontname=\"Helvetica\";\n  node [shape=box, style=\"rounded,filled\", fillcolor=white, fontname=\"Helvetica\", fontsize=11];\n  edge [arrowsize=0.6, color=\"#555555\"];\n");
        let mut anchors: HashMap<&ClassId, String> = HashMap::new();
        let mut edges = String::new();
        for (c, class) in drawn_classes.iter().enumerate() {
            // a class is a merge when two e-nodes in it were, last step, in classes of
            // different names: told here by the class's name being new although e-nodes
            // in it are old
            let class_key = view.keys[*class];
            class_keys.insert(class_key);
            let members = &view.members[*class];
            let old_members = members.iter().filter(|id| self.seen_nodes.contains(&view.node_key(id))).count();
            let is_merge = !first && old_members >= 2 && !self.seen_classes.contains(&class_key);
            merged += usize::from(is_merge);
            let _ = writeln!(
                dot,
                "  subgraph cluster_{c} {{\n    label=\"{}\\n{}\"; style=\"rounded\"; color=\"{}\"; penwidth={}; fontsize=10; fontcolor=\"#333333\";",
                dot_escape(&format!("sort {}", view.sort_of(class))),
                dot_escape(&view.term(class, 96)),
                if is_merge { "#c62828" } else { "#888888" },
                if is_merge { 2 } else { 1 },
            );
            for (n, id) in members.iter().enumerate() {
                let key = view.node_key(id);
                node_keys.insert(key);
                let new = !first && !self.seen_nodes.contains(&key);
                let label = view.node_label(id, sorts);
                if new {
                    added.push(format!("{}   in class {}", label, view.term(class, 60)));
                }
                let name = format!("n{c}_{n}");
                let _ = writeln!(dot, "    {name} [label=\"{}\"{}];", dot_wrapped(&label, 64), if new { ", fillcolor=\"#c8e6c9\"" } else { "" });
                anchors.entry(class).or_insert_with(|| name.clone());
                let mut arrow = 0;
                for child in &view.graph.nodes[id].children {
                    let target = view.class_of(child);
                    if sorts.contains(view.sort_of(target)) {
                        arrow += 1;
                        // an operand in the e-node's own class: the arrow stays inside the box
                        let own = if target == *class { "own" } else { "other" };
                        let _ = writeln!(edges, "{name}\t{arrow}\t{own}\t{target}");
                    }
                }
            }
            dot.push_str("  }\n");
        }
        let class_index: HashMap<&ClassId, usize> = drawn_classes.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        for line in edges.lines() {
            let mut fields = line.splitn(4, '\t');
            let (from, arrow, own, target) = (fields.next().unwrap(), fields.next().unwrap(), fields.next().unwrap(), fields.next().unwrap());
            let target = view.members.keys().find(|c| c.to_string() == target).unwrap();
            if let (Some(anchor), Some(c)) = (anchors.get(target), class_index.get(target)) {
                // the number sits at the arrow's tail (`taillabel`): a `label` on an edge
                // between clusters makes graphviz fail to rank some graphs
                let head = if own == "own" { String::new() } else { format!("lhead=cluster_{c}, ") };
                let _ = writeln!(dot, "  {from} -> {anchor} [{head}taillabel=\"#{arrow}\", labelfontsize=9, labelfontcolor=\"#555555\"];");
            }
        }
        dot.push_str("}\n");
        self.seen_nodes = node_keys;
        self.seen_classes = class_keys;
        // an unchanged step draws what the step before it drew
        let mut drawn = false;
        if (changed || index == 1) && node_count <= self.options.max_nodes {
            let path = self.options.out.join(format!("step_{index:03}.dot"));
            std::fs::write(&path, &dot).unwrap();
            // graphviz's default ranking fails on some clustered graphs ("trouble in
            // init_rank"); its other ranking is the second try
            let svg = path.with_extension("svg");
            let mut failure = String::new();
            for extra in [None, Some("-Gnewrank=true")] {
                let mut command = std::process::Command::new("dot");
                command.arg("-Tsvg").arg(&path).arg("-o").arg(&svg).args(extra);
                match command.output() {
                    Ok(output) if output.status.success() => {
                        drawn = true;
                        break;
                    }
                    Ok(output) => failure = String::from_utf8_lossy(&output.stderr).lines().find(|l| l.starts_with("Error")).unwrap_or("dot failed").to_string(),
                    Err(error) => failure = format!("dot did not run: {error}"),
                }
            }
            if !drawn {
                let _ = std::fs::remove_file(&svg);
                eprintln!("step {index:>3} NOT DRAWN: {failure} (the graph is in {})", path.display());
            }
        }
        (drawn_classes.len(), node_count, added, merged, drawn)
    }

    fn write_index(&self, source: &str, schedule: &str) {
        let mut tsv = String::from("step\tpath\tchanged\ttuples\tclasses\tenodes\tadded\tmerged\trules_matched\tseconds\n");
        for (i, s) in self.steps.iter().enumerate() {
            let _ = writeln!(tsv, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2}", i + 1, s.path, s.changed, s.tuples, s.classes, s.nodes, s.added.len(), s.merged, s.rules.len(), s.seconds);
        }
        std::fs::write(self.options.out.join("steps.tsv"), tsv).unwrap();
        let mut data = String::from("[");
        let mut image = 0;
        for (i, s) in self.steps.iter().enumerate() {
            // a step whose drawing could not be made shows none, not an earlier step's
            if s.drawn {
                image = i + 1;
            } else if s.failed {
                image = 0;
            }
            let rules: Vec<String> = s.rules.iter().take(40).map(|(m, r)| serde_json::json!([m, r.chars().take(220).collect::<String>()]).to_string()).collect();
            let _ = write!(
                data,
                "{}{{\"path\":{},\"ruleset\":{},\"changed\":{},\"tuples\":{},\"classes\":{},\"nodes\":{},\"merged\":{},\"image\":{},\"rules\":[{}],\"ruleCount\":{},\"added\":{}}}",
                if i > 0 { "," } else { "" },
                serde_json::json!(s.path),
                serde_json::json!(s.ruleset),
                s.changed,
                s.tuples,
                s.classes,
                s.nodes,
                s.merged,
                image,
                rules.join(","),
                s.rules.len(),
                serde_json::json!(s.added),
            );
        }
        data.push(']');
        let note = if self.stopped {
            format!("Cut at {} steps (--max-steps): the schedule had not converged.", self.steps.len())
        } else {
            format!("The schedule converged in {} steps.", self.steps.len())
        };
        let page = INDEX_HTML
            .replace("__SOURCE__", &html_escape(source))
            .replace("__SCHEDULE__", &html_escape(schedule))
            .replace("__NOTE__", &html_escape(&note))
            .replace("__SORTS__", &html_escape(&self.options.sorts.iter().cloned().collect::<Vec<_>>().join(", ")))
            .replace("__RUN__", &std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()).to_string())
            .replace("__DATA__", &data.replace("</", "<\\/"));
        std::fs::write(self.options.out.join("index.html"), page.replace("__IMAGES__", "null")).unwrap();
        if let Some(single) = &self.options.single_file {
            // every drawing's SVG text, by step; `<` escaped so no SVG can end the script
            let mut images = serde_json::Map::new();
            for (i, s) in self.steps.iter().enumerate() {
                if s.drawn {
                    let svg = std::fs::read_to_string(self.options.out.join(format!("step_{:03}.svg", i + 1))).unwrap();
                    images.insert((i + 1).to_string(), serde_json::Value::String(svg));
                }
            }
            let images = serde_json::Value::Object(images).to_string().replace('<', "\\u003c");
            std::fs::write(single, page.replace("__IMAGES__", &images)).unwrap();
            eprintln!("single file: {} ({:.1} MB)", single.display(), std::fs::metadata(single).map_or(0.0, |m| m.len() as f64 / 1e6));
        }
    }
}

const INDEX_HTML: &str = r##"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-graph steps</title>
<style>
:root { --bg:#fafafa; --panel:#ffffff; --ink:#1c1c1c; --muted:#666; --line:#d8d8d8; --accent:#1565c0; --new:#2e7d32; --quiet:#9e9e9e; --sel:#e3f2fd; }
@media (prefers-color-scheme: dark) { :root { --bg:#161616; --panel:#1f1f1f; --ink:#e8e8e8; --muted:#a0a0a0; --line:#3a3a3a; --accent:#90caf9; --new:#81c784; --quiet:#707070; --sel:#263238; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }
header { padding:12px 16px; border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0 0 4px; }
header code, .rule { font:12px ui-monospace, monospace; }
header p { margin:2px 0; color:var(--muted); }
main { display:grid; grid-template-columns:minmax(260px, 340px) 1fr; height:calc(100vh - 170px); min-height:480px; }
#steps { overflow:auto; border-right:1px solid var(--line); background:var(--panel); }
#steps div { padding:4px 10px; border-bottom:1px solid var(--line); cursor:pointer; display:flex; gap:8px; align-items:baseline; }
#steps div.sel { background:var(--sel); }
#steps .n { width:30px; text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; }
#steps .quiet { color:var(--quiet); }
#steps .plus { margin-left:auto; color:var(--new); font-variant-numeric:tabular-nums; white-space:nowrap; }
#detail { overflow:auto; padding:12px 16px; }
#path { font-weight:600; }
#facts { color:var(--muted); margin:4px 0 10px; }
#image { background:#fff; border:1px solid var(--line); border-radius:6px; overflow:hidden; height:62vh; position:relative; cursor:grab; touch-action:none; }
#image.dragging { cursor:grabbing; }
#image img { display:block; position:absolute; left:0; top:0; max-width:none; transform-origin:0 0; user-select:none; -webkit-user-drag:none; }
#image p { color:#666; margin:8px; }
#imagenote { color:var(--muted); margin:4px 0 0; min-height:1.4em; }
h2 { font-size:13px; margin:14px 0 4px; }
ul { margin:0; padding-left:18px; }
li { margin:1px 0; }
.count { display:inline-block; min-width:44px; text-align:right; color:var(--muted); font-variant-numeric:tabular-nums; margin-right:8px; }
.controls { margin:0 0 8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
button { font:inherit; padding:3px 10px; border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:4px; cursor:pointer; }
label { color:var(--muted); }
@media (max-width: 760px) { main { grid-template-columns:1fr; height:auto; } #steps { max-height:32vh; } }
</style></head><body>
<header>
<h1>E-graph after each ruleset application: <code>__SOURCE__</code></h1>
<p>Schedule: <code>__SCHEDULE__</code></p>
<p><b>Reading a drawing:</b> a grey box is an e-class (terms proved equal), titled by its sort and, under it, its smallest term (cut short with … when long). Each rounded row inside is one e-node, written as its egglog term. An operand that is itself a drawn e-class is written <code>#1</code>, <code>#2</code>, … and the arrow with that number leaves the e-node for that class; every other operand (shape, index map, dtype) is written out in full as the smallest term of its class.</p>
<p>A step is one <code>(run ruleset)</code>. <b>L1, L2, …</b> are the schedule's <code>saturate</code>/<code>repeat</code> loops in the order they are first entered; a loop repeats its body until a whole pass changes nothing. __NOTE__ Drawn sorts: __SORTS__.</p>
</header>
<main>
<div id="steps"></div>
<div id="detail">
<div class="controls"><button id="prev">◀ prev</button><button id="next">next ▶</button><button id="nextchange">next change ▶▶</button>
<button id="fit">fit drawing</button>
<label><input type="checkbox" id="onlychanged"> list only steps that changed the database</label></div>
<div id="path"></div><div id="facts"></div>
<div id="image"></div>
<div id="imagenote"></div>
<h2 id="addedtitle"></h2><ul id="added"></ul>
<h2 id="rulestitle"></h2><ul id="rules"></ul>
</div>
</main>
<script>
const steps = __DATA__;
// the drawings' SVG text by step when the page carries them itself (--single-file), else null
const images = __IMAGES__;
const addresses = {};
function imageAddress(step) {
  // files beside the page: the run's stamp in the address, so a browser does not show an earlier run's drawing
  if (!images) return `step_${String(step).padStart(3, '0')}.svg?run=__RUN__`;
  if (!addresses[step]) addresses[step] = URL.createObjectURL(new Blob([images[step]], {type: 'image/svg+xml'}));
  return addresses[step];
}
let at = 0;
const el = id => document.getElementById(id);
function list() {
  const only = el('onlychanged').checked;
  el('steps').innerHTML = '';
  steps.forEach((s, i) => {
    if (only && !s.changed) return;
    const row = document.createElement('div');
    row.className = (i === at ? 'sel ' : '') + (s.changed ? '' : 'quiet');
    const short = s.path.replace(/ pass /g, '·').replace(/run /, '');
    row.innerHTML = `<span class="n">${i + 1}</span><span>${short}</span><span class="plus">${s.added.length ? '+' + s.added.length : ''}${s.merged ? ' ⧉' + s.merged : ''}</span>`;
    row.onclick = () => show(i);
    el('steps').appendChild(row);
    if (i === at) row.scrollIntoView({block: 'nearest'});
  });
}
function show(i) {
  at = Math.max(0, Math.min(steps.length - 1, i));
  const s = steps[at];
  el('path').textContent = `Step ${at + 1} of ${steps.length}: ${s.path}`;
  el('facts').textContent = `${s.changed ? 'changed the database' : 'changed nothing'} · ${s.tuples} tuples in all · ${s.nodes} e-nodes in ${s.classes} classes drawn` + (s.merged ? ` · ${s.merged} merged classes (red outline)` : '');
  const n = String(s.image).padStart(3, '0');
  if (s.image) {
    // the same image element from step to step, so the zoom and position stay
    let img = el('image').querySelector('img');
    if (!img) {
      el('image').innerHTML = '<img draggable="false" alt="">';
      img = el('image').querySelector('img');
      img.onload = () => { if (!view.set) fit(); else place(); };
    }
    const src = imageAddress(s.image);
    if (img.getAttribute('src') !== src) img.src = src;
    img.alt = `e-graph after step ${s.image}`;
    el('imagenote').textContent = 'scroll to zoom, drag to move, double-click to fit' + (s.image !== at + 1 ? ` · unchanged since step ${s.image}` : '');
  } else {
    el('image').innerHTML = '<p>No drawing for this step: it has more e-nodes than --max-nodes, or graphviz is missing or failed on it (the output of the run says which). The lists below and steps.tsv still describe it.</p>';
    el('imagenote').textContent = '';
  }
  el('addedtitle').textContent = s.added.length ? `E-nodes added (${s.added.length}, green in the drawing)` : '';
  el('added').innerHTML = s.added.map(a => `<li class="rule">${a.replace(/&/g, '&amp;').replace(/</g, '&lt;')}</li>`).join('');
  el('rulestitle').textContent = `Rules that matched (${s.ruleCount}${s.ruleCount > s.rules.length ? ', top ' + s.rules.length : ''})`;
  el('rules').innerHTML = s.rules.map(r => `<li><span class="count">${r[0]}</span><span class="rule">${r[1].replace(/&/g, '&amp;').replace(/</g, '&lt;')}</span></li>`).join('');
  list();
}
// the drawing's zoom (k) and offset, kept across steps
const view = {x: 0, y: 0, k: 1, set: false};
function place() {
  const img = el('image').querySelector('img');
  if (img) img.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.k})`;
}
function fit() {
  const img = el('image').querySelector('img');
  if (!img || !img.naturalWidth) return;
  const box = el('image').getBoundingClientRect();
  view.k = Math.min(1, (box.width - 16) / img.naturalWidth, (box.height - 16) / img.naturalHeight);
  view.x = (box.width - img.naturalWidth * view.k) / 2;
  view.y = (box.height - img.naturalHeight * view.k) / 2;
  view.set = true;
  place();
}
el('image').addEventListener('wheel', e => {
  if (!el('image').querySelector('img')) return;
  e.preventDefault();
  const box = el('image').getBoundingClientRect();
  const cx = e.clientX - box.left, cy = e.clientY - box.top;
  const k = Math.max(0.02, Math.min(20, view.k * Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015))));
  // the point under the cursor stays under it
  view.x = cx - (cx - view.x) * (k / view.k);
  view.y = cy - (cy - view.y) * (k / view.k);
  view.k = k;
  view.set = true;
  place();
}, {passive: false});
let drag = null;
el('image').addEventListener('pointerdown', e => {
  drag = {x: e.clientX - view.x, y: e.clientY - view.y};
  el('image').setPointerCapture(e.pointerId);
  el('image').classList.add('dragging');
});
el('image').addEventListener('pointermove', e => {
  if (!drag) return;
  view.x = e.clientX - drag.x;
  view.y = e.clientY - drag.y;
  view.set = true;
  place();
});
for (const name of ['pointerup', 'pointercancel']) el('image').addEventListener(name, () => { drag = null; el('image').classList.remove('dragging'); });
el('image').addEventListener('dblclick', fit);
el('fit').onclick = fit;
el('prev').onclick = () => show(at - 1);
el('next').onclick = () => show(at + 1);
el('nextchange').onclick = () => { let i = at + 1; while (i < steps.length && !steps[i].added.length && !steps[i].merged) i++; show(i < steps.length ? i : at); };
el('onlychanged').onchange = list;
document.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { show(at + 1); e.preventDefault(); } if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { show(at - 1); e.preventDefault(); } });
show(0);
</script></body></html>
"##;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut options = Options {
        out: "egraph-steps".into(),
        max_steps: 60,
        max_nodes: 400,
        sorts: BTreeSet::from(["LogicalTensor".to_string()]),
        hide: BTreeSet::from(["logical-of".to_string()]),
        abbreviate: false,
        single_file: None,
    };
    let mut standalone = false;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--out" => options.out = value("--out")?.into(),
            "--max-steps" => options.max_steps = value("--max-steps")?.parse()?,
            "--max-nodes" => options.max_nodes = value("--max-nodes")?.parse()?,
            "--sorts" => options.sorts = value("--sorts")?.split(',').map(str::to_string).collect(),
            "--hide" => options.hide = value("--hide")?.split(',').map(str::to_string).collect(),
            "--abbreviate" => options.abbreviate = true,
            "--single-file" => options.single_file = Some(value("--single-file")?.into()),
            "--standalone" => standalone = true,
            other if path.is_none() && !other.starts_with("--") => path = Some(other.to_string()),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let path = path.ok_or("usage: egraph_steps <facts.egg> [--out DIR] [--max-steps N] [--max-nodes N] [--sorts A,B] [--hide op,op] [--abbreviate] [--single-file FILE.html] [--standalone]")?;
    let source = std::fs::read_to_string(&path)?;
    let forms = top_level_forms(&source);
    let facts: String = forms.iter().filter(|f| !f.starts_with("(run-schedule")).map(|f| format!("{f}\n")).collect();
    let schedules: Vec<&str> = forms.iter().copied().filter(|f| f.starts_with("(run-schedule")).collect();
    let schedule = if schedules.is_empty() { DEFAULT_SCHEDULE.to_string() } else { schedules.join(" ") };
    std::fs::create_dir_all(&options.out)?;
    // an earlier run's drawings must not stand in for this run's
    for entry in std::fs::read_dir(&options.out)? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with("step_") && (name.ends_with(".dot") || name.ends_with(".svg")) {
            std::fs::remove_file(&path)?;
        }
    }

    let mut egraph = luminal::egglog_snippet::new_egraph();
    if !standalone {
        egraph.parse_and_run_program(None, &luminal::egglog_snippet::assembled_program_for(&[]))?;
    }
    egraph.parse_and_run_program(Some(path.clone()), &facts)?;
    let out = options.out.clone();
    let mut stepper = Stepper {
        egraph,
        options,
        steps: Vec::new(),
        loops: HashMap::new(),
        seen_nodes: BTreeSet::new(),
        seen_classes: BTreeSet::new(),
        stopped: false,
        start: std::time::Instant::now(),
    };
    // the facts as loaded are what the first step's additions are told against
    let (classes, nodes, _, _, _) = stepper.draw(0, true);
    eprintln!("loaded: {} tuples, {nodes} e-nodes in {classes} classes drawn (step_000)", stepper.egraph.num_tuples());
    stepper.run(&parse_sexp(&format!("(seq {schedule})")), "");
    stepper.write_index(&path, &schedule);
    eprintln!(
        "{}; open {}",
        if stepper.stopped { format!("cut at {} steps, not converged", stepper.steps.len()) } else { format!("converged in {} steps", stepper.steps.len()) },
        out.join("index.html").display()
    );
    Ok(())
}
