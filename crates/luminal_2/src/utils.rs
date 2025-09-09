#![allow(unused)]

use std::{collections::HashMap, io::Write};

use egglog::{CommandOutput, EGraph, Error, Term, prelude::exprs::var};
use egui::Color32;
use itertools::Itertools;
use luminal::{
    prelude::{
        NodeIndex,
        petgraph::{
            Directed, Direction,
            algo::toposort,
            dot::{Config, Dot},
            prelude::StableGraph,
            visit::{EdgeRef, Topo},
        },
    },
    shape::Expression,
};
use regex::Regex;
use rustc_hash::FxHashSet;

pub fn unary(
    a: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, ());
    tmp
}

pub fn binary(
    a: NodeIndex,
    b: NodeIndex,
    term: GraphTerm,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    let tmp = graph.add_node(term);
    graph.add_edge(a, tmp, ());
    graph.add_edge(b, tmp, ());
    tmp
}

pub fn loop_in(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    marker: impl ToString,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopIn {
            range: range.into(),
            stride: stride.into(),
            marker: marker.to_string(),
        },
        graph,
    )
}

pub fn loop_out(
    node: NodeIndex,
    range: impl Into<Expression>,
    stride: impl Into<Expression>,
    marker: impl ToString,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
) -> NodeIndex {
    unary(
        node,
        GraphTerm::LoopOut {
            range: range.into(),
            stride: stride.into(),
            marker: marker.to_string(),
        },
        graph,
    )
}

pub fn pad_in(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    levels: usize,
) -> NodeIndex {
    for i in 0..levels {
        node = loop_in(node, 1, 0, "pad".to_string(), graph);
    }
    node
}

pub fn pad_out(
    mut node: NodeIndex,
    graph: &mut StableGraph<GraphTerm, (), Directed>,
    levels: usize,
) -> NodeIndex {
    for i in (0..levels).rev() {
        node = loop_out(node, 1, 0, "pad".to_string(), graph);
    }
    node
}

use crate::{GraphTerm, Kernel, debug::display_graph};

pub fn validate_graph(graph: &StableGraph<(GraphTerm, usize), (), Directed>) {
    // walk the graph and make sure loopins -> next loop level (or loopout) and prev loop (or loopin) -> loopout
    for node in graph.node_indices() {
        let (curr_term, curr_level) = graph.node_weight(node).unwrap();
        if matches!(curr_term, GraphTerm::LoopIn { .. }) {
            // All loopins must have outputs that are one level more, unless they are loopouts
            for new_node in graph.neighbors_directed(node, Direction::Outgoing) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopOut { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(graph);
                        panic!("incorrect levels");
                    }
                }
            }
        } else if matches!(curr_term, GraphTerm::LoopOut { .. }) {
            // All loopouts must have inputs that are one level more, unless they are loopins
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(new_term, GraphTerm::LoopIn { .. }) {
                    if *new_level != *curr_level + 1 {
                        display_graph(graph);
                        panic!("incorrect levels");
                    }
                }
            }
        } else {
            for new_node in graph.neighbors_directed(node, Direction::Incoming) {
                let (new_term, new_level) = graph.node_weight(new_node).unwrap();
                if !matches!(
                    new_term,
                    GraphTerm::LoopIn { .. } | GraphTerm::LoopOut { .. }
                ) {
                    if *new_level != *curr_level {
                        display_graph(graph);
                        panic!("incorrect levels");
                    }
                }
            }

            // if graph
            //     .neighbors_directed(node, Direction::Incoming)
            //     .next()
            //     .is_none()
            //     && !matches!(graph.node_weight(node).unwrap().0, GraphTerm::SMEM)
            // {
            //     if *curr_level != 0 {
            //         display_graph(graph);
            //         panic!("Inputs must have level 0, found {curr_level}");
            //     }
            // }
        }
    }
}

pub fn build_search_space(
    graph: &StableGraph<GraphTerm, (), Directed>,
    iters: usize,
) -> egraph_serialize::EGraph {
    let (rendered, root) = render_egglog(graph, "t");
    if option_env!("PRINT_EGGLOG").is_some() {
        println!("{rendered}");
        // println!("{}", render_egglog(graph, "a").0);
    }
    let code = include_str!("code.lisp");

    let mut final_code = code
        .replace("{code}", &rendered)
        .replace("{iters}", &iters.to_string());
    if option_env!("SAVE_EGGLOG").is_some() {
        std::fs::write("egglog.txt", &final_code).unwrap();
    }
    let serialized = run_egglog_program(&final_code, &root).unwrap();
    if option_env!("DEBUG").is_some() || option_env!("PRINT_EGGLOG").is_some() {
        println!("Done building search space.");
    }
    serialized
}

pub fn render_egglog(
    graph: &StableGraph<GraphTerm, (), Directed>,
    prefix: &str,
) -> (String, String) {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    // 1. Topo-order with tie-break: lower NodeIndex first
    let mut indeg: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|n| (n, graph.neighbors_directed(n, Direction::Incoming).count()))
        .collect();

    let mut ready: BinaryHeap<(Reverse<usize>, NodeIndex)> = BinaryHeap::new();
    for (n, &d) in &indeg {
        if d == 0 {
            ready.push((Reverse(n.index()), *n));
        }
    }

    let mut topo_order: Vec<NodeIndex> = Vec::with_capacity(indeg.len());
    while let Some((_, n)) = ready.pop() {
        topo_order.push(n);
        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            let e = indeg.get_mut(&succ).unwrap();
            *e -= 1;
            if *e == 0 {
                ready.push((Reverse(succ.index()), succ));
            }
        }
    }

    // 2. Map <node-id> → <egglog var name>
    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut next_id = 0usize;
    let mut out = String::new();

    // helper to fetch operand text (sort lower-id edges first)
    let operand = |n: NodeIndex,
                   names: &HashMap<NodeIndex, String>,
                   g: &StableGraph<GraphTerm, (), Directed>|
     -> Vec<String> {
        g.edges_directed(n, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| names[&e.source()].clone())
            .collect()
    };

    for n in topo_order {
        let var = format!("{prefix}{next_id}");
        next_id += 1;
        let code = match &graph[n] {
            GraphTerm::GMEM { label } => {
                format!("(GMEM \"{label}\")")
            }
            GraphTerm::SMEM => "(SMEM)".into(),

            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => {
                let [ref src] = operand(n, &names, &graph)[..] else {
                    panic!("LoopIn expects 1 child");
                };
                format!(
                    "(LoopIn {src} (Loop \"{marker}\" {}) {})",
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => {
                let [ref body] = operand(n, &names, &graph)[..] else {
                    panic!("LoopOut expects 1 child");
                };
                format!(
                    "(LoopOut {body} (Loop \"{marker}\" {}) {})",
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }
            GraphTerm::TCMatmul {
                a_k_stride,
                b_k_stride,
                a_inner_stride,
                b_inner_stride,
                c_inner_stride,
                k_outer_loops,
            } => {
                let [ref a, ref b] = operand(n, &names, &graph)[..] else {
                    panic!("LoopOut expects 1 child");
                };
                format!(
                    "(TCMatmul {a} {b} {} {} {} {} {} {})",
                    a_k_stride.to_egglog(),
                    b_k_stride.to_egglog(),
                    a_inner_stride.to_egglog(),
                    b_inner_stride.to_egglog(),
                    c_inner_stride.to_egglog(),
                    k_outer_loops.to_egglog()
                )
            }
            GraphTerm::Custom(_) => "(Custom)".into(),
            GraphTerm::Diff(_) => "(Diff)".into(),
            GraphTerm::Break => "(Break)".into(),

            GraphTerm::Add
            | GraphTerm::Mul
            | GraphTerm::Max
            | GraphTerm::Exp2
            | GraphTerm::Log2
            | GraphTerm::Mod
            | GraphTerm::LessThan
            | GraphTerm::Recip
            | GraphTerm::Sin
            | GraphTerm::Neg
            | GraphTerm::Sqrt
            | GraphTerm::SMEMLoad
            | GraphTerm::SMEMRead => {
                let mut ops = operand(n, &names, &graph);
                let op = match &graph[n] {
                    GraphTerm::Add => "Add",
                    GraphTerm::Mul => "Mul",
                    GraphTerm::Max => "Max",
                    GraphTerm::Exp2 => "Exp2",
                    GraphTerm::Log2 => "Log2",
                    GraphTerm::Recip => "Recip",
                    GraphTerm::Sin => "Sin",
                    GraphTerm::Neg => "Neg",
                    GraphTerm::Sqrt => "Sqrt",
                    GraphTerm::Mod => "Mod",
                    GraphTerm::LessThan => "LessThan",
                    GraphTerm::SMEMLoad => "SMEMLoad",
                    GraphTerm::SMEMRead => "SMEMRead",
                    _ => unreachable!(),
                };
                if ops.len() == 1 {
                    format!("(Unary ({op}) {})", ops.pop().unwrap())
                } else {
                    format!("(Binary ({op}) {})", ops.join(" "))
                }
            }
        };

        out.push_str(&format!("(let {var} {code})\n"));
        names.insert(n, var);
    }

    let root = graph
        .node_indices()
        .find(|&idx| {
            graph
                .neighbors_directed(idx, Direction::Outgoing)
                .next()
                .is_none()
        })
        .and_then(|idx| names.get(&idx))
        .cloned()
        .unwrap_or_else(|| "t0".into());
    (out, root)
}

pub fn render_egglog_inline(
    graph: &StableGraph<GraphTerm, (), Directed>,
    no_loop_markers: bool,
) -> String {
    fn render_node(
        n: NodeIndex,
        graph: &StableGraph<GraphTerm, (), Directed>,
        cache: &mut HashMap<NodeIndex, String>,
        no_loop_markers: bool,
    ) -> String {
        if let Some(expr) = cache.get(&n) {
            return expr.clone();
        }
        // recurse into operands
        let children: Vec<String> = graph
            .neighbors_directed(n, Direction::Incoming)
            .sorted_by_key(|c| graph.find_edge(*c, n).unwrap().index())
            .map(|c| render_node(c, graph, cache, no_loop_markers))
            .collect();

        let expr = match &graph[n] {
            GraphTerm::GMEM { label } => format!("(GMEM \"{label}\")"),
            GraphTerm::SMEM => "(SMEM)".into(),
            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => {
                let src = &children[0];
                format!(
                    "(LoopIn {src} (Loop \"{}\" {}) {})",
                    if no_loop_markers {
                        "".to_string()
                    } else {
                        marker.to_string()
                    },
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => {
                let body = &children[0];
                format!(
                    "(LoopOut {body} (Loop \"{}\" {}) {})",
                    if no_loop_markers {
                        "".to_string()
                    } else {
                        marker.to_string()
                    },
                    range.to_egglog(),
                    stride.to_egglog()
                )
            }
            GraphTerm::TCMatmul {
                a_k_stride,
                b_k_stride,
                a_inner_stride,
                b_inner_stride,
                c_inner_stride,
                k_outer_loops,
            } => {
                let a = &children[0];
                let b = &children[1];
                format!(
                    "(TCMatmul {a} {b} {} {} {} {} {} {})",
                    a_k_stride.to_egglog(),
                    b_k_stride.to_egglog(),
                    a_inner_stride.to_egglog(),
                    b_inner_stride.to_egglog(),
                    c_inner_stride.to_egglog(),
                    k_outer_loops.to_egglog()
                )
            }
            GraphTerm::Custom(_) => "(Custom)".into(),
            GraphTerm::Diff(_) => "(Diff)".into(),
            GraphTerm::Break => "(Break)".into(),
            GraphTerm::Add
            | GraphTerm::Mul
            | GraphTerm::Max
            | GraphTerm::Exp2
            | GraphTerm::Log2
            | GraphTerm::Mod
            | GraphTerm::LessThan
            | GraphTerm::Recip
            | GraphTerm::Sin
            | GraphTerm::Neg
            | GraphTerm::Sqrt
            | GraphTerm::SMEMLoad
            | GraphTerm::SMEMRead => {
                let op = match &graph[n] {
                    GraphTerm::Add => "Add",
                    GraphTerm::Mul => "Mul",
                    GraphTerm::Max => "Max",
                    GraphTerm::Exp2 => "Exp2",
                    GraphTerm::Log2 => "Log2",
                    GraphTerm::Recip => "Recip",
                    GraphTerm::Sin => "Sin",
                    GraphTerm::Neg => "Neg",
                    GraphTerm::Sqrt => "Sqrt",
                    GraphTerm::Mod => "Mod",
                    GraphTerm::LessThan => "LessThan",
                    GraphTerm::SMEMLoad => "SMEMLoad",
                    GraphTerm::SMEMRead => "SMEMRead",
                    _ => unreachable!(),
                };
                if children.len() == 1 {
                    format!("({op} {})", children[0])
                } else {
                    format!("({op} {})", children.join(" "))
                }
            }
        };
        cache.insert(n, expr.clone());
        expr
    }

    // find sink node
    let root = graph
        .node_indices()
        .find(|&idx| {
            graph
                .neighbors_directed(idx, Direction::Outgoing)
                .next()
                .is_none()
        })
        .expect("no sink node");
    render_node(root, graph, &mut HashMap::new(), no_loop_markers)
}

/// Runs an Egglog program from a string and returns its output messages.
fn run_egglog_program(code: &str, root: &str) -> Result<egraph_serialize::EGraph, Error> {
    // Create a fresh EGraph with all the defaults
    let mut egraph = egglog_experimental::new_experimental_egraph();
    let commands = egraph.parser.get_program_from_string(None, code)?;
    let start = std::time::Instant::now();
    let msgs = egraph.run_program(commands)?;
    if option_env!("PRINT_EGGLOG")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!("Took {}ms", start.elapsed().as_millis());
        println!("Run Report:  {}", egraph.get_overall_run_report());
    }
    let (sort, value) = egraph.eval_expr(&egglog::var!(root))?;
    // let (_petgraph, _root_idx) = dag_to_petgraph(&termdag, termdag.lookup(&root));
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        ..Default::default()
    });
    if option_env!("PRINT_EGGLOG")
        .map(|s| s.parse::<i32>().map(|i| i == 1).unwrap_or_default())
        .unwrap_or_default()
    {
        println!(
            "Nodes: {} Roots: {} Class Data: {}",
            s.egraph.nodes.len(),
            s.egraph.root_eclasses.len(),
            s.egraph.class_data.len()
        );
        println!("Messages:");
        for m in msgs {
            println!("{m}");
        }
    }
    Ok(s.egraph)
}

pub fn print_kernels(kernels: &StableGraph<Kernel, (usize, usize), Directed>) -> String {
    let mut s = format!("Kernels: {}", kernels.node_count() - 2);
    for (i, node) in toposort(&kernels, None).unwrap().into_iter().enumerate() {
        let Kernel {
            code,
            grid,
            threadblock,
            smem,
            outputs,
        } = kernels.node_weight(node).unwrap();
        if !code.starts_with("Inputs") && code != "Outputs" {
            s.push_str(&format!(
                "\nKernel {i} Grid: {grid:?} Threadblock: {threadblock:?} Smem: {smem}"
            ));
            s.push_str(&format!("\n{code}"));
            s.push_str(&format!("\nOutputs: {:?}", outputs));
        }
    }
    s
}

use crossterm::{
    cursor::Show,
    event::{self, Event, KeyCode},
    execute,
    terminal::{self, LeaveAlternateScreen},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Gauge, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
    },
};
use std::{
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

enum Msg {
    Set(u16),
    Text(String),
    Title(String),
    Exit,
}

fn expand_tabs(input: &str, tabstop: usize) -> String {
    let mut out = String::with_capacity(input.len());
    let mut col = 0usize;
    for ch in input.chars() {
        match ch {
            '\n' => {
                out.push(ch);
                col = 0;
            }
            '\t' => {
                let spaces = tabstop - (col % tabstop);
                for _ in 0..spaces {
                    out.push(' ');
                }
                col += spaces;
            }
            _ => {
                out.push(ch);
                col += unicode_width::UnicodeWidthChar::width(ch)
                    .unwrap_or(1)
                    .max(1);
            }
        }
    }
    out
}

pub fn search_ui() -> (
    impl Fn(u16) + Send + 'static,    // set_progress
    impl Fn(String) + Send + 'static, // set_log_text
    impl Fn(String) + Send + 'static, // set_title
    impl Fn() + Send + 'static,       // exit (blocks until teardown done)
) {
    let (tx, rx) = mpsc::channel::<Msg>();

    // one-shot to wait for teardown
    let (done_tx, done_rx) = mpsc::channel::<()>();

    let set_progress = {
        let tx = tx.clone();
        move |pct: u16| {
            let _ = tx.send(Msg::Set(pct.min(100)));
        }
    };
    let set_log_text = {
        let tx = tx.clone();
        move |text: String| {
            let _ = tx.send(Msg::Text(text));
        }
    };
    let set_title = {
        let tx = tx.clone();
        move |title: String| {
            let _ = tx.send(Msg::Title(title));
        }
    };

    let exit = {
        let tx = tx.clone();
        move || {
            let _ = tx.send(Msg::Exit);
            // wait for UI thread to finish teardown
            let _ = done_rx.recv();
        }
    };

    std::thread::spawn(move || {
        // setup
        let mut stdout = std::io::stdout();
        let _ = crossterm::terminal::enable_raw_mode();
        let _ = crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen);
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        let green = ratatui::style::Style::default().fg(ratatui::style::Color::Green);

        let mut progress: u16 = 0;
        let mut log_text = String::new();
        let mut title = String::from("Logs");
        let mut quit = false;

        'ui: loop {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Set(p) => progress = p,
                    Msg::Text(t) => log_text = expand_tabs(&t, 8),
                    Msg::Title(t) => title = t,
                    Msg::Exit => break 'ui,
                }
            }

            if crossterm::event::poll(std::time::Duration::from_millis(8)).unwrap_or(false) {
                if let Ok(crossterm::event::Event::Key(k)) = crossterm::event::read() {
                    if k.code == KeyCode::Char('c')
                        && k.modifiers.contains(event::KeyModifiers::CONTROL)
                    {
                        quit = true;
                        break 'ui;
                    }
                }
            }

            let _ = terminal.draw(|f| {
                let chunks = ratatui::layout::Layout::default()
                    .direction(ratatui::layout::Direction::Vertical)
                    .constraints([
                        ratatui::layout::Constraint::Min(1),
                        ratatui::layout::Constraint::Length(3),
                    ])
                    .split(f.area());

                let log_widget = ratatui::widgets::Paragraph::new(log_text.clone())
                    .block(
                        ratatui::widgets::Block::default()
                            .borders(ratatui::widgets::Borders::ALL)
                            .title(ratatui::text::Span::styled(
                                title.clone(),
                                green.add_modifier(ratatui::style::Modifier::BOLD),
                            ))
                            .border_style(green)
                            .title_style(green),
                    )
                    .style(green)
                    .wrap(ratatui::widgets::Wrap { trim: false });
                f.render_widget(log_widget, chunks[0]);

                let gauge = ratatui::widgets::Gauge::default()
                    .block(
                        ratatui::widgets::Block::default()
                            .borders(ratatui::widgets::Borders::ALL)
                            .title(ratatui::text::Span::styled(
                                "Progress",
                                green.add_modifier(ratatui::style::Modifier::BOLD),
                            ))
                            .border_style(green)
                            .title_style(green),
                    )
                    .gauge_style(green)
                    .ratio((progress as f64 / 100.0).clamp(0.0, 1.0))
                    .label(format!("{progress}%"));
                f.render_widget(gauge, chunks[1]);
            });
        }

        // teardown (ordered + flush)
        // 1) drop Terminal to flush backend buffers
        drop(terminal);
        // 2) disable raw mode
        let _ = crossterm::terminal::disable_raw_mode();
        // 3) leave alt screen + show cursor
        let mut out = std::io::stdout();
        let _ = crossterm::execute!(
            out,
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::cursor::Show
        );
        let _ = out.flush();

        // signal we're done so exit() can return
        let _ = done_tx.send(());
        if quit {
            std::process::exit(0);
        }
    });

    (set_progress, set_log_text, set_title, exit)
}

pub fn generate_proof(
    graph1: &StableGraph<GraphTerm, (), Directed>,
    graph2: &StableGraph<GraphTerm, (), Directed>,
) {
    let egglog1 = render_egglog_inline(graph1, true);
    let egglog2 = render_egglog_inline(graph2, true);
    let mut egraph = egglog_proof::EGraph::with_tracing();
    egraph
        .parse_and_run_program(
            None,
            &include_str!("code.lisp")
                .split("\n")
                .take_while(|l| *l != "{code}")
                .join("\n"),
        )
        .unwrap();
    let expr1 = egraph.parser.get_expr_from_string(None, &egglog1).unwrap();
    let expr2 = egraph.parser.get_expr_from_string(None, &egglog2).unwrap();
    let (_, v1) = egraph.eval_expr(&expr1).unwrap();
    let (_, v2) = egraph.eval_expr(&expr2).unwrap();

    egraph
        .parse_and_run_program(
            None,
            "
(run-schedule
	(run ir-generic)
	(repeat 3
		(run ir)
		(repeat 3 ir-prop)
		(repeat 3 expr)
		(run ir-generic)
	)
)

(ruleset loop-blank)
(rewrite (Loop ?s ?r) (Loop \"\" ?r) :ruleset loop-blank)
(run-schedule (run loop-blank))
    ",
        )
        .unwrap();
    if let Ok(proof) = egraph.backend.explain_terms_equal(v1, v2) {
        println!("Proof: {:#?}", proof);
    } else {
        println!("Couldn't find proof");
        println!("First Expression: {egglog1}");
        println!("\n\nSecond Expression: {egglog2}");
    }
}
