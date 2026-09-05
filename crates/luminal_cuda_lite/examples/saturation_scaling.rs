//! HOW SATURATION SCALES WITH MODEL SIZE — a script, not a test
//! (2026-09-05).
//!
//! It grows a released model ONE TRANSFORMER LAYER AT A TIME, records
//! the same graph the CUDA-lite application records, saturates it
//! through [`luminal_cuda_lite::CudaRuntime::saturation_stats`], and
//! prints what egglog itself says the run cost: iterations, per-ruleset
//! search+apply/merge/rebuild, the slowest rules with their match
//! counts, and three cuts of the e-graph's size. No search runs — the
//! genetic loop, the extractor and the planner are all off the clock,
//! so every number here belongs to saturation.
//!
//! THE POINT IS THE TREND, not any single N. The closing table fits the
//! measured points three ways — a power law `a·N^b`, an exponential
//! `a·e^(bN)`, and a line `c + m·N` — reports each fit's residual, and
//! extrapolates all three to the full model's layer count, which is the
//! question a scaling study is actually asked: does a 36-layer
//! saturation cost 36 times a 1-layer one, or 2^36? THE RESIDUALS ARE
//! THE ANSWER, not the extrapolations: three curves through four points
//! agree on the points and disagree wildly at N=36, so the honest use
//! of this table is to read which fit the measurements actually pick
//! out — and, when the range is short, to say that they do not.
//!
//! HOST-ONLY. Nothing here touches a device; `--features device` is not
//! needed and would change nothing.
//!
//! Run:
//!   cargo run --release -p luminal_cuda_lite --example saturation_scaling \
//!     -- --model qwen3 --layers 1,2,3,4 --cap-secs 300

use anyhow::{Result, bail};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal_cuda_lite::{CudaRuntime, SaturationStats};

/// The KV-cache slot count the applications record with
/// (`crates/luminal_cuda_lite/examples/qwen3.rs`). Kept identical so
/// this study's graph is the application's graph with a different layer
/// count, and nothing else.
const SLOTS: usize = 4;

/// The full-size layer counts the trend extrapolates to.
const QWEN3_4B_LAYERS: usize = 36;
const GEMMA3_4B_LAYERS: usize = 34;

fn main() {
    if let Err(error) = run() {
        eprintln!("saturation_scaling: FAIL: {error:#}");
        std::process::exit(1);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    Qwen3,
    Gemma3,
}

impl Model {
    fn name(self) -> &'static str {
        match self {
            Model::Qwen3 => "qwen3",
            Model::Gemma3 => "gemma3",
        }
    }

    fn full_layers(self) -> usize {
        match self {
            Model::Qwen3 => QWEN3_4B_LAYERS,
            Model::Gemma3 => GEMMA3_4B_LAYERS,
        }
    }
}

struct Options {
    model: Model,
    layers: Vec<usize>,
    cap_secs: f64,
    extract: bool,
    seed: u64,
    /// How many characters of a rule's name print. MOST OF OUR RULES
    /// ARE ANONYMOUS, so egglog names them by their whole source text,
    /// and the arms of a rule FAMILY share a long common prefix — the
    /// cuBLASLt marker rules agree for ~200 characters and differ only
    /// in their `CoordVar` indices. 100 shows which family; naming the
    /// exact arm needs several hundred.
    rule_chars: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: Model::Qwen3,
            layers: vec![1, 2, 3, 4],
            cap_secs: 300.0,
            extract: false,
            seed: 0,
            rule_chars: 100,
        }
    }
}

fn parse_args() -> Result<Options> {
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--model" => {
                options.model = match value()?.as_str() {
                    "qwen3" => Model::Qwen3,
                    "gemma3" => Model::Gemma3,
                    other => bail!("unknown model '{other}' (qwen3 | gemma3)"),
                }
            }
            "--layers" => {
                options.layers = value()?
                    .split(',')
                    .map(|n| n.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
                if options.layers.contains(&0) {
                    bail!("--layers must be positive");
                }
            }
            "--cap-secs" => options.cap_secs = value()?.parse()?,
            "--seed" => options.seed = value()?.parse()?,
            "--rule-chars" => options.rule_chars = value()?.parse()?,
            "--extract" => options.extract = true,
            "--help" | "-h" => {
                println!(
                    "saturation_scaling [--model qwen3|gemma3] [--layers 1,2,3,4] \
                     [--cap-secs 300] [--seed 0] [--rule-chars 100] [--extract]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument '{other}' (try --help)"),
        }
    }
    if options.layers.is_empty() {
        bail!("--layers listed nothing");
    }
    Ok(options)
}

/// ONE MEASURED POINT.
struct Row {
    layers: usize,
    saturation_ms: f64,
    serialize_ms: f64,
    iterations: usize,
    tuples: usize,
    classes: usize,
    nodes: usize,
    top_ruleset: String,
    top_rule: String,
    extract_ms: Option<f64>,
}

fn run() -> Result<()> {
    let options = parse_args()?;
    println!(
        "saturation_scaling: model {} | layers {:?} | cap {:.0}s | extract {}",
        options.model.name(),
        options.layers,
        options.cap_secs,
        options.extract
    );
    // THE SCHEDULE THE NUMBERS BELONG TO: printed once so a reader can
    // map the per-ruleset rows below onto the strata that ran them.
    println!(
        "saturation_scaling: run-schedule {}",
        luminal_cuda_lite::CudaBindings::SCHEDULE.trim()
    );

    let mut rows: Vec<Row> = Vec::new();
    for layers in &options.layers {
        let layers = *layers;
        let build_start = std::time::Instant::now();
        let cx = build_graph(options.model, layers)?;
        let build_ms = build_start.elapsed().as_secs_f64() * 1e3;

        let mut runtime = CudaRuntime::load(&cx)?;
        // The applications' dyn discipline: pin every dynamic variable
        // the recorder minted to the value the harness recorded it at.
        let mut vars: Vec<_> = cx.dyn_map.iter().collect();
        vars.sort();
        for (var, value) in vars {
            runtime.bind_dyn_range(*var, *value as u64, *value as u64)?;
        }

        let stats = runtime.saturation_stats()?;
        let extract_ms = if options.extract {
            let extraction = runtime.extraction_stats(&stats.serialized, options.seed)?;
            print_extraction(layers, &extraction);
            Some(extraction.extract_nanos as f64 / 1e6)
        } else {
            None
        };
        print_block(options.model, layers, build_ms, &stats, options.rule_chars);

        let saturation_ms = stats.saturation_nanos as f64 / 1e6;
        rows.push(Row {
            layers,
            saturation_ms,
            serialize_ms: stats.serialize_nanos as f64 / 1e6,
            iterations: stats.iterations,
            tuples: stats.num_tuples,
            classes: stats.classes,
            nodes: stats.nodes,
            top_ruleset: stats
                .report
                .rulesets
                .first()
                .map(|r| ruleset_name(&r.name).to_string())
                .unwrap_or_else(|| "-".to_string()),
            top_rule: stats
                .report
                .top_rules
                .first()
                .map(|r| flatten(&r.name, 48))
                .unwrap_or_else(|| "-".to_string()),
            extract_ms,
        });

        if saturation_ms / 1e3 > options.cap_secs {
            println!(
                "\nsaturation_scaling: STOP — layers={layers} saturation {:.1}s exceeded the \
                 {:.0}s cap; not growing further.",
                saturation_ms / 1e3,
                options.cap_secs
            );
            break;
        }
    }

    print_trend(options.model, &rows);
    Ok(())
}

/// The application graph at `layers` transformer layers — the same
/// recording sequence as `examples/{qwen3,gemma3}.rs`, minus the host
/// tables and seeded payloads, which saturation never reads.
fn build_graph(model: Model, layers: usize) -> Result<Graph> {
    let mut cx = Graph::new();
    match model {
        Model::Qwen3 => {
            use qwen3::{
                Qwen, QwenDims,
                model_support::{Namespace, named_kv_cache_pool},
            };
            let dims = QwenDims {
                layers,
                ..QwenDims::qwen3_4b()
            };
            let model = Qwen::init(&mut cx, &dims);
            let token = cx.tensor(1, DType::Int);
            let q_pos = cx.tensor(1, DType::Int);
            let rope_cos = cx.tensor((1, dims.head_dim), DType::F32);
            let rope_sin = cx.tensor((1, dims.head_dim), DType::F32);
            let rope_rot = cx.tensor((dims.head_dim, dims.head_dim), DType::F32);
            let gather_idx = cx.tensor(SLOTS, DType::Int);
            let scatter_idx = cx.tensor(1, DType::Int);
            let pool = named_kv_cache_pool(
                &mut cx,
                dims.layers,
                SLOTS,
                dims.kv_dim(),
                DType::F32,
                &Namespace::root().child("cache"),
            );
            let (logits, _) = model.forward(
                token,
                q_pos,
                rope_cos,
                rope_sin,
                rope_rot,
                &pool.layers,
                gather_idx,
                scatter_idx,
            );
            let _ = logits.output();
        }
        Model::Gemma3 => {
            use gemma3::{
                Gemma3, Gemma3Dims,
                model_support::{Namespace, named_kv_cache_pool},
            };
            let dims = Gemma3Dims {
                layers,
                ..Gemma3Dims::gemma3_4b()
            };
            let model = Gemma3::init(&mut cx, &dims);
            let token = cx.tensor(1, DType::Int);
            let q_pos = cx.tensor(1, DType::Int);
            let local_cos = cx.tensor((1, dims.head_dim), DType::F32);
            let local_sin = cx.tensor((1, dims.head_dim), DType::F32);
            let global_cos = cx.tensor((1, dims.head_dim), DType::F32);
            let global_sin = cx.tensor((1, dims.head_dim), DType::F32);
            let rope_rot = cx.tensor((dims.head_dim, dims.head_dim), DType::F32);
            let gather_idx = cx.tensor(SLOTS, DType::Int);
            let scatter_idx = cx.tensor(1, DType::Int);
            let pool = named_kv_cache_pool(
                &mut cx,
                dims.layers,
                SLOTS,
                dims.kv_dim(),
                DType::F32,
                &Namespace::root().child("cache"),
            );
            let (logits, _) = model.forward(
                token,
                q_pos,
                (local_cos, local_sin),
                (global_cos, global_sin),
                rope_rot,
                &pool,
                gather_idx,
                scatter_idx,
            );
            let _ = logits.output();
        }
    }
    Ok(cx)
}

/// egglog names the ruleset carrying no `:ruleset` with the empty
/// string. Say so.
fn ruleset_name(name: &str) -> &str {
    if name.is_empty() { "(default)" } else { name }
}

/// A rule name on one line at most `width` chars. Most of our rules are
/// ANONYMOUS, so egglog names them by their whole source text.
fn flatten(name: &str, width: usize) -> String {
    let flat = name.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(width) {
        Some((cut, _)) => format!("{}...", &flat[..cut]),
        None => flat,
    }
}

fn print_block(
    model: Model,
    layers: usize,
    build_ms: f64,
    stats: &SaturationStats,
    rule_chars: usize,
) {
    let ms = |n: u128| n as f64 / 1e6;
    println!(
        "\n===== {} layers={layers} =====",
        model.name().to_uppercase()
    );
    println!(
        "  record        {build_ms:>10.0}ms\n  \
         saturation    {:>10.0}ms\n  \
         serialize     {:>10.0}ms",
        ms(stats.saturation_nanos),
        ms(stats.serialize_nanos)
    );
    println!(
        "  size          iterations {} | tuples {} | classes {} | nodes {}",
        stats.iterations, stats.num_tuples, stats.classes, stats.nodes
    );
    println!(
        "  rules         {} timed, {:.0}ms summed over rules",
        stats.report.rules_reported,
        ms(stats.report.rule_total_nanos)
    );
    for ruleset in &stats.report.rulesets {
        println!(
            "  ruleset {:<26} search+apply {:>9.0}ms  merge {:>7.0}ms  rebuild {:>8.0}ms",
            ruleset_name(&ruleset.name),
            ms(ruleset.search_and_apply_nanos),
            ms(ruleset.merge_nanos),
            ms(ruleset.rebuild_nanos)
        );
    }
    for (rank, rule) in stats.report.top_rules.iter().enumerate() {
        println!(
            "  rule #{:<2} {:>9.0}ms {:>12} matches  {}",
            rank + 1,
            ms(rule.search_and_apply_nanos),
            rule.matches,
            flatten(&rule.name, rule_chars)
        );
    }
    for (rank, (op, count)) in stats.nodes_per_constructor.iter().take(15).enumerate() {
        println!("  ctor #{:<2} {count:>10} nodes  {op}", rank + 1);
    }
    for (rank, (name, size)) in stats.function_sizes.iter().take(15).enumerate() {
        println!("  func #{:<2} {size:>10} rows   {name}", rank + 1);
    }
}

fn print_extraction(layers: usize, stats: &luminal_cuda_lite::ExtractionStats) {
    let ms = |n: u128| n as f64 / 1e6;
    println!(
        "\n  extract layers={layers}: analysis {:.0}ms | space {:.0}ms | sample {:.0}ms | \
         extract {:.0}ms (discovery {:.0}ms, relax {:.0}ms over {} pass(es), assemble {:.0}ms) | \
         producer classes {} | extracted nodes {}",
        ms(stats.analysis_nanos),
        ms(stats.space_nanos),
        ms(stats.sample_nanos),
        ms(stats.extract_nanos),
        ms(stats.sub.discovery_nanos),
        ms(stats.sub.relax_nanos),
        stats.sub.relax_passes,
        ms(stats.sub.assemble_nanos),
        stats.producer_classes,
        stats
            .extracted_nodes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "refused".to_string())
    );
}

/// A LEAST-SQUARES FIT of `y = a·exp(b·u)`, done on `ln y` (so it is a
/// straight-line fit in the transformed space, not a nonlinear solve).
/// With `u = ln N` this is the power law `a·N^b`; with `u = N` it is the
/// exponential `a·e^(bN)`.
///
/// The residual reported is RMS in LOG SPACE, which is the honest one
/// for a fit done there: a 10% miss costs the same whether it is on the
/// smallest point or the largest. `max_rel` re-expresses the worst
/// single point as a percentage, for readers who want a number in the
/// units they measured.
struct Fit {
    a: f64,
    b: f64,
    rms_log: f64,
    max_rel: f64,
}

impl Fit {
    fn at(&self, u: f64) -> f64 {
        self.a * (self.b * u).exp()
    }
}

/// Fit `points` as `(u, y)`; `None` if fewer than two distinct `u` or any
/// non-positive `y` (the log is undefined and a zero measurement is not
/// a measurement).
fn fit(points: &[(f64, f64)]) -> Option<Fit> {
    if points.len() < 2 || points.iter().any(|(_, y)| *y <= 0.0) {
        return None;
    }
    let n = points.len() as f64;
    let mean_u = points.iter().map(|(u, _)| *u).sum::<f64>() / n;
    let mean_ly = points.iter().map(|(_, y)| y.ln()).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (u, y) in points {
        cov += (u - mean_u) * (y.ln() - mean_ly);
        var += (u - mean_u) * (u - mean_u);
    }
    if var == 0.0 {
        return None;
    }
    let b = cov / var;
    let a = (mean_ly - b * mean_u).exp();
    let mut sq = 0.0;
    let mut max_rel: f64 = 0.0;
    for (u, y) in points {
        let resid = y.ln() - (a.ln() + b * u);
        sq += resid * resid;
        max_rel = max_rel.max((resid.abs().exp() - 1.0) * 100.0);
    }
    Some(Fit {
        a,
        b,
        rms_log: (sq / n).sqrt(),
        max_rel,
    })
}

/// AN ORDINARY LEAST-SQUARES LINE, `y = c + m·N`, fitted in LINEAR
/// space (2026-09-05).
///
/// It is here as the THIRD null hypothesis, and specifically as the one
/// the other two cannot express: a fixed cost plus a per-layer cost is
/// what a schedule that parses a base program once and then does a
/// layer's work per layer would look like, and neither a power law nor
/// an exponential can represent a constant term. A power law fitted
/// through `c + m·N` reads back a spuriously SUB-linear exponent,
/// because the constant flattens the log-log slope. Printing all three
/// residuals side by side is what makes the answer readable: the fit
/// with the small residual over a wide N range is the shape, and over a
/// narrow one near N=1 the line and the power law are not
/// distinguishable.
///
/// Its residual is reported IN LOG SPACE like the other two, so the
/// three numbers are comparable.
struct Line {
    c: f64,
    m: f64,
    rms_log: f64,
    max_rel: f64,
}

impl Line {
    fn at(&self, x: f64) -> f64 {
        self.c + self.m * x
    }
}

fn fit_line(points: &[(f64, f64)]) -> Option<Line> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (x, y) in points {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x) * (x - mean_x);
    }
    if var == 0.0 {
        return None;
    }
    let m = cov / var;
    let c = mean_y - m * mean_x;
    let mut sq = 0.0;
    let mut max_rel: f64 = 0.0;
    for (x, y) in points {
        let predicted = c + m * x;
        // A fitted line can dip non-positive on small x; a log residual
        // is undefined there, so the fit reports no residual rather
        // than a wrong one.
        if predicted <= 0.0 || *y <= 0.0 {
            return None;
        }
        let resid = y.ln() - predicted.ln();
        sq += resid * resid;
        max_rel = max_rel.max((resid.abs().exp() - 1.0) * 100.0);
    }
    Some(Line {
        c,
        m,
        rms_log: (sq / n).sqrt(),
        max_rel,
    })
}

/// All three fits for one measured quantity, with the extrapolation to
/// `target` layers.
fn print_fits(label: &str, unit: &str, rows: &[Row], target: usize, value: impl Fn(&Row) -> f64) {
    let power: Vec<(f64, f64)> = rows
        .iter()
        .map(|row| ((row.layers as f64).ln(), value(row)))
        .collect();
    let expo: Vec<(f64, f64)> = rows
        .iter()
        .map(|row| (row.layers as f64, value(row)))
        .collect();
    let t = target as f64;
    match fit(&power) {
        Some(f) => println!(
            "  {label:<12} power law    {:>12.4} * N^{:<8.4}  rms(log) {:.4}  max err {:>6.1}%  \
             → N={target}: {:.1} {unit}",
            f.a,
            f.b,
            f.rms_log,
            f.max_rel,
            f.at(t.ln())
        ),
        None => println!("  {label:<12} power law    (not fittable)"),
    }
    match fit(&expo) {
        Some(f) => println!(
            "  {label:<12} exponential  {:>12.4} * e^({:.4}·N)  rms(log) {:.4}  max err {:>6.1}%  \
             → N={target}: {:.1} {unit}",
            f.a,
            f.b,
            f.rms_log,
            f.max_rel,
            f.at(t)
        ),
        None => println!("  {label:<12} exponential  (not fittable)"),
    }
    match fit_line(&expo) {
        Some(f) => println!(
            "  {label:<12} linear       {:>12.4} + {:.4}*N        rms(log) {:.4}  max err {:>6.1}%  \
             \u{2192} N={target}: {:.1} {unit}",
            f.c,
            f.m,
            f.rms_log,
            f.max_rel,
            f.at(t)
        ),
        None => println!("  {label:<12} linear       (not fittable)"),
    }
    // THE LOCAL SLOPE, from the last two points only. A global fit over
    // a range that spans a fixed startup cost AND an asymptotic regime
    // fits neither, and says so through a large residual; the log-log
    // slope between the two LARGEST measured N is the exponent the
    // curve is actually running at where it matters, and extrapolating
    // from the largest measured point along it is the least
    // unreasonable thing this data supports.
    if let [.., prev, last] = rows {
        let (n0, n1) = (prev.layers as f64, last.layers as f64);
        let (y0, y1) = (value(prev), value(last));
        if n1 > n0 && y0 > 0.0 && y1 > 0.0 {
            let slope = (y1 / y0).ln() / (n1 / n0).ln();
            println!(
                "  {label:<12} local slope  N^{slope:<10.4} over N={}..{}                         \
                 \u{2192} N={target}: {:.1} {unit}",
                prev.layers,
                last.layers,
                y1 * (t / n1).powf(slope)
            );
        }
    }
}

fn print_trend(model: Model, rows: &[Row]) {
    println!("\n===== TREND: {} =====", model.name());
    println!(
        "{:>3} | {:>12} | {:>7} | {:>5} | {:>10} | {:>9} | {:>9} | {:>26} | top rule",
        "N", "saturate ms", "ratio", "iters", "tuples", "classes", "nodes", "top ruleset"
    );
    for (index, row) in rows.iter().enumerate() {
        let ratio = if index == 0 {
            "-".to_string()
        } else {
            format!("{:.2}x", row.saturation_ms / rows[index - 1].saturation_ms)
        };
        println!(
            "{:>3} | {:>12.0} | {:>7} | {:>5} | {:>10} | {:>9} | {:>9} | {:>26} | {}",
            row.layers,
            row.saturation_ms,
            ratio,
            row.iterations,
            row.tuples,
            row.classes,
            row.nodes,
            row.top_ruleset,
            row.top_rule
        );
    }
    if rows.iter().any(|row| row.extract_ms.is_some()) {
        println!("\n  extraction (one sampled genome), ms per N:");
        for row in rows {
            if let Some(ms) = row.extract_ms {
                println!("    N={:<3} {ms:>10.0}ms", row.layers);
            }
        }
    }
    println!("\n  serialize ms per N:");
    for row in rows {
        println!("    N={:<3} {:>10.0}ms", row.layers, row.serialize_ms);
    }

    let target = model.full_layers();
    if rows.len() < 2 {
        println!("\n  (fewer than two points — no fit)");
        return;
    }
    println!(
        "\n  FITS over N in {:?}, extrapolated to the full model (N={target}):",
        rows.iter().map(|row| row.layers).collect::<Vec<_>>()
    );
    print_fits("saturate", "s", rows, target, |row| row.saturation_ms / 1e3);
    print_fits("tuples", "tuples", rows, target, |row| row.tuples as f64);
    print_fits("nodes", "nodes", rows, target, |row| row.nodes as f64);
    println!(
        "\n  (the three fits are least-squares over all {} points — the first two in log space, \
         the line in linear space; the local slope uses only the last two. READ THE RESIDUALS: a \
         fit with a large one is not describing this curve, and none of these are predictions \
         with error bars.)",
        rows.len()
    );
}

#[cfg(test)]
mod tests {
    use super::{fit, fit_line};

    /// The fit recovers a clean power law exactly: y = 3·N^2 sampled at
    /// N = 1..4, fitted on (ln N, y), gives a = 3, b = 2, zero residual.
    #[test]
    fn power_law_fit_recovers_its_own_exponent() {
        let points: Vec<(f64, f64)> = (1..=4)
            .map(|n| ((n as f64).ln(), 3.0 * (n as f64).powi(2)))
            .collect();
        let f = fit(&points).expect("fittable");
        assert!((f.a - 3.0).abs() < 1e-9, "a = {}", f.a);
        assert!((f.b - 2.0).abs() < 1e-9, "b = {}", f.b);
        assert!(f.rms_log < 1e-9, "residual = {}", f.rms_log);
    }

    /// The affine fit recovers a line exactly, and the power law
    /// CANNOT — which is the reason the line is printed beside it: a
    /// constant term reads back as a sub-linear exponent.
    #[test]
    fn line_fit_recovers_an_affine_law_the_power_law_misreads() {
        let points: Vec<(f64, f64)> = (1..=4)
            .map(|n| (n as f64, 100.0 + 7.0 * n as f64))
            .collect();
        let line = fit_line(&points).expect("fittable");
        assert!((line.c - 100.0).abs() < 1e-9, "c = {}", line.c);
        assert!((line.m - 7.0).abs() < 1e-9, "m = {}", line.m);
        assert!(line.rms_log < 1e-12, "residual = {}", line.rms_log);
        let power: Vec<(f64, f64)> = points.iter().map(|(x, y)| (x.ln(), *y)).collect();
        let power = fit(&power).expect("fittable");
        assert!(
            power.b < 0.3,
            "a constant-dominated line must read back as a badly sub-linear exponent, got {}",
            power.b
        );
    }
}
