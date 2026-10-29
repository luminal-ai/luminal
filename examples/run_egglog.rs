//! Run a self-contained .egg file with Luminal's registered primitives.
//! cargo run --release --example run_egglog -- path/to/repro.egg

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("expected a .egg file path")?;
    let source = std::fs::read_to_string(&path)?;
    let mut graph = luminal::egglog_snippet::new_egraph();
    let start = std::time::Instant::now();
    graph.parse_and_run_program(Some(path), &source)?;
    println!(
        "completed in {:.3}s; {} tuples",
        start.elapsed().as_secs_f64(),
        graph.num_tuples()
    );
    Ok(())
}
