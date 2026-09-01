//! THE SCRIPT CORPUS GATE (restored 2026-08-14 for the subst-primitive
//! experiment; moved here with the reference registry in Step B): every
//! test_scripts/*.egg is self-driving (carries its own run-schedule and
//! checks), so each runs verbatim against the assembled program on a
//! fresh e-graph. This is the merge-tree home of the old prototype's
//! `cargo run corpus` gate; subst_example P1-P9 and
//! subst_range_guard_example live here and pin the substitution guard
//! semantics.

/// The fixture corpus lives in the WORKSPACE-ROOT Egglog core tree; this
/// crate runs two directories below it.
const SCRIPTS_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../src/egglog_core/test_scripts"
);

#[test]
fn corpus_scripts_all_green() {
    let dir = SCRIPTS_DIR;
    let mut scripts: Vec<_> = std::fs::read_dir(dir)
        .expect("test_scripts dir")
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            name.ends_with(".egg").then_some(name)
        })
        .collect();
    scripts.sort();
    // CORPUS_ONLY=a.egg,b.egg filters to a subset — the targeted-run
    // knob for diagnosing a single script without the whole sweep.
    if let Ok(only) = std::env::var("CORPUS_ONLY") {
        let keep: std::collections::HashSet<&str> = only.split(',').collect();
        scripts.retain(|s| keep.contains(s.as_str()));
    }
    assert!(!scripts.is_empty(), "corpus found no scripts");
    // The corpus assembles against THE REFERENCE REGISTRY — not a
    // superset. Every script here must be expressible in the vocabulary
    // the reference runtime actually ships; a script that needs a
    // view/fused/mutating spelling belongs in that runtime's own fixture
    // tree, gated by that runtime's own corpus. That is a property of
    // the file's HOME, never of a name filter.
    let program_head = luminal::egglog_snippet::assembled_program_for(
        &luminal_reference::ops::built_in_matchers(),
    );
    let mut failures = Vec::new();
    for script in &scripts {
        let started = std::time::Instant::now();
        eprintln!("[corpus] running {script}");
        let source = std::fs::read_to_string(format!("{dir}/{script}")).expect("script readable");
        let program = format!("{program_head}\n\n{source}");
        let mut egraph = luminal::egglog_snippet::new_egraph();
        if let Err(err) = egraph.parse_and_run_program(Some(script.clone()), &program) {
            failures.push(format!("{script}: {err}"));
        }
        eprintln!(
            "[corpus]   {script} done in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
    assert!(
        failures.is_empty(),
        "corpus scripts failed ({}/{}):\n  {}",
        failures.len(),
        scripts.len(),
        failures.join("\n  ")
    );
    eprintln!("[corpus] {} scripts green", scripts.len());
}

/// Dump THE assembled program (core preamble + spliced reference-op
/// snippets — exactly what every run executes) to
/// target/assembled_program.egg under this crate.
/// Run: cargo test -p luminal_reference --release dump_assembled_program -- --ignored --nocapture
#[test]
#[ignore = "utility — run explicitly by name"]
fn dump_assembled_program() {
    let program = luminal_reference::assembled_program();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/assembled_program.egg");
    std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/target")).expect("target dir");
    std::fs::write(path, program).expect("dump written");
    eprintln!("[dump] {} lines -> {path}", program.lines().count());
}
