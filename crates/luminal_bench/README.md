# `luminal_bench`

Benchmarks and debugging utilities for Luminal (Criterion benchmarks + egglog lowering debug).

## Running Benchmarks

The benches in this crate are typically run with the Metal backend enabled via a feature flag.

```bash
# L1: micro (single op / HLIR primitive)
cargo bench -p luminal_bench --features metal --bench micro

# L2: patterns (composed patterns)
cargo bench -p luminal_bench --features metal --bench patterns
```

### Outputs (Criterion)

After running, common outputs are under:

- HTML report: `target/criterion/report/index.html`
- micro metrics mapping: `target/criterion/bench_metrics.json`
- micro full report: `target/criterion/bench_report.json`
- patterns metrics mapping: `target/criterion/pattern_metrics.json`
- patterns full report: `target/criterion/pattern_report.json`

These JSON files (constant metrics such as bytes/flops) can be combined with Criterion timing to
compute derived throughput metrics (MBU/MFU/etc.).

## Coverage (Overview)

### L1 micro (single op)

Measures single-op performance for HLIR primitives (currently includes):

- Unary: `Exp2` / `Log2` / `Sin` / `Recip` / `Sqrt`
- Binary: `Add` / `Mul` / `Mod` / `LessThan`
- Indexing: `Gather` / `Cast`
- Reduction: `Sum` / `Max`

### L2 patterns (composed patterns)

Covers common composed patterns (currently includes):

- `MatMul`
- `Softmax`
- `GeLU`
- `Attention`
- `LayerNorm` (currently skipped in the Metal bench: requires unsupported HLIR primitives)

## egglog Debug Tool: `debug_ops`

`examples/debug_ops.rs` is a general egglog / lowering debug tool to help diagnose:

- Why a particular HLIR op failed to lower into backend dialect ops (and cleanup triggers
  `No valid graphs present in the e-graph!`)
- Why a particular egglog function fact (e.g. `dtype`) is missing for some nodes

### Common Commands (Metal examples)

```bash
# Default: print summaries (HLIR/egglog op counts + root) and try build_search_space
# (which prints egglog rule match counts)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner

# Explicit op coverage check: provide HLIR:Backend mapping(s)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --inspect-op Add:MetalAdd

# Print full analysis output (HLIR-only + Backend+HLIR)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --analyze --inspect-op Add:MetalAdd

# Trace an egglog function fact for a specific var (HLIR-only)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --trace-fact dtype t24

# Scan all vars whose op-head is Add, find the first missing dtype, then trace it (HLIR-only)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner \
  --trace-first-missing-fact dtype --within-op Add

# Inspect a var's eclass/enodes/children and dtype facts (HLIR-only)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --inspect-var t24

# Dump the raw egglog program (the `(let tN ...)` program from `hlir_to_egglog`)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner \
  --dump-egglog target/gelu-inner.egg

# Export structured JSON (useful for repro/diffing)
cargo run -p luminal_bench --features metal --example debug_ops -- --case gelu-inner --json target/debug_ops.json
```

Notes:
- `--trace-fact` can only evaluate functions that exist in the egglog program (e.g. `dtype`).
  Many values such as shape/strides are encoded as IR term parameters, not as function facts.

For more options, see:

```bash
cargo run -p luminal_bench --features metal --example debug_ops -- --help
```
