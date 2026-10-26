//! Compare the two approved accumulating calls through normal CUDA search.
//! Run with `cargo bench -p luminal_cuda_lite --features device --bench
//! cublaslt_accumulation`. Pass `-- --quick` for one seed and seven samples,
//! or `-- --case <name>` to select a workload. `--allow-materialization`
//! includes the production layout-copy candidates. Results are TSV on stdout.
use anyhow::{Context, Result, ensure};
use luminal::bufferize::BufferNode;
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal_cuda_lite::device::{CudaDevice, CudaStaging};
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::CublasLtAccumulateInPlace;
use luminal_cuda_lite::ops::cublaslt::approved::ApprovedCublasLt;
use luminal_cuda_lite::{CompileOptions, CudaBindings, CudaRuntime, HostBuffer};
use std::time::Instant;

#[path = "../tests/support/mod.rs"]
mod support;
use support::{Allocation, TestTransfers};

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    rows: usize,
    width: usize,
    depth: usize,
    writable: bool,
    self_residual: bool,
    keep_intermediates: bool,
}

const CASES: &[Case] = &[
    Case {
        name: "decode_read_only",
        rows: 1,
        width: 1024,
        depth: 4,
        writable: false,
        self_residual: false,
        keep_intermediates: false,
    },
    Case {
        name: "prefill_read_only",
        rows: 128,
        width: 512,
        depth: 4,
        writable: false,
        self_residual: false,
        keep_intermediates: false,
    },
    Case {
        name: "prefill_donated",
        rows: 128,
        width: 512,
        depth: 4,
        writable: true,
        self_residual: false,
        keep_intermediates: false,
    },
    Case {
        name: "deep_donated",
        rows: 128,
        width: 512,
        depth: 8,
        writable: true,
        self_residual: false,
        keep_intermediates: false,
    },
    Case {
        name: "self_residual",
        rows: 128,
        width: 512,
        depth: 4,
        writable: true,
        self_residual: true,
        keep_intermediates: false,
    },
    Case {
        name: "live_intermediates",
        rows: 128,
        width: 512,
        depth: 4,
        writable: true,
        self_residual: false,
        keep_intermediates: true,
    },
    Case {
        name: "large_donated",
        rows: 512,
        width: 512,
        depth: 4,
        writable: true,
        self_residual: false,
        keep_intermediates: false,
    },
];

#[derive(Clone, Copy)]
enum Mode {
    Separate,
    InPlace,
    Both,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::InPlace => "in_place",
            Self::Both => "both",
        }
    }
}

struct Fixture {
    graph: Graph,
    bindings: CudaBindings,
    inputs: FxHashMap<NodeIndex, HostBuffer>,
    outputs: Vec<(NodeIndex, Vec<f32>)>,
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 37 + seed * 101 + i / 13) % 127) as f32 - 63.) / 256.)
        .collect()
}

fn fixture(case: Case) -> Fixture {
    let mut graph = Graph::new();
    let initial = graph.tensor([case.rows, case.width], DType::F32);
    let weight = graph.tensor([case.width, case.width], DType::F32);
    let mut bindings = CudaBindings::new();
    let initial_buffer = bindings.input(initial.id);
    if case.writable {
        bindings.declare(initial_buffer, Access::ReadWrite, FreedBy::Caller);
    }
    bindings.input(weight.id);
    // Dense storage/dispatch, with three nonzeros per column. An independent
    // O(rows * width) reference can then validate every output of deep chains.
    let mut weights = vec![0.; case.width * case.width];
    for col in 0..case.width {
        for (offset, scale) in [(0, 0.125), (1, -0.0625), (17, 0.03125)] {
            weights[((col + offset) % case.width) * case.width + col] = scale;
        }
    }
    let mut expected = values(case.rows * case.width, 0);
    let mut inputs: FxHashMap<_, _> = [
        (initial.id, HostBuffer::from(expected.clone())),
        (weight.id, HostBuffer::from(weights)),
    ]
    .into_iter()
    .collect();
    let mut residual = initial;
    let mut outputs = Vec::new();
    for layer in 0..case.depth {
        if case.keep_intermediates {
            bindings.output(residual.id);
            outputs.push((residual.id, expected.clone()));
        }
        let branch_values;
        let branch = if case.self_residual {
            branch_values = expected.clone();
            residual
        } else {
            let branch = graph.tensor([case.rows, case.width], DType::F32);
            branch_values = values(case.rows * case.width, layer + 1);
            inputs.insert(branch.id, HostBuffer::from(branch_values.clone()));
            bindings.input(branch.id);
            branch
        };
        residual = branch.matmul(weight) + residual;
        for row in 0..case.rows {
            for col in 0..case.width {
                let product: f64 = [(0, 0.125), (1, -0.0625), (17, 0.03125)]
                    .into_iter()
                    .map(|(offset, scale)| {
                        f64::from(branch_values[row * case.width + (col + offset) % case.width])
                            * scale
                    })
                    .sum();
                expected[row * case.width + col] += product as f32;
            }
        }
    }
    bindings.output(residual.id);
    outputs.push((residual.id, expected));
    Fixture {
        graph,
        bindings,
        inputs,
        outputs,
    }
}

struct Run {
    allow_materialization: bool,
    mode: Mode,
    runtime: CudaRuntime,
    arena: Allocation,
    staging: CudaStaging,
    search_ms: f64,
    search_profile_ms: f64,
    candidates: usize,
    duplicates: usize,
    extract_refusals: usize,
    build_refusals: usize,
    in_place: usize,
    separate: usize,
    copies: usize,
    copied_bytes: usize,
    materializations: usize,
    slab_bytes: usize,
    peak_live_bytes: usize,
    max_error: f32,
    samples: Vec<u128>,
}

fn prepare(
    device: &CudaDevice,
    fixture: &Fixture,
    case: Case,
    mode: Mode,
    seed: u64,
    allow_materialization: bool,
) -> Result<Run> {
    // All authored boundaries already have callable native matrix layouts.
    // By default, keep approved GEMMs and zero-copy views and omit layout
    // materialization to isolate C/D behavior. The optional broader comparison
    // includes layout copies. Bufferizer repair/delivery copies are unrestricted.
    // Every call is still selected by production egglog rules.
    let registry = luminal_cuda_lite::cuda_registry_filtered(|row| {
        !matches!(row.label(), "AddFunctionalGeneric" | "ReduceSumGeneric")
            && (allow_materialization
                || !matches!(row.label(), "CopyGeneric" | "IndexMapApplyMaterialize"))
            && match mode {
                Mode::Separate => row.label() != "CublasLtAccumulateInPlace",
                Mode::InPlace => row.label() != "CublasLtCall",
                Mode::Both => true,
            }
    });
    let mut runtime = CudaRuntime::load_with(&fixture.graph, fixture.bindings.clone(), registry)?
        .with_device(device)?;
    let options = CompileOptions {
        seed,
        search_log: false,
        ..Default::default()
    };
    let start = Instant::now();
    let outcome = runtime.search(
        &Default::default(),
        &Default::default(),
        &fixture.inputs,
        &options,
    )?;
    // Includes assembly, saturation, serialization, extraction, profiling and
    // finalist validation; the component timing fields omit some of these.
    let search_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        outcome.refusal_breakdown.execute_refusals == 0,
        "timed execution refusal"
    );
    ensure!(outcome.finalist_rejections == 0, "finalist rejected");
    let mut in_place = 0;
    let mut separate = 0;
    let mut copies = 0;
    let mut copied_bytes = 0;
    let mut materializations = 0;
    let plan = runtime.plan().unwrap();
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::Compute {
                op, result_info, ..
            } => {
                if op.as_any().is::<CublasLtAccumulateInPlace>() {
                    in_place += 1;
                } else if op.as_any().is::<ApprovedCublasLt>() {
                    separate += 1;
                } else if matches!(op.label(), "CopyGeneric" | "IndexMapApplyMaterialize") {
                    materializations += 1;
                    copies += 1;
                    for result in result_info {
                        copied_bytes += luminal_cuda_lite::symbolic::bytes(
                            &result.layout,
                            &Default::default(),
                        )?;
                    }
                }
            }
            BufferNode::BufferCopy { src, .. } => {
                copies += 1;
                copied_bytes += luminal_cuda_lite::arena::buffer_bytes(&plan.buffers[src])?;
            }
            BufferNode::BufferCopyValue { layout, .. } => {
                copies += 1;
                copied_bytes += luminal_cuda_lite::symbolic::bytes(layout, &Default::default())?;
            }
            _ => {}
        }
    }
    ensure!(
        in_place + separate == case.depth,
        "expected {} GEMMs, got {in_place} in-place + {separate} separate",
        case.depth
    );
    let memory = runtime.memory_plan()?;
    let slab_bytes = memory.slab_bytes;
    let peak_live_bytes = memory.peak_live_bytes;
    let arena = support::allocate_arena(&runtime)?;
    let staging = runtime.allocate_staging()?;
    let mut run = Run {
        allow_materialization,
        mode,
        runtime,
        arena,
        staging,
        search_ms,
        search_profile_ms: outcome.timings.profile_nanos as f64 / 1e6,
        candidates: outcome.plans_profiled,
        duplicates: outcome.fingerprint_hits,
        extract_refusals: outcome.refusal_breakdown.extract_refusals,
        build_refusals: outcome.refusal_breakdown.plan_build_refusals,
        in_place,
        separate,
        copies,
        copied_bytes,
        materializations,
        slab_bytes,
        peak_live_bytes,
        max_error: 0.,
        samples: Vec::new(),
    };
    for _ in 0..5 {
        run.restore(fixture)?;
        run.runtime.execute(run.arena.arena(), &mut run.staging)?;
    }
    run.validate(fixture)?;
    Ok(run)
}

impl Run {
    fn restore(&mut self, fixture: &Fixture) -> Result<()> {
        // Stable upload order; input donation must not turn repeated replay
        // into a recurrence. Transfers/synchronization are outside timing.
        for binding in fixture.bindings.inputs() {
            self.runtime.upload(
                &mut self.arena,
                binding.value,
                fixture.inputs[&binding.value].clone(),
            )?;
        }
        Ok(())
    }

    fn sample(&mut self, fixture: &Fixture) -> Result<()> {
        self.restore(fixture)?;
        let start = Instant::now();
        self.runtime
            .execute(self.arena.arena(), &mut self.staging)?;
        self.samples.push(start.elapsed().as_nanos());
        Ok(())
    }

    fn validate(&mut self, fixture: &Fixture) -> Result<()> {
        for (id, expected) in &fixture.outputs {
            let (data, slot) = self.runtime.download(&self.arena, *id)?;
            let actual = luminal_cuda_lite::layouts::dense_f32(&data.as_f32()?, &slot.layout)?;
            ensure!(actual.len() == expected.len(), "output size mismatch");
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let error = (actual - expected).abs();
                ensure!(
                    actual.is_finite() && error <= 2e-5 + 2e-5 * expected.abs(),
                    "{} output {id:?}[{index}]: {actual} vs {expected}",
                    self.mode.name()
                );
                self.max_error = self.max_error.max(error);
            }
        }
        Ok(())
    }

    fn report(&self, case: Case, seed: u64) {
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let percentile = |percent: usize| sorted[(sorted.len() - 1) * percent / 100] as f64 / 1000.;
        let samples = self
            .samples
            .iter()
            .map(u128::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
            if self.allow_materialization {
                "materialization_allowed"
            } else {
                "native_layouts"
            },
            case.name,
            case.rows,
            case.width,
            case.depth,
            self.mode.name(),
            seed,
            self.search_ms,
            self.search_profile_ms,
            self.candidates,
            self.duplicates,
            self.extract_refusals,
            self.build_refusals,
            self.in_place,
            self.separate,
            self.copies,
            self.copied_bytes,
            self.materializations,
            self.slab_bytes,
            self.peak_live_bytes,
            percentile(10),
            percentile(50),
            percentile(90),
            self.max_error,
            samples
        );
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args()
        .skip(1)
        .filter(|arg| arg != "--bench")
        .collect();
    let mut quick = false;
    let mut selected = None;
    let mut allow_materialization = false;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => quick = true,
            "--allow-materialization" => allow_materialization = true,
            "--case" => selected = Some(args.next().context("--case requires a name")?.as_str()),
            _ => anyhow::bail!("unknown argument {arg}"),
        }
    }
    ensure!(
        selected.is_none_or(|name| CASES.iter().any(|case| case.name == name)),
        "unknown case"
    );
    let seeds = if quick { 1 } else { 3 };
    let samples = if quick { 7 } else { 31 };
    let options = CompileOptions::default();
    eprintln!(
        "search budget: {} generations, {} candidates/generation, {} profiling trials",
        options.generations, options.generation_size, options.trials
    );
    let device = CudaDevice::new(0)?;
    println!(
        "layout_candidates\tcase\trows\twidth\tdepth\tmode\tseed\tsearch_ms\tprofile_ms\tcandidates\tduplicates\textract_refusals\tbuild_refusals\tin_place_calls\tseparate_calls\tcopies\tcopied_bytes\tmaterializations\tslab_bytes\tpeak_live_bytes\tp10_us\tmedian_us\tp90_us\tmax_error\tsamples_ns"
    );
    for &case in CASES
        .iter()
        .filter(|case| selected.is_none_or(|name| case.name == name))
    {
        let fixture = fixture(case);
        for seed in 0..seeds {
            // Rotate search order between seeds and execution order between
            // trials, sharing one device/module cache across all three modes.
            let modes = [Mode::Separate, Mode::InPlace, Mode::Both];
            let mut runs = Vec::new();
            for offset in 0..3 {
                let mode = modes[(seed as usize + offset) % 3];
                eprintln!("{} {} seed {seed}: searching", case.name, mode.name());
                runs.push(
                    prepare(&device, &fixture, case, mode, seed, allow_materialization)
                        .with_context(|| format!("{} {} seed {seed}", case.name, mode.name()))?,
                );
            }
            for trial in 0..samples {
                for offset in 0..runs.len() {
                    let index = (trial + offset) % runs.len();
                    runs[index].sample(&fixture)?;
                }
            }
            for mut run in runs {
                run.validate(&fixture)?;
                run.report(case, seed);
            }
        }
    }
    Ok(())
}
