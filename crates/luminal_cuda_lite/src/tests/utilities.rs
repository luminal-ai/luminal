use candle_core::{Device, Tensor, WithDType};
use cudarc::driver::CudaContext;
use half::{bf16, f16};
use itertools::Itertools;
use luminal::egglog_utils::{
    EGraphChoiceSet, egglog_to_llir, extract_generation, hash_choice_set, random_initial_choice,
    validate_choice_set,
};
use luminal::prelude::{
    petgraph::{Direction, algo::toposort, visit::EdgeRef},
    *,
};
use num_traits::{Num, Signed};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use crate::runtime::{CudaRuntime, ToCudaInput};

/// Safety factor multiplied with epsilon for tolerance calculations
pub const TOLERANCE_SAFETY_FACTOR: f32 = 2.0;

/// Number of genomes to fuzz per op test invocation.
pub const GENOME_FUZZ_COUNT: usize = 20;

/// Trait for test-compatible data types that can be used in generic test functions.
/// Bridges luminal's runtime types with candle's tensor types.
pub trait TestDType:
    Clone + Sized + WithDType + PartialEq + Copy + std::fmt::Debug + 'static
where
    Vec<Self>: ToCudaInput,
{
    /// The corresponding luminal DType
    const DTYPE: luminal::dtype::DType;

    /// Retrieve data from the runtime in this dtype
    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self>;
    /// Extract a Vec from a candle Tensor
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self>;
    /// Compare two result vectors. Float types use tolerance; exact types use equality.
    fn assert_match(a: &[Self], b: &[Self], rtol: f32, atol: f32);
}

impl TestDType for f32 {
    const DTYPE: luminal::dtype::DType = luminal::dtype::DType::F32;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_f32(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<f32>().unwrap()
    }
    fn assert_match(a: &[Self], b: &[Self], rtol: f32, atol: f32) {
        assert_close(a, b, rtol, atol);
    }
}

impl TestDType for f16 {
    const DTYPE: luminal::dtype::DType = luminal::dtype::DType::F16;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_f16(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<f16>().unwrap()
    }
    fn assert_match(a: &[Self], b: &[Self], rtol: f32, atol: f32) {
        assert_close(a, b, f16::from_f32(rtol), f16::from_f32(atol));
    }
}

impl TestDType for bf16 {
    const DTYPE: luminal::dtype::DType = luminal::dtype::DType::Bf16;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_bf16(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<bf16>().unwrap()
    }
    fn assert_match(a: &[Self], b: &[Self], rtol: f32, atol: f32) {
        assert_close(a, b, bf16::from_f32(rtol), bf16::from_f32(atol));
    }
}

impl TestDType for i32 {
    const DTYPE: luminal::dtype::DType = luminal::dtype::DType::Int;

    fn get_from_runtime(rt: &CudaRuntime, id: NodeIndex) -> Vec<Self> {
        rt.get_i32(id)
    }
    fn candle_to_vec(tensor: &Tensor) -> Vec<Self> {
        tensor.to_vec1::<i32>().unwrap()
    }
    fn assert_match(a: &[Self], b: &[Self], _rtol: f32, _atol: f32) {
        assert_eq!(a, b);
    }
}

#[allow(dead_code)]
pub fn random_i32_vec(n: usize, seed: u64, low: i32, high: i32) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(low..=high)).collect()
}

pub fn random_f32_vec(n: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(low..high)).collect()
}

/// Assert two vectors are close following NumPy/PyTorch conventions.
/// Formula: |a - b| <= atol + rtol * |b|
/// Generic version that works with any Float type (f32, f16, bf16).
pub fn assert_close<T: Num + Signed + PartialOrd + Copy + std::fmt::Display>(
    a_vec: &[T],
    b_vec: &[T],
    rtol: T,
    atol: T,
) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let diff = (*a - *b).abs();
        let tolerance = atol + rtol * b.abs();

        if diff > tolerance {
            panic!("{a} is not close to {b}, index {i}, diff: {diff}, tolerance: {tolerance}");
        }
    }
}

pub fn get_cuda_stream() -> Option<Arc<cudarc::driver::CudaStream>> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.bind_to_thread().ok()?;
    Some(ctx.default_stream())
}

#[derive(Debug, Clone)]
pub enum CudaFuzzInput {
    F32(NodeIndex, Vec<f32>),
    Bf16(NodeIndex, Vec<bf16>),
    I32(NodeIndex, Vec<i32>),
}

impl CudaFuzzInput {
    fn apply(&self, rt: &mut CudaRuntime) {
        match self {
            Self::F32(id, data) => rt.set_data(*id, data.clone()),
            Self::Bf16(id, data) => rt.set_data(*id, data.clone()),
            Self::I32(id, data) => rt.set_data(*id, data.clone()),
        }
    }

    fn apply_reference(&self, rt: &mut ReferenceRuntime) {
        match self {
            Self::F32(id, data) => rt.set_data(*id, data.clone()),
            Self::Bf16(id, data) => rt.set_data(*id, data.clone()),
            Self::I32(id, data) => rt.set_data(*id, data.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct F32OutputCheck {
    pub id: NodeIndex,
    pub name: String,
    pub rtol: f32,
    pub atol: f32,
}

impl F32OutputCheck {
    pub fn new(id: NodeIndex, name: impl Into<String>, rtol: f32, atol: f32) -> Self {
        Self {
            id,
            name: name.into(),
            rtol,
            atol,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchEquivalenceFuzzConfig {
    pub seed: u64,
    pub samples: usize,
    pub generation_size: usize,
    pub mutations: usize,
    pub max_attempts: usize,
    pub build_options: CompileOptions,
    pub reference: SearchEquivalenceReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchEquivalenceReference {
    FirstCudaExtraction,
    ReferenceRuntime,
}

impl Default for SearchEquivalenceFuzzConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            samples: 32,
            generation_size: 16,
            mutations: 2,
            max_attempts: 1_000,
            build_options: CompileOptions::default(),
            reference: SearchEquivalenceReference::FirstCudaExtraction,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchEquivalenceFuzzReport {
    pub tested: usize,
    pub skipped_invalid: usize,
}

struct ChoiceRun {
    outputs: Vec<Vec<f32>>,
    llir_summary: String,
}

pub struct CudaSearchEquivalenceFuzzer<'a> {
    cx: &'a mut Graph,
    stream: &'a Arc<cudarc::driver::CudaStream>,
    inputs: Vec<CudaFuzzInput>,
    outputs: Vec<F32OutputCheck>,
    config: SearchEquivalenceFuzzConfig,
}

impl<'a> CudaSearchEquivalenceFuzzer<'a> {
    pub fn new(cx: &'a mut Graph, stream: &'a Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            cx,
            stream,
            inputs: Vec::new(),
            outputs: Vec::new(),
            config: SearchEquivalenceFuzzConfig::default(),
        }
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    pub fn samples(mut self, samples: usize) -> Self {
        self.config.samples = samples;
        self
    }

    pub fn generation_size(mut self, generation_size: usize) -> Self {
        self.config.generation_size = generation_size;
        self
    }

    pub fn mutations(mut self, mutations: usize) -> Self {
        self.config.mutations = mutations;
        self
    }

    pub fn reference_runtime(mut self) -> Self {
        self.config.reference = SearchEquivalenceReference::ReferenceRuntime;
        self
    }

    pub fn input_f32(mut self, id: NodeIndex, data: Vec<f32>) -> Self {
        self.inputs.push(CudaFuzzInput::F32(id, data));
        self
    }

    pub fn input_bf16(mut self, id: NodeIndex, data: Vec<bf16>) -> Self {
        self.inputs.push(CudaFuzzInput::Bf16(id, data));
        self
    }

    pub fn input_i32(mut self, id: NodeIndex, data: Vec<i32>) -> Self {
        self.inputs.push(CudaFuzzInput::I32(id, data));
        self
    }

    pub fn output_f32(
        mut self,
        id: NodeIndex,
        name: impl Into<String>,
        rtol: f32,
        atol: f32,
    ) -> Self {
        self.outputs.push(F32OutputCheck::new(id, name, rtol, atol));
        self
    }

    pub fn run(self) -> SearchEquivalenceFuzzReport {
        fuzz_cuda_search_space_equivalence(
            self.cx,
            self.stream,
            &self.inputs,
            &self.outputs,
            self.config,
        )
    }
}

/// End-to-end search-space equivalence fuzzing for CUDA.
///
/// This builds the normal CUDA e-graph search space, extracts random selectable
/// LLIR graphs, runs each with identical inputs, and verifies every requested
/// f32 output matches the first valid extraction. The reference is intentionally
/// another selected LLIR graph, not a hand-written CPU implementation: this
/// catches cases where supposedly equivalent e-graph choices diverge, including
/// candidates that produce non-finite outputs.
pub fn fuzz_cuda_search_space_equivalence(
    cx: &mut Graph,
    stream: &Arc<cudarc::driver::CudaStream>,
    inputs: &[CudaFuzzInput],
    outputs: &[F32OutputCheck],
    config: SearchEquivalenceFuzzConfig,
) -> SearchEquivalenceFuzzReport {
    assert!(
        !outputs.is_empty(),
        "fuzz harness needs at least one output"
    );

    let reference_runtime_outputs =
        if config.reference == SearchEquivalenceReference::ReferenceRuntime {
            cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
            let mut reference_rng = StdRng::seed_from_u64(config.seed);
            let mut reference_rt = cx.search_with_rng(
                ReferenceRuntime::default(),
                CompileOptions::default().search_graph_limit(1),
                &mut reference_rng,
            );
            for input in inputs {
                input.apply_reference(&mut reference_rt);
            }
            reference_rt.execute(&cx.dyn_map);
            Some(
                outputs
                    .iter()
                    .map(|out| reference_rt.get_f32(out.id).clone())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

    cx.build_search_space::<CudaRuntime>(config.build_options);

    let egraph = cx.egraph().expect("search space should be built");
    let ops = cx.egglog_ops().expect("search ops should be built");
    let seed = if reference_runtime_outputs.is_some() {
        config.seed.wrapping_add(0xC0DA_C0DA)
    } else {
        config.seed
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut prev_selected = FxHashSet::default();
    let mut base = random_initial_choice(egraph, &mut rng);
    prev_selected.insert(hash_choice_set(&base));

    let mut skipped_invalid = 0usize;
    let reference_is_cuda = reference_runtime_outputs.is_none();
    let (reference_hash, reference_outputs, reference_llir_summary, mut tested) =
        if let Some(reference_outputs) = reference_runtime_outputs {
            (0, reference_outputs, None, 0usize)
        } else {
            let mut attempts = 0usize;
            let (reference_hash, reference_run) = loop {
                attempts += 1;
                if attempts > config.max_attempts {
                    panic!(
                        "failed to extract a valid reference LLIR after {} attempts",
                        config.max_attempts
                    );
                }
                if validate_choice_set(egraph, &base, ops).is_err() {
                    skipped_invalid += 1;
                } else {
                    let hash = hash_choice_set(&base);
                    match run_choice_outputs(cx, stream, inputs, outputs, &base) {
                        Ok(run) => break (hash, run),
                        Err(err) => panic!("reference candidate hash={hash} failed: {err}"),
                    }
                }
                base = random_initial_choice(egraph, &mut rng);
                prev_selected.insert(hash_choice_set(&base));
            };
            (
                reference_hash,
                reference_run.outputs,
                Some(reference_run.llir_summary),
                1usize,
            )
        };

    let mut attempts = 0usize;
    while tested < config.samples && attempts < config.max_attempts {
        attempts += 1;
        let mut candidates = extract_generation(
            egraph,
            &base,
            config.generation_size,
            config.mutations,
            &mut prev_selected,
            &mut rng,
        );
        if candidates.is_empty() {
            let next = random_initial_choice(egraph, &mut rng);
            prev_selected.insert(hash_choice_set(&next));
            candidates.push(next);
        }

        for candidate in candidates {
            if tested >= config.samples {
                break;
            }
            let candidate_hash = hash_choice_set(&candidate);
            if reference_is_cuda && candidate_hash == reference_hash {
                continue;
            }
            if validate_choice_set(egraph, &candidate, ops).is_err() {
                skipped_invalid += 1;
                continue;
            }

            let candidate_run = run_choice_outputs(cx, stream, inputs, outputs, &candidate)
                .unwrap_or_else(|err| panic!("candidate hash={candidate_hash} failed: {err}"));
            assert_fuzz_outputs_close(
                outputs,
                &reference_outputs,
                &candidate_run.outputs,
                &candidate_run.llir_summary,
                reference_llir_summary.as_deref(),
                reference_hash,
                candidate_hash,
            );
            base = candidate;
            tested += 1;
        }
    }

    assert_eq!(
        tested, config.samples,
        "only tested {tested}/{} LLIR samples before exhausting attempts",
        config.samples
    );
    SearchEquivalenceFuzzReport {
        tested,
        skipped_invalid,
    }
}

fn run_choice_outputs<'a>(
    cx: &'a Graph,
    stream: &Arc<cudarc::driver::CudaStream>,
    inputs: &[CudaFuzzInput],
    outputs: &[F32OutputCheck],
    choices: &EGraphChoiceSet<'a>,
) -> Result<ChoiceRun, String> {
    let egraph = cx.egraph().ok_or("search space was not built")?;
    let ops = cx.egglog_ops().ok_or("search ops were not built")?;
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();
    let mut llir_graph = egglog_to_llir(
        egraph,
        choices.clone(),
        ops,
        &cx.custom_ops,
        &mut list_cache,
        &mut expr_cache,
        None,
    );
    unroll_loops_in_llir(&mut llir_graph);
    let llir_summary = summarize_llir(&llir_graph);

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.load_llir(&llir_graph);
    rt.preserve_intermediate_buffers_for_debug();
    for input in inputs {
        input.apply(&mut rt);
    }
    if std::env::var_os("LUMINAL_FUZZ_DUMP_LAST_LLIR").is_some() {
        let _ = std::fs::write("/tmp/luminal_fuzz_last_candidate_llir.txt", &llir_summary);
    }
    rt.execute(&cx.dyn_map);
    let topo_order = toposort(&llir_graph, None).map_err(|cycle| {
        format!(
            "extracted LLIR contains cycle at node {:?}",
            cycle.node_id()
        )
    })?;
    if let Some(report) = rt.first_nonfinite_f32_buffer_in_nodes(topo_order) {
        let dump_path = "/tmp/luminal_fuzz_bad_candidate_llir.txt";
        let _ = std::fs::write(dump_path, &llir_summary);
        let op = llir_graph
            .node_weight(report.node)
            .map(|op| format!("{op:?}"))
            .unwrap_or_else(|| "unknown op".to_string());
        return Err(format!(
            "LLIR produced non-finite F32 buffer node={} index={} value={} op={}; llir={dump_path}",
            report.node.index(),
            report.index,
            report.value,
            op
        ));
    }

    let values = outputs
        .iter()
        .map(|out| rt.get_f32(out.id))
        .collect::<Vec<_>>();
    for (spec, values) in outputs.iter().zip(&values) {
        if let Some((idx, value)) = values
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            let dump_path = "/tmp/luminal_fuzz_bad_candidate_llir.txt";
            let _ = std::fs::write(dump_path, &llir_summary);
            let internal = rt
                .first_nonfinite_f32_buffer()
                .map(|report| {
                    let op = llir_graph
                        .node_weight(report.node)
                        .map(|op| format!("{op:?}"))
                        .unwrap_or_else(|| "unknown op".to_string());
                    format!(
                        "; first observed non-finite buffer node={} index={} value={} op={}",
                        report.node.index(),
                        report.index,
                        report.value,
                        op
                    )
                })
                .unwrap_or_default();
            return Err(format!(
                "output {} produced non-finite value {value} at index {idx}{internal}; llir={dump_path}",
                spec.name
            ));
        }
    }
    Ok(ChoiceRun {
        outputs: values,
        llir_summary,
    })
}

fn assert_fuzz_outputs_close(
    outputs: &[F32OutputCheck],
    expected: &[Vec<f32>],
    actual: &[Vec<f32>],
    candidate_llir_summary: &str,
    reference_llir_summary: Option<&str>,
    reference_hash: u64,
    candidate_hash: u64,
) {
    for ((spec, expected), actual) in outputs.iter().zip(expected.iter()).zip(actual.iter()) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "output {} length mismatch for candidate hash={candidate_hash} reference hash={reference_hash}",
            spec.name
        );
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut worst = 0usize;
        for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                a.is_finite(),
                "output {} candidate hash={candidate_hash} produced non-finite value {a} at index {i}",
                spec.name
            );
            assert!(
                b.is_finite(),
                "output {} reference hash={reference_hash} produced non-finite value {b} at index {i}",
                spec.name
            );
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-12);
            if abs > max_abs {
                max_abs = abs;
                max_rel = rel;
                worst = i;
            }
            if abs > spec.atol + spec.rtol * b.abs() {
                let dump_path = "/tmp/luminal_fuzz_bad_candidate_llir.txt";
                let _ = std::fs::write(dump_path, candidate_llir_summary);
                if let Some(reference_llir_summary) = reference_llir_summary {
                    let _ = std::fs::write(
                        "/tmp/luminal_fuzz_bad_reference_llir.txt",
                        reference_llir_summary,
                    );
                }
                panic!(
                    "output {} mismatch candidate hash={candidate_hash} reference hash={reference_hash} index={i} actual={a} expected={b} abs={abs} rel={rel} tolerance={} candidate_llir={dump_path}",
                    spec.name,
                    spec.atol + spec.rtol * b.abs()
                );
            }
        }
        eprintln!(
            "fuzz output {} ok: candidate hash={candidate_hash} max_abs={max_abs} max_rel={max_rel} worst={worst}",
            spec.name
        );
    }
}

pub(crate) fn summarize_llir(llir_graph: &LLIRGraph) -> String {
    llir_graph
        .node_indices()
        .map(|idx| {
            let inputs = llir_graph
                .edges_directed(idx, Direction::Incoming)
                .sorted_by_key(|edge| edge.id())
                .map(|edge| edge.source().index().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("{} <- [{}]: {:?}", idx.index(), inputs, &llir_graph[idx])
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Get the GPU compute capability as (major, minor).
pub fn gpu_compute_cap() -> Option<(i32, i32)> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.compute_capability().ok()
}

/// FlashInfer needs Ampere+ (sm_80; its kernels use cp.async). Tests that
/// directly execute FlashInfer (bypassing the search, which gates the rule
/// itself) must skip on older arches like the T4 (sm_75), where the kernel
/// symbol is absent at launch (CUDA_ERROR_NOT_FOUND).
pub fn gpu_supports_flashinfer() -> bool {
    crate::device_compute_major() >= 8
}

/// Check if the current GPU supports the given dtype for tensor core / WMMA operations.
pub fn gpu_supports_dtype(dtype: luminal::dtype::DType) -> bool {
    let Some((major, minor)) = gpu_compute_cap() else {
        return false;
    };
    match dtype {
        luminal::dtype::DType::Bf16 => major >= 8, // Ampere (sm_80+)
        luminal::dtype::DType::F8E4M3 | luminal::dtype::DType::F8E5M2 => {
            major > 8 || (major == 8 && minor >= 9)
        } // Ada/Hopper (sm_89+)
        luminal::dtype::DType::F4E2M1 | luminal::dtype::DType::F8UE8M0 => major >= 10, // Blackwell (sm_100+)
        _ => true,
    }
}

/// Machine epsilon for each dtype (approximate)
pub fn dtype_epsilon(dtype: luminal::dtype::DType) -> f32 {
    match dtype {
        luminal::dtype::DType::F32 => 1.19e-7,  // 2^-23
        luminal::dtype::DType::F16 => 9.77e-4,  // 2^-10
        luminal::dtype::DType::Bf16 => 7.81e-3, // 2^-7
        luminal::dtype::DType::Int => 0.0,
        luminal::dtype::DType::Bool => 0.0,
        other => todo!("dtype_epsilon not implemented for {other}"),
    }
}

/// Map a luminal DType to the corresponding candle DType.
pub fn to_candle_dtype(dtype: luminal::dtype::DType) -> candle_core::DType {
    match dtype {
        luminal::dtype::DType::F32 => candle_core::DType::F32,
        luminal::dtype::DType::F16 => candle_core::DType::F16,
        luminal::dtype::DType::Bf16 => candle_core::DType::BF16,
        luminal::dtype::DType::Int => candle_core::DType::I32,
        luminal::dtype::DType::Bool => candle_core::DType::U8,
        other => todo!("candle dtype mapping not implemented for {other}"),
    }
}

/// Base unary test function with input generator (CUDA version)
/// Generic over dtype T - comparison happens in native precision.
pub fn test_unary_cuda<T: TestDType>(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
    generator: impl Fn(usize, u64) -> Vec<T>,
    seed: u64,
) where
    Vec<T>: ToCudaInput,
{
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let shape: Vec<usize> = shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let n_elements: usize = shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(shape.clone());
    let b = func(a).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());

    let input_data = generator(n_elements, seed);
    rt.set_data(a, input_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);

    let result = T::get_from_runtime(&rt, b.id);

    // Reference using candle on CUDA
    let device = Device::new_cuda(0).expect("Candle CUDA device required for test");
    let ref_a = Tensor::from_slice(&input_data, shape, &device).unwrap();
    let ref_b = ref_func(ref_a).flatten_all().unwrap();
    let ref_vec = T::candle_to_vec(&ref_b);

    let eps = dtype_epsilon(<T as TestDType>::DTYPE);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    T::assert_match(&result, &ref_vec, tol, tol);

    // Fuzz genomes: verify multiple graph rewrites produce consistent results
    fuzz_genomes::<T>(
        &cx,
        &stream,
        |rt| rt.set_data(a, input_data.clone()),
        b.id,
        &ref_vec,
        tol,
        tol,
        GENOME_FUZZ_COUNT,
        seed,
    );
}

/// Base binary test function with input generators
/// Generic over dtype T - comparison happens in native precision.
/// Requires explicit rtol and atol tolerances (as f32, converted to T internally).
#[allow(clippy::too_many_arguments)]
pub fn test_binary_cuda<T: TestDType>(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
    a_generator: impl Fn(usize, u64) -> Vec<T>,
    b_generator: impl Fn(usize, u64) -> Vec<T>,
    seed: u64,
    rtol: f32,
    atol: f32,
) where
    Vec<T>: ToCudaInput,
{
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let a_shape: Vec<usize> = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let a_elements: usize = a_shape.iter().product();
    let b_elements: usize = b_shape.iter().product();

    let mut cx = Graph::default();
    let a: GraphTensor = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());

    let a_data = a_generator(a_elements, seed);
    let b_data = b_generator(b_elements, seed.wrapping_add(1));
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);

    let result = T::get_from_runtime(&rt, c.id);

    // Reference using candle on CUDA
    let device = Device::new_cuda(0).expect("Candle CUDA device required for test");
    let ref_a = Tensor::from_slice(&a_data, a_shape, &device).unwrap();
    let ref_b = Tensor::from_slice(&b_data, b_shape, &device).unwrap();
    let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();
    let ref_vec = T::candle_to_vec(&ref_c);

    T::assert_match(&result, &ref_vec, rtol, atol);

    // Fuzz genomes: verify multiple graph rewrites produce consistent results
    fuzz_genomes::<T>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(a, a_data.clone());
            rt.set_data(b, b_data.clone());
        },
        c.id,
        &ref_vec,
        rtol,
        atol,
        GENOME_FUZZ_COUNT,
        seed,
    );
}

/// Test mod operation with element-wise reference using Rust's % operator
pub fn test_mod(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    seed: u64,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let a_shape: Vec<usize> = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let a_elements: usize = a_shape.iter().product();
    let b_elements: usize = b_shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = CudaRuntime::initialize(stream.clone());

    let a_data = random_f32_vec(a_elements, seed, -0.5, 0.5);
    // Generate divisor values away from zero (0.1 to 0.5) to avoid division issues
    let b_data = random_f32_vec(b_elements, seed.wrapping_add(1), 0.1, 0.5);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(5));
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);

    // Reference: Rust's % operator matches CUDA's fmodf (IEEE 754 remainder)
    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x % y)
        .collect();

    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    let rtol = eps * TOLERANCE_SAFETY_FACTOR;
    let atol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, rtol, atol);

    // Fuzz genomes: verify multiple graph rewrites produce consistent results
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(a, a_data.clone());
            rt.set_data(b, b_data.clone());
        },
        c.id,
        &expected,
        rtol,
        atol,
        GENOME_FUZZ_COUNT,
        seed,
    );
}

/// Generate a slice range for an axis of given size.
/// If do_start is true, randomly choose a start offset (leaving at least 1 element).
/// If do_end is true, randomly choose an end before the axis end.
pub fn gen_slice_range(
    size: usize,
    do_start: bool,
    do_end: bool,
    rng: &mut impl Rng,
) -> (usize, usize) {
    let start = if do_start && size > 1 {
        rng.random_range(0..size)
    } else {
        0
    };
    let remaining = size - start;
    let end = if do_end && remaining > 1 {
        start + rng.random_range(1..remaining)
    } else {
        size
    };
    (start, end)
}

/// Fuzz test multiple genomes from the e-graph search space.
///
/// After a graph has been built and compared against a reference, this function
/// extracts random genomes via mutation and verifies they all produce results
/// matching the expected reference output. This catches bugs where graph rewrites
/// produce incorrect computation.
///
/// `setup_inputs` is called for each genome's fresh runtime to load input data.
#[allow(clippy::too_many_arguments)]
pub fn fuzz_genomes<T: TestDType>(
    cx: &Graph,
    stream: &Arc<cudarc::driver::CudaStream>,
    setup_inputs: impl Fn(&mut CudaRuntime),
    output_id: NodeIndex,
    expected: &[T],
    rtol: f32,
    atol: f32,
    num_genomes: usize,
    seed: u64,
) where
    Vec<T>: ToCudaInput,
{
    let Some(egraph) = cx.egraph() else {
        return;
    };
    let Some(ops) = cx.egglog_ops() else {
        return;
    };

    // Check if there are alternative genomes to explore
    let mutable_eclasses: usize = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .count();
    if mutable_eclasses == 0 {
        return; // Only one valid graph, nothing to fuzz
    }

    // Use a different seed offset to avoid correlating with the search seed
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(7777));
    let mut prev_selected: FxHashSet<u64> = FxHashSet::default();

    let initial = random_initial_choice(egraph, &mut rng);
    prev_selected.insert(hash_choice_set(&initial));

    let mut base = initial;
    let mut tested = 0;

    for _ in 0..100 {
        let offspring = extract_generation(egraph, &base, 10, 2, &mut prev_selected, &mut rng);

        if offspring.is_empty() {
            break;
        }

        for genome in offspring {
            if validate_choice_set(egraph, &genome, ops).is_err() {
                continue;
            }

            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();
            let mut llir_graph = egglog_to_llir(
                egraph,
                genome.clone(),
                ops,
                &cx.custom_ops,
                &mut list_cache,
                &mut expr_cache,
                None,
            );
            // Same finalization as `Graph::search` performs on the chosen
            // best LLIR: collapse the rolled body's loop markers into a
            // fully-unrolled LLIR. The runtime cannot execute LoopStart /
            // LoopEnd / LoopInput / LoopOutput markers — they exist only as
            // a search-time scaffold the auto-roll prepass introduces.
            unroll_loops_in_llir(&mut llir_graph);

            // The search catches candidates that fail to load/materialize
            // (e.g. the GEMM-chain "missing cached buffer" corner case) and
            // skips them; mirror that here so the fuzz exercises the same
            // candidate set the search can actually select.
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut rt = CudaRuntime::initialize(stream.clone());
                rt.load_llir(&llir_graph);
                setup_inputs(&mut rt);
                rt.execute(&cx.dyn_map);
                T::get_from_runtime(&rt, output_id)
            }));
            let Ok(result) = run else {
                continue;
            };
            T::assert_match(&result, expected, rtol, atol);

            tested += 1;
            base = genome;

            if tested >= num_genomes {
                return;
            }
        }
    }
}
