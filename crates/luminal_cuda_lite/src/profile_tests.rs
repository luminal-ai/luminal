use super::*;
use luminal::op::{CustomOp, EgglogOp, LLIROp};
use rand::{SeedableRng, rngs::SmallRng};
use std::sync::Mutex;

fn runtime() -> CudaRuntime {
    CudaRuntime::initialize(
        cudarc::driver::CudaContext::new(0)
            .unwrap()
            .default_stream(),
    )
}
fn dims(s: usize) -> DynMap {
    [(Symbol::from('s'), s)].into_iter().collect()
}

#[test]
fn profile_budget_never_substitutes_warmup_for_a_timed_trial() {
    let mut cx = Graph::new();
    let input = cx.tensor(4).persist();
    let output = (input + 1.).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let space = cx.search_space().unwrap();
    let contexts = space.bucket_contexts(&cx.dyn_map);
    let llir = luminal::search::extract_one(space, &contexts[0], &mut SmallRng::seed_from_u64(73));
    let mut rt = runtime();
    rt.load_llir(&llir);
    rt.set_data(input, vec![2f32; 4]);
    let before = rt.next_execution_id;
    rt.profile_loaded_cuda_graph(&llir, &cx.dyn_map, 3, Some(Duration::ZERO), None);
    assert_eq!(
        rt.next_execution_id - before,
        2,
        "even an exhausted profiling budget requires a warmup and one timed trial"
    );
    assert_eq!(rt.get_f32(output), vec![3f32; 4]);
}

#[test]
fn profile_dense_exact_cases_and_unseen_shapes() {
    let mut cx = Graph::new();
    let input = cx.tensor(('s', 4)).persist();
    let bias = cx.tensor(4).persist();
    let output = (input + bias.expand_dim(0, 's')).output();
    cx.set_dim('s', 2);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default().dim_buckets(
        's',
        &[
            DimBucket::new(1, 4).representative(2),
            DimBucket::new(5, 8).representative(6),
        ],
    ));
    let mut rt = runtime();
    rt.set_data(input, vec![9f32; 8]);
    rt.set_data(bias, vec![1f32, 2., 3., 4.]);
    let original = rt.current_hlir_device_binding(input.id).unwrap();
    let workload = ProfileWorkload::new()
        .device_snapshots(true)
        .shared_input(bias)
        .case(
            "small",
            dims(1),
            ProfileInputs::new().input(input, vec![2f32; 4]),
            1.,
        )
        .case(
            "same-bucket",
            dims(4),
            ProfileInputs::new().input(input, vec![5f32; 16]),
            8.,
        )
        .case(
            "large",
            dims(7),
            ProfileInputs::new().input(input, vec![3f32; 28]),
            1.,
        );
    rt.set_profile_workload(&cx, workload).unwrap();
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(3).trials(2),
        &mut SmallRng::seed_from_u64(43),
    );
    assert_eq!(rt.current_hlir_device_binding(input.id).unwrap(), original);
    assert!(!rt.profile_evaluations().is_empty());
    assert!(rt.profile_evaluations().iter().all(|e| e.cuda_graph));
    for evaluation in rt.profile_evaluations() {
        let expected = if evaluation.bucket == 0 {
            vec![1, 4]
        } else {
            vec![7]
        };
        assert_eq!(
            evaluation
                .cases
                .iter()
                .map(|c| c.dims[&'s'.into()])
                .collect::<Vec<_>>(),
            expected
        );
        let score: f64 = evaluation
            .cases
            .iter()
            .map(|c| c.duration.as_secs_f64() * c.weight / 10.)
            .sum();
        assert!((evaluation.weighted_cost.as_secs_f64() - score).abs() < 1e-9);
    }
    rt.begin_cuda_graph_warmup(&[]);
    for s in [2, 8, 3, 6, 1] {
        cx.set_dim('s', s);
        rt.set_data(input, vec![10f32; s * 4]);
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_f32(output), [11., 12., 13., 14.].repeat(s));
    }
}

#[test]
fn profile_capture_aliases_and_restore_mirrors() {
    let mut cx = Graph::new();
    let a = cx.tensor(4).persist();
    let b = cx.tensor(4).persist();
    let meta = cx.tensor(1).as_dtype(DType::Int).persist();
    (a + b).output();
    meta.output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    let backing = rt
        .cuda_stream
        .clone_htod(&[1f32, 2., 3., 4., 5., 6.])
        .unwrap();
    let ptr = backing.device_ptr(&rt.cuda_stream).0;
    unsafe {
        rt.set_device_ptr(a, ptr, 16);
        rt.set_device_ptr(b, ptr + 8, 16);
    }
    rt.set_data_with_host_mirror(meta, vec![17i32]);
    let inputs = rt
        .capture_profile_inputs(&[a, b, meta], &DynMap::default())
        .unwrap();
    assert_eq!(inputs.groups().len(), 2);
    rt.set_profile_workload(
        &cx,
        ProfileWorkload::new()
            .device_snapshots(true)
            .case("alias", DynMap::default(), inputs.clone(), 1.)
            .case("alias-again", DynMap::default(), inputs, 1.),
    )
    .unwrap();
    let contexts = cx
        .search_space()
        .unwrap()
        .bucket_contexts(&DynMap::default());
    rt.begin_profile_replay(&contexts).unwrap();
    assert!(rt.profile_replay.as_ref().unwrap().snapshots.is_empty());
    rt.activate_profile_case(0).unwrap();
    assert_eq!(rt.profile_replay.as_ref().unwrap().snapshots.len(), 2);
    let replay_a = rt.current_hlir_device_binding(a.id).unwrap().0;
    let replay_b = rt.current_hlir_device_binding(b.id).unwrap().0;
    assert_eq!(replay_b, replay_a + 8);
    unsafe {
        result::memcpy_htod_sync(replay_b, &[99f32; 4]).unwrap();
    }
    rt.hlir_host_mirrors.insert(meta.id, vec![0; 4]);
    rt.restore_profile_inputs().unwrap();
    let mut actual = [0f32; 6];
    unsafe {
        result::memcpy_dtoh_sync(&mut actual, replay_a).unwrap();
    }
    assert_eq!(actual, [1., 2., 3., 4., 5., 6.]);
    assert_eq!(rt.hlir_host_mirrors[&meta.id], 17i32.to_ne_bytes());
    rt.finish_profile_replay();
    assert_eq!(rt.current_hlir_device_binding(a.id).unwrap().0, ptr);
}

#[derive(Debug, Clone, Default)]
struct Probe {
    seen: Arc<Mutex<Vec<(i32, i32, usize)>>>,
    hidden: Arc<Mutex<usize>>,
    fail: bool,
}
impl EgglogOp for Probe {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        luminal::egglog_utils::api::sort(luminal::egglog_utils::base::OP_KIND, "ReplayProbe", &[])
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn n_inputs(&self) -> usize {
        2
    }
}
impl CustomOp for Probe {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
impl HostOp for Probe {
    fn output_size(&self) -> Expression {
        1.into()
    }
    fn output_bytes(&self) -> Expression {
        4.into()
    }
    fn capture_profile_state(
        &self,
        _: &Arc<CudaStream>,
    ) -> anyhow::Result<Box<dyn crate::host::ProfileState>> {
        struct Snapshot(Arc<Mutex<usize>>, usize);
        impl crate::host::ProfileState for Snapshot {
            fn restore(&self, _: &Arc<CudaStream>) -> anyhow::Result<()> {
                *self.0.lock().unwrap() = self.1;
                Ok(())
            }
        }
        Ok(Box::new(Snapshot(
            self.hidden.clone(),
            *self.hidden.lock().unwrap(),
        )))
    }
    fn profile_mutated_inputs(&self) -> Vec<usize> {
        vec![0]
    }
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _: &DynMap,
    ) -> anyhow::Result<()> {
        stream.synchronize()?;
        let mut state = [0i32];
        let mut meta = [0i32];
        unsafe {
            result::memcpy_dtoh_sync(&mut state, buffers[&inputs[0]].ptr())?;
            result::memcpy_dtoh_sync(&mut meta, buffers[&inputs[1]].ptr())?;
        }
        assert_eq!(
            buffers[&inputs[1]].host_bytes().unwrap(),
            meta[0].to_ne_bytes()
        );
        let mut hidden = self.hidden.lock().unwrap();
        self.seen.lock().unwrap().push((state[0], meta[0], *hidden));
        *hidden += 1;
        unsafe {
            result::memcpy_htod_sync(buffers[&inputs[0]].ptr(), &[state[0] + 1])?;
            result::memcpy_htod_sync(buffers[&node].ptr(), &[state[0] as f32])?;
        }
        anyhow::ensure!(!self.fail, "deliberate probe failure after input mutation");
        Ok(())
    }
}

#[test]
fn profile_custom_op_state_rng_and_candidate_order() {
    let mut cx = Graph::new();
    let state = cx.tensor(1).as_dtype(DType::Int).persist();
    let meta = cx.tensor(1).as_dtype(DType::Int).persist();
    let probe = Probe::default();
    let out = cx
        .custom_op(probe.clone(), (state.id, meta.id), 1, DType::F32)
        .output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(state, vec![101i32]);
    rt.set_data_with_host_mirror(meta, vec![202i32]);
    let workload = ProfileWorkload::new()
        .case(
            "a",
            DynMap::default(),
            ProfileInputs::new()
                .input(state, vec![7i32])
                .mirrored_input(meta, vec![11i32]),
            1.,
        )
        .case(
            "b",
            DynMap::default(),
            ProfileInputs::new()
                .input(state, vec![19i32])
                .mirrored_input(meta, vec![23i32]),
            3.,
        );
    rt.set_profile_workload(&cx, workload.clone()).unwrap();
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(2).trials(3),
        &mut SmallRng::seed_from_u64(1),
    );
    let seen = probe.seen.lock().unwrap().clone();
    assert!(
        seen.len() >= 16,
        "warmup and trials in both execution modes"
    );
    assert!(
        seen.iter().all(|v| *v == (7, 11, 0) || *v == (19, 23, 0)),
        "{seen:?}"
    );
    probe.seen.lock().unwrap().clear();
    let mut reversed = workload;
    reversed.cases.reverse();
    rt.set_profile_workload(&cx, reversed).unwrap();
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(2).trials(3),
        &mut SmallRng::seed_from_u64(2),
    );
    let seen = probe.seen.lock().unwrap().clone();
    assert!(seen.iter().all(|v| *v == (7, 11, 0) || *v == (19, 23, 0)));
    assert!(rt.profile_replay.is_none());
    rt.execute(&DynMap::default());
    assert_eq!(rt.get_f32(out), vec![101.]);
}

#[test]
fn profile_rejects_invalid_bindings_and_uncovered_bucket() {
    let mut cx = Graph::new();
    let input = cx.tensor('s');
    (input + input).output();
    cx.set_dim('s', 1);
    cx.build_search_space::<CudaRuntime>(
        CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 4), DimBucket::new(5, 8)]),
    );
    let mut rt = runtime();
    for (d, data, weight) in [
        (dims(2), vec![1f32], 1.),
        (DynMap::default(), vec![1f32], 1.),
        (dims(1), vec![1f32], f64::NAN),
    ] {
        assert!(
            rt.set_profile_workload(
                &cx,
                ProfileWorkload::new().case(
                    "bad",
                    d,
                    ProfileInputs::new().input(input, data),
                    weight
                )
            )
            .is_err()
        );
    }
    let storage = ProfileStorage::new(vec![0u8; 8]);
    assert!(
        rt.set_profile_workload(
            &cx,
            ProfileWorkload::new().case(
                "unaligned",
                dims(1),
                ProfileInputs::new().view(input, &storage, 1..5, false),
                1.
            )
        )
        .is_err()
    );
    rt.set_profile_workload(
        &cx,
        ProfileWorkload::new().case(
            "only",
            dims(2),
            ProfileInputs::new().input(input, vec![1f32; 2]),
            1.,
        ),
    )
    .unwrap();
    let contexts = cx.search_space().unwrap().bucket_contexts(&cx.dyn_map);
    assert!(rt.begin_profile_replay(&contexts).is_err());
    assert!(rt.profile_replay.is_none());
}

// Two semantically identical test algorithms with deliberately different
// data-dependent costs. Sleeping makes the ranking deterministic without relying
// on a particular GPU kernel implementation or hardcoding a production winner.
#[derive(Debug, Clone, Default)]
struct Choice(usize);
impl CustomOp for Choice {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
impl EgglogOp for Choice {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        use luminal::egglog_utils::{
            api::sort,
            base::{I64, OP_KIND},
        };
        sort(OP_KIND, "ProfileChoice", &[("algorithm", I64)])
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<luminal::egglog_utils::api::Rule> {
        vec![luminal::egglog_utils::api::Rule::raw(
            r#"
            (rule ((= ?x (Op (CustomOpKind 0 (F32)) ?inputs)))
                  ((union ?x (Op (ProfileChoice 0) ?inputs))
                   (union ?x (Op (ProfileChoice 1) ?inputs))) :ruleset kernel_lower)
        "#,
        )]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let algorithm = egraph.enodes[children[0]].0.parse().unwrap();
        (Choice(algorithm).to_llir_op(), inputs)
    }
}
impl HostOp for Choice {
    fn output_size(&self) -> Expression {
        1.into()
    }
    fn output_bytes(&self) -> Expression {
        4.into()
    }
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        _: &DynMap,
    ) -> anyhow::Result<()> {
        stream.synchronize()?;
        let mut data = [0f32; 4];
        unsafe {
            result::memcpy_dtoh_sync(&mut data, buffers[&inputs[0]].ptr())?;
        }
        let preferred = usize::from(data[0] != 0.);
        std::thread::sleep(Duration::from_millis(if self.0 == preferred {
            1
        } else {
            6
        }));
        unsafe {
            result::memcpy_htod_sync(buffers[&node].ptr(), &[data.iter().sum::<f32>()])?;
        }
        Ok(())
    }
}

#[test]
fn profile_supplied_data_and_weights_change_search_winner() {
    type TestRuntime = CudaRuntimeImpl<(DefaultCudaOps, Choice)>;
    let mut cx = Graph::new();
    let input = cx.tensor(4).persist();
    let output = cx.custom_op(Choice(0), input, 1, DType::F32).output();
    cx.build_search_space::<TestRuntime>(CompileOptions::default());
    let mut rt = TestRuntime::initialize(
        cudarc::driver::CudaContext::new(0)
            .unwrap()
            .default_stream(),
    );
    rt.set_data(input, vec![1f32, 2., 3., 4.]);
    for (zero_weight, nonzero_weight, expected) in [(9., 1., 0), (1., 9., 1)] {
        let workload = ProfileWorkload::new()
            .timing_method(if expected == 0 {
                luminal::op::TimingMethod::DeviceTimestamp
            } else {
                luminal::op::TimingMethod::WallClock
            })
            .case(
                "zero",
                DynMap::default(),
                ProfileInputs::new().input(input, vec![0f32, 2., 3., 4.]),
                zero_weight,
            )
            .case(
                "nonzero",
                DynMap::default(),
                ProfileInputs::new().input(input, vec![1f32, 2., 3., 4.]),
                nonzero_weight,
            );
        rt.set_profile_workload(&cx, workload).unwrap();
        rt = cx.search_with_rng(
            rt,
            CompileOptions::default()
                .search_graph_limit(8)
                .trials(2)
                .keep_best(8),
            &mut SmallRng::seed_from_u64(17),
        );
        let choices: Vec<_> = rt.compiled_buckets[0]
            .exec_graph
            .node_weights()
            .filter_map(|e| e.internal.as_any().downcast_ref::<Choice>())
            .collect();
        assert_eq!(choices.len(), 1);
        assert_eq!(
            choices[0].0, expected,
            "supplied workload must drive selection"
        );
        assert!(rt.profile_evaluations().iter().all(|e| e.cases.len() == 2));
        rt.execute(&DynMap::default());
        assert_eq!(rt.get_f32(output), vec![10.]);
    }
}

#[test]
fn profile_research_remeasures_the_saved_program() {
    let mut graph = Graph::new();
    let input = graph.tensor(32).persist();
    let output = (input.sin() * input + 1.0).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(input, vec![2f32; 32]);
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(8).trials(2),
        &mut SmallRng::seed_from_u64(20260910),
    );
    let previous = serde_json::to_value(rt.selected_schedule.as_ref().unwrap()).unwrap();
    rt.set_profile_workload(
        &graph,
        ProfileWorkload::new().case(
            "new-input",
            DynMap::default(),
            ProfileInputs::new().input(input, vec![3f32; 32]),
            1.,
        ),
    )
    .unwrap();
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(1).trials(2),
        &mut SmallRng::seed_from_u64(9271),
    );
    assert_eq!(
        serde_json::to_value(rt.selected_schedule.as_ref().unwrap()).unwrap(),
        previous
    );
    let measurements = rt.profile_evaluations();
    assert_eq!(
        measurements.len(),
        2,
        "remeasure the seed and its deployment finalist"
    );
    assert!(measurements.iter().all(|m| m.cuda_graph
        && m.cases.len() == 1
        && m.cases[0].case_id == "new-input"
        && m.cases[0].duration > Duration::ZERO));
    rt.clear_profile_workload();
    rt.execute(&DynMap::default());
    for value in rt.get_f32(output) {
        assert!((value - (2f32.sin() * 2. + 1.)).abs() < 1e-5);
    }
}

#[test]
fn profile_failure_restores_caller_inputs_outputs_and_hidden_state() {
    let mut cx = Graph::new();
    let state = cx.tensor(1).as_dtype(DType::Int).persist();
    let meta = cx.tensor(1).as_dtype(DType::Int).persist();
    let probe = Probe {
        fail: true,
        ..Probe::default()
    };
    let output = cx
        .custom_op(probe.clone(), (state, meta), 1, DType::F32)
        .output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(state, vec![101i32]);
    rt.set_data_with_host_mirror(meta, vec![202i32]);
    let original = rt.current_hlir_device_binding(state.id).unwrap();
    let output_buffer = rt.cuda_stream.clone_htod(&[42f32]).unwrap();
    let output_ptr = output_buffer.device_ptr(&rt.cuda_stream).0;
    unsafe {
        rt.set_output_device_ptr(output, output_ptr, 4);
    }
    rt.set_profile_workload(
        &cx,
        ProfileWorkload::new().case(
            "fails",
            DynMap::default(),
            ProfileInputs::new()
                .input(state, vec![7i32])
                .mirrored_input(meta, vec![11i32]),
            1.,
        ),
    )
    .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // compile borrows the runtime so it remains available to inspect after
        // search correctly reports that no executable candidate succeeded.
        rt.compile(
            cx.search_space().unwrap(),
            &cx.dyn_map,
            &CompileOptions::default().search_graph_limit(1),
            &mut SmallRng::seed_from_u64(1),
        );
    }));
    assert!(result.is_err());
    assert!(rt.profile_replay.is_none());
    assert!(!rt.profiling);
    assert_eq!(rt.current_hlir_device_binding(state.id).unwrap(), original);
    let mut state_bytes = [0i32];
    unsafe {
        result::memcpy_dtoh_sync(&mut state_bytes, original.0).unwrap();
    }
    assert_eq!(state_bytes, [101]);
    assert_eq!(*probe.hidden.lock().unwrap(), 0);
    assert_eq!(rt.output_ptr_registrations[&output.id], (output_ptr, 4));
    assert_eq!(
        rt.cuda_stream.clone_dtoh(&output_buffer).unwrap(),
        vec![42f32]
    );
    assert_eq!(rt.hlir_host_mirrors[&meta.id], 202i32.to_ne_bytes());
}

#[test]
fn profile_indexed_update_keeps_original_state() {
    let mut cx = Graph::new();
    let destination = cx.tensor(4).persist();
    let indices = cx.tensor(2).as_dtype(DType::Int).persist();
    let values = cx.tensor(2).persist();
    let result = (destination.gather(indices) + values)
        .scatter(indices, destination)
        .output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(destination, vec![1f32, 2., 3., 4.]);
    rt.set_data(indices, vec![1i32, 3]);
    rt.set_data(values, vec![4f32, 5.]);
    let samples = ProfileWorkload::new()
        .case(
            "indices-a",
            DynMap::default(),
            ProfileInputs::new()
                .input(destination, vec![0f32; 4])
                .input(indices, vec![1i32, 3])
                .input(values, vec![4f32, 5.]),
            1.,
        )
        .case(
            "indices-b",
            DynMap::default(),
            ProfileInputs::new()
                .input(destination, vec![10f32; 4])
                .input(indices, vec![0i32, 2])
                .input(values, vec![2f32, 3.]),
            1.,
        );
    rt.set_profile_workload(&cx, samples).unwrap();
    rt = cx.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(5).trials(3),
        &mut SmallRng::seed_from_u64(12),
    );
    rt.execute(&DynMap::default());
    assert_eq!(rt.get_f32(result), vec![1., 6., 3., 9.]);
}

#[test]
fn profile_rejects_shared_mutation_and_stale_search_space() {
    let mut cx = Graph::new();
    let state = cx.tensor(1).as_dtype(DType::Int);
    let meta = cx.tensor(1).as_dtype(DType::Int);
    let probe = Probe::default();
    cx.custom_op(probe.clone(), (state, meta), 1, DType::F32)
        .output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(state, vec![1i32]);
    rt.set_profile_workload(
        &cx,
        ProfileWorkload::new().shared_input(state).case(
            "readonly",
            DynMap::default(),
            ProfileInputs::new().mirrored_input(meta, vec![2i32]),
            1.,
        ),
    )
    .unwrap();
    let mut llir = LLIRGraph::default();
    let s = llir.add_node(LLIROp::new(Box::new(Input {
        node: state.id.index(),
        label: String::new(),
        dtype: DType::Int,
    })));
    let m = llir.add_node(LLIROp::new(Box::new(Input {
        node: meta.id.index(),
        label: String::new(),
        dtype: DType::Int,
    })));
    let op = llir.add_node(probe.to_llir_op());
    llir.add_edge(s, op, ());
    llir.add_edge(m, op, ());
    assert!(
        rt.validate_profile_effects(&llir)
            .unwrap_err()
            .to_string()
            .contains("read-only")
    );
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let contexts = cx.search_space().unwrap().bucket_contexts(&cx.dyn_map);
    assert!(
        rt.begin_profile_replay(&contexts)
            .unwrap_err()
            .to_string()
            .contains("another search space")
    );
}

#[test]
fn profile_bf16_strided_graph_uses_physical_input_abi() {
    let mut graph = Graph::new();
    let input = graph.tensor((2, 3)).as_dtype(DType::Bf16).persist();
    let output = input.permute((1, 0)).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let values: Vec<_> = (1..=6).map(|i| bf16::from_f32(i as f32)).collect();
    let mut rt = runtime();
    rt.set_data(input, values.clone());
    rt.set_profile_workload(
        &graph,
        ProfileWorkload::new().case(
            "bf16",
            DynMap::default(),
            ProfileInputs::new().input(input, values),
            1.,
        ),
    )
    .unwrap();
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(2),
        &mut SmallRng::seed_from_u64(5),
    );
    rt.execute(&DynMap::default());
    assert_eq!(
        rt.get_bf16(output)
            .iter()
            .map(|x| x.to_f32())
            .collect::<Vec<_>>(),
        vec![1., 4., 2., 5., 3., 6.]
    );
}

#[derive(Debug, Clone, Default)]
struct MetadataCopy {
    direct_launches: Arc<std::sync::atomic::AtomicUsize>,
    preparation_delay: Duration,
}
impl EgglogOp for MetadataCopy {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        luminal::egglog_utils::api::sort(luminal::egglog_utils::base::OP_KIND, "MetadataCopy", &[])
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn n_inputs(&self) -> usize {
        2
    }
}
impl CustomOp for MetadataCopy {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
impl HostOp for MetadataCopy {
    fn output_size(&self) -> Expression {
        's'.into()
    }
    fn output_bytes(&self) -> Expression {
        Expression::from('s') * 4
    }
    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        Some(2)
    }
    fn cuda_graph_capture_dyn_dims(&self) -> Vec<Symbol> {
        vec!['s'.into()]
    }
    fn prepare_cuda_graph_capture(
        &self,
        stream: &Arc<CudaStream>,
        _: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dims: &DynMap,
    ) -> anyhow::Result<()> {
        let rows = dims[&'s'.into()];
        anyhow::ensure!(
            buffers[&inputs[0]].len() == rows * 4,
            "input length disagrees with capture dimensions"
        );
        stream.synchronize()?;
        let mut metadata = [0i32];
        unsafe {
            result::memcpy_dtoh_sync(&mut metadata, buffers[&inputs[1]].ptr())?;
        }
        anyhow::ensure!(
            metadata[0] == rows as i32,
            "metadata disagrees with capture dimensions"
        );
        std::thread::sleep(self.preparation_delay);
        Ok(())
    }
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dims: &DynMap,
    ) -> anyhow::Result<()> {
        if stream.capture_status()?
            == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            self.direct_launches
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        unsafe {
            result::memcpy_dtod_async(
                buffers[&node].ptr(),
                buffers[&inputs[0]].ptr(),
                dims[&'s'.into()] * 4,
                stream.cu_stream(),
            )?;
        }
        Ok(())
    }
}

#[test]
fn profile_budget_accepts_complete_workload_but_rejects_missing_cases() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let output = graph
        .custom_op(
            MetadataCopy {
                preparation_delay: Duration::from_millis(100),
                ..Default::default()
            },
            (input.id, metadata.id),
            's',
            DType::Int,
        )
        .output();
    graph.set_dim('s', 4);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let llir = luminal::search::extract_one(space, &contexts[0], &mut SmallRng::seed_from_u64(17));
    for (cases, warmups) in [(1, 1), (2, 1), (1, 3), (2, 3)] {
        let mut rt = runtime();
        rt.set_data(input, vec![19i32; 4]);
        rt.set_data_with_host_mirror(metadata, vec![4i32]);
        let mut workload = ProfileWorkload::new().warmup_trials(warmups);
        for case in 0..cases {
            workload = workload.case(
                format!("case-{case}"),
                dims(4),
                ProfileInputs::new()
                    .input(input, vec![19i32; 4])
                    .mirrored_input(metadata, vec![4i32]),
                1.0,
            );
        }
        rt.set_profile_workload(&graph, workload).unwrap();
        rt.begin_profile_replay(&contexts).unwrap();
        let before = rt.next_execution_id;
        let result = rt.evaluate_profile_workload(
            &llir,
            &contexts[0],
            &CompileOptions::default()
                .trials(1)
                .execution_timeout(Duration::from_millis(50)),
            false,
        );
        assert_eq!(
            rt.next_execution_id - before,
            warmups as u64 + 1,
            "configured warmups and one full timed trial"
        );
        assert_eq!(rt.get_i32(output), vec![19; 4]);
        if cases == 1 {
            assert!(
                result.is_ok(),
                "completed workload must be rankable despite slow warmup: {result:?}"
            );
            assert_eq!(rt.profile_evaluations().len(), 1);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("before every case was measured")
            );
            assert!(
                rt.profile_evaluations().is_empty(),
                "incomplete workloads must not be ranked"
            );
        }
        rt.finish_profile_replay();
    }
}

#[test]
fn profile_capture_preparation_uses_exact_metadata() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let copy = MetadataCopy::default();
    let direct_launches = copy.direct_launches.clone();
    let output = graph
        .custom_op(copy, (input.id, metadata.id), 's', DType::Int)
        .output();
    graph.set_dim('s', 1);
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 8).representative(8)]),
    );
    let mut rt = runtime();
    rt.set_data(input, vec![19i32]);
    rt.set_data_with_host_mirror(metadata, vec![1i32]);
    let mut workload = ProfileWorkload::new();
    for n in [2, 6] {
        workload = workload.case(
            format!("rows-{n}"),
            dims(n),
            ProfileInputs::new()
                .input(input, vec![7i32; n])
                .mirrored_input(metadata, vec![n as i32]),
            1.,
        );
    }
    rt.set_profile_workload(&graph, workload).unwrap();
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(1),
        &mut SmallRng::seed_from_u64(13),
    );
    assert!(
        rt.profile_evaluations()
            .iter()
            .any(|e| e.cuda_graph && e.cases.len() == 2)
    );
    assert_eq!(direct_launches.load(std::sync::atomic::Ordering::SeqCst), 0);
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_i32(output), vec![19]);
}

#[test]
fn search_accepts_larger_equivalent_graph_after_small_candidate() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let output = graph
        .custom_op(
            MetadataCopy::default(),
            (input.id, metadata.id),
            's',
            DType::Int,
        )
        .output();
    graph.set_dim('s', 4);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let contexts = space.bucket_contexts(&graph.dyn_map);
    let mut llir =
        luminal::search::extract_one(space, &contexts[0], &mut SmallRng::seed_from_u64(17));
    let copy = llir
        .node_indices()
        .find(|&n| {
            llir[n]
                .to_dialect::<dyn HostOp>()
                .is_some_and(|op| op.as_any().is::<MetadataCopy>())
        })
        .unwrap();
    let mut rt = runtime();
    rt.set_data(input, vec![19i32; 4]);
    rt.set_data_with_host_mirror(metadata, vec![4i32]);
    drop(
        rt.compile_and_validate_profile_candidate(&llir, &graph.dyn_map, &contexts[0])
            .unwrap(),
    );
    let metadata_node = llir
        .edges_directed(copy, petgraph::Direction::Incoming)
        .max_by_key(|edge| edge.id())
        .unwrap()
        .source();
    let consumers: Vec<_> = llir
        .edges_directed(copy, petgraph::Direction::Outgoing)
        .map(|edge| (edge.id(), edge.target()))
        .collect();
    let mut previous = copy;
    // Identity copies keep the program's result unchanged while exercising a
    // larger valid candidate, independent of the first candidate's node count.
    for _ in 0..1025 {
        let next = llir.add_node(llir[copy].clone());
        llir.add_edge(previous, next, ());
        llir.add_edge(metadata_node, next, ());
        previous = next;
    }
    for (edge, target) in consumers {
        llir.remove_edge(edge);
        llir.add_edge(previous, target, ());
    }
    let compiled = rt
        .compile_and_validate_profile_candidate(&llir, &graph.dyn_map, &contexts[0])
        .expect("valid graph growth must not be rejected based on an earlier candidate's size");
    rt.install_validated_bucket_set(&space.dim_buckets, compiled.buckets)
        .unwrap();
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_i32(output), vec![19; 4]);
}

#[test]
fn synthetic_search_profiles_materialized_graphs() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let copy = MetadataCopy::default();
    let direct_launches = copy.direct_launches.clone();
    let output = graph
        .custom_op(copy, (input.id, metadata.id), 's', DType::Int)
        .output();
    graph.set_dim('s', 4);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(input, vec![19i32; 4]);
    rt.set_data_with_host_mirror(metadata, vec![4i32]);
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(1).trials(3),
        &mut SmallRng::seed_from_u64(17),
    );
    assert_eq!(direct_launches.load(std::sync::atomic::Ordering::SeqCst), 0);
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_i32(output), vec![19; 4]);
}

#[test]
fn profile_replay_reclaims_previous_arena_and_rematerializes_original_program() {
    let mut graph = Graph::new();
    let input = graph.tensor(256).persist();
    let output = (input + 2.0).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(input, vec![3f32; 256]);
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(2).trials(1),
        &mut SmallRng::seed_from_u64(831),
    );
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![5.; 256]);
    assert!(rt.shared_arena.is_some(), "test requires a resident arena");
    assert!(rt.cuda_graphs().any(CudaGraphOp::is_materialized));
    let original = rt.current_hlir_device_binding(input.id).unwrap();
    rt.set_profile_workload(
        &graph,
        ProfileWorkload::new().device_snapshots(true).case(
            "replacement",
            DynMap::default(),
            ProfileInputs::new().input(input, vec![7f32; 256]),
            1.,
        ),
    )
    .unwrap();
    let contexts = graph
        .search_space()
        .unwrap()
        .bucket_contexts(&graph.dyn_map);
    rt.begin_profile_replay(&contexts).unwrap();
    assert!(
        rt.shared_arena.is_none(),
        "previous arena overlaps replay storage"
    );
    assert!(!rt.cuda_graphs().any(CudaGraphOp::is_materialized));
    assert!(rt.profile_replay.as_ref().unwrap().snapshots.is_empty());
    rt.activate_profile_case(0).unwrap();
    assert_eq!(rt.profile_replay.as_ref().unwrap().snapshots.len(), 1);
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![9.; 256]);
    rt.finish_profile_replay();
    assert_eq!(rt.current_hlir_device_binding(input.id).unwrap(), original);
    rt.clear_profile_workload();
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![5.; 256]);
}

#[derive(Debug, Clone, Default)]
struct WorkspaceCopy {
    metadata: MetadataCopy,
    cache: Arc<crate::host::workspace::WorkspaceCache>,
    owners: Arc<Mutex<Vec<std::sync::Weak<crate::host::workspace::Workspace>>>>,
}
impl EgglogOp for WorkspaceCopy {
    fn sort(&self) -> luminal::egglog_utils::api::SortDef {
        luminal::egglog_utils::api::sort(luminal::egglog_utils::base::OP_KIND, "WorkspaceCopy", &[])
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn n_inputs(&self) -> usize {
        2
    }
}
impl CustomOp for WorkspaceCopy {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new(Box::new(self.clone()) as Box<dyn HostOp>)
    }
}
impl HostOp for WorkspaceCopy {
    fn output_size(&self) -> Expression {
        's'.into()
    }
    fn output_bytes(&self) -> Expression {
        Expression::from('s') * 4
    }
    fn output_dtype(&self) -> DType {
        DType::Int
    }
    fn cuda_graph_capture_arity(&self) -> Option<usize> {
        Some(2)
    }
    fn cuda_graph_capture_dyn_dims(&self) -> Vec<Symbol> {
        vec!['s'.into()]
    }
    fn prepare_cuda_graph_capture_resources(
        &self,
        capture: &Arc<CudaStream>,
        execution: &Arc<CudaStream>,
        node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dims: &DynMap,
    ) -> anyhow::Result<crate::host::CudaGraphCaptureResources> {
        self.metadata
            .prepare_cuda_graph_capture(capture, node, inputs, buffers, dims)?;
        let owner = self
            .cache
            .acquire(capture, execution, dims[&'s'.into()] * 4)?;
        self.owners.lock().unwrap().push(Arc::downgrade(&owner));
        Ok(vec![owner])
    }
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dims: &DynMap,
    ) -> anyhow::Result<()> {
        let bytes = dims[&'s'.into()] * 4;
        let owner = if stream.capture_status()?
            != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            self.cache.prepared(stream, bytes)?
        } else {
            self.cache.acquire(stream, stream, bytes)?
        };
        unsafe {
            result::memcpy_dtod_async(
                owner.ptr(),
                buffers[&inputs[0]].ptr(),
                bytes,
                stream.cu_stream(),
            )?;
            result::memcpy_dtod_async(
                buffers[&node].ptr(),
                owner.ptr(),
                bytes,
                stream.cu_stream(),
            )?;
        }
        Ok(())
    }
}

#[test]
fn captured_workspace_owners_follow_resident_graph_lifetimes() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let copy = WorkspaceCopy::default();
    let owners = copy.owners.clone();
    let output = graph
        .custom_op(copy, (input.id, metadata.id), 's', DType::Int)
        .output();
    graph.set_dim('s', 1);
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 8).representative(1)]),
    );
    let mut rt = runtime();
    rt.set_data_with_capacity(input, vec![19i32; 1], 32);
    rt.set_data_with_host_mirror(metadata, vec![1i32]);
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(1).trials(1),
        &mut SmallRng::seed_from_u64(221),
    );
    // Keep the generic output arena stable while the private workspace grows;
    // arena relocation deliberately retires every resident graph generation.
    rt.ensure_shared_arena_capacity(16 * 1024 * 1024);
    rt.begin_cuda_graph_warmup(&['s'.into()]);
    for s in [2, 7, 3, 2, 7, 3] {
        graph.set_dim('s', s);
        rt.set_data(input, vec![s as i32; s]);
        rt.set_data_with_host_mirror(metadata, vec![s as i32]);
        rt.execute(&graph.dyn_map);
        assert_eq!(rt.get_i32(output), vec![s as i32; s]);
    }
    let live_allocations: FxHashSet<_> = owners
        .lock()
        .unwrap()
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .map(|w| w.ptr())
        .collect();
    assert!(
        live_allocations.len() >= 2,
        "test must retain multiple allocation generations"
    );
    rt.release_all_bucket_cuda_graphs();
    assert!(
        owners.lock().unwrap().iter().all(|w| w.upgrade().is_none()),
        "retired graphs retain workspace owners"
    );
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_i32(output), vec![3; 3]);
    drop(rt);
    assert!(
        owners.lock().unwrap().iter().all(|w| w.upgrade().is_none()),
        "dropping runtime retains workspace owners"
    );
}

#[test]
fn prepared_input_survives_replay_and_clears_on_ordinary_write() {
    let mut graph = Graph::new();
    let input = graph.tensor(4).persist();
    let output = (input + 1.).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    let bytes: Vec<u8> = [2f32, 3., 4., 5., 99., 98., 97., 96.]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let allocation = rt.cuda_stream.clone_htod(&bytes).unwrap();
    rt.set_prepared_buffer(input, allocation, 16, "test.dual.v1");
    let binding = rt.current_hlir_device_binding(input.id).unwrap();
    let before = rt.current_resource_input_signature();
    assert_eq!(before[&input.id].owned_capacity_bytes, Some(32));
    assert!(rt.capture_profile_inputs(&[input], &graph.dyn_map).is_err());
    assert!(
        rt.set_profile_workload(
            &graph,
            ProfileWorkload::new().case(
                "bad",
                graph.dyn_map.clone(),
                ProfileInputs::new().input(input, vec![0f32; 4]),
                1.
            )
        )
        .is_err()
    );
    rt.set_profile_workload(
        &graph,
        ProfileWorkload::new().shared_input(input).case(
            "shared",
            graph.dyn_map.clone(),
            ProfileInputs::new(),
            1.,
        ),
    )
    .unwrap();
    let contexts = graph
        .search_space()
        .unwrap()
        .bucket_contexts(&graph.dyn_map);
    rt.begin_profile_replay(&contexts).unwrap();
    assert_eq!(rt.profile_replay.as_ref().unwrap().owned_device_bytes(), 32);
    let view = CudaRuntime::input_device_buffer(
        input.id,
        &rt.cuda_stream,
        &rt.hlir_buffers,
        &rt.external_buffers,
        &rt.prepared_inputs,
    )
    .unwrap();
    assert_eq!(
        (view.ptr(), view.len(), view.capacity(), view.input_format()),
        (binding.0, 16, 32, Some("test.dual.v1"))
    );
    rt.finish_profile_replay();
    assert_eq!(before, rt.current_resource_input_signature());
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(2).trials(1),
        &mut SmallRng::seed_from_u64(671),
    );
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![3., 4., 5., 6.]);
    rt.set_data(input, vec![7f32; 4]);
    assert!(!rt.prepared_inputs.contains_key(&input.id));
    assert_eq!(rt.current_hlir_device_binding(input.id).unwrap(), binding);
    assert_ne!(before, rt.current_resource_input_signature());
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![8.; 4]);
}

#[test]
fn prepared_input_rejects_writes_and_overlapping_output() {
    let mut rt = runtime();
    let input = NodeIndex::new(721);
    let allocation = rt.cuda_stream.alloc_zeros::<u8>(64).unwrap();
    let ptr = allocation.device_ptr(&rt.cuda_stream).0;
    rt.set_prepared_buffer(input, allocation, 16, "test.dual.v1");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        rt.set_output_device_ptr(NodeIndex::new(722), ptr + 32, 4);
    }));
    assert!(result.is_err());
    let mut llir = LLIRGraph::default();
    let source = llir.add_node(LLIROp::new(Box::new(Input {
        node: input.index(),
        label: String::new(),
        dtype: DType::Int,
    })));
    let op = llir.add_node(Probe::default().to_llir_op());
    llir.add_edge(source, op, ());
    llir.add_edge(source, op, ());
    assert!(rt.validate_prepared_input_effects(&llir).is_err());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rt.copy_output_to_input(NodeIndex::new(722), input);
    }));
    assert!(result.is_err());
}

#[test]
fn reused_replay_inputs_reset_each_trial_and_restore_after_failure() {
    for fail in [false, true] {
        let mut graph = Graph::new();
        let state = graph.tensor(1).as_dtype(DType::Int).persist();
        let meta = graph.tensor(1).as_dtype(DType::Int).persist();
        let probe = Probe {
            fail,
            ..Probe::default()
        };
        graph
            .custom_op(probe.clone(), (state.id, meta.id), 1, DType::F32)
            .output();
        graph.build_search_space::<CudaRuntime>(CompileOptions::default());
        let mut rt = runtime();
        rt.set_data(state, vec![101i32]);
        rt.set_data_with_host_mirror(meta, vec![202i32]);
        let original = rt.current_hlir_device_binding(state.id).unwrap();
        let restore = rt.capture_profile_inputs(&[state], &graph.dyn_map).unwrap();
        let workload = ProfileWorkload::new()
            .device_snapshots(true)
            .reuse_input_buffers(restore)
            .case(
                "a",
                DynMap::default(),
                ProfileInputs::new()
                    .input(state, vec![7i32])
                    .mirrored_input(meta, vec![11i32]),
                1.,
            )
            .case(
                "b",
                DynMap::default(),
                ProfileInputs::new()
                    .input(state, vec![19i32])
                    .mirrored_input(meta, vec![23i32]),
                1.,
            );
        rt.set_profile_workload(&graph, workload).unwrap();
        let space = graph.search_space().unwrap();
        let contexts = space.bucket_contexts(&graph.dyn_map);
        let llir =
            luminal::search::extract_one(space, &contexts[0], &mut SmallRng::seed_from_u64(12));
        rt.begin_profile_replay(&contexts).unwrap();
        assert_eq!(
            rt.profile_replay
                .as_ref()
                .unwrap()
                .slots
                .iter()
                .flatten()
                .count(),
            1
        );
        assert!(rt.profile_replay.as_ref().unwrap().snapshots.is_empty());
        rt.activate_profile_case(0).unwrap();
        assert_eq!(rt.profile_replay.as_ref().unwrap().snapshots.len(), 2);
        let first_keys: FxHashSet<_> = rt
            .profile_replay
            .as_ref()
            .unwrap()
            .snapshots
            .keys()
            .copied()
            .collect();
        rt.activate_profile_case(1).unwrap();
        let second_keys: FxHashSet<_> = rt
            .profile_replay
            .as_ref()
            .unwrap()
            .snapshots
            .keys()
            .copied()
            .collect();
        assert_eq!(second_keys.len(), 2);
        assert!(
            first_keys.is_disjoint(&second_keys),
            "inactive snapshots must be released"
        );
        rt.activate_profile_case(0).unwrap();
        assert_eq!(rt.current_hlir_device_binding(state.id).unwrap(), original);
        for _ in 0..2 {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rt.evaluate_profile_workload(
                    &llir,
                    &contexts[0],
                    &CompileOptions::default().trials(3),
                    false,
                )
            }));
        }
        rt.finish_profile_replay();
        assert_eq!(rt.current_hlir_device_binding(state.id).unwrap(), original);
        assert_eq!(rt.get_i32(state), vec![101]);
        assert_eq!(rt.hlir_host_mirrors[&meta.id], 202i32.to_ne_bytes());
        let seen = probe.seen.lock().unwrap();
        assert!(!seen.is_empty());
        assert!(
            seen.iter().all(|v| *v == (7, 11, 0) || *v == (19, 23, 0)),
            "{seen:?}"
        );
    }
}

#[test]
fn reused_replay_rejects_external_aliases_and_oversized_cases() {
    let mut graph = Graph::new();
    let a = graph.tensor('s').persist();
    let b = graph.tensor(1).persist();
    (a * 2.).output();
    b.output();
    graph.set_dim('s', 1);
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    rt.set_data(a, vec![3f32]);
    rt.set_data(b, vec![4f32]);
    let restore = rt.capture_profile_inputs(&[a], &graph.dyn_map).unwrap();
    for oversized in [false, true] {
        let ptr = rt.current_hlir_device_binding(a.id).unwrap().0;
        if !oversized {
            unsafe {
                rt.set_device_ptr(b, ptr, 4);
            }
        } else {
            rt.set_data(b, vec![4f32]);
        }
        rt.set_profile_workload(
            &graph,
            ProfileWorkload::new()
                .reuse_input_buffers(restore.clone())
                .shared_input(b)
                .case(
                    "bad",
                    dims(if oversized { 2 } else { 1 }),
                    ProfileInputs::new().input(a, vec![1f32; if oversized { 2 } else { 1 }]),
                    1.,
                ),
        )
        .unwrap();
        let contexts = graph
            .search_space()
            .unwrap()
            .bucket_contexts(&graph.dyn_map);
        assert!(rt.begin_profile_replay(&contexts).is_err());
        assert!(rt.profile_replay.is_none());
        let mut actual = [0f32];
        unsafe {
            result::memcpy_dtoh_sync(&mut actual, ptr).unwrap();
        }
        assert_eq!(actual, [3.]);
    }
}

#[test]
fn prepared_unified_owner_survives_replay_and_releases_on_replacement() {
    let mut graph = Graph::new();
    let input = graph.tensor(4).persist();
    let output = (input + 1.).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default());
    let mut rt = runtime();
    let mut buffer = unsafe {
        rt.cuda_stream
            .context()
            .alloc_unified::<u8>(32, true)
            .unwrap()
    };
    let bytes: Vec<_> = [5f32; 8].iter().flat_map(|x| x.to_ne_bytes()).collect();
    rt.cuda_stream.memcpy_htod(&bytes, &mut buffer).unwrap();
    let ptr = buffer.device_ptr(&rt.cuda_stream).0;
    rt.set_prepared_unified_buffer(input, buffer, 16, "test.dual.f32");
    assert_eq!(rt.input_buffer(input).unwrap().capacity(), 32);
    rt.set_profile_workload(
        &graph,
        ProfileWorkload::new().shared_input(input).case(
            "shared",
            DynMap::default(),
            ProfileInputs::new(),
            1.,
        ),
    )
    .unwrap();
    let contexts = graph
        .search_space()
        .unwrap()
        .bucket_contexts(&graph.dyn_map);
    let original_limits = rt.device_resource_limits;
    rt.device_resource_limits
        .as_mut()
        .unwrap()
        .max_candidate_memory_bytes = 16;
    // The 32-byte managed owner is allowed to exceed the simulated physical
    // budget. It must not eliminate room for non-evictable graph allocations.
    assert_eq!(
        rt.candidate_device_resource_limits()
            .unwrap()
            .max_candidate_memory_bytes,
        16
    );
    let before = rt
        .candidate_device_resource_limits()
        .unwrap()
        .max_candidate_memory_bytes;
    rt.begin_profile_replay(&contexts).unwrap();
    assert_eq!(
        rt.candidate_device_resource_limits()
            .unwrap()
            .max_candidate_memory_bytes,
        before
    );
    assert!(rt.prepared_unified_owners.is_empty());
    assert_eq!(rt.profile_replay.as_ref().unwrap().owned_device_bytes(), 32);
    rt.finish_profile_replay();
    assert_eq!(rt.prepared_unified_owners.len(), 1);
    rt.device_resource_limits = original_limits;
    rt = graph.search_with_rng(
        rt,
        CompileOptions::default().search_graph_limit(1).trials(2),
        &mut SmallRng::seed_from_u64(74),
    );
    assert_eq!(rt.input_buffer(input).unwrap().ptr(), ptr);
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![6.; 4]);
    rt.clear_profile_workload();
    rt.set_data(input, vec![9f32; 4]);
    assert!(rt.prepared_unified_owners.is_empty());
    assert!(rt.prepared_inputs.is_empty());
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_f32(output), vec![10.; 4]);
}

#[test]
fn profile_storage_retains_owned_allocation_and_shared_alias_identity() {
    let bytes = vec![0u8; 1024 * 1024];
    let address = bytes.as_ptr();
    let storage = ProfileStorage::new(bytes);
    assert_eq!(
        storage.0.as_ptr(),
        address,
        "moving a snapshot must not copy its bytes"
    );
    let alias = storage.clone();
    assert!(Arc::ptr_eq(&storage.0, &alias.0));
    assert_eq!(alias.0.len(), 1024 * 1024);
    drop(storage);
    assert!(alias.0.iter().all(|&b| b == 0));
}

#[test]
fn large_zero_capture_preserves_overlapping_views_after_device_mutation() {
    let mut graph = Graph::new();
    let n = 256 * 1024;
    let a = graph.tensor(n).persist();
    let b = graph.tensor(n).persist();
    (a + b).output();
    graph.build_search_space::<CudaRuntime>(CompileOptions::default().search_log(false));
    let mut rt = runtime();
    let mut backing = rt.cuda_stream.alloc_zeros::<f32>(n + 2).unwrap();
    let ptr = backing.device_ptr(&rt.cuda_stream).0;
    unsafe {
        rt.set_device_ptr(a, ptr, n * 4);
        rt.set_device_ptr(b, ptr + 8, n * 4);
    }
    let captured = rt.capture_profile_inputs(&[a, b], &graph.dyn_map).unwrap();
    let groups = captured.groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].0.len(), (n + 2) * 4);
    assert_eq!(captured.bindings[0].range, 0..n * 4);
    assert_eq!(captured.bindings[1].range, 8..(n + 2) * 4);
    rt.cuda_stream
        .memcpy_htod(&vec![7.0f32; n + 2], &mut backing)
        .unwrap();
    rt.cuda_stream.synchronize().unwrap();
    assert!(groups[0].0.iter().all(|&byte| byte == 0));
    assert!(Arc::ptr_eq(
        &captured.bindings[0].storage.0,
        &captured.bindings[1].storage.0
    ));
}
