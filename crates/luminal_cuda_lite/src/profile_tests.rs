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
    for graph in [false, true] {
        let evaluations: Vec<_> = rt
            .profile_evaluations()
            .iter()
            .filter(|e| e.cuda_graph == graph)
            .collect();
        assert!(!evaluations.is_empty());
        for evaluation in evaluations {
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
    assert_eq!(rt.profile_replay.as_ref().unwrap().snapshots.len(), 2);
    rt.activate_profile_case(0).unwrap();
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
struct MetadataCopy;
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
fn profile_capture_preparation_uses_exact_metadata() {
    let mut graph = Graph::new();
    let input = graph.tensor('s').as_dtype(DType::Int).persist();
    let metadata = graph.tensor(1).as_dtype(DType::Int).persist();
    let output = graph
        .custom_op(MetadataCopy, (input.id, metadata.id), 's', DType::Int)
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
    rt.execute(&graph.dyn_map);
    assert_eq!(rt.get_i32(output), vec![19]);
}
