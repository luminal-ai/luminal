use crate::{kernel::lower_expression_for_metal, runtime::MetalRuntime};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use half::{bf16, f16};
use luminal::prelude::*;
use proptest::prelude::*;
use rand::{SeedableRng, rngs::StdRng};
use safetensors::{Dtype, tensor::TensorView};
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static SAFETENSORS_TEST_FILE_ID: AtomicUsize = AtomicUsize::new(0);

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_err = diff / e.abs().max(1.0);
        assert!(
            rel_err < tolerance,
            "Mismatch at index {}: got {}, expected {}, rel_err={}",
            i,
            a,
            e,
            rel_err
        );
    }
}

fn bytes_of<T: bytemuck::NoUninit>(values: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn search_candidates(cx: &mut Graph, rt: MetalRuntime, limit: usize) -> MetalRuntime {
    let mut rng = StdRng::seed_from_u64(0);
    cx.search_options(rt, SearchOptions::new(limit), &mut rng)
}

fn egraph_has_op(cx: &Graph, op_name: &str) -> bool {
    cx.egraph()
        .expect("search space should be built")
        .enodes
        .values()
        .any(|(label, _)| label == op_name)
}

fn assert_matmul_options(cx: &Graph, mps_op_name: &str) {
    assert!(
        egraph_has_op(cx, mps_op_name),
        "expected {mps_op_name} rewrite option in e-graph"
    );
    assert!(
        egraph_has_op(cx, "GenericMatmul"),
        "expected GenericMatmul rewrite option in e-graph"
    );
}

fn write_test_safetensors(tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> PathBuf {
    let tensor_views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                (*name).to_string(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::serialize(&tensor_views, None).unwrap();
    let id = SAFETENSORS_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!(
        "luminal_metal_runtime_{}_{}.safetensors",
        std::process::id(),
        id
    ));
    std::fs::write(&path, serialized).unwrap();
    path
}

const TRANSFORMER_SEQ: usize = 4;
const TRANSFORMER_HIDDEN: usize = 16;
const TRANSFORMER_INTERMEDIATE: usize = 32;

fn rms_norm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let normed = x.std_norm(x.shape.last_axis(), eps);
    normed * weight.expand_lhs(&x.dims()[..x.dims().len() - 1])
}

fn self_attention(
    x: GraphTensor,
    wq: GraphTensor,
    wk: GraphTensor,
    wv: GraphTensor,
    wo: GraphTensor,
) -> GraphTensor {
    let q = x.matmul(wq.t());
    let k = x.matmul(wk.t());
    let v = x.matmul(wv.t());

    let scale = 1.0 / (TRANSFORMER_HIDDEN as f32).sqrt();
    let scores = q.matmul(k.t()) * scale;
    let attn_weights = scores.softmax(1);
    attn_weights.matmul(v).matmul(wo.t())
}

fn swiglu_mlp(
    x: GraphTensor,
    w_gate: GraphTensor,
    w_up: GraphTensor,
    w_down: GraphTensor,
) -> GraphTensor {
    let gate = x.matmul(w_gate.t()).swish();
    let up = x.matmul(w_up.t());
    (gate * up).matmul(w_down.t())
}

struct MiniTransformerLayer {
    attn_norm_w: GraphTensor,
    wq: GraphTensor,
    wk: GraphTensor,
    wv: GraphTensor,
    wo: GraphTensor,
    mlp_norm_w: GraphTensor,
    w_gate: GraphTensor,
    w_up: GraphTensor,
    w_down: GraphTensor,
}

impl MiniTransformerLayer {
    fn init(cx: &mut Graph) -> Self {
        Self {
            attn_norm_w: cx.tensor(TRANSFORMER_HIDDEN),
            wq: cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN)),
            wk: cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN)),
            wv: cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN)),
            wo: cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN)),
            mlp_norm_w: cx.tensor(TRANSFORMER_HIDDEN),
            w_gate: cx.tensor((TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN)),
            w_up: cx.tensor((TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN)),
            w_down: cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_INTERMEDIATE)),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let normed = rms_norm(x, self.attn_norm_w, 1e-5);
        let attn_out = self_attention(normed, self.wq, self.wk, self.wv, self.wo);
        let x = x + attn_out;

        let normed = rms_norm(x, self.mlp_norm_w, 1e-5);
        let mlp_out = swiglu_mlp(normed, self.w_gate, self.w_up, self.w_down);
        x + mlp_out
    }

    fn weights(&self) -> Vec<(GraphTensor, usize)> {
        vec![
            (self.attn_norm_w, TRANSFORMER_HIDDEN),
            (self.wq, TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN),
            (self.wk, TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN),
            (self.wv, TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN),
            (self.wo, TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN),
            (self.mlp_norm_w, TRANSFORMER_HIDDEN),
            (self.w_gate, TRANSFORMER_INTERMEDIATE * TRANSFORMER_HIDDEN),
            (self.w_up, TRANSFORMER_INTERMEDIATE * TRANSFORMER_HIDDEN),
            (self.w_down, TRANSFORMER_HIDDEN * TRANSFORMER_INTERMEDIATE),
        ]
    }
}

fn rms_norm_ref(x: &CandleTensor, weight: &CandleTensor, eps: f64) -> CandleTensor {
    let dims = x.dims();
    let last_dim = dims[dims.len() - 1];
    let sq_mean = x.sqr().unwrap().mean_keepdim(dims.len() - 1).unwrap();
    let rsqrt = (sq_mean + eps).unwrap().sqrt().unwrap().recip().unwrap();
    let normed = x.broadcast_mul(&rsqrt).unwrap();
    normed
        .broadcast_mul(&weight.reshape((1, last_dim)).unwrap())
        .unwrap()
}

fn self_attention_ref(
    x: &CandleTensor,
    wq: &CandleTensor,
    wk: &CandleTensor,
    wv: &CandleTensor,
    wo: &CandleTensor,
) -> CandleTensor {
    let q = x.matmul(&wq.t().unwrap()).unwrap();
    let k = x.matmul(&wk.t().unwrap()).unwrap();
    let v = x.matmul(&wv.t().unwrap()).unwrap();

    let scale = 1.0 / (TRANSFORMER_HIDDEN as f64).sqrt();
    let scores = q.matmul(&k.t().unwrap()).unwrap();
    let scores = (scores * scale).unwrap();

    let max_val = scores.max(1).unwrap().unsqueeze(1).unwrap();
    let shifted = scores.broadcast_sub(&max_val).unwrap();
    let exps = shifted.exp().unwrap();
    let sum_exps = exps.sum(1).unwrap().unsqueeze(1).unwrap();
    let attn_weights = exps.broadcast_div(&sum_exps).unwrap();

    attn_weights
        .matmul(&v)
        .unwrap()
        .matmul(&wo.t().unwrap())
        .unwrap()
}

fn swiglu_mlp_ref(
    x: &CandleTensor,
    w_gate: &CandleTensor,
    w_up: &CandleTensor,
    w_down: &CandleTensor,
) -> CandleTensor {
    let gate = x.matmul(&w_gate.t().unwrap()).unwrap().silu().unwrap();
    let up = x.matmul(&w_up.t().unwrap()).unwrap();
    (gate * up).unwrap().matmul(&w_down.t().unwrap()).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn transformer_layer_ref(
    x: &CandleTensor,
    attn_norm_w: &CandleTensor,
    wq: &CandleTensor,
    wk: &CandleTensor,
    wv: &CandleTensor,
    wo: &CandleTensor,
    mlp_norm_w: &CandleTensor,
    w_gate: &CandleTensor,
    w_up: &CandleTensor,
    w_down: &CandleTensor,
) -> CandleTensor {
    let normed = rms_norm_ref(x, attn_norm_w, 1e-5);
    let attn_out = self_attention_ref(&normed, wq, wk, wv, wo);
    let x = (x + attn_out).unwrap();

    let normed = rms_norm_ref(&x, mlp_norm_w, 1e-5);
    let mlp_out = swiglu_mlp_ref(&normed, w_gate, w_up, w_down);
    (x + mlp_out).unwrap()
}

fn seeded_data(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + 11) % 97) as f32 / 97.0) * scale + bias)
        .collect()
}

fn to_f16_vec(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

fn generate_layer_weights(layer: &MiniTransformerLayer) -> Vec<(GraphTensor, Vec<f32>)> {
    layer
        .weights()
        .iter()
        .enumerate()
        .map(|(i, (tensor, size))| {
            let data = seeded_data(*size, 0.8 - i as f32 * 0.03, -0.4 + i as f32 * 0.02);
            let data = if *size == TRANSFORMER_HIDDEN {
                data.iter().map(|x| x + 1.0).collect::<Vec<_>>()
            } else {
                data
            };
            (*tensor, data)
        })
        .collect()
}

/// dynamic symbols in kernel expressions should route through dyn buffer.
#[test]
fn dynamic_const_codegen_uses_dyn_buffer() {
    let expr = (Expression::from('a') * 2 + Expression::from('z')).simplify();
    let code = lower_expression_for_metal(&expr, "idx");

    assert!(
        !code.contains("*const_"),
        "dynamic symbols should be lowered via dyn buffer, got: {code}"
    );
    assert!(
        code.contains("dyn["),
        "expected generated kernel expression to reference dyn buffer, got: {code}"
    );
}

/// dynamic-dimension reduction should compile and execute on Metal.
#[test]
fn dynamic_dim_sum_reduce_runs() {
    let mut cx = Graph::default();
    cx.set_dim('a', 3);
    let input = cx.tensor(('a', 2));
    let output = input.sum(0).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[9.0, 12.0], 0.001);
}

#[test]
fn metal_bucketed_dynamic_dim_dispatches_correct_graph() {
    let mut cx = Graph::default();
    let input = cx.tensor(('s', 4));
    let output = (input + input).output();

    cx.set_dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]);
    cx.set_dim('s', 1);
    cx.build_search_space::<MetalRuntime>();

    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, vec![1.0f32; 4]);
    rt = cx.search(rt, 5);

    cx.set_dim('s', 1);
    let s1_input = vec![1.0, 2.0, 3.0, 4.0];
    rt.set_data(input, s1_input.clone());
    rt.execute(&cx.dyn_map);
    let s1_out = rt.get_f32(output);
    assert_close(&s1_out[..4], &[2.0, 4.0, 6.0, 8.0], 0.001);

    cx.set_dim('s', 3);
    let s3_input: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let s3_expected: Vec<f32> = s3_input.iter().map(|v| v * 2.0).collect();
    rt.set_data(input, s3_input);
    rt.execute(&cx.dyn_map);
    let s3_out = rt.get_f32(output);
    assert_close(&s3_out[..12], &s3_expected, 0.001);
}

#[test]
fn metal_int_arithmetic_preserves_large_values() {
    let mut cx = Graph::default();
    let token = cx.tensor(1).as_dtype(DType::Int);
    let large_index = (token * 1024) + 123;
    let mod_output = (large_index % 65_537).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(token, &[16_385i32]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_eq!(rt.get_f32(mod_output), vec![891.0]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Test basic addition: input + input = 2 * input
    #[test]
    fn metal_add_test(len in 1usize..32, values in proptest::collection::vec(-5.0f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input + input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * 2.0).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test basic multiplication: input * input = input^2
    #[test]
    fn metal_mul_test(len in 1usize..32, values in proptest::collection::vec(0.1f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input * input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * v).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test exp2: 2^x
    #[test]
    fn metal_exp2_test(len in 1usize..32, values in proptest::collection::vec(-3.0f32..3.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = input.exp2().output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| 2.0f32.powf(*v)).collect();
        assert_close(&out, &expected, 0.001);
    }
}

#[test]
fn metal_build_search_space_accepts_memory_budget() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    (a * b).output();

    cx.build_search_space_with_options::<MetalRuntime>(
        BuildSearchSpaceOptions::new().max_memory_mib(1),
    );
}

/// Simple deterministic test for add
#[test]
fn metal_simple_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a + b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
}

/// Simple deterministic test for mul
#[test]
fn metal_simple_mul() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a * b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
}

/// Simple deterministic test for exp2
#[test]
fn metal_simple_exp2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.exp2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[0.0, 1.0, 2.0, 3.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 4.0, 8.0], 0.001);
}

#[test]
fn metal_simple_log2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.log2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 2.0, 3.0], 0.001);
}

#[test]
fn metal_simple_sin() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sin().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(
        input,
        &[
            0.0,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            3.0 * std::f32::consts::FRAC_PI_2,
        ],
    );
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 0.0, -1.0], 0.01);
}

#[test]
fn metal_simple_sqrt() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sqrt().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 4.0, 9.0, 16.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.001);
}

#[test]
fn metal_simple_recip() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.reciprocal().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 0.5, 0.25, 0.2], 0.001);
}

#[test]
fn metal_simple_mod() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a % b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[7.0, 10.0, 15.0, 8.5]);
    rt.set_data(b, &[3.0, 4.0, 6.0, 2.5]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 1.0], 0.001);
}

#[test]
fn metal_simple_less_than() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = a.lt(b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 5.0, 3.0, 4.0]);
    rt.set_data(b, &[2.0, 3.0, 3.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    // 1 < 2 = true (1.0), 5 < 3 = false (0.0), 3 < 3 = false (0.0), 4 < 5 = true (1.0)
    assert_eq!(out, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn metal_simple_sum_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // sum over axis 1
    let output = input.sum(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,2,3,4], [5,6,7,8]] -> [10, 26]
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[10.0, 26.0], 0.001);
}

#[test]
fn metal_simple_max_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // max over axis 1
    let output = input.max(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,4,2,3], [8,5,7,6]] -> [4, 8]
    rt.set_data(input, &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[4.0, 8.0], 0.001);
}

#[test]
fn metal_f16_cast_roundtrip() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.cast(DType::F16).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, -2.5, 3.25, 4.75]);
    rt = cx.search(rt, 3);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, -2.5, 3.25, 4.75], 0.002);
}

#[test]
fn metal_f16_intermediate_add_roundtrip() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a.cast(DType::F16) + b.cast(DType::F16))
        .cast(DType::F32)
        .output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, -3.0, 4.5]);
    rt.set_data(b, &[0.5, -1.0, 3.0, 0.25]);
    rt = cx.search(rt, 3);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.5, 1.0, 0.0, 4.75], 0.003);
}

#[test]
fn metal_specialized_matmul() {
    let mut cx = Graph::default();
    let a = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let b = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN));
    let output = a.matmul(b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let b_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.8, -0.4);

    rt.set_data(a, &a_data);
    rt.set_data(b, &b_data);
    rt = search_candidates(&mut cx, rt, 32);
    assert!(
        rt.contains_matmul(),
        "expected Metal runtime to fuse matmul, kernels: {:?}",
        rt.debug_kernel_ops()
    );
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_a =
        CandleTensor::from_vec(a_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_b =
        CandleTensor::from_vec(b_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let expected = ref_a.matmul(&ref_b).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_regular_tiled_matmul_path() {
    let mut cx = Graph::default();
    let m = 64;
    let k = 64;
    let n = 64;
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let output = a.matmul(b).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSMatmul");
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(m * k, 0.4, -0.2);
    let b_data = seeded_data(k * n, 0.3, -0.15);

    rt.set_data(a, &a_data);
    rt.set_data(b, &b_data);
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
    let ref_b = CandleTensor::from_vec(b_data, (k, n), &device).unwrap();
    let expected = ref_a.matmul(&ref_b).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 2e-3);
}

#[test]
fn metal_mps_matmul_transposed_rhs_weight_layout() {
    let mut cx = Graph::default();
    let m = 7;
    let k = 11;
    let n = 13;
    let a = cx.tensor((m, k));
    let weight = cx.tensor((n, k));
    let output = a.matmul(weight.t()).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSMatmul");
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(m * k, 0.35, -0.17);
    let weight_data = seeded_data(n * k, 0.21, -0.09);

    rt.set_data(a, &a_data);
    rt.set_data(weight, &weight_data);
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
    let ref_weight = CandleTensor::from_vec(weight_data, (n, k), &device).unwrap();
    let expected = ref_a.matmul(&ref_weight.t().unwrap()).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_mps_matmul_transposed_lhs_layout() {
    let mut cx = Graph::default();
    let m = 5;
    let k = 9;
    let n = 6;
    let lhs_storage = cx.tensor((k, m));
    let rhs = cx.tensor((k, n));
    let output = lhs_storage.t().matmul(rhs).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSMatmul");
    let mut rt = MetalRuntime::initialize(());

    let lhs_data = seeded_data(k * m, 0.31, -0.12);
    let rhs_data = seeded_data(k * n, 0.27, -0.08);

    rt.set_data(lhs_storage, &lhs_data);
    rt.set_data(rhs, &rhs_data);
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_lhs = CandleTensor::from_vec(lhs_data, (k, m), &device)
        .unwrap()
        .t()
        .unwrap();
    let ref_rhs = CandleTensor::from_vec(rhs_data, (k, n), &device).unwrap();
    let expected = ref_lhs.matmul(&ref_rhs).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_mps_batched_matmul_row_row_layout() {
    let mut cx = Graph::default();
    let batch = 3;
    let m = 4;
    let k = 5;
    let n = 6;
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let output = a.matmul(b).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSBatchedMatmul");
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(batch * m * k, 0.17, -0.08);
    let b_data = seeded_data(batch * k * n, 0.11, -0.05);
    rt.set_data(a, &a_data);
    rt.set_data(b, &b_data);
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(output);

    let mut expected = vec![0.0; batch * m * n];
    for batch_idx in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += a_data[batch_idx * m * k + row * k + inner]
                        * b_data[batch_idx * k * n + inner * n + col];
                }
                expected[batch_idx * m * n + row * n + col] = sum;
            }
        }
    }

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_generic_matmul_covers_noncontiguous_merged_head_projection() {
    let mut cx = Graph::default();
    let heads = 3;
    let seq = 4;
    let head_dim = 5;
    let hidden = heads * head_dim;
    let out_dim = 7;
    let attn = cx.tensor((heads, seq, head_dim));
    let weight = cx.tensor((out_dim, hidden));
    let merged = attn.transpose(0, 1).merge_dims(1, 2);
    let output = merged.matmul(weight.t()).output();

    cx.build_search_space::<MetalRuntime>();
    assert!(
        egraph_has_op(&cx, "GenericMatmul"),
        "expected GenericMatmul rewrite option in e-graph"
    );
    let mut rt = MetalRuntime::initialize(());

    let attn_data = seeded_data(heads * seq * head_dim, 0.19, -0.09);
    let weight_data = seeded_data(out_dim * hidden, 0.14, -0.06);
    rt.set_data(attn, &attn_data);
    rt.set_data(weight, &weight_data);
    rt = search_candidates(&mut cx, rt, 32);

    let kernels = rt.debug_kernel_ops();
    assert!(
        kernels.iter().any(|k| k.contains("GenericMatmul")),
        "expected generic matmul fallback for non-contiguous merged-head projection, kernels: {:?}",
        kernels
    );
    assert!(
        !kernels.iter().any(|k| {
            k.contains("MetalMul") && k.contains(&format!("shape: [{seq}, {out_dim}, {hidden}]"))
        }),
        "generic fallback should remove the broadcast multiply intermediate, kernels: {:?}",
        kernels
    );

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(output);

    let mut expected = vec![0.0; seq * out_dim];
    for token in 0..seq {
        for out_col in 0..out_dim {
            let mut sum = 0.0;
            for inner in 0..hidden {
                let head = inner / head_dim;
                let dim = inner % head_dim;
                let attn_idx = head * seq * head_dim + token * head_dim + dim;
                sum += attn_data[attn_idx] * weight_data[out_col * hidden + inner];
            }
            expected[token * out_dim + out_col] = sum;
        }
    }

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_mps_batched_matmul_transposed_rhs_layout() {
    let mut cx = Graph::default();
    let batch = 4;
    let m = 3;
    let k = 7;
    let n = 5;
    let a = cx.tensor((batch, m, k));
    let weight = cx.tensor((batch, n, k));
    let output = a.matmul(weight.permute((0, 2, 1))).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSBatchedMatmul");
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(batch * m * k, 0.13, -0.06);
    let weight_data = seeded_data(batch * n * k, 0.09, -0.04);
    rt.set_data(a, &a_data);
    rt.set_data(weight, &weight_data);
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(output);

    let mut expected = vec![0.0; batch * m * n];
    for batch_idx in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += a_data[batch_idx * m * k + row * k + inner]
                        * weight_data[batch_idx * n * k + col * k + inner];
                }
                expected[batch_idx * m * n + row * n + col] = sum;
            }
        }
    }

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_mps_matmul_f16_transposed_rhs_weight_layout() {
    let mut cx = Graph::default();
    let m = 6;
    let k = 10;
    let n = 7;
    let a = cx.tensor((m, k)).as_dtype(DType::F16);
    let weight = cx.tensor((n, k)).as_dtype(DType::F16);
    let output = a.matmul(weight.t()).cast(DType::F32).output();

    cx.build_search_space::<MetalRuntime>();
    assert_matmul_options(&cx, "MPSMatmul");
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(m * k, 0.22, -0.07);
    let weight_data = seeded_data(n * k, 0.18, -0.05);

    rt.set_data(a, to_f16_vec(&a_data));
    rt.set_data(weight, to_f16_vec(&weight_data));
    rt = search_candidates(&mut cx, rt, 32);

    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_a = CandleTensor::from_vec(a_data, (m, k), &device).unwrap();
    let ref_weight = CandleTensor::from_vec(weight_data, (n, k), &device).unwrap();
    let expected = ref_a.matmul(&ref_weight.t().unwrap()).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 5e-3);
}

#[test]
fn metal_rms_norm() {
    let mut cx = Graph::default();
    let input = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let weight = cx.tensor(TRANSFORMER_HIDDEN);
    let output = rms_norm(input, weight, 1e-5).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let weight_data: Vec<f32> = seeded_data(TRANSFORMER_HIDDEN, 0.5, 0.75);

    rt.set_data(input, &input_data);
    rt.set_data(weight, &weight_data);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_weight = CandleTensor::from_vec(weight_data, TRANSFORMER_HIDDEN, &device).unwrap();
    let expected = rms_norm_ref(&ref_input, &ref_weight, 1e-5);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_self_attention() {
    let mut cx = Graph::default();
    let input = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let wq = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN));
    let wk = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN));
    let wv = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN));
    let wo = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN));
    let output = self_attention(input, wq, wk, wv, wo).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let wq_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.8, -0.4);
    let wk_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.7, -0.35);
    let wv_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.6, -0.3);
    let wo_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.5, -0.25);

    rt.set_data(input, &input_data);
    rt.set_data(wq, &wq_data);
    rt.set_data(wk, &wk_data);
    rt.set_data(wv, &wv_data);
    rt.set_data(wo, &wo_data);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wq =
        CandleTensor::from_vec(wq_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wk =
        CandleTensor::from_vec(wk_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wv =
        CandleTensor::from_vec(wv_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wo =
        CandleTensor::from_vec(wo_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let expected = self_attention_ref(&ref_input, &ref_wq, &ref_wk, &ref_wv, &ref_wo);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2);
}

#[test]
fn metal_self_attention_f16_weights() {
    let mut cx = Graph::default();
    let input = cx
        .tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN))
        .as_dtype(DType::F16);
    let wq = cx
        .tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN))
        .as_dtype(DType::F16);
    let wk = cx
        .tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN))
        .as_dtype(DType::F16);
    let wv = cx
        .tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN))
        .as_dtype(DType::F16);
    let wo = cx
        .tensor((TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN))
        .as_dtype(DType::F16);
    let output = self_attention(input, wq, wk, wv, wo)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let wq_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.8, -0.4);
    let wk_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.7, -0.35);
    let wv_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.6, -0.3);
    let wo_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_HIDDEN, 0.5, -0.25);

    rt.set_data(input, to_f16_vec(&input_data));
    rt.set_data(wq, to_f16_vec(&wq_data));
    rt.set_data(wk, to_f16_vec(&wk_data));
    rt.set_data(wv, to_f16_vec(&wv_data));
    rt.set_data(wo, to_f16_vec(&wo_data));
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wq =
        CandleTensor::from_vec(wq_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wk =
        CandleTensor::from_vec(wk_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wv =
        CandleTensor::from_vec(wv_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_wo =
        CandleTensor::from_vec(wo_data, (TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN), &device).unwrap();
    let expected = self_attention_ref(&ref_input, &ref_wq, &ref_wk, &ref_wv, &ref_wo);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 2e-2);
}

#[test]
fn metal_swiglu_mlp() {
    let mut cx = Graph::default();
    let input = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let w_gate = cx.tensor((TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN));
    let w_up = cx.tensor((TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN));
    let w_down = cx.tensor((TRANSFORMER_HIDDEN, TRANSFORMER_INTERMEDIATE));
    let output = swiglu_mlp(input, w_gate, w_up, w_down).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let gate_data = seeded_data(TRANSFORMER_INTERMEDIATE * TRANSFORMER_HIDDEN, 0.8, -0.4);
    let up_data = seeded_data(TRANSFORMER_INTERMEDIATE * TRANSFORMER_HIDDEN, 0.7, -0.35);
    let down_data = seeded_data(TRANSFORMER_HIDDEN * TRANSFORMER_INTERMEDIATE, 0.6, -0.3);

    rt.set_data(input, &input_data);
    rt.set_data(w_gate, &gate_data);
    rt.set_data(w_up, &up_data);
    rt.set_data(w_down, &down_data);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let ref_gate = CandleTensor::from_vec(
        gate_data,
        (TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN),
        &device,
    )
    .unwrap();
    let ref_up = CandleTensor::from_vec(
        up_data,
        (TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN),
        &device,
    )
    .unwrap();
    let ref_down = CandleTensor::from_vec(
        down_data,
        (TRANSFORMER_HIDDEN, TRANSFORMER_INTERMEDIATE),
        &device,
    )
    .unwrap();
    let expected = swiglu_mlp_ref(&ref_input, &ref_gate, &ref_up, &ref_down);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3);
}

#[test]
fn metal_mini_transformer_layer() {
    let mut cx = Graph::default();
    let input = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let layer = MiniTransformerLayer::init(&mut cx);
    let output = layer.forward(input).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let weight_data = generate_layer_weights(&layer);

    rt.set_data(input, &input_data);
    for (tensor, data) in &weight_data {
        rt.set_data(*tensor, data);
    }
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let w = |idx: usize, shape: &[usize]| {
        CandleTensor::from_vec(weight_data[idx].1.clone(), shape, &device).unwrap()
    };
    let expected = transformer_layer_ref(
        &ref_input,
        &w(0, &[TRANSFORMER_HIDDEN]),
        &w(1, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(2, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(3, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(4, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(5, &[TRANSFORMER_HIDDEN]),
        &w(6, &[TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN]),
        &w(7, &[TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN]),
        &w(8, &[TRANSFORMER_HIDDEN, TRANSFORMER_INTERMEDIATE]),
    );
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2);
}

#[test]
fn metal_mini_transformer_layer_f16_intermediate() {
    let mut cx = Graph::default();
    let input = cx.tensor((TRANSFORMER_SEQ, TRANSFORMER_HIDDEN));
    let layer = MiniTransformerLayer::init(&mut cx);

    let normed = rms_norm(input, layer.attn_norm_w, 1e-5).cast(DType::F16);
    let attn_out = self_attention(
        normed,
        layer.wq.cast(DType::F16),
        layer.wk.cast(DType::F16),
        layer.wv.cast(DType::F16),
        layer.wo.cast(DType::F16),
    )
    .cast(DType::F32);
    let x = input + attn_out;

    let normed = rms_norm(x, layer.mlp_norm_w, 1e-5).cast(DType::F16);
    let mlp_out = swiglu_mlp(
        normed,
        layer.w_gate.cast(DType::F16),
        layer.w_up.cast(DType::F16),
        layer.w_down.cast(DType::F16),
    )
    .cast(DType::F32);
    let output = (x + mlp_out).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());

    let input_data = seeded_data(TRANSFORMER_SEQ * TRANSFORMER_HIDDEN, 1.0, -0.5);
    let weight_data = generate_layer_weights(&layer);

    rt.set_data(input, &input_data);
    for (tensor, data) in &weight_data {
        rt.set_data(*tensor, data);
    }
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let device = CandleDevice::Cpu;
    let ref_input =
        CandleTensor::from_vec(input_data, (TRANSFORMER_SEQ, TRANSFORMER_HIDDEN), &device).unwrap();
    let w = |idx: usize, shape: &[usize]| {
        CandleTensor::from_vec(weight_data[idx].1.clone(), shape, &device).unwrap()
    };
    let expected = transformer_layer_ref(
        &ref_input,
        &w(0, &[TRANSFORMER_HIDDEN]),
        &w(1, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(2, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(3, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(4, &[TRANSFORMER_HIDDEN, TRANSFORMER_HIDDEN]),
        &w(5, &[TRANSFORMER_HIDDEN]),
        &w(6, &[TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN]),
        &w(7, &[TRANSFORMER_INTERMEDIATE, TRANSFORMER_HIDDEN]),
        &w(8, &[TRANSFORMER_HIDDEN, TRANSFORMER_INTERMEDIATE]),
    );
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 3e-2);
}

#[test]
fn test_scatter_basic() {
    let mut cx = Graph::default();
    let src = cx.tensor(3);
    let indexes = cx.tensor(3).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[10.0, 20.0, 30.0]);
    rt.set_data(indexes, &[1.0, 3.0, 4.0]);
    rt.set_data(dest, &[0.0, 0.0, 0.0, 0.0, 0.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[0.0, 10.0, 0.0, 20.0, 30.0], 0.001);
}

#[test]
fn test_scatter_buffer_roundtrip() {
    let mut cx = Graph::default();
    let src = cx.tensor(1);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let cache = cx.tensor(4).persist();
    let cache_out = src.scatter(indexes, cache);
    let read = cache_out.output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[0.0]);
    rt.set_data(indexes, &[0.0]);
    rt.set_zeros(cache, 4 * std::mem::size_of::<f32>());
    rt = cx.search(rt, 1);

    for (pos, value, expected) in [
        (0, 10.0, [10.0, 0.0, 0.0, 0.0]),
        (1, 20.0, [10.0, 20.0, 0.0, 0.0]),
        (2, 30.0, [10.0, 20.0, 30.0, 0.0]),
    ] {
        rt.set_data(src, &[value]);
        rt.set_data(indexes, &[pos as f32]);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(read), &expected, 0.001);

        let updated_cache = rt.remove_buffer(cache_out);
        rt.set_buffer(cache, updated_cache);
    }
}

#[test]
fn test_load_safetensors_f32_survives_search_and_overrides_input_data() {
    let mut cx = Graph::default();
    let weights = cx.named_tensor("weights", 3);
    let bias = cx.named_tensor("bias", 3);
    let out = (weights + bias).output();

    let weight_values = [1.25f32, -2.5, 4.0];
    let tensors = [("weights", Dtype::F32, vec![3], bytes_of(&weight_values))];
    let path = write_test_safetensors(&tensors);

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(weights, &[99.0, 99.0, 99.0]);
    rt.set_data(bias, &[0.5, 1.0, -1.5]);
    rt.load_safetensors(&cx, path.to_str().unwrap());
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out), &[1.75, -1.5, 2.5], 0.001);
    std::fs::remove_file(path).ok();
}

#[test]
fn test_load_safetensors_converts_supported_float_dtypes() {
    let mut cx = Graph::default();
    let f16_to_f32 = cx.named_tensor("f16_to_f32", 2);
    let bf16_to_f32 = cx.named_tensor("bf16_to_f32", 2);
    let f16_to_f16 = cx.named_tensor("f16_to_f16", 2).as_dtype(DType::F16);
    let f32_to_f16 = cx.named_tensor("f32_to_f16", 2).as_dtype(DType::F16);
    let bf16_to_f16 = cx.named_tensor("bf16_to_f16", 2).as_dtype(DType::F16);

    let f16_to_f32_out = (f16_to_f32 + 0.0).output();
    let bf16_to_f32_out = (bf16_to_f32 + 0.0).output();
    let f16_to_f16_out = f16_to_f16.cast(DType::F32).output();
    let f32_to_f16_out = f32_to_f16.cast(DType::F32).output();
    let bf16_to_f16_out = bf16_to_f16.cast(DType::F32).output();

    let f16_to_f32_values = [f16::from_f32(1.5), f16::from_f32(-2.25)];
    let bf16_to_f32_values = [bf16::from_f32(3.5), bf16::from_f32(-4.25)];
    let f16_to_f16_values = [f16::from_f32(5.5), f16::from_f32(-6.25)];
    let f32_to_f16_values = [7.5f32, -8.25];
    let bf16_to_f16_values = [bf16::from_f32(9.5), bf16::from_f32(-10.25)];
    let tensors = [
        (
            "f16_to_f32",
            Dtype::F16,
            vec![2],
            bytes_of(&f16_to_f32_values),
        ),
        (
            "bf16_to_f32",
            Dtype::BF16,
            vec![2],
            bytes_of(&bf16_to_f32_values),
        ),
        (
            "f16_to_f16",
            Dtype::F16,
            vec![2],
            bytes_of(&f16_to_f16_values),
        ),
        (
            "f32_to_f16",
            Dtype::F32,
            vec![2],
            bytes_of(&f32_to_f16_values),
        ),
        (
            "bf16_to_f16",
            Dtype::BF16,
            vec![2],
            bytes_of(&bf16_to_f16_values),
        ),
    ];
    let path = write_test_safetensors(&tensors);

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.load_safetensors(&cx, path.to_str().unwrap());
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(f16_to_f32_out), &[1.5, -2.25], 0.001);
    assert_close(&rt.get_f32(bf16_to_f32_out), &[3.5, -4.25], 0.001);
    assert_close(&rt.get_f32(f16_to_f16_out), &[5.5, -6.25], 0.001);
    assert_close(&rt.get_f32(f32_to_f16_out), &[7.5, -8.25], 0.001);
    assert_close(&rt.get_f32(bf16_to_f16_out), &[9.5, -10.25], 0.001);
    std::fs::remove_file(path).ok();
}

#[test]
fn test_gather_noncontiguous_data_uses_data_shape() {
    let mut cx = Graph::default();
    let input = cx.tensor((4, 3));
    let data = input.transpose(0, 1);
    let indexes = cx.tensor((2, 2)).as_dtype(DType::Int);
    let out = data.gather(indexes).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(
        input,
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );
    rt.set_data(indexes, &[0.0, 3.0, 4.0, 7.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out), &[0.0, 9.0, 1.0, 10.0], 0.001);
}

#[test]
fn test_scatter_into_nonzero_dest() {
    let mut cx = Graph::default();
    let src = cx.tensor(1);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[99.0]);
    rt.set_data(indexes, &[2f32]);
    rt.set_data(dest, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    rt = cx.search(rt, 1);
    let kernels = rt.debug_kernel_ops();
    assert!(
        kernels.iter().any(|k| k.contains("MetalScatterNoCopy")),
        "expected no-copy scatter for consumed destination, kernels: {:?}",
        kernels
    );
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[1.0, 2.0, 99.0, 4.0, 5.0], 0.001);
}

#[test]
fn test_scatter_no_copy_remove_buffer_aliases_dest() {
    let mut cx = Graph::default();
    let src = cx.tensor(2);
    let indexes = cx.tensor(2).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[7.0, 8.0]);
    rt.set_data(indexes, &[1.0, 3.0]);
    rt.set_data(dest, &[10.0, 20.0, 30.0, 40.0, 50.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let moved = rt.remove_buffer(result);
    let moved_values = unsafe {
        std::slice::from_raw_parts(
            moved.contents() as *const f32,
            moved.length() as usize / std::mem::size_of::<f32>(),
        )
        .to_vec()
    };
    assert_close(&moved_values, &[10.0, 7.0, 30.0, 8.0, 50.0], 0.001);
    rt.set_buffer(dest.id, moved);
}

#[test]
fn test_scatter_no_copy_handles_2d_destination() {
    let mut cx = Graph::default();
    let src = cx.tensor(2);
    let indexes = cx.tensor(2).as_dtype(DType::Int);
    let dest = cx.tensor((2, 3));
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[9.0, 8.0]);
    rt.set_data(indexes, &[2.0, 4.0]);
    rt.set_data(dest, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt = cx.search(rt, 1);
    let kernels = rt.debug_kernel_ops();
    assert!(
        kernels.iter().any(|k| k.contains("MetalScatterNoCopy")),
        "expected no-copy scatter for 2D destination, kernels: {:?}",
        kernels
    );
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(result), &[1.0, 2.0, 9.0, 4.0, 8.0, 6.0], 0.001);
}

#[test]
fn test_scatter_no_copy_not_selected_when_dest_has_another_consumer() {
    let mut cx = Graph::default();
    let src = cx.tensor(1);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let dest = cx.tensor(4);
    let scatter = src.scatter(indexes, dest).output();
    let dest_plus_one = (dest + 1.0).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[99.0]);
    rt.set_data(indexes, &[1.0]);
    rt.set_data(dest, &[10.0, 20.0, 30.0, 40.0]);
    rt = cx.search(rt, 1);
    let kernels = rt.debug_kernel_ops();
    assert!(
        !kernels.iter().any(|k| k.contains("MetalScatterNoCopy")),
        "no-copy scatter should not be selected when dest is also consumed, kernels: {:?}",
        kernels
    );
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(scatter), &[10.0, 99.0, 30.0, 40.0], 0.001);
    assert_close(&rt.get_f32(dest_plus_one), &[11.0, 21.0, 31.0, 41.0], 0.001);
}

#[test]
fn test_scatter_all_positions() {
    let mut cx = Graph::default();
    let src = cx.tensor(4);
    let indexes = cx.tensor(4).as_dtype(DType::Int);
    let dest = cx.tensor(4);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[40.0, 30.0, 20.0, 10.0]);
    rt.set_data(indexes, &[3.0, 2.0, 1.0, 0.0]);
    rt.set_data(dest, &[1.0, 2.0, 3.0, 4.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[10.0, 20.0, 30.0, 40.0], 0.001);
}

#[test]
fn test_gather_preserves_data_dtype() {
    let mut cx = Graph::default();
    let data = cx.tensor(2);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let out = data.gather(indexes).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(data, &[1.25, 2.5]);
    rt.set_data(indexes, &[1.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out), &[2.5], 0.001);
}
