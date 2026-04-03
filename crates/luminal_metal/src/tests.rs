use crate::{kernel::lower_expression_for_metal, runtime::MetalRuntime};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use half::f16;
use luminal::prelude::*;
use proptest::prelude::*;

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
    rt = cx.search(rt, 1);
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
    let mut rt = MetalRuntime::initialize(());

    let a_data = seeded_data(m * k, 0.4, -0.2);
    let b_data = seeded_data(k * n, 0.3, -0.15);

    rt.set_data(a, &a_data);
    rt.set_data(b, &b_data);
    rt = cx.search(rt, 1);

    let kernels = rt.debug_kernel_ops();
    assert!(
        kernels.iter().any(|k| k.contains("family: RegularTiled")),
        "expected regular tiled matmul path, kernels: {:?}",
        kernels
    );

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
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[1.0, 2.0, 99.0, 4.0, 5.0], 0.001);
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
