use half::bf16;
use luminal::{dtype::DType, prelude::*, shape::Expression};

use super::utilities::{assert_close, get_cuda_stream, random_f32_vec};
use crate::{
    host::{
        HostOp,
        moe::{GLUMoE, GLUMoEMode},
    },
    runtime::CudaRuntime,
};

const SEQ: usize = 2;
const HIDDEN: usize = 16;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const MOE_INTERMEDIATE: usize = 6;
const RMS_NORM_EPS: f32 = 1e-6;

struct QwenMoeGraph {
    graph: Graph,
    x: GraphTensor,
    router: GraphTensor,
    gate_up_weights: GraphTensor,
    down_weights: GraphTensor,
    output: GraphTensor,
}

struct GemmaMoeGraph {
    graph: Graph,
    router_input: GraphTensor,
    expert_input: GraphTensor,
    router_scale: GraphTensor,
    router_proj: GraphTensor,
    per_expert_scale: GraphTensor,
    gate_up_weights: GraphTensor,
    down_weights: GraphTensor,
    output: GraphTensor,
}

fn build_qwen_moe_graph() -> QwenMoeGraph {
    let mut cx = Graph::default();
    let x = cx.tensor(('s', HIDDEN));
    let router = cx.tensor((NUM_EXPERTS, HIDDEN));
    let gate_up_weights = cx
        .tensor((NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = x.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let routing_weights = x.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);

    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);

    let gate_up_gathered = gather_experts(x, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = x.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(x, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let output = (down_out * top_k_values.unsqueeze(top_k_values.dims().len()))
        .sum(n - 1)
        .output();

    QwenMoeGraph {
        graph: cx,
        x,
        router,
        gate_up_weights,
        down_weights,
        output,
    }
}

fn build_gemma_moe_graph() -> GemmaMoeGraph {
    let mut cx = Graph::default();
    let router_input = cx.tensor(('s', HIDDEN));
    let expert_input = cx.tensor(('s', HIDDEN));
    let router_scale = cx.tensor(HIDDEN);
    let router_proj = cx.tensor((NUM_EXPERTS, HIDDEN));
    let per_expert_scale = cx.tensor(NUM_EXPERTS);
    let gate_up_weights = cx
        .tensor((NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = router_input.dims().len();
    let e_dim = *router_proj.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let router_hidden = router_input.std_norm(n - 1, RMS_NORM_EPS)
        * router_scale.expand_lhs(&router_input.dims()[..n - 1])
        * (HIDDEN as f32).sqrt().recip();
    let routing_weights = router_hidden.matmul(router_proj.t()).softmax(n - 1);

    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);
    let row_offsets = router_input
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_norm = top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);
    let top_k_weights = (top_k_values / top_k_norm) * per_expert_scale.gather(top_k_indices);

    let gate_up_gathered =
        gather_experts(expert_input, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = expert_input.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gemma_gelu(gate) * up;

    let down_gathered = gather_experts(expert_input, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let output = (down_out * top_k_weights.unsqueeze(top_k_weights.dims().len()))
        .sum(n - 1)
        .output();

    GemmaMoeGraph {
        graph: cx,
        router_input,
        expert_input,
        router_scale,
        router_proj,
        per_expert_scale,
        gate_up_weights,
        down_weights,
        output,
    }
}

fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (axis, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(axis, *dim);
    }
    let expert_flat_idx = exp_base + exp_within;
    weights.gather(expert_flat_idx)
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn glumoe_modes(rt: &CudaRuntime) -> Vec<GLUMoEMode> {
    rt.llir_graph()
        .node_weights()
        .filter_map(|node| {
            let op = node.to_dialect::<dyn HostOp>()?;
            op.as_any()
                .downcast_ref::<GLUMoE>()
                .map(|glumoe| glumoe.mode)
        })
        .collect()
}

fn run_qwen_moe(use_glumoe: bool) -> (Vec<f32>, Vec<GLUMoEMode>) {
    let Some(stream) = get_cuda_stream() else {
        return (vec![], vec![]);
    };

    let mut model = build_qwen_moe_graph();
    model.graph.set_dim('s', SEQ);
    if use_glumoe {
        model.graph.build_search_space::<CudaRuntime>();
    } else {
        model
            .graph
            .build_search_space_exclude_ops::<CudaRuntime, GLUMoE>();
    }

    let x_data = random_f32_vec(SEQ * HIDDEN, 11, -0.15, 0.15);
    let router_data = random_f32_vec(NUM_EXPERTS * HIDDEN, 12, -0.2, 0.2);
    let gate_up_data = random_f32_vec(NUM_EXPERTS * MOE_INTERMEDIATE * 2 * HIDDEN, 13, -0.1, 0.1)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let down_data = random_f32_vec(NUM_EXPERTS * HIDDEN * MOE_INTERMEDIATE, 14, -0.1, 0.1)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(model.x, x_data);
    rt.set_data(model.router, router_data);
    rt.set_data(model.gate_up_weights, gate_up_data);
    rt.set_data(model.down_weights, down_data);
    rt = model.graph.search(rt, 10);
    rt.execute(&model.graph.dyn_map);

    (rt.get_f32(model.output.id), glumoe_modes(&rt))
}

fn run_gemma_moe(use_glumoe: bool) -> (Vec<f32>, Vec<GLUMoEMode>) {
    let Some(stream) = get_cuda_stream() else {
        return (vec![], vec![]);
    };

    let mut model = build_gemma_moe_graph();
    model.graph.set_dim('s', SEQ);
    if use_glumoe {
        model.graph.build_search_space::<CudaRuntime>();
    } else {
        model
            .graph
            .build_search_space_exclude_ops::<CudaRuntime, GLUMoE>();
    }

    let router_input_data = random_f32_vec(SEQ * HIDDEN, 21, -0.15, 0.15);
    let expert_input_data = random_f32_vec(SEQ * HIDDEN, 22, -0.15, 0.15);
    let router_scale_data = random_f32_vec(HIDDEN, 23, 0.7, 1.3);
    let router_proj_data = random_f32_vec(NUM_EXPERTS * HIDDEN, 24, -0.2, 0.2);
    let per_expert_scale_data = random_f32_vec(NUM_EXPERTS, 25, 0.5, 1.5);
    let gate_up_data = random_f32_vec(NUM_EXPERTS * MOE_INTERMEDIATE * 2 * HIDDEN, 26, -0.1, 0.1)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    let down_data = random_f32_vec(NUM_EXPERTS * HIDDEN * MOE_INTERMEDIATE, 27, -0.1, 0.1)
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();

    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(model.router_input, router_input_data);
    rt.set_data(model.expert_input, expert_input_data);
    rt.set_data(model.router_scale, router_scale_data);
    rt.set_data(model.router_proj, router_proj_data);
    rt.set_data(model.per_expert_scale, per_expert_scale_data);
    rt.set_data(model.gate_up_weights, gate_up_data);
    rt.set_data(model.down_weights, down_data);
    rt = model.graph.search(rt, 10);
    rt.execute(&model.graph.dyn_map);

    (rt.get_f32(model.output.id), glumoe_modes(&rt))
}

#[test]
fn test_glumoe_matches_qwen_swiglu_pattern() {
    let (_result, modes) = run_qwen_moe(true);
    if modes.is_empty() {
        return;
    }

    assert_eq!(modes, vec![GLUMoEMode::SwiGLU]);
}

#[test]
fn test_glumoe_matches_gemma_gelu_pattern() {
    let (_result, modes) = run_gemma_moe(true);
    if modes.is_empty() {
        return;
    }

    assert_eq!(modes, vec![GLUMoEMode::GemmaGELU]);
}

#[test]
fn test_glumoe_swiglu_matches_unfused_output() {
    let (expected, baseline_modes) = run_qwen_moe(false);
    if expected.is_empty() {
        return;
    }
    assert!(baseline_modes.is_empty());

    let (actual, fused_modes) = run_qwen_moe(true);
    assert_eq!(fused_modes, vec![GLUMoEMode::SwiGLU]);
    assert_close(&actual, &expected, 3e-2, 3e-2);
}

#[test]
fn test_glumoe_gemma_gelu_matches_unfused_output() {
    let (expected, baseline_modes) = run_gemma_moe(false);
    if expected.is_empty() {
        return;
    }
    assert!(baseline_modes.is_empty());

    let (actual, fused_modes) = run_gemma_moe(true);
    assert_eq!(fused_modes, vec![GLUMoEMode::GemmaGELU]);
    assert_close(&actual, &expected, 3e-2, 3e-2);
}
